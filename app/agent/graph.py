"""The investigation workflow as a LangGraph state machine.

        START -> plan -> execute -> evaluate -+-> plan      (follow-up round, bounded)
                  |                           |
                  +--(no plan / stop)---------+-> report -> END

- plan:     the LLM proposes a structured plan (validated against the registry). It may also ask
            for clarification or decline; those end the run with a clear message.
- execute:  runs each approved step through the controlled analytics layer, one query budget
            shared across rounds. Step failures are recorded; the run continues where it can.
- evaluate: deterministic. Judges hypotheses from evidence and decides whether a follow-up
            round is worthwhile (for example: we found WHEN a metric fell but not WHERE).
- report:   assembles the final report from evidence only. No numbers come from the model.

Hard limits (all from Settings): rounds, queries, LLM calls, wall-clock time. When one is hit the
run finishes with what it has and says so in the caveats.
"""

import logging
import re
import time
from dataclasses import dataclass, replace
from datetime import date
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.evaluation import build_findings, evaluate_hypotheses
from app.agent.evidence import EvidenceItem, extract_evidence
from app.agent.report import (
    InvestigationReport,
    ReportHypothesis,
    ReportStep,
    RunStats,
)
from app.analytics.errors import AnalyticsError, DatabaseUnavailableError
from app.analytics.executor import QueryExecutor
from app.analytics.operations import run_step
from app.core.config import Settings
from app.llm.errors import (
    InvalidQuestionError,
    LLMAuthError,
    LLMBlockedError,
    LLMBudgetExceededError,
    LLMConfigError,
    LLMError,
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.service import LLMService
from app.retail.generate import add_months
from app.retail.registry import OPERATION_QUERY_COST
from app.schemas.plan import InvestigationPlan

logger = logging.getLogger(__name__)

LIMITATIONS = [
    "Findings show where a change is concentrated in the data, not why it happened; the "
    "dataset has no causal information.",
    "Hypothesis verdicts use fixed heuristic thresholds (see app/agent/evaluation.py), not "
    "statistical significance tests.",
    "The dataset is synthetic and covers a fixed 24-month window.",
]

_LLM_ERROR_CODES: dict[type[LLMError], str] = {
    LLMTimeoutError: "llm_timeout",
    LLMUnavailableError: "llm_unavailable",
    LLMAuthError: "llm_auth",
    LLMBlockedError: "llm_blocked",
    LLMOutputError: "llm_invalid_output",
    LLMBudgetExceededError: "llm_budget_exceeded",
    LLMConfigError: "llm_config",
}


@dataclass
class Dependencies:
    llm: LLMService
    executor: QueryExecutor
    settings: Settings
    data_start: date
    data_end_exclusive: date
    deadline: float  # time.monotonic() value after which no new work starts


class InvestigationState(TypedDict, total=False):
    question: str
    started: float
    iteration: int  # 0-based index of the CURRENT/most recent round
    current_plan: InvestigationPlan | None
    plan_outcome: str | None  # plan | needs_clarification | out_of_scope | error
    plan_message: str | None
    prior_findings: list[dict]
    queries_run: int
    stop: bool
    error_code: str | None
    error_message: str | None
    evidence: Annotated[list[EvidenceItem], add]
    hypotheses: Annotated[list[ReportHypothesis], add]
    steps: Annotated[list[ReportStep], add]
    caveats: Annotated[list[str], add]
    events: Annotated[list[dict], add]
    report: InvestigationReport | None


def _event(node: str, started: float, status: str, **detail: Any) -> dict:
    return {
        "node": node,
        "status": status,
        "duration_ms": round((time.monotonic() - started) * 1000),
        "detail": detail,
    }


def build_graph(deps: Dependencies):
    settings = deps.settings

    # ------------------------------------------------------------------- plan
    def plan_node(state: InvestigationState) -> dict:
        t0 = time.monotonic()
        iteration = state.get("iteration", -1) + 1
        followup = iteration > 0
        try:
            reply = deps.llm.plan(
                state["question"],
                data_start=deps.data_start,
                data_end_exclusive=deps.data_end_exclusive,
                prior_findings=state.get("prior_findings") or None,
            )
        except InvalidQuestionError as exc:
            return {
                "iteration": iteration,
                "current_plan": None,
                "plan_outcome": "needs_clarification",
                "plan_message": exc.public_message,
                "events": [_event("plan", t0, "rejected", iteration=iteration)],
            }
        except LLMError as exc:
            code = _LLM_ERROR_CODES.get(type(exc), "llm_error")
            logger.warning("plan_failed", extra={"error_code": code, "iteration": iteration})
            update: dict = {
                "iteration": iteration,
                "current_plan": None,
                "plan_outcome": "error",
                "events": [_event("plan", t0, "failed", iteration=iteration, error_code=code)],
            }
            if followup:
                update["caveats"] = [
                    "A follow-up round could not be planned "
                    f"({exc.public_message}); results are from earlier rounds only."
                ]
            else:
                update["error_code"] = code
                update["error_message"] = exc.public_message
            return update

        if reply.outcome != "plan":
            update = {
                "iteration": iteration,
                "current_plan": None,
                "plan_outcome": reply.outcome,
                "plan_message": reply.message,
                "events": [_event("plan", t0, reply.outcome, iteration=iteration)],
            }
            if followup:
                update["caveats"] = [
                    "No follow-up analysis was proposed; results are from earlier rounds only."
                ]
            return update

        assert reply.plan is not None
        return {
            "iteration": iteration,
            "current_plan": reply.plan,
            "plan_outcome": "plan",
            "events": [
                _event(
                    "plan",
                    t0,
                    "ok",
                    iteration=iteration,
                    steps=len(reply.plan.steps),
                    hypotheses=len(reply.plan.hypotheses),
                )
            ],
        }

    def route_after_plan(state: InvestigationState) -> str:
        return "execute" if state.get("plan_outcome") == "plan" else "report"

    # ---------------------------------------------------------------- execute
    def execute_node(state: InvestigationState) -> dict:
        t0 = time.monotonic()
        plan = state["current_plan"]
        iteration = state["iteration"]
        assert plan is not None
        queries_run = state.get("queries_run", 0)
        next_index = len(state.get("evidence", [])) + 1
        evidence: list[EvidenceItem] = []
        steps: list[ReportStep] = []
        caveats: list[str] = []
        update: dict = {}

        def record(step, status, note=None):
            steps.append(
                ReportStep(
                    iteration=iteration,
                    id=step.id,
                    operation=step.operation,
                    metric=step.metric,
                    dimension=step.dimension,
                    purpose=step.purpose,
                    status=status,
                    note=note,
                )
            )

        for step in plan.steps:
            cost = OPERATION_QUERY_COST[step.operation]
            if time.monotonic() > deps.deadline:
                record(step, "skipped", "time limit reached")
                if not any("time limit" in c for c in caveats):
                    caveats.append("The time limit was reached; some steps were not run.")
                continue
            if queries_run + cost > settings.max_total_queries:
                record(step, "skipped", "query budget reached")
                if not any("query budget" in c for c in caveats):
                    caveats.append("The query budget was reached; some steps were not run.")
                continue
            try:
                out = run_step(step, deps.executor)
            except DatabaseUnavailableError as exc:
                record(step, "failed", exc.public_message)
                update["error_code"] = "database_unavailable"
                update["error_message"] = exc.public_message
                update["stop"] = True
                break
            except AnalyticsError as exc:
                queries_run += cost
                record(step, "failed", exc.public_message)
                caveats.append(f"Step {step.id} could not be completed: {exc.public_message}")
                continue

            queries_run += len(out.queries)
            record(step, out.status)
            caveats.extend(f"{out.step_id}: {c}" for c in out.caveats)
            for item in extract_evidence(out, next_index):
                source = {**item.source_operation, "iteration": iteration}
                evidence.append(replace(item, source_operation=source))
                next_index += 1

        update.update(
            {
                "queries_run": queries_run,
                "evidence": evidence,
                "steps": steps,
                "caveats": caveats,
                "events": [
                    _event(
                        "execute",
                        t0,
                        "failed" if update.get("stop") else "ok",
                        iteration=iteration,
                        steps_run=sum(1 for s in steps if s.status in ("ok", "no_data")),
                        evidence_items=len(evidence),
                        queries_run=queries_run,
                    )
                ],
            }
        )
        return update

    # --------------------------------------------------------------- evaluate
    def evaluate_node(state: InvestigationState) -> dict:
        t0 = time.monotonic()
        plan = state["current_plan"]
        iteration = state["iteration"]
        assert plan is not None
        this_round = [
            e for e in state.get("evidence", []) if e.source_operation.get("iteration") == iteration
        ]
        by_step: dict[str, list[EvidenceItem]] = {}
        for item in this_round:
            by_step.setdefault(item.step_id, []).append(item)
        failed = {
            s.id
            for s in state.get("steps", [])
            if s.iteration == iteration and s.status == "failed"
        }
        hypotheses = evaluate_hypotheses(
            list(plan.hypotheses),
            by_step,
            failed,
            first_number=len(state.get("hypotheses", [])) + 1,
        )

        follow_up, prior, reason = _decide_follow_up(state, plan, this_round)
        update: dict = {
            "hypotheses": hypotheses,
            "stop": not follow_up,
            "prior_findings": prior if follow_up else state.get("prior_findings", []),
            "events": [
                _event(
                    "evaluate",
                    t0,
                    "continue" if follow_up else "finish",
                    iteration=iteration,
                    reason=reason,
                    hypotheses={h.id: h.status for h in hypotheses},
                )
            ],
        }
        return update

    def _decide_follow_up(
        state: InvestigationState, plan: InvestigationPlan, this_round: list[EvidenceItem]
    ) -> tuple[bool, list[dict], str]:
        if state.get("stop"):
            return False, [], "run was stopped"
        if state.get("iteration", 0) + 1 >= settings.max_iterations:
            return False, [], "iteration limit"
        if time.monotonic() > deps.deadline:
            return False, [], "time limit"
        if settings.max_total_queries - state.get("queries_run", 0) < 2:
            return False, [], "query budget"
        if deps.llm.usage.calls >= settings.max_llm_calls:
            return False, [], "llm call budget"
        if any(s.comparison_period is not None for s in plan.steps):
            return False, [], "plan already compared periods"
        declines = [
            e
            for e in this_round
            if e.detail.get("kind") == "largest_decline" and e.value is not None
        ]
        if not declines:
            return False, [], "nothing to follow up"
        prior = []
        for e in declines:
            focus_start = date.fromisoformat(e.detail["focus_month"])
            base_start = date.fromisoformat(e.detail["baseline_month"])
            prior.append(
                {
                    "evidence_key": e.key,
                    "metric": e.metric,
                    "finding": e.description,
                    "largest_decline": {
                        "focus": {
                            "start": focus_start.isoformat(),
                            "end": add_months(focus_start, 1).isoformat(),
                        },
                        "baseline": {
                            "start": base_start.isoformat(),
                            "end": add_months(base_start, 1).isoformat(),
                        },
                    },
                }
            )
        return True, prior, "found when the metric fell; asking where"

    def route_after_evaluate(state: InvestigationState) -> str:
        return "report" if state.get("stop") else "plan"

    # ----------------------------------------------------------------- report
    def report_node(state: InvestigationState) -> dict:
        t0 = time.monotonic()
        report = _assemble_report(state, deps)
        return {
            "report": report,
            "events": [_event("report", t0, report.status, evidence=len(report.evidence))],
        }

    graph = StateGraph(InvestigationState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("report", report_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", route_after_plan, {"execute": "execute", "report": "report"}
    )
    graph.add_edge("execute", "evaluate")
    graph.add_conditional_edges(
        "evaluate", route_after_evaluate, {"plan": "plan", "report": "report"}
    )
    graph.add_edge("report", END)
    return graph.compile()


def _assemble_report(state: InvestigationState, deps: Dependencies) -> InvestigationReport:
    evidence = state.get("evidence", [])
    usage = deps.llm.usage
    stats = RunStats(
        iterations=len({st.iteration for st in state.get("steps", [])}),
        queries_run=state.get("queries_run", 0),
        llm_calls=usage.calls,
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        model=usage.model,
        duration_ms=round((time.monotonic() - state["started"]) * 1000),
    )
    base = {
        "question": state["question"],
        "steps": state.get("steps", []),
        "caveats": _group_caveats(state.get("caveats", [])),
        "stats": stats,
    }

    if state.get("error_code"):
        return InvestigationReport(
            status="failed",
            error_code=state["error_code"],
            message=state.get("error_message") or "The investigation could not be completed.",
            **base,
        )
    outcome = state.get("plan_outcome")
    if not evidence and outcome in ("needs_clarification", "out_of_scope"):
        return InvestigationReport(
            status="needs_clarification",
            message=state.get("plan_message") or "Please rephrase your question.",
            **base,
        )

    usable = [e for e in evidence if e.result_status not in ("no_data",)]
    if not usable:
        timed_out = time.monotonic() > deps.deadline
        if timed_out:
            return InvestigationReport(
                status="failed",
                error_code="timeout",
                message="The investigation ran out of time before producing results.",
                **base,
            )
        return InvestigationReport(
            status="insufficient_evidence",
            message="The analysis ran but returned no usable data for this question.",
            limitations=LIMITATIONS,
            evidence=[e.to_dict() for e in evidence],
            **base,
        )

    hypotheses = state.get("hypotheses", [])
    findings = build_findings(evidence, hypotheses)
    by_key = {e.key: e for e in evidence}
    has_decline = any(e.detail.get("kind") == "largest_decline" for e in evidence)
    headline = [
        f
        for f in findings
        if not (has_decline and by_key[f.evidence_keys[0]].detail.get("kind") == "overall_change")
    ]
    summary = " ".join(f.text for f in headline[:3]) or "No notable change was found."
    return InvestigationReport(
        status="completed",
        summary=summary,
        findings=findings,
        hypotheses=hypotheses,
        evidence=[e.to_dict() for e in evidence],
        limitations=LIMITATIONS,
        **base,
    )


_STEP_PREFIX = re.compile(r"^(S\d+): (.*)$")


def _group_caveats(caveats: list[str]) -> list[str]:
    """Merge identical step caveats: 'S2: x', 'S3: x' -> 'S2, S3: x'. Order of first use kept."""
    grouped: dict[str, list[str]] = {}
    for caveat in caveats:
        match = _STEP_PREFIX.match(caveat)
        text, step = (match[2], match[1]) if match else (caveat, "")
        ids = grouped.setdefault(text, [])
        if step and step not in ids:
            ids.append(step)
    return [f"{', '.join(ids)}: {text}" if ids else text for text, ids in grouped.items()]
