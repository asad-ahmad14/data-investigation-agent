"""The LangGraph workflow without a database: run_step is replaced by a canned fake.

These tests pin down routing, limits and failure handling. Numbers in the real pipeline are
checked against pandas in test_agent_integration.py.
"""

import time

import pytest

from app.agent import graph as graph_module
from app.agent.runner import run_investigation
from app.analytics import stats
from app.analytics.errors import DatabaseUnavailableError, QueryTimeoutError
from app.analytics.operations import ExecutedQuery, StepOutput
from app.core.config import Settings
from app.llm.errors import LLMOutputError, LLMTimeoutError
from tests.test_llm_service import ScriptedLLM

FOCUS = "2025-06-01 to 2025-07-01 (end exclusive)"
BASE = "2025-05-01 to 2025-06-01 (end exclusive)"


def _period(start, end):
    return {"start": start, "end": end}


def series_plan():
    return {
        "outcome": "plan",
        "plan": {
            "steps": [
                {
                    "id": "S1",
                    "operation": "metric_by_period",
                    "metric": "net_revenue",
                    "period": _period("2025-01-01", "2025-09-01"),
                }
            ]
        },
    }


def compare_plan(dims=("region", "category")):
    steps = [
        {
            "id": f"S{i}",
            "operation": "contribution_to_change",
            "metric": "net_revenue",
            "dimension": dim,
            "period": _period("2025-06-01", "2025-07-01"),
            "comparison_period": _period("2025-05-01", "2025-06-01"),
        }
        for i, dim in enumerate(dims, 1)
    ]
    hyps = [
        {"id": f"H{i}", "statement": f"Concentrated by {d}", "step_ids": [f"S{i}"]}
        for i, d in enumerate(dims, 1)
    ]
    return {"outcome": "plan", "plan": {"steps": steps, "hypotheses": hyps}}


def fake_run_step(step, executor):
    """Canned StepOutput shaped like the real operations' results."""
    queries = [ExecutedQuery("q", "SELECT 1", {}, 1, 3, False)] * (
        1 if step.operation == "metric_by_period" else 2
    )
    if step.operation == "metric_by_period":
        series = [
            {"label": "2025-05-01", "value": 100.0, "n_orders": 50},
            {"label": "2025-06-01", "value": 60.0, "n_orders": 40},
        ]
        changes = stats.period_over_period(
            [(p["label"], p["value"]) for p in series], unit="currency"
        )
        result = {
            "series": series,
            "largest_decline": stats.largest_decline(changes).to_dict(),
        }
    else:
        before, after = {"West": 50.0, "East": 50.0}, {"West": 10.0, "East": 50.0}
        result = {
            "focus_period": FOCUS,
            "baseline_period": BASE,
            "dimension": step.dimension,
            **stats.contribution_to_change(before, after).to_dict(),
        }
    return StepOutput(
        step.id, step.operation, step.metric, "currency", step.dimension, {}, queries, result
    )


@pytest.fixture(autouse=True)
def canned_steps(monkeypatch):
    monkeypatch.setattr(graph_module, "run_step", fake_run_step)


def settings(**kw):
    return Settings(_env_file=None, **kw)


def run(*replies, question="why did net revenue drop?", **kw):
    llm = ScriptedLLM(*replies)
    report, events = run_investigation(
        question, settings=settings(**kw), analyst_engine=object(), llm_client=llm
    )
    return report, events, llm


def test_two_round_investigation_finds_when_then_where():
    report, events, llm = run(series_plan(), compare_plan())
    assert report.status == "completed"
    assert [e["node"] for e in events] == [
        "plan", "execute", "evaluate", "plan", "execute", "evaluate", "report",
    ]  # fmt: skip
    assert report.stats.iterations == 2 and report.stats.llm_calls == 2
    assert report.stats.queries_run == 1 + 2 + 2
    # The second planning call received the trusted finding from round one.
    assert "largest_decline" in llm.requests[1].user and "May 2025" in llm.requests[1].user
    assert [h.id for h in report.hypotheses] == ["H1", "H2"]
    assert all(h.status == "supported" for h in report.hypotheses)
    keys = [e["key"] for e in report.evidence]
    assert keys == ["E1", "E2", "E3"]
    # every cited key exists
    cited = {k for f in report.findings for k in f.evidence_keys}
    assert cited <= set(keys)
    assert "limitations" and report.limitations


def test_single_round_when_plan_already_compares_periods():
    report, events, _ = run(compare_plan(("region",)))
    assert report.status == "completed"
    assert report.stats.iterations == 1
    assert [e["node"] for e in events].count("plan") == 1


def test_iteration_limit_stops_follow_up():
    report, events, _ = run(series_plan(), max_iterations=1)
    assert report.status == "completed"
    assert report.stats.iterations == 1
    assert any(e["detail"].get("reason") == "iteration limit" for e in events)


def test_clarification_ends_run_without_running_queries():
    reply = {"outcome": "needs_clarification", "message": "Which metric?"}
    report, events, _ = run(reply)
    assert report.status == "needs_clarification"
    assert report.message == "Which metric?"
    assert report.stats.queries_run == 0
    assert [e["node"] for e in events] == ["plan", "report"]


def test_out_of_scope_is_reported_as_needs_clarification_with_message():
    report, _, _ = run({"outcome": "out_of_scope", "message": "Not about this dataset."})
    assert report.status == "needs_clarification"
    assert report.message == "Not about this dataset."


def test_empty_question_is_rejected_before_any_llm_call():
    report, _, llm = run(question="   ")
    assert report.status == "needs_clarification"
    assert llm.requests == []


@pytest.mark.parametrize(
    ("error", "code"),
    [(LLMTimeoutError(), "llm_timeout"), (LLMOutputError(), "llm_invalid_output")],
)
def test_llm_failure_becomes_failed_report_with_safe_message(error, code):
    report, _, _ = run(error)
    assert report.status == "failed" and report.error_code == code
    assert report.message == error.public_message


def test_followup_planning_failure_keeps_first_round_results():
    report, _, _ = run(series_plan(), LLMTimeoutError())
    assert report.status == "completed"
    assert any("follow-up round could not be planned" in c for c in report.caveats)
    assert report.error_code is None


def test_followup_that_declines_keeps_first_round_results():
    report, _, _ = run(series_plan(), {"outcome": "out_of_scope", "message": "no"})
    assert report.status == "completed"
    assert any("No follow-up analysis" in c for c in report.caveats)


def test_database_unavailable_fails_the_run(monkeypatch):
    def boom(step, executor):
        raise DatabaseUnavailableError()

    monkeypatch.setattr(graph_module, "run_step", boom)
    report, events, _ = run(series_plan())
    assert report.status == "failed" and report.error_code == "database_unavailable"
    assert report.steps[0].status == "failed"
    assert "SELECT" not in (report.message or "")


def test_one_failed_step_does_not_stop_the_others(monkeypatch):
    def flaky(step, executor):
        if step.dimension == "region":
            raise QueryTimeoutError()
        return fake_run_step(step, executor)

    monkeypatch.setattr(graph_module, "run_step", flaky)
    report, _, _ = run(compare_plan(("region", "category")))
    assert report.status == "completed"
    statuses = {s.id: s.status for s in report.steps}
    assert statuses == {"S1": "failed", "S2": "ok"}
    by_id = {h.id: h for h in report.hypotheses}
    assert by_id["H1"].status == "not_testable"
    assert by_id["H2"].status == "supported"
    assert any("S1 could not be completed" in c for c in report.caveats)


def test_query_budget_is_shared_across_rounds_and_skips_steps():
    # Round 1 spends 1 query; round 2's plan fits the budget on its own (4) but not what is left.
    report, _, _ = run(series_plan(), compare_plan(("region", "category")), max_total_queries=4)
    assert [(s.iteration, s.status) for s in report.steps] == [(0, "ok"), (1, "ok"), (1, "skipped")]
    assert any("query budget" in c for c in report.caveats)
    assert report.stats.queries_run == 3


def test_plan_that_exceeds_the_budget_is_rejected_by_validation_not_run():
    big = compare_plan(("region", "category", "customer_segment"))  # cost 6
    report, _, llm = run(big, big, max_total_queries=4)
    assert report.status == "failed" and report.error_code == "llm_invalid_output"
    assert report.stats.queries_run == 0
    assert len(llm.requests) == 2  # original + one repair attempt


def test_time_limit_before_execution_fails_with_timeout(monkeypatch):
    real = time.monotonic
    start = real()
    # After planning, the clock jumps past the deadline.
    ticks = iter([start, start + 1000])
    monkeypatch.setattr(
        graph_module.time, "monotonic", lambda: next(ticks, start + 1000), raising=True
    )
    report, _, _ = run(compare_plan(("region",)), investigation_timeout_seconds=5)
    assert report.status == "failed"
    assert report.error_code == "timeout"


def test_unexpected_crash_returns_generic_failure(monkeypatch):
    def bug(step, executor):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(graph_module, "run_step", bug)
    report, events, _ = run(series_plan())
    assert report.status == "failed" and report.error_code == "internal_error"
    assert "secret" not in report.model_dump_json()
    assert events == []
