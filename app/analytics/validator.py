"""Plan validation: checks an LLM-produced plan against the registry and resource limits.

This is the policy gate in front of the compiler. It never trusts names from the plan: a metric,
dimension or filter must exist in app.retail.registry. The compiler re-checks the same things
(defence in depth), and the database role is the final backstop.

Limits of this module: it validates STRUCTURE and PERMISSIONS of a plan. It does not judge
whether the plan is a good investigation; that is the planner's and evaluator's job.
"""

from dataclasses import dataclass

from app.retail.registry import (
    DIMENSIONS,
    FILTERABLE_DIMENSIONS,
    METRICS,
    OPERATION_QUERY_COST,
)
from app.schemas.plan import AnalysisStep, InvestigationPlan


@dataclass(frozen=True)
class PlanIssue:
    code: str
    step_id: str | None
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "step_id": self.step_id, "message": self.message}


def estimate_query_cost(plan: InvestigationPlan) -> int:
    return sum(OPERATION_QUERY_COST[s.operation] for s in plan.steps)


def _validate_step(step: AnalysisStep) -> list[PlanIssue]:
    issues: list[PlanIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(PlanIssue(code, step.id, message))

    metric = METRICS.get(step.metric)
    if metric is None:
        add("unknown_metric", f"Metric '{step.metric}' is not available.")

    if step.dimension is not None and step.dimension not in DIMENSIONS:
        add("unknown_dimension", f"Dimension '{step.dimension}' is not available.")

    op = step.operation
    if op == "metric_by_period":
        if step.dimension not in (None, "month"):
            add("bad_dimension", "metric_by_period groups by month; omit dimension.")
        if step.comparison_period is not None:
            add("unexpected_comparison", "metric_by_period does not take a comparison_period.")
    elif op == "metric_by_dimension":
        if step.dimension is None:
            add("missing_dimension", "metric_by_dimension requires a dimension.")
        if step.comparison_period is not None:
            add("unexpected_comparison", "metric_by_dimension does not take a comparison_period.")
    elif op in ("compare_periods", "contribution_to_change"):
        if step.comparison_period is None:
            add("missing_comparison", f"{op} requires a comparison_period (the baseline).")
        elif _overlap(step):
            add("periods_overlap", "period and comparison_period must not overlap.")
        if op == "contribution_to_change":
            if step.dimension is None:
                add("missing_dimension", "contribution_to_change requires a dimension.")
            elif step.dimension == "month":
                add("bad_dimension", "contribution_to_change cannot group by month.")
            if metric is not None:
                if metric.is_ratio:
                    add(
                        "metric_not_additive",
                        f"'{metric.name}' is a ratio; its change cannot be split additively "
                        "across groups.",
                    )
                elif (
                    step.dimension is not None and step.dimension not in metric.additive_dimensions
                ):
                    add(
                        "metric_not_additive",
                        f"'{metric.name}' does not add up across '{step.dimension}' groups.",
                    )

    seen_filter_dims: set[str] = set()
    for f in step.filters:
        if f.dimension not in FILTERABLE_DIMENSIONS:
            add("bad_filter_dimension", f"Cannot filter on '{f.dimension}'.")
        if f.dimension in seen_filter_dims:
            add("duplicate_filter", f"Dimension '{f.dimension}' is filtered more than once.")
        seen_filter_dims.add(f.dimension)
    return issues


def _overlap(step: AnalysisStep) -> bool:
    a, b = step.period, step.comparison_period
    return a.start < b.end and b.start < a.end


def validate_plan(
    plan: InvestigationPlan, *, max_steps: int, max_total_queries: int
) -> list[PlanIssue]:
    """Return every problem found; an empty list means the plan may be executed."""
    issues: list[PlanIssue] = []

    if len(plan.steps) > max_steps:
        issues.append(
            PlanIssue(
                "too_many_steps", None, f"Plan has {len(plan.steps)} steps; max is {max_steps}."
            )
        )

    ids = [s.id for s in plan.steps]
    if len(set(ids)) != len(ids):
        issues.append(PlanIssue("duplicate_step_id", None, "Step ids must be unique."))

    for step in plan.steps:
        issues.extend(_validate_step(step))

    cost = estimate_query_cost(plan)
    if cost > max_total_queries:
        issues.append(
            PlanIssue(
                "query_budget_exceeded",
                None,
                f"Plan needs about {cost} queries; the limit is {max_total_queries}.",
            )
        )

    known = set(ids)
    for h in plan.hypotheses:
        missing = [sid for sid in h.step_ids if sid not in known]
        if missing:
            issues.append(
                PlanIssue(
                    "hypothesis_unknown_step",
                    None,
                    f"Hypothesis {h.id} references unknown steps: {', '.join(missing)}.",
                )
            )
    return issues
