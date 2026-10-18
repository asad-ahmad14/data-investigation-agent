"""Analysis operations: run one validated plan step and return structured, JSON-safe results.

Each operation composes one or two compiled aggregate queries with deterministic statistics
(app.analytics.stats). Nothing here calls an LLM. Phase 5 converts these outputs into Evidence
objects; this module only guarantees that every number originates from a query result.
"""

from dataclasses import dataclass, field
from datetime import date

from app.analytics import stats
from app.analytics.compiler import compile_aggregate
from app.analytics.errors import PlanValidationError
from app.analytics.executor import ExecutionResult, QueryExecutor
from app.retail.generate import add_months
from app.retail.registry import METRICS, MetricDef
from app.schemas.plan import AnalysisStep, Period


@dataclass(frozen=True)
class ExecutedQuery:
    label: str
    sql: str
    params: dict
    duration_ms: int
    row_count: int
    truncated: bool

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "sql": self.sql,
            "params": self.params,
            "duration_ms": self.duration_ms,
            "row_count": self.row_count,
            "truncated": self.truncated,
        }


@dataclass
class StepOutput:
    step_id: str
    operation: str
    metric: str
    unit: str
    dimension: str | None
    filters: dict[str, list[str]]
    queries: list[ExecutedQuery]
    result: dict
    caveats: list[str] = field(default_factory=list)
    status: str = "ok"  # "ok" | "no_data"

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "metric": self.metric,
            "unit": self.unit,
            "dimension": self.dimension,
            "filters": self.filters,
            "queries": [q.to_dict() for q in self.queries],
            "result": self.result,
            "caveats": self.caveats,
            "status": self.status,
        }


def _row_value(metric: MetricDef, row: dict) -> float | None:
    if not row.get("n_orders"):
        return None
    if metric.is_ratio:
        return stats.safe_ratio(row["numerator"], row["denominator"])
    return row["numerator"]


def _group(metric: MetricDef, row: dict) -> dict:
    return {
        "group": row.get("dimension"),
        "value": _row_value(metric, row),
        "numerator": row["numerator"],
        "denominator": row["denominator"],
        "n_orders": row["n_orders"],
        "small_sample": stats.is_small_sample(row["n_orders"]),
    }


class _Runner:
    """Executes aggregates for one step and records what was run."""

    def __init__(self, step: AnalysisStep, metric: MetricDef, executor: QueryExecutor) -> None:
        self.step = step
        self.metric = metric
        self.executor = executor
        self.queries: list[ExecutedQuery] = []
        self.caveats: list[str] = []
        self.filters = {f.dimension: list(f.values) for f in step.filters}

    def aggregate(self, label: str, period: Period, dimension: str | None) -> list[dict]:
        query = compile_aggregate(
            metric=self.step.metric,
            dimension=dimension,
            start=period.start,
            end=period.end,
            filters=self.filters,
            max_rows=self.executor.max_rows,
        )
        result: ExecutionResult = self.executor.execute(query)
        self.queries.append(
            ExecutedQuery(
                label,
                result.sql,
                result.params,
                result.duration_ms,
                len(result.rows),
                result.truncated,
            )
        )
        if result.truncated:
            self.caveats.append(
                f"Results for '{label}' were truncated at {self.executor.max_rows} rows."
            )
        return result.rows


def _month_labels(period: Period) -> list[str]:
    labels = []
    cursor = period.start.replace(day=1)
    while cursor < period.end:
        labels.append(cursor.isoformat())
        cursor = add_months(cursor, 1)
    return labels


def _note_small_samples(runner: _Runner, groups: list[dict]) -> None:
    small = [g["group"] for g in groups if g["small_sample"] and g["group"] is not None]
    if small:
        shown = ", ".join(str(g) for g in small[:5])
        more = f" (+{len(small) - 5} more)" if len(small) > 5 else ""
        runner.caveats.append(
            f"Small samples (fewer than {stats.SMALL_SAMPLE_THRESHOLD} orders): {shown}{more}. "
            "Values for these groups are unreliable."
        )


def _note_period_mismatch(runner: _Runner, focus: Period, baseline: Period) -> None:
    if focus.days != baseline.days:
        runner.caveats.append(
            f"The periods have different lengths ({focus.days} vs {baseline.days} days); totals "
            "are not directly comparable. Compare rates or per-day values instead."
        )


def _metric_by_period(runner: _Runner) -> dict:
    step, metric = runner.step, runner.metric
    rows = runner.aggregate("series", step.period, "month")
    by_label = {r["dimension"]: r for r in rows}
    series = []
    for label in _month_labels(step.period):
        row = by_label.get(label)
        series.append(
            {
                "label": label,
                "value": _row_value(metric, row) if row else None,
                "n_orders": row["n_orders"] if row else 0,
            }
        )
    if step.period.start.day != 1 or step.period.end.day != 1:
        runner.caveats.append("The period does not align to month boundaries; partial months.")
    changes = stats.period_over_period([(p["label"], p["value"]) for p in series], unit=metric.unit)
    decline = stats.largest_decline(changes)
    return {
        "series": series,
        "period_over_period": [c.to_dict() for c in changes],
        "largest_decline": decline.to_dict() if decline else None,
    }


def _metric_by_dimension(runner: _Runner) -> dict:
    step = runner.step
    rows = runner.aggregate("groups", step.period, step.dimension)
    groups = [_group(runner.metric, r) for r in rows]
    groups.sort(key=lambda g: (g["value"] is None, -(g["value"] or 0)))
    _note_small_samples(runner, groups)
    return {"period": step.period.label, "groups": groups}


def _compare_periods(runner: _Runner) -> dict:
    step, metric = runner.step, runner.metric
    focus, baseline = step.period, step.comparison_period
    _note_period_mismatch(runner, focus, baseline)
    focus_rows = runner.aggregate("focus", focus, step.dimension)
    base_rows = runner.aggregate("baseline", baseline, step.dimension)

    if step.dimension is None:
        f = _row_value(metric, focus_rows[0]) if focus_rows else None
        b = _row_value(metric, base_rows[0]) if base_rows else None
        return {
            "focus_period": focus.label,
            "baseline_period": baseline.label,
            "overall": {
                "n_orders_focus": focus_rows[0]["n_orders"] if focus_rows else 0,
                "n_orders_baseline": base_rows[0]["n_orders"] if base_rows else 0,
                **stats.compare_values(b, f, unit=metric.unit).to_dict(),
            },
        }

    focus_by = {r["dimension"]: r for r in focus_rows}
    base_by = {r["dimension"]: r for r in base_rows}
    groups = []
    for name in sorted(set(focus_by) | set(base_by)):
        fr, br = focus_by.get(name), base_by.get(name)
        fv = _row_value(metric, fr) if fr else None
        bv = _row_value(metric, br) if br else None
        n_focus = fr["n_orders"] if fr else 0
        n_base = br["n_orders"] if br else 0
        groups.append(
            {
                "group": name,
                "n_orders_focus": n_focus,
                "n_orders_baseline": n_base,
                "small_sample": stats.is_small_sample(min(n_focus, n_base)),
                **stats.compare_values(bv, fv, unit=metric.unit).to_dict(),
            }
        )
    groups.sort(key=lambda g: (g["absolute_change"] is None, -abs(g["absolute_change"] or 0)))
    _note_small_samples(runner, groups)
    return {"focus_period": focus.label, "baseline_period": baseline.label, "groups": groups}


def _contribution_to_change(runner: _Runner) -> dict:
    step, metric = runner.step, runner.metric
    focus, baseline = step.period, step.comparison_period
    _note_period_mismatch(runner, focus, baseline)
    after_rows = runner.aggregate("focus", focus, step.dimension)
    before_rows = runner.aggregate("baseline", baseline, step.dimension)
    after = {r["dimension"]: _row_value(metric, r) for r in after_rows}
    before = {r["dimension"]: _row_value(metric, r) for r in before_rows}
    result = stats.contribution_to_change(before, after)
    if result.mixed_direction:
        runner.caveats.append(
            "Groups moved in opposite directions, so individual shares can exceed 100% or be "
            "negative."
        )
    return {
        "focus_period": focus.label,
        "baseline_period": baseline.label,
        "dimension": step.dimension,
        **result.to_dict(),
    }


_DISPATCH = {
    "metric_by_period": _metric_by_period,
    "metric_by_dimension": _metric_by_dimension,
    "compare_periods": _compare_periods,
    "contribution_to_change": _contribution_to_change,
}


def run_step(step: AnalysisStep, executor: QueryExecutor) -> StepOutput:
    metric = METRICS.get(step.metric)
    handler = _DISPATCH.get(step.operation)
    if metric is None or handler is None:
        raise PlanValidationError("The analysis step is not allowed.")
    if step.operation != "metric_by_period" and step.comparison_period is None:
        if step.operation in ("compare_periods", "contribution_to_change"):
            raise PlanValidationError("A comparison_period is required for this operation.")

    runner = _Runner(step, metric, executor)
    result = handler(runner)
    has_data = any(q.row_count > 0 for q in runner.queries)
    return StepOutput(
        step_id=step.id,
        operation=step.operation,
        metric=metric.name,
        unit=metric.unit,
        dimension=step.dimension,
        filters=runner.filters,
        queries=runner.queries,
        result=result,
        caveats=runner.caveats,
        status="ok" if has_data else "no_data",
    )


__all__ = ["ExecutedQuery", "StepOutput", "run_step", "date"]
