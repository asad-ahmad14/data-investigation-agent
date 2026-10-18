"""Operation logic with a fake executor: no database needed."""

from datetime import date

import pytest

from app.analytics.errors import PlanValidationError
from app.analytics.executor import ExecutionResult
from app.analytics.operations import run_step
from app.schemas.plan import AnalysisStep, Period


class FakeExecutor:
    """Returns canned rows in call order; records the compiled queries it was given."""

    def __init__(self, *responses, max_rows=500, truncated=False):
        self.responses = list(responses)
        self.max_rows = max_rows
        self.truncated = truncated
        self.queries = []

    def execute(self, query):
        self.queries.append(query)
        rows = self.responses.pop(0)
        return ExecutionResult(rows, self.truncated, 1, query.sql_text(), query.params())


def P(a, b):
    return Period(start=date.fromisoformat(a), end=date.fromisoformat(b))


def row(dim, num, den=None, n=100):
    return {"dimension": dim, "numerator": num, "denominator": den, "n_orders": n}


def mk(operation, **kw):
    base = {
        "id": "S1",
        "operation": operation,
        "metric": "net_revenue",
        "period": P("2025-02-01", "2025-03-01"),
    }
    base.update(kw)
    return AnalysisStep(**base)


def test_metric_by_period_fills_gaps_with_none_not_zero():
    step = mk("metric_by_period", period=P("2025-01-01", "2025-04-01"))
    ex = FakeExecutor([row("2025-01-01", 100.0), row("2025-03-01", 50.0)])
    out = run_step(step, ex)
    values = [p["value"] for p in out.result["series"]]
    assert values == [100.0, None, 50.0]
    assert out.result["largest_decline"] is None  # gaps give no_data, never a fabricated drop


def test_metric_by_period_reports_largest_decline():
    step = mk("metric_by_period", period=P("2025-01-01", "2025-04-01"))
    ex = FakeExecutor([row("2025-01-01", 100.0), row("2025-02-01", 120.0), row("2025-03-01", 60.0)])
    decline = run_step(step, ex).result["largest_decline"]
    assert (decline["previous_label"], decline["label"]) == ("2025-02-01", "2025-03-01")


def test_partial_month_caveat():
    step = mk("metric_by_period", period=P("2025-01-15", "2025-03-01"))
    out = run_step(step, FakeExecutor([row("2025-01-01", 1.0), row("2025-02-01", 2.0)]))
    assert any("month boundaries" in c for c in out.caveats)


def test_ratio_metric_uses_numerator_over_denominator_and_guards_zero():
    step = mk("metric_by_dimension", metric="refund_rate", dimension="category")
    ex = FakeExecutor([row("Apparel", 20.0, 200.0), row("Home", 5.0, 0.0)])
    groups = {g["group"]: g["value"] for g in run_step(step, ex).result["groups"]}
    assert groups["Apparel"] == pytest.approx(0.10)
    assert groups["Home"] is None


def test_small_sample_groups_are_flagged_with_caveat():
    step = mk("metric_by_dimension", dimension="region")
    ex = FakeExecutor([row("West", 10.0, n=5), row("East", 99.0, n=500)])
    out = run_step(step, ex)
    assert any("Small samples" in c and "West" in c for c in out.caveats)


def test_compare_periods_overall_and_period_length_caveat():
    step = mk(
        "compare_periods",
        period=P("2025-02-01", "2025-03-01"),  # 28 days
        comparison_period=P("2025-01-01", "2025-02-01"),  # 31 days
    )
    ex = FakeExecutor([row(None, 90.0)], [row(None, 100.0)])
    out = run_step(step, ex)
    overall = out.result["overall"]
    assert overall["old"] == 100.0 and overall["new"] == 90.0
    assert overall["relative_change"] == pytest.approx(-0.10)
    assert any("different lengths" in c for c in out.caveats)


def test_compare_refund_rate_reports_percentage_points():
    step = mk(
        "compare_periods",
        metric="refund_rate",
        period=P("2025-02-01", "2025-03-01"),
        comparison_period=P("2025-01-01", "2025-02-01"),
    )
    ex = FakeExecutor([row(None, 8.0, 100.0)], [row(None, 5.0, 100.0)])
    overall = run_step(step, ex).result["overall"]
    assert overall["percentage_point_change"] == pytest.approx(3.0)


def test_contribution_to_change_by_region_and_mixed_direction_caveat():
    step = mk(
        "contribution_to_change",
        dimension="region",
        period=P("2025-02-01", "2025-03-01"),
        comparison_period=P("2025-01-01", "2025-02-01"),
    )
    ex = FakeExecutor(
        [row("West", 40.0), row("East", 120.0)], [row("West", 100.0), row("East", 100.0)]
    )
    out = run_step(step, ex)
    top = out.result["contributions"][0]
    assert top["group"] == "West" and top["change"] == -60.0
    assert any("opposite directions" in c for c in out.caveats)


def test_no_rows_gives_no_data_status():
    out = run_step(mk("metric_by_dimension", dimension="region"), FakeExecutor([]))
    assert out.status == "no_data"


def test_truncation_is_reported_not_hidden():
    out = run_step(
        mk("metric_by_dimension", dimension="product"),
        FakeExecutor([row("A", 1.0)], max_rows=1, truncated=True),
    )
    assert any("truncated" in c for c in out.caveats)
    assert out.queries[0].truncated is True


def test_run_step_refuses_disallowed_metric_and_missing_baseline():
    with pytest.raises(PlanValidationError):
        run_step(
            mk("metric_by_dimension", metric="profit_margin", dimension="region"), FakeExecutor([])
        )
    with pytest.raises(PlanValidationError):
        run_step(mk("compare_periods"), FakeExecutor([], []))
