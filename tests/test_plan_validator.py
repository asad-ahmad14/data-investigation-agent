from datetime import date

import pytest
from pydantic import ValidationError

from app.analytics.validator import validate_plan
from app.schemas.plan import AnalysisStep, FilterSpec, InvestigationPlan, Period


def P(start, end):
    return Period(start=date.fromisoformat(start), end=date.fromisoformat(end))


def step(sid="S1", **kw):
    base = {
        "id": sid,
        "operation": "metric_by_dimension",
        "metric": "net_revenue",
        "dimension": "region",
        "period": P("2025-01-01", "2025-02-01"),
    }
    base.update(kw)
    return AnalysisStep(**base)


def plan(*steps, hypotheses=()):
    return InvestigationPlan(steps=list(steps), hypotheses=list(hypotheses))


def codes(p, max_steps=6, max_queries=12):
    return {i.code for i in validate_plan(p, max_steps=max_steps, max_total_queries=max_queries)}


def test_valid_plan_has_no_issues():
    assert codes(plan(step())) == set()


def test_unknown_metric_is_rejected():
    assert "unknown_metric" in codes(plan(step(metric="profit_margin")))


def test_unknown_dimension_is_rejected():
    assert "unknown_dimension" in codes(plan(step(dimension="shoe_size")))


def test_cannot_filter_on_month_or_unknown_dimension():
    bad = [FilterSpec(dimension="month", values=["2025-01-01"])]
    assert "bad_filter_dimension" in codes(plan(step(filters=bad)))
    bad = [FilterSpec(dimension="password", values=["x"])]
    assert "bad_filter_dimension" in codes(plan(step(filters=bad)))


def test_duplicate_filter_dimension_rejected():
    f = [
        FilterSpec(dimension="region", values=["West"]),
        FilterSpec(dimension="region", values=["East"]),
    ]
    assert "duplicate_filter" in codes(plan(step(filters=f)))


def test_compare_requires_comparison_period():
    assert "missing_comparison" in codes(plan(step(operation="compare_periods", dimension=None)))


def test_overlapping_periods_rejected():
    s = step(
        operation="compare_periods",
        dimension=None,
        period=P("2025-02-01", "2025-04-01"),
        comparison_period=P("2025-03-01", "2025-05-01"),
    )
    assert "periods_overlap" in codes(plan(s))


def test_adjacent_periods_are_not_overlapping():
    s = step(
        operation="compare_periods",
        dimension=None,
        period=P("2025-02-01", "2025-03-01"),
        comparison_period=P("2025-01-01", "2025-02-01"),
    )
    assert codes(plan(s)) == set()


def test_contribution_rejects_ratio_metric():
    s = step(
        operation="contribution_to_change",
        metric="refund_rate",
        period=P("2025-02-01", "2025-03-01"),
        comparison_period=P("2025-01-01", "2025-02-01"),
    )
    assert "metric_not_additive" in codes(plan(s))


def test_contribution_rejects_order_count_across_categories():
    # An order can contain several categories, so per-category order counts do not add up.
    s = step(
        operation="contribution_to_change",
        metric="order_count",
        dimension="category",
        period=P("2025-02-01", "2025-03-01"),
        comparison_period=P("2025-01-01", "2025-02-01"),
    )
    assert "metric_not_additive" in codes(plan(s))


def test_contribution_ok_for_revenue_by_region_but_not_by_month():
    kw = {
        "operation": "contribution_to_change",
        "period": P("2025-02-01", "2025-03-01"),
        "comparison_period": P("2025-01-01", "2025-02-01"),
    }
    assert codes(plan(step(dimension="region", **kw))) == set()
    assert "bad_dimension" in codes(plan(step(dimension="month", **kw)))


def test_metric_by_period_rules():
    assert "bad_dimension" in codes(plan(step(operation="metric_by_period", dimension="region")))
    assert codes(plan(step(operation="metric_by_period", dimension=None))) == set()


def test_metric_by_dimension_requires_dimension():
    assert "missing_dimension" in codes(plan(step(dimension=None)))


def test_too_many_steps_duplicate_ids_and_budget():
    many = [step(f"S{i}") for i in range(1, 8)]
    assert "too_many_steps" in codes(plan(*many), max_steps=6)
    assert "duplicate_step_id" in codes(plan(step("S1"), step("S1")))
    costly = [
        step(
            f"S{i}",
            operation="compare_periods",
            dimension=None,
            period=P("2025-02-01", "2025-03-01"),
            comparison_period=P("2025-01-01", "2025-02-01"),
        )
        for i in range(1, 5)
    ]
    assert "query_budget_exceeded" in codes(plan(*costly), max_queries=6)  # 4 steps x 2 queries


def test_hypothesis_must_reference_existing_steps():
    from app.schemas.plan import HypothesisSpec

    h = HypothesisSpec(id="H1", statement="West fell", step_ids=["S9"])
    assert "hypothesis_unknown_step" in codes(plan(step(), hypotheses=[h]))


# --- shape/bounds enforced by the Pydantic models (LLM output is parsed through these) ---


def test_extra_fields_are_forbidden_so_raw_sql_cannot_ride_along():
    with pytest.raises(ValidationError):
        AnalysisStep(
            id="S1",
            operation="metric_by_dimension",
            metric="net_revenue",
            dimension="region",
            period=P("2025-01-01", "2025-02-01"),
            sql="DROP TABLE retail.orders",
        )


def test_unknown_operation_rejected():
    with pytest.raises(ValidationError):
        step(operation="run_sql")


def test_period_must_be_ordered_and_bounded():
    with pytest.raises(ValidationError):
        P("2025-02-01", "2025-02-01")
    with pytest.raises(ValidationError):
        P("2000-01-01", "2025-01-01")


def test_step_id_format_and_filter_limits():
    with pytest.raises(ValidationError):
        step(sid="step-1")
    with pytest.raises(ValidationError):
        FilterSpec(dimension="region", values=[])
    with pytest.raises(ValidationError):
        FilterSpec(dimension="region", values=["x" * 61])
    with pytest.raises(ValidationError):
        FilterSpec(dimension="region", values=[str(i) for i in range(11)])


def test_plan_must_have_at_least_one_step():
    with pytest.raises(ValidationError):
        InvestigationPlan(steps=[])
