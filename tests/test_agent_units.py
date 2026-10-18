"""Agent building blocks without a database: formatting, evidence, hypothesis rules."""

import pytest

from app.agent.evaluation import build_findings, evaluate_hypotheses
from app.agent.evidence import EvidenceItem, extract_evidence
from app.agent.formatting import fmt_change, fmt_pct, fmt_signed, fmt_value, period_text
from app.agent.graph import _group_caveats
from app.analytics import stats
from app.analytics.operations import ExecutedQuery, StepOutput
from app.schemas.plan import HypothesisSpec

# ------------------------------------------------------------------ formatting


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (1234.5, "currency", "$1,234"),
        (0.0725, "fraction", "7.2%"),
        (3565, "count", "3,565"),
        (None, "currency", "n/a"),
    ],
)
def test_fmt_value(value, unit, expected):
    assert fmt_value(value, unit) == expected


def test_fmt_signed_and_change():
    assert fmt_signed(-16583.4, "currency") == "-$16,583"
    assert fmt_signed(0.032, "fraction") == "+3.2 pp"
    assert fmt_signed(0, "count") == "0"
    assert fmt_pct(-0.28) == "-28.0%"
    assert fmt_change(-100, -0.25, "currency") == "-$100, -25.0%"
    assert fmt_change(0.03, 0.5, "fraction") == "+3.0 pp"  # relative change hidden for rates
    assert fmt_change(None, None, "count") == "n/a"


def test_period_text():
    assert period_text("2025-06-01 to 2025-07-01 (end exclusive)") == "June 2025"
    assert period_text("2025-01-01 to 2025-04-01 (end exclusive)") == "2025-01-01 to 2025-04-01"
    assert period_text("2025-12-01 to 2026-01-01 (end exclusive)") == "December 2025"
    assert period_text(None) == "the period"


# -------------------------------------------------------------------- evidence


def _out(operation, result, *, dimension=None, metric="net_revenue", unit="currency", status="ok"):
    return StepOutput(
        step_id="S1",
        operation=operation,
        metric=metric,
        unit=unit,
        dimension=dimension,
        filters={},
        queries=[ExecutedQuery("focus", "SELECT 1", {}, 1, 1, False)],
        result=result,
        status=status,
    )


def _series_result(values):
    series = [
        {"label": f"2025-{m:02d}-01", "value": v, "n_orders": 10} for m, v in enumerate(values, 1)
    ]
    changes = stats.period_over_period([(p["label"], p["value"]) for p in series], unit="currency")
    decline = stats.largest_decline(changes)
    return {"series": series, "largest_decline": decline.to_dict() if decline else None}


def test_evidence_from_series_with_decline():
    items = extract_evidence(_out("metric_by_period", _series_result([100, 120, 60])), 5)
    assert [i.key for i in items] == ["E5"]
    item = items[0]
    assert item.value == -60
    assert item.detail["kind"] == "largest_decline"
    assert "February 2025 to March 2025" in item.description
    assert "$120 to $60" in item.description


def test_evidence_from_series_without_decline():
    (item,) = extract_evidence(_out("metric_by_period", _series_result([10, 20, 30])), 1)
    assert item.result_status == "no_decline" and item.value is None


def test_evidence_no_data_is_explicit_not_zero():
    out = _out("metric_by_period", {"series": [], "largest_decline": None}, status="no_data")
    (item,) = extract_evidence(out, 1)
    assert item.result_status == "no_data" and item.value is None


def _contribution(before, after, dimension="region"):
    res = stats.contribution_to_change(before, after).to_dict()
    res.update(
        focus_period="2025-06-01 to 2025-07-01 (end exclusive)",
        baseline_period="2025-05-01 to 2025-06-01 (end exclusive)",
        dimension=dimension,
    )
    return _out("contribution_to_change", res, dimension=dimension)


def test_evidence_from_contribution():
    (item,) = extract_evidence(_contribution({"West": 50, "East": 50}, {"West": 10, "East": 48}), 3)
    assert item.value == -40
    assert item.filters == {"region": ["West"]}
    assert item.detail["share_of_total_change"] == pytest.approx(40 / 42)
    assert item.detail["baseline_share"] == pytest.approx(0.5)
    assert "West" in item.description and "95% of the total change" in item.description


def test_evidence_from_overall_comparison_rate_in_points():
    change = stats.compare_values(0.04, 0.07, unit="fraction").to_dict()
    res = {
        "focus_period": "2025-10-01 to 2025-11-01 (end exclusive)",
        "baseline_period": "2025-09-01 to 2025-10-01 (end exclusive)",
        "overall": {"n_orders_focus": 100, "n_orders_baseline": 90, **change},
    }
    (item,) = extract_evidence(
        _out("compare_periods", res, metric="refund_rate", unit="fraction"), 1
    )
    assert "4.0% to 7.0%" in item.description
    assert "+3.0 pp" in item.description
    assert "September 2025 to October 2025" in item.description


def test_evidence_from_group_comparison_records_runner_up():
    groups = [
        {"group": "A", "absolute_change": 0.2, "relative_change": 1.0, "old": 0.2, "new": 0.4,
         "status": "ok", "small_sample": False, "n_orders_focus": 80, "n_orders_baseline": 80},
        {"group": "B", "absolute_change": 0.05, "relative_change": 0.5, "old": 0.1, "new": 0.15,
         "status": "ok", "small_sample": False, "n_orders_focus": 80, "n_orders_baseline": 80},
    ]  # fmt: skip
    res = {"focus_period": "x", "baseline_period": "y", "groups": groups}
    (item,) = extract_evidence(
        _out("compare_periods", res, dimension="category", metric="refund_rate", unit="fraction"), 1
    )
    assert item.detail["group"] == "A" and item.detail["second_abs_change"] == 0.05


# ------------------------------------------------------------------ hypotheses


def _contrib_item(share, base, *, mixed=False, group="West", key="E1", step="S2"):
    return EvidenceItem(
        key=key, step_id=step, metric="net_revenue", unit="currency", value=-10.0,
        period={}, filters={}, source_operation={}, result_status="ok", description="d",
        detail={"kind": "top_contributor", "group": group, "share_of_total_change": share,
                "baseline_share": base, "mixed_direction": mixed, "dimension": "region"},
    )  # fmt: skip


def _group_item(top, second, *, small=False, n_groups=4, key="E1", step="S2"):
    return EvidenceItem(
        key=key, step_id=step, metric="refund_rate", unit="fraction", value=top,
        period={}, filters={}, source_operation={}, result_status="ok", description="d",
        detail={"kind": "top_group_change", "group": "Apparel", "small_sample": small,
                "second_abs_change": second, "n_groups": n_groups},
    )  # fmt: skip


def _judge(item, step="S2"):
    spec = HypothesisSpec(id="H1", statement="concentrated", step_ids=[step])
    (h,) = evaluate_hypotheses([spec], {step: [item]}, set(), first_number=1)
    return h


def test_concentrated_and_disproportionate_is_supported():
    h = _judge(_contrib_item(0.85, 0.25))
    assert h.status == "supported" and "85%" in h.reason and "25%" in h.reason


def test_large_share_that_matches_baseline_share_is_proportionate():
    h = _judge(_contrib_item(0.62, 0.64, group="Consumer"))
    assert h.status == "contradicted" and "proportionate" in h.reason


def test_small_share_is_contradicted_unless_mixed():
    assert _judge(_contrib_item(0.3, 0.1)).status == "contradicted"
    assert _judge(_contrib_item(0.3, 0.1, mixed=True)).status == "inconclusive"


def test_mixed_direction_but_dominant_share_still_supported_with_note():
    h = _judge(_contrib_item(1.4, 0.3, mixed=True))
    assert h.status == "supported" and "offset" in h.reason


def test_undefined_share_is_inconclusive():
    assert _judge(_contrib_item(None, 0.3)).status == "inconclusive"


def test_group_comparison_rules():
    assert _judge(_group_item(0.20, 0.05)).status == "supported"
    assert _judge(_group_item(0.20, 0.15)).status == "contradicted"
    assert _judge(_group_item(0.20, 0.0)).status == "supported"
    assert _judge(_group_item(0.20, 0.01, small=True)).status == "inconclusive"
    assert _judge(_group_item(0.20, 0.0, n_groups=1)).status == "inconclusive"


def test_no_data_evidence_is_inconclusive_and_missing_steps_not_testable():
    empty = EvidenceItem("E1", "S2", "m", "count", None, {}, {}, {}, "no_data", "d")
    assert _judge(empty).status == "inconclusive"
    spec = HypothesisSpec(id="H1", statement="x", step_ids=["S9"])
    (a,) = evaluate_hypotheses([spec], {}, {"S9"}, first_number=1)
    (b,) = evaluate_hypotheses([spec], {}, set(), first_number=1)
    assert a.status == b.status == "not_testable"
    assert "could not be completed" in a.reason and "No analysis" in b.reason


def test_multi_step_hypothesis_supported_if_any_supports_and_ids_renumbered():
    spec = HypothesisSpec(id="H1", statement="x", step_ids=["S2", "S3"])
    items = {
        "S2": [_contrib_item(0.2, 0.1, key="E1", step="S2")],
        "S3": [_contrib_item(0.9, 0.2, key="E2", step="S3")],
    }
    (h,) = evaluate_hypotheses([spec], items, set(), first_number=4)
    assert h.id == "H4" and h.status == "supported" and h.evidence_keys == ["E1", "E2"]


def test_findings_cite_evidence_and_skip_no_data():
    decline = EvidenceItem(
        "E1", "S1", "net_revenue", "currency", -5.0, {}, {}, {}, "ok", "fell",
        {"kind": "largest_decline"},
    )  # fmt: skip
    nodata = EvidenceItem("E2", "S2", "m", "count", None, {}, {}, {}, "no_data", "nothing")
    top = _contrib_item(0.9, 0.2, key="E3")
    h = _judge(top)
    findings = build_findings([decline, nodata, top], [h])
    assert [f.evidence_keys for f in findings] == [["E1"], ["E3"]]


def test_group_caveats_merges_identical_text():
    merged = _group_caveats(["S1: a", "S2: a", "S2: b", "plain", "S1: a"])
    assert merged == ["S1, S2: a", "S2: b", "plain"]
