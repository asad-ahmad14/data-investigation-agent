import math

import pytest

from app.analytics import stats


def test_pct_change_basic_and_negative():
    assert stats.pct_change(100, 75) == pytest.approx(-0.25)
    assert stats.pct_change(100, 150) == pytest.approx(0.5)


@pytest.mark.parametrize("old,new", [(0, 5), (None, 5), (5, None), (math.nan, 1), (1, math.nan)])
def test_pct_change_undefined_cases_return_none(old, new):
    assert stats.pct_change(old, new) is None


def test_pp_change_is_not_relative_change():
    # 5% -> 8% is +3.0 percentage points but a +60% RELATIVE increase. They must not be mixed up.
    assert stats.pp_change(0.05, 0.08) == pytest.approx(3.0)
    assert stats.pct_change(0.05, 0.08) == pytest.approx(0.6)


def test_pp_change_missing():
    assert stats.pp_change(None, 0.1) is None


def test_safe_ratio_division_by_zero_and_missing():
    assert stats.safe_ratio(1, 0) is None
    assert stats.safe_ratio(None, 2) is None
    assert stats.safe_ratio(1, 4) == 0.25


def test_small_sample_threshold():
    assert stats.is_small_sample(29)
    assert not stats.is_small_sample(30)
    assert stats.is_small_sample(None)


def test_compare_values_fraction_reports_percentage_points():
    c = stats.compare_values(0.05, 0.08, unit="fraction")
    assert c.percentage_point_change == pytest.approx(3.0)
    assert c.relative_change == pytest.approx(0.6)
    assert c.status == "ok"


def test_compare_values_currency_has_no_pp_change():
    c = stats.compare_values(200.0, 150.0, unit="currency")
    assert c.absolute_change == -50.0
    assert c.percentage_point_change is None


def test_compare_values_zero_baseline_and_no_data():
    zero = stats.compare_values(0.0, 10.0, unit="currency")
    assert zero.relative_change is None and zero.absolute_change == 10.0
    assert zero.status == "baseline_zero"
    assert stats.compare_values(None, 10.0, unit="currency").status == "no_data"


def test_contribution_shares_sum_to_one_when_groups_move_together():
    r = stats.contribution_to_change({"A": 100, "B": 100}, {"A": 60, "B": 90})
    assert r.total_change == -50
    assert sum(c.share_of_total_change for c in r.contributions) == pytest.approx(1.0)
    assert r.contributions[0].group == "A"  # largest absolute change first
    assert r.mixed_direction is False


def test_contribution_group_missing_in_one_period_counts_as_zero():
    r = stats.contribution_to_change({"A": 100}, {"A": 100, "B": 40})
    by = {c.group: c for c in r.contributions}
    assert by["B"].before == 0 and by["B"].change == 40


def test_contribution_mixed_direction_flagged_and_zero_total_has_no_share():
    mixed = stats.contribution_to_change({"A": 100, "B": 100}, {"A": 150, "B": 60})
    assert mixed.mixed_direction is True
    flat = stats.contribution_to_change({"A": 100, "B": 100}, {"A": 120, "B": 80})
    assert flat.total_change == 0
    assert all(c.share_of_total_change is None for c in flat.contributions)


def test_period_over_period_and_largest_decline():
    series = [("m1", 100.0), ("m2", 120.0), ("m3", 80.0), ("m4", 70.0)]
    changes = stats.period_over_period(series, unit="currency")
    assert len(changes) == 3
    worst = stats.largest_decline(changes)
    assert (worst.previous_label, worst.label) == ("m2", "m3")


def test_largest_decline_none_when_nothing_fell():
    changes = stats.period_over_period([("a", 1.0), ("b", 2.0)], unit="currency")
    assert stats.largest_decline(changes) is None


def test_period_over_period_with_gap_does_not_invent_zero():
    changes = stats.period_over_period([("a", 1.0), ("b", None), ("c", 3.0)], unit="currency")
    assert [c.change.status for c in changes] == ["no_data", "no_data"]
