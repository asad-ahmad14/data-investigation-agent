"""Deterministic statistics. No LLM involvement: every number in a report comes from here
or from a database aggregate.

Conventions
-----------
- Relative change is a fraction: -0.25 means a 25% decrease.
- Rates are fractions (0.05 = 5%). The change between two rates is reported in PERCENTAGE
  POINTS (pp), which is not the same thing as a relative percent change.
- Missing data is None, never silently 0. Division by zero yields None.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

SMALL_SAMPLE_THRESHOLD = 30


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if _missing(numerator) or _missing(denominator) or denominator == 0:
        return None
    return numerator / denominator


def pct_change(old: float | None, new: float | None) -> float | None:
    """Relative change (new - old) / |old|. None if either value is missing or old is 0."""
    if _missing(old) or _missing(new) or old == 0:
        return None
    return (new - old) / abs(old)


def pp_change(old_rate: float | None, new_rate: float | None) -> float | None:
    """Difference between two rates in percentage points (0.05 -> 0.08 is +3.0 pp)."""
    if _missing(old_rate) or _missing(new_rate):
        return None
    return (new_rate - old_rate) * 100.0


def is_small_sample(n: int | None, minimum: int = SMALL_SAMPLE_THRESHOLD) -> bool:
    return n is None or n < minimum


@dataclass(frozen=True)
class Change:
    old: float | None
    new: float | None
    absolute_change: float | None
    relative_change: float | None
    percentage_point_change: float | None
    status: str  # "ok" | "no_data" | "baseline_zero"

    def to_dict(self) -> dict:
        return asdict(self)


def compare_values(old: float | None, new: float | None, *, unit: str) -> Change:
    """Compare a baseline (`old`) with a focus value (`new`).

    For unit == "fraction" the headline change is `percentage_point_change`; a relative change
    is also given when the baseline is non-zero, but the two must not be confused.
    """
    if _missing(old) or _missing(new):
        return Change(old, new, None, None, None, "no_data")
    relative = pct_change(old, new)
    pp = pp_change(old, new) if unit == "fraction" else None
    status = "baseline_zero" if relative is None else "ok"
    return Change(old, new, new - old, relative, pp, status)


@dataclass(frozen=True)
class Contribution:
    group: str
    before: float
    after: float
    change: float
    share_of_total_change: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContributionResult:
    total_before: float
    total_after: float
    total_change: float
    contributions: tuple[Contribution, ...]
    mixed_direction: bool

    def to_dict(self) -> dict:
        return {
            "total_before": self.total_before,
            "total_after": self.total_after,
            "total_change": self.total_change,
            "mixed_direction": self.mixed_direction,
            "contributions": [c.to_dict() for c in self.contributions],
        }


def contribution_to_change(
    before: Mapping[str, float | None], after: Mapping[str, float | None]
) -> ContributionResult:
    """Decompose the change in an ADDITIVE metric into per-group changes.

    A group missing from one period is treated as 0 there (it genuinely had no activity).
    The share is None when the total change is exactly zero. When groups move in opposite
    directions shares can exceed 100% or be negative; `mixed_direction` flags this.
    """
    groups = sorted(set(before) | set(after))
    rows = []
    for g in groups:
        b = before.get(g) or 0.0
        a = after.get(g) or 0.0
        rows.append((g, b, a, a - b))
    total_before = sum(r[1] for r in rows)
    total_after = sum(r[2] for r in rows)
    total_change = total_after - total_before
    contributions = tuple(
        Contribution(g, b, a, c, safe_ratio(c, total_change))
        for g, b, a, c in sorted(rows, key=lambda r: abs(r[3]), reverse=True)
    )
    changes = [c.change for c in contributions if c.change != 0]
    mixed = any(c > 0 for c in changes) and any(c < 0 for c in changes)
    return ContributionResult(total_before, total_after, total_change, contributions, mixed)


@dataclass(frozen=True)
class PeriodChange:
    previous_label: str
    label: str
    change: Change

    def to_dict(self) -> dict:
        return {
            "previous_label": self.previous_label,
            "label": self.label,
            **self.change.to_dict(),
        }


def period_over_period(
    series: Sequence[tuple[str, float | None]], *, unit: str
) -> list[PeriodChange]:
    """Change between each consecutive pair of points in a time series."""
    return [
        PeriodChange(prev[0], cur[0], compare_values(prev[1], cur[1], unit=unit))
        for prev, cur in zip(series, series[1:], strict=False)
    ]


def largest_decline(changes: Sequence[PeriodChange]) -> PeriodChange | None:
    """The consecutive pair with the most negative absolute change (None if nothing fell)."""
    declines = [c for c in changes if c.change.absolute_change is not None]
    declines = [c for c in declines if c.change.absolute_change < 0]
    if not declines:
        return None
    return min(declines, key=lambda c: c.change.absolute_change)
