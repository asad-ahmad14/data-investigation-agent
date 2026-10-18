"""Number formatting for evidence descriptions and reports.

Every number shown to a user is formatted here from a computed value, never typed by a model.
"""

import re
from datetime import date

from app.retail.generate import add_months


def fmt_value(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    if unit == "fraction":
        return f"{value * 100:.1f}%"
    if unit == "currency":
        return f"${value:,.0f}"
    return f"{value:,.0f}"


def fmt_signed(value: float | None, unit: str) -> str:
    """Signed absolute change. Fractions are shown in percentage points."""
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else "-" if value < 0 else ""
    mag = abs(value)
    if unit == "fraction":
        return f"{sign}{mag * 100:.1f} pp"
    if unit == "currency":
        return f"{sign}${mag:,.0f}"
    return f"{sign}{mag:,.0f}"


def fmt_pct(value: float | None) -> str:
    """Signed relative change given as a fraction (0.25 -> +25.0%)."""
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}{abs(value) * 100:.1f}%"


def fmt_change(abs_change: float | None, rel_change: float | None, unit: str) -> str:
    if abs_change is None:
        return "n/a"
    text = fmt_signed(abs_change, unit)
    if unit != "fraction" and rel_change is not None:
        text += f", {fmt_pct(rel_change)}"
    return text


def month_name(iso: str) -> str:
    """'2025-06-01' -> 'June 2025'. Falls back to the input if it is not an ISO date."""
    try:
        return date.fromisoformat(iso).strftime("%B %Y")
    except ValueError:
        return iso


_PERIOD = re.compile(r"^(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})")


def period_text(label: str | None) -> str:
    """'2025-06-01 to 2025-07-01 (end exclusive)' -> 'June 2025'; other spans stay as dates."""
    if not label:
        return "the period"
    match = _PERIOD.match(label)
    if not match:
        return label
    start, end = date.fromisoformat(match[1]), date.fromisoformat(match[2])
    if start.day == 1 and end == add_months(start, 1):
        return start.strftime("%B %Y")
    return f"{start.isoformat()} to {end.isoformat()}"
