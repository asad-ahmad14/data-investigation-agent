"""Structured investigation plan: the ONLY thing the LLM may produce to request analysis.

These models check shape and bounds. Whether a metric or dimension is actually permitted is
decided by app.analytics.validator against the schema registry; the LLM naming a column
never makes it exist.

Conventions
-----------
- Periods are half-open: `start` is included, `end` is excluded.
- For compare_periods and contribution_to_change, `period` is the FOCUS period and
  `comparison_period` is the BASELINE. Change is reported as focus versus baseline.
"""

import re
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

Operation = Literal[
    "metric_by_period",
    "metric_by_dimension",
    "compare_periods",
    "contribution_to_change",
]

MAX_PERIOD_DAYS = 3660

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _strip_control(value):
    """Drop control characters (notably NUL, which PostgreSQL text/JSONB cannot store) from
    free text that a model produced. Non-strings are left for normal validation to reject."""
    return _CONTROL.sub("", value) if isinstance(value, str) else value


CleanText = Annotated[str, BeforeValidator(_strip_control)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Period(StrictModel):
    start: date
    end: date

    @model_validator(mode="after")
    def _check_order_and_span(self) -> "Period":
        if self.end <= self.start:
            raise ValueError("period end must be after start (end is exclusive)")
        if (self.end - self.start).days > MAX_PERIOD_DAYS:
            raise ValueError(f"period may not span more than {MAX_PERIOD_DAYS} days")
        return self

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    @property
    def label(self) -> str:
        return f"{self.start.isoformat()} to {self.end.isoformat()} (end exclusive)"


class FilterSpec(StrictModel):
    dimension: str = Field(max_length=40)
    values: list[Annotated[CleanText, Field(min_length=1, max_length=60)]] = Field(
        min_length=1, max_length=10
    )


class AnalysisStep(StrictModel):
    id: str = Field(pattern=r"^S\d{1,2}$")
    operation: Operation
    metric: str = Field(max_length=40)
    dimension: str | None = Field(default=None, max_length=40)
    period: Period
    comparison_period: Period | None = None
    filters: list[FilterSpec] = Field(default_factory=list, max_length=4)
    purpose: CleanText = Field(default="", max_length=300)


class HypothesisSpec(StrictModel):
    id: str = Field(pattern=r"^H\d{1,2}$")
    statement: CleanText = Field(min_length=1, max_length=300)
    step_ids: list[str] = Field(min_length=1, max_length=6)


class InvestigationPlan(StrictModel):
    hypotheses: list[HypothesisSpec] = Field(default_factory=list, max_length=5)
    steps: list[AnalysisStep] = Field(min_length=1, max_length=10)
