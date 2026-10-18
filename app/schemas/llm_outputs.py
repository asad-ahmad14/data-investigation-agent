"""Structured outputs the LLM is allowed to produce.

Every model reply is parsed into one of these strict Pydantic models. A reply that does not
fit (extra fields, wrong types, out-of-range values) is rejected, never "best-effort" accepted.
"""

from typing import Literal

from pydantic import Field, model_validator

from app.schemas.plan import CleanText, InvestigationPlan, StrictModel


class PlannerResponse(StrictModel):
    """The planner either proposes an analysis plan, asks for clarification, or declines."""

    outcome: Literal["plan", "needs_clarification", "out_of_scope"]
    plan: InvestigationPlan | None = None
    message: CleanText | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def _consistent(self) -> "PlannerResponse":
        if self.outcome == "plan":
            if self.plan is None:
                raise ValueError("outcome 'plan' requires a plan")
        else:
            if self.plan is not None:
                raise ValueError(f"outcome '{self.outcome}' must not include a plan")
            if not self.message:
                raise ValueError(f"outcome '{self.outcome}' requires a message")
        return self
