"""Request and response models for the investigations API."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal[
    "queued", "running", "completed", "insufficient_evidence", "needs_clarification", "failed"
]


class InvestigationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The exact limit is enforced from settings in the route; this is a hard outer bound.
    question: str = Field(min_length=1, max_length=2000, examples=["Why did net revenue drop?"])


class InvestigationAccepted(BaseModel):
    id: uuid.UUID
    status: Status
    status_url: str


class InvestigationOut(BaseModel):
    id: uuid.UUID
    question: str
    status: Status
    error_code: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    llm_calls: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str | None = None
    report: dict | None = None


class InvestigationList(BaseModel):
    items: list[InvestigationOut]
    total: int
    limit: int
    offset: int


class StepOut(BaseModel):
    seq: int
    node: str
    status: str
    duration_ms: int
    detail: dict
