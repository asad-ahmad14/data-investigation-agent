"""Investigation endpoints.

POST   /investigations              validate, save as 'queued', run in the background -> 202
GET    /investigations/{id}         status and, once finished, the full report
GET    /investigations/{id}/steps   the workflow trace (plan / execute / evaluate / report)
GET    /investigations              recent investigations, newest first

Clients poll GET /investigations/{id} until status is no longer 'queued' or 'running'.
"""

import uuid
from functools import partial
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request

from app.agent.report import InvestigationReport
from app.agent.runner import run_investigation
from app.agent.service import RunFn, execute_investigation
from app.api.schemas import (
    InvestigationAccepted,
    InvestigationCreate,
    InvestigationList,
    InvestigationOut,
    Status,
    StepOut,
)
from app.core.config import Settings, get_settings
from app.core.errors import BusyError, NotFoundError, RateLimitedError, ValidationFailedError
from app.core.security import SlidingWindowLimiter
from app.db.engines import get_analyst_engine, get_app_engine
from app.llm.errors import InvalidQuestionError
from app.llm.service import clean_question
from app.repositories.investigations import InvestigationStore

router = APIRouter(prefix="/investigations", tags=["investigations"])


def get_store() -> InvestigationStore:
    return InvestigationStore(get_app_engine())


def get_runner(settings: Annotated[Settings, Depends(get_settings)]) -> RunFn:
    def _run(question: str) -> tuple[InvestigationReport, list[dict]]:
        return run_investigation(question, settings=settings, analyst_engine=get_analyst_engine())

    return _run


def get_limiter(request: Request) -> SlidingWindowLimiter:
    return request.app.state.limiter


def enforce_rate_limit(
    request: Request, limiter: Annotated[SlidingWindowLimiter, Depends(get_limiter)]
) -> None:
    client = request.client.host if request.client else "unknown"
    wait = limiter.check(client)
    if wait is not None:
        raise RateLimitedError(wait)


Store = Annotated[InvestigationStore, Depends(get_store)]
Cfg = Annotated[Settings, Depends(get_settings)]


@router.post(
    "",
    status_code=202,
    response_model=InvestigationAccepted,
    dependencies=[Depends(enforce_rate_limit)],
)
def create_investigation(
    body: InvestigationCreate,
    request: Request,
    background: BackgroundTasks,
    store: Store,
    settings: Cfg,
    run: Annotated[RunFn, Depends(get_runner)],
) -> InvestigationAccepted:
    try:
        question = clean_question(body.question, max_length=settings.max_question_length)
    except InvalidQuestionError as exc:
        raise ValidationFailedError(exc.public_message) from exc

    # Bound concurrent work (database load and LLM cost). Rows older than twice the investigation
    # timeout are treated as abandoned and do not count.
    if store.count_active(settings.investigation_timeout_seconds * 2) >= (
        settings.max_active_investigations
    ):
        raise BusyError()

    investigation_id = store.create(
        question,
        {
            "llm_provider": settings.llm_provider,
            "max_iterations": settings.max_iterations,
            "max_total_queries": settings.max_total_queries,
        },
    )
    background.add_task(
        partial(execute_investigation, investigation_id, question, store=store, run=run)
    )
    return InvestigationAccepted(
        id=investigation_id,
        status="queued",
        status_url=str(request.url_for("get_investigation", investigation_id=investigation_id)),
    )


@router.get("/{investigation_id}", response_model=InvestigationOut, name="get_investigation")
def get_investigation(investigation_id: uuid.UUID, store: Store) -> dict:
    row = store.get(investigation_id)
    if row is None:
        raise NotFoundError("Investigation not found.")
    return row


@router.get("/{investigation_id}/steps", response_model=list[StepOut])
def get_investigation_steps(investigation_id: uuid.UUID, store: Store) -> list[dict]:
    if store.get(investigation_id) is None:
        raise NotFoundError("Investigation not found.")
    return store.steps(investigation_id)


@router.get("", response_model=InvestigationList)
def list_investigations(
    store: Store,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=100_000)] = 0,
    status: Status | None = None,
) -> dict:
    items, total = store.list_recent(limit=limit, offset=offset, status=status)
    return {"items": items, "total": total, "limit": limit, "offset": offset}
