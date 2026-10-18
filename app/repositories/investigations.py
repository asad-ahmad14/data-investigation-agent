"""Persistence for investigations: the only code that writes to the `app` schema.

Uses the app_rw database role (read/write on `app`, no access to `retail`). Each public method
is one transaction, so a half-saved investigation is never visible.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session

from app.agent.report import InvestigationReport
from app.db.models import Evidence, Hypothesis, Investigation, InvestigationStep

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("queued", "running")


def _now() -> datetime:
    return datetime.now(UTC)


def investigation_dict(row: Investigation) -> dict:
    return {
        "id": row.id,
        "question": row.question,
        "status": row.status,
        "error_code": row.error_code,
        "created_at": row.created_at,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "report": row.report,
        "llm_calls": row.llm_calls,
        "tokens_in": row.tokens_in,
        "tokens_out": row.tokens_out,
        "model": row.model,
    }


class InvestigationStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------ writes
    def create(self, question: str, params: dict) -> uuid.UUID:
        with Session(self._engine) as session, session.begin():
            row = Investigation(question=question, status="queued", params=params)
            session.add(row)
            session.flush()
            return row.id

    def mark_running(self, investigation_id: uuid.UUID) -> None:
        with Session(self._engine) as session, session.begin():
            session.execute(
                update(Investigation)
                .where(Investigation.id == investigation_id)
                .values(status="running", started_at=_now())
            )

    def save_result(
        self, investigation_id: uuid.UUID, report: InvestigationReport, events: list[dict]
    ) -> None:
        """Store the outcome, the workflow trace, the evidence rows and hypothesis verdicts."""
        with Session(self._engine) as session, session.begin():
            row = session.get(Investigation, investigation_id)
            if row is None:
                logger.error("save_result_missing_investigation")
                return
            row.status = report.status
            row.error_code = report.error_code
            row.report = report.model_dump(mode="json")
            row.finished_at = _now()
            row.llm_calls = report.stats.llm_calls
            row.tokens_in = report.stats.tokens_in
            row.tokens_out = report.stats.tokens_out
            row.model = report.stats.model

            for seq, event in enumerate(events, start=1):
                session.add(
                    InvestigationStep(
                        investigation_id=investigation_id,
                        seq=seq,
                        node=event["node"],
                        status=str(event["status"])[:20],
                        duration_ms=int(event["duration_ms"]),
                        detail=event["detail"],
                    )
                )
            for item in report.evidence:
                session.add(
                    Evidence(
                        investigation_id=investigation_id,
                        evidence_key=item["key"],
                        metric=item["metric"],
                        value=None if item["value"] is None else Decimal(str(item["value"])),
                        unit=item["unit"],
                        period=item["period"],
                        filters=item["filters"],
                        source_operation={
                            **item["source_operation"],
                            "description": item["description"],
                            "detail": item["detail"],
                        },
                        result_status=item["result_status"],
                    )
                )
            for hyp in report.hypotheses:
                session.add(
                    Hypothesis(
                        investigation_id=investigation_id,
                        label=hyp.statement[:200],
                        status=hyp.status,
                        evidence_keys=hyp.evidence_keys,
                    )
                )

    def save_failure(self, investigation_id: uuid.UUID, error_code: str) -> None:
        with Session(self._engine) as session, session.begin():
            session.execute(
                update(Investigation)
                .where(Investigation.id == investigation_id)
                .values(status="failed", error_code=error_code, finished_at=_now())
            )

    def fail_stale(self, older_than_seconds: int) -> int:
        """Mark investigations stuck in queued/running (for example after a restart) as failed."""
        cutoff = _now() - timedelta(seconds=older_than_seconds)
        with Session(self._engine) as session, session.begin():
            result = session.execute(
                update(Investigation)
                .where(Investigation.status.in_(ACTIVE_STATUSES), Investigation.created_at < cutoff)
                .values(status="failed", error_code="interrupted", finished_at=_now())
            )
            return result.rowcount or 0

    # ------------------------------------------------------------------- reads
    def count_active(self, max_age_seconds: int) -> int:
        cutoff = _now() - timedelta(seconds=max_age_seconds)
        with Session(self._engine) as session:
            return (
                session.scalar(
                    select(func.count())
                    .select_from(Investigation)
                    .where(
                        Investigation.status.in_(ACTIVE_STATUSES), Investigation.created_at > cutoff
                    )
                )
                or 0
            )

    def get(self, investigation_id: uuid.UUID) -> dict | None:
        with Session(self._engine) as session:
            row = session.get(Investigation, investigation_id)
            return investigation_dict(row) if row else None

    def list_recent(self, *, limit: int, offset: int, status: str | None) -> tuple[list[dict], int]:
        with Session(self._engine) as session:
            query = select(Investigation)
            count = select(func.count()).select_from(Investigation)
            if status:
                query = query.where(Investigation.status == status)
                count = count.where(Investigation.status == status)
            rows = session.scalars(
                query.order_by(Investigation.created_at.desc()).limit(limit).offset(offset)
            ).all()
            total = session.scalar(count) or 0
            return [investigation_dict(r) for r in rows], total

    def steps(self, investigation_id: uuid.UUID) -> list[dict]:
        with Session(self._engine) as session:
            rows = session.scalars(
                select(InvestigationStep)
                .where(InvestigationStep.investigation_id == investigation_id)
                .order_by(InvestigationStep.seq)
            ).all()
            return [
                {
                    "seq": r.seq,
                    "node": r.node,
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                    "detail": r.detail,
                }
                for r in rows
            ]
