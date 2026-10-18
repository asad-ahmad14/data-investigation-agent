"""Background execution of one stored investigation."""

import logging
import uuid
from collections.abc import Callable

from app.agent.report import InvestigationReport
from app.repositories.investigations import InvestigationStore

logger = logging.getLogger(__name__)

RunFn = Callable[[str], tuple[InvestigationReport, list[dict]]]


def execute_investigation(
    investigation_id: uuid.UUID, question: str, *, store: InvestigationStore, run: RunFn
) -> None:
    """Run the workflow and save the outcome. Never raises: a background task has no caller to
    handle errors, so anything unexpected is logged and recorded as status 'failed'."""
    try:
        store.mark_running(investigation_id)
        report, events = run(question)
        store.save_result(investigation_id, report, events)
    except Exception:
        logger.exception(
            "investigation_task_failed", extra={"investigation_id": str(investigation_id)}
        )
        try:
            store.save_failure(investigation_id, "internal_error")
        except Exception:
            logger.exception("investigation_failure_not_saved")
