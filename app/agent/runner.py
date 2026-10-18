"""Public entry point: run one investigation end to end and return a report."""

import logging
import time

from sqlalchemy import Engine

from app.agent.graph import Dependencies, build_graph
from app.agent.report import InvestigationReport, RunStats
from app.analytics.executor import QueryExecutor
from app.core.config import Settings
from app.llm.base import LLMClient
from app.llm.factory import build_llm_service
from app.retail.generate import FIRST_MONTH, N_MONTHS, add_months

logger = logging.getLogger(__name__)


def run_investigation(
    question: str,
    *,
    settings: Settings,
    analyst_engine: Engine,
    llm_client: LLMClient | None = None,
) -> tuple[InvestigationReport, list[dict]]:
    """Run an investigation. Returns (report, trace events). Never raises for expected failures
    (bad question, LLM or database trouble): those become a report with status 'failed' or
    'needs_clarification'. Unexpected bugs are logged and also returned as a generic failure."""
    started = time.monotonic()
    deps = Dependencies(
        llm=build_llm_service(settings, llm_client),
        executor=QueryExecutor(
            analyst_engine,
            max_rows=settings.max_result_rows,
            timeout_ms=settings.query_timeout_ms,
        ),
        settings=settings,
        data_start=FIRST_MONTH,
        data_end_exclusive=add_months(FIRST_MONTH, N_MONTHS),
        deadline=started + settings.investigation_timeout_seconds,
    )
    graph = build_graph(deps)
    try:
        final = graph.invoke({"question": question, "started": started, "iteration": -1})
    except Exception:
        logger.exception("investigation_crashed")
        report = InvestigationReport(
            question=question,
            status="failed",
            error_code="internal_error",
            message="The investigation failed unexpectedly.",
            stats=RunStats(duration_ms=round((time.monotonic() - started) * 1000)),
        )
        return report, []
    return final["report"], final.get("events", [])
