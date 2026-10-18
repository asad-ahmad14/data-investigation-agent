"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import routes_health, routes_investigations
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.security import (
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    SlidingWindowLimiter,
)
from app.db.engines import get_app_engine
from app.repositories.investigations import InvestigationStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "web" / "static"

STATIC_DIR = Path(__file__).parent / "web" / "static"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """On startup, mark investigations left unfinished by a previous run as failed.

    Best effort: the API still starts when the database is down (readiness reports that).
    """
    settings = get_settings()
    try:
        store = InvestigationStore(get_app_engine())
        stale = store.fail_stale(settings.investigation_timeout_seconds * 2)
        if stale:
            logger.warning("stale_investigations_failed", extra={"count": stale})
    except Exception:
        logger.warning("startup_reconcile_skipped", exc_info=True)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Autonomous Data Analyst & Investigation Agent",
        version=__version__,
        lifespan=lifespan,
        # The interactive docs and schema are for development; production does not expose them.
        docs_url=None if settings.app_env == "production" else "/docs",
        redoc_url=None,
        openapi_url=None if settings.app_env == "production" else "/openapi.json",
        description="Evidence-driven, LangGraph-orchestrated investigations over a retail dataset.",
    )
    register_exception_handlers(app)
    app.state.limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(routes_health.router)
    app.include_router(routes_investigations.router)
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse("/ui/")

    logger.info(
        "app_started", extra={"env": settings.app_env, "llm_provider": settings.llm_provider}
    )
    return app


app = create_app()
