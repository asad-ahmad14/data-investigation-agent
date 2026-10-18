"""Health endpoints.

/health        liveness: the process is up (no dependencies checked)
/health/ready  readiness: the database is reachable
"""

from collections.abc import Callable

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import Engine

from app import __version__
from app.db.engines import check_database, get_app_engine

router = APIRouter(tags=["health"])


def get_db_checker() -> Callable[[], bool]:
    """Dependency returning a zero-arg callable; overridable in tests."""

    def _check() -> bool:
        engine: Engine = get_app_engine()
        return check_database(engine)

    return _check


@router.get("/health")
def liveness() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/health/ready")
def readiness(checker: Callable[[], bool] = Depends(get_db_checker)) -> JSONResponse:
    db_ok = checker()
    status_code = 200 if db_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if db_ok else "unavailable",
            "database": "ok" if db_ok else "down",
        },
    )
