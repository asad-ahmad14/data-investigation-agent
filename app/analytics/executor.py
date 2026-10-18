"""Read-only query execution.

Defence in depth for the analytical path:
1. only CompiledQuery objects (built by the compiler from the registry) are accepted;
2. the engine logs in as `analyst_ro`, which has SELECT on approved tables and nothing else;
3. each query runs in an explicitly READ ONLY transaction with a statement timeout;
4. the result is capped, with truncation reported rather than hidden.

Driver errors are logged in full but converted to errors with generic public messages, so SQL
text and database internals never reach an API response.
"""

import logging
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import psycopg
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from app.analytics.compiler import CompiledQuery
from app.analytics.errors import (
    DatabaseUnavailableError,
    QueryExecutionError,
    QueryTimeoutError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionResult:
    rows: list[dict]
    truncated: bool
    duration_ms: int
    sql: str
    params: dict


def _clean(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


class QueryExecutor:
    def __init__(self, engine: Engine, *, max_rows: int, timeout_ms: int) -> None:
        self._engine = engine
        self.max_rows = max_rows
        self._timeout_ms = timeout_ms

    def execute(self, query: CompiledQuery) -> ExecutionResult:
        started = time.perf_counter()
        try:
            with self._engine.connect() as conn, conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))
                conn.execute(
                    text("SELECT set_config('statement_timeout', :ms, true)"),
                    {"ms": str(self._timeout_ms)},
                )
                fetched = conn.execute(query.statement).mappings().all()
        except DBAPIError as exc:
            self._raise_mapped(exc)

        duration_ms = int((time.perf_counter() - started) * 1000)
        truncated = len(fetched) > self.max_rows
        rows = [{k: _clean(v) for k, v in row.items()} for row in fetched[: self.max_rows]]
        return ExecutionResult(rows, truncated, duration_ms, query.sql_text(), query.params())

    @staticmethod
    def _raise_mapped(exc: DBAPIError):
        original = exc.orig
        if isinstance(original, psycopg.errors.QueryCanceled):
            logger.warning("query_timeout")
            raise QueryTimeoutError() from exc
        if isinstance(original, psycopg.OperationalError):
            logger.error("database_unavailable", exc_info=exc)
            raise DatabaseUnavailableError() from exc
        logger.error("query_failed", exc_info=exc)
        raise QueryExecutionError() from exc
