"""Application errors and FastAPI handlers.

Handlers return a stable, safe JSON body. Details (stack traces, DB messages) go to logs only.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 500
    code = "internal_error"
    public_message = "An internal error occurred."

    def __init__(self, public_message: str | None = None) -> None:
        super().__init__(public_message or self.public_message)
        if public_message:
            self.public_message = public_message


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    public_message = "The requested resource was not found."


class BusyError(AppError):
    status_code = 429
    code = "too_many_investigations"
    public_message = "Too many investigations are running. Please try again shortly."


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"
    public_message = "Too many requests. Please slow down."

    def __init__(self, retry_after: float = 10.0) -> None:
        super().__init__()
        self.retry_after = max(1, int(retry_after + 0.999))


class ValidationFailedError(AppError):
    status_code = 422
    code = "validation_failed"
    public_message = "The request could not be validated."


def _body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("app_error", extra={"code": exc.code}, exc_info=exc)
        headers = None
        if exc.status_code == 429:
            headers = {"Retry-After": str(getattr(exc, "retry_after", 10))}
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(exc.code, exc.public_message),
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_body("internal_error", "An internal error occurred."),
        )
