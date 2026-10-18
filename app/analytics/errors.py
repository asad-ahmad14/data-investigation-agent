"""Errors raised by the analytics layer.

`public_message` is safe to show to a user. Internal details (SQL text, driver messages) are
logged, never placed in the exception message that reaches an API response.
"""


class AnalyticsError(Exception):
    public_message = "The analysis could not be completed."
    retryable = False

    def __init__(self, public_message: str | None = None) -> None:
        super().__init__(public_message or self.public_message)
        if public_message:
            self.public_message = public_message


class PlanValidationError(AnalyticsError):
    """The plan or query request references something that is not allowed."""

    public_message = "The analysis request is not allowed."


class QueryTimeoutError(AnalyticsError):
    public_message = "The analytical query exceeded its time limit."


class QueryExecutionError(AnalyticsError):
    public_message = "The analytical query failed."


class DatabaseUnavailableError(AnalyticsError):
    """Connection-level failure. The only analytics error that is sensible to retry."""

    public_message = "The analytical database is currently unavailable."
    retryable = True
