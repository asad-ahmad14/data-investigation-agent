"""Errors raised by the LLM layer.

`public_message` is safe to show to a user. Provider response bodies, prompts and API keys are
never placed in exception messages.
"""


class LLMError(Exception):
    public_message = "The language model could not be used for this request."
    retryable = False

    def __init__(self, public_message: str | None = None) -> None:
        super().__init__(public_message or self.public_message)
        if public_message:
            self.public_message = public_message


class LLMConfigError(LLMError):
    public_message = "The language model is not configured correctly."


class LLMAuthError(LLMError):
    public_message = "The language model rejected the configured credentials."


class LLMTimeoutError(LLMError):
    public_message = "The language model took too long to respond."
    retryable = True


class LLMUnavailableError(LLMError):
    """Rate limit, server error or network failure."""

    public_message = "The language model is temporarily unavailable."
    retryable = True


class LLMBlockedError(LLMError):
    """The provider refused to answer (safety block, empty reply)."""

    public_message = "The language model declined to answer this request."


class LLMOutputError(LLMError):
    """The reply was not valid JSON, or did not match the required structure."""

    public_message = "The language model returned an unusable answer."


class LLMBudgetExceededError(LLMError):
    public_message = "The limit on language model calls for this investigation was reached."


class InvalidQuestionError(LLMError):
    public_message = "The question is empty or too long."
