"""Gemini provider over the public REST API (httpx), so there is no vendor SDK to install.

Security notes: the API key travels in the `x-goog-api-key` header, never in the URL, so it
cannot appear in URL logs or httpx error messages. Prompts, replies and response bodies are
never logged or put into exception messages.

Not verified against the live API from the development sandbox (no outbound access there);
the request/response handling is tested with a mocked transport. Run the opt-in real_llm
test with your own key to verify it.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import SecretStr

from app.llm.base import LLMRequest, LLMResponse, LLMUsage
from app.llm.errors import (
    LLMAuthError,
    LLMBlockedError,
    LLMConfigError,
    LLMError,
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_BLOCKED_FINISH = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "OTHER"}
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_MAX_BACKOFF_SECONDS = 8.0


class GeminiClient:
    provider = "gemini"

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        timeout_seconds: float = 30,
        max_retries: int = 2,
        max_output_tokens: int = 4096,
        base_url: str = BASE_URL,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.get_secret_value():
            raise LLMAuthError("No LLM API key is configured.")
        self.model = model
        self._key = api_key
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._sleep = sleep
        self._url = f"{base_url}/models/{model}:generateContent"
        self._http = httpx.Client(timeout=timeout_seconds, transport=transport)

    def __repr__(self) -> str:  # never include the key
        return f"GeminiClient(model={self.model!r})"

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ public
    def generate_json(self, request: LLMRequest) -> LLMResponse:
        body = {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": [{"role": "user", "parts": [{"text": request.user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": self._max_output_tokens,
            },
        }
        headers = {"x-goog-api-key": self._key.get_secret_value()}

        last_error: LLMError | None = None
        for attempt in range(self._max_retries + 1):
            retry_after: float | None = None
            try:
                response = self._http.post(self._url, json=body, headers=headers)
            except httpx.TimeoutException:
                last_error = LLMTimeoutError()
            except httpx.TransportError:
                last_error = LLMUnavailableError()
            else:
                if response.status_code == 200:
                    return self._parse(response)
                last_error = self._error_for(response)
                if response.status_code not in _RETRYABLE_STATUS:
                    raise last_error
                retry_after = _retry_after(response)
            logger.warning(
                "llm_transport_retry",
                extra={
                    "provider": "gemini",
                    "attempt": attempt + 1,
                    "error": type(last_error).__name__,
                },
            )
            if attempt < self._max_retries:
                self._sleep(retry_after or min(_MAX_BACKOFF_SECONDS, 2.0**attempt))
        assert last_error is not None
        raise last_error

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _error_for(response: httpx.Response) -> LLMError:
        status = response.status_code
        if status in (401, 403):
            return LLMAuthError()
        if status == 400:
            # Gemini reports an invalid key as 400 INVALID_ARGUMENT, so look for it.
            if "api key" in response.text.lower():
                return LLMAuthError()
            return LLMConfigError("The language model rejected the request (bad request).")
        if status == 404:
            return LLMConfigError("The configured language model name was not found.")
        if status == 429:
            return LLMUnavailableError("The language model rate limit was reached.")
        if status in _RETRYABLE_STATUS:
            return LLMUnavailableError()
        return LLMError()

    @staticmethod
    def _parse(response: httpx.Response) -> LLMResponse:
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise LLMOutputError() from exc

        feedback = payload.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise LLMBlockedError()
        candidates = payload.get("candidates") or []
        if not candidates:
            raise LLMBlockedError()
        candidate = candidates[0]
        finish = candidate.get("finishReason")
        if finish in _BLOCKED_FINISH:
            raise LLMBlockedError()
        if finish == "MAX_TOKENS":
            raise LLMOutputError("The language model reply was cut off before it finished.")

        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise LLMBlockedError()
        fenced = _FENCE.match(text)
        if fenced:
            text = fenced.group(1)
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise LLMOutputError() from exc
        if not isinstance(data, dict):
            raise LLMOutputError()

        meta = payload.get("usageMetadata") or {}
        tokens_in = meta.get("promptTokenCount")
        out_parts = [meta.get("candidatesTokenCount"), meta.get("thoughtsTokenCount")]
        tokens_out = (
            sum(v for v in out_parts if isinstance(v, int))
            if any(isinstance(v, int) for v in out_parts)
            else None
        )
        return LLMResponse(
            data=data,
            model=str(payload.get("modelVersion") or ""),
            usage=LLMUsage(tokens_in if isinstance(tokens_in, int) else None, tokens_out),
        )


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    if seconds is None or seconds <= 0:
        return None
    return min(seconds, _MAX_BACKOFF_SECONDS)
