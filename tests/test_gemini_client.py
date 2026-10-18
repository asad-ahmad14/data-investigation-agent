"""Gemini client behaviour against a mocked HTTP transport (no network, no real key)."""

import json

import httpx
import pytest
from pydantic import SecretStr

from app.llm.base import LLMRequest
from app.llm.errors import (
    LLMAuthError,
    LLMBlockedError,
    LLMConfigError,
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.gemini import GeminiClient
from app.schemas.llm_outputs import PlannerResponse

KEY = "sk-test-SECRET-12345"
REQ = LLMRequest(task="plan", system="SYS", user="USER", schema=PlannerResponse)


def ok_body(text: str, **extra) -> dict:
    body = {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 5,
        },
        "modelVersion": "gemini-test-001",
    }
    body.update(extra)
    return body


def make(handler, *, retries=2, sleeps=None) -> GeminiClient:
    return GeminiClient(
        api_key=SecretStr(KEY),
        model="gemini-test",
        max_retries=retries,
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else (lambda _s: None)),
    )


def test_success_parses_json_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body('{"outcome": "out_of_scope", "message": "x"}'))

    reply = make(handler).generate_json(REQ)
    assert reply.data == {"outcome": "out_of_scope", "message": "x"}
    assert reply.usage.tokens_in == 100
    assert reply.usage.tokens_out == 25  # candidates + thinking tokens
    assert reply.model == "gemini-test-001"


def test_request_shape_and_key_only_in_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body("{}"))

    make(handler).generate_json(REQ)
    assert seen["url"].endswith("/models/gemini-test:generateContent")
    assert KEY not in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == KEY
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert seen["body"]["contents"][0]["parts"][0]["text"] == "USER"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "tools" not in seen["body"]  # the model gets no tools


def test_markdown_fenced_json_is_accepted():
    client = make(lambda r: httpx.Response(200, json=ok_body('```json\n{"a": 1}\n```')))
    assert client.generate_json(REQ).data == {"a": 1}


def test_thought_parts_are_ignored():
    body = ok_body("")
    body["candidates"][0]["content"]["parts"] = [
        {"text": "reasoning...", "thought": True},
        {"text": '{"a": 2}'},
    ]
    assert make(lambda r: httpx.Response(200, json=body)).generate_json(REQ).data == {"a": 2}


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '"str"', ""])
def test_non_object_or_invalid_json_is_an_output_error(text):
    client = make(lambda r: httpx.Response(200, json=ok_body(text)))
    with pytest.raises((LLMOutputError, LLMBlockedError)):
        client.generate_json(REQ)


def test_truncated_reply_is_an_output_error():
    body = ok_body('{"a":')
    body["candidates"][0]["finishReason"] = "MAX_TOKENS"
    with pytest.raises(LLMOutputError):
        make(lambda r: httpx.Response(200, json=body)).generate_json(REQ)


def test_prompt_block_and_safety_finish_are_blocked_errors():
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(LLMBlockedError):
        make(lambda r: httpx.Response(200, json=blocked)).generate_json(REQ)
    body = ok_body("{}")
    body["candidates"][0]["finishReason"] = "SAFETY"
    with pytest.raises(LLMBlockedError):
        make(lambda r: httpx.Response(200, json=body)).generate_json(REQ)


def test_transient_errors_are_retried_with_backoff_then_succeed():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json=ok_body('{"a": 1}'))

    assert make(handler, sleeps=sleeps).generate_json(REQ).data == {"a": 1}
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_retries_are_bounded_and_final_error_is_typed():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={})

    with pytest.raises(LLMUnavailableError):
        make(handler, retries=2).generate_json(REQ)
    assert len(calls) == 3  # first try + 2 retries


def test_retry_after_header_is_honoured_but_capped():
    sleeps = []

    def handler(request):
        return httpx.Response(429, headers={"retry-after": "999"}, json={})

    with pytest.raises(LLMUnavailableError):
        make(handler, retries=1, sleeps=sleeps).generate_json(REQ)
    assert sleeps == [8.0]


def test_timeout_and_network_errors_are_typed_and_retried():
    calls = []

    def timeout(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMTimeoutError):
        make(timeout, retries=1).generate_json(REQ)
    assert len(calls) == 2

    def down(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(LLMUnavailableError):
        make(down, retries=0).generate_json(REQ)


@pytest.mark.parametrize(
    ("status", "body", "exc"),
    [
        (401, {}, LLMAuthError),
        (403, {}, LLMAuthError),
        (
            400,
            {"error": {"message": "API key not valid. Please pass a valid API key."}},
            LLMAuthError,
        ),
        (400, {"error": {"message": "bad field"}}, LLMConfigError),
        (404, {}, LLMConfigError),
    ],
)
def test_permanent_errors_are_not_retried(status, body, exc):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, json=body)

    with pytest.raises(exc):
        make(handler).generate_json(REQ)
    assert len(calls) == 1


def test_key_never_appears_in_errors_repr_or_provider_text():
    def handler(request):
        return httpx.Response(401, text=f"bad key {KEY}")

    client = make(handler)
    with pytest.raises(LLMAuthError) as exc:
        client.generate_json(REQ)
    assert KEY not in str(exc.value)
    assert KEY not in repr(client)


def test_missing_key_is_rejected_at_construction():
    with pytest.raises(LLMAuthError):
        GeminiClient(api_key=SecretStr(""), model="m")
