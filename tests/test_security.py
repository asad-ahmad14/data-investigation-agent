"""Security tests that need no database: headers, limits, hostile model output, secret hygiene."""

import copy
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.routes_investigations import get_store
from app.core.config import Settings
from app.core.security import PAGE_CSP, SlidingWindowLimiter
from app.llm.base import LLMRequest
from app.llm.errors import LLMOutputError
from app.llm.gemini import GeminiClient
from app.main import create_app
from app.schemas.llm_outputs import PlannerResponse
from tests.test_llm_service import GOOD_PLAN, ScriptedLLM, ask, service

# ------------------------------------------------------------------- headers


def test_page_has_strict_csp_and_frame_protection():
    r = TestClient(create_app()).get("/ui/")
    assert r.headers["content-security-policy"] == PAGE_CSP
    assert "script-src 'self'" in PAGE_CSP and "unsafe-inline" not in PAGE_CSP
    assert "frame-ancestors 'none'" in PAGE_CSP
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"


class _EmptyStore:
    """Stands in for the database so this unit test never needs one."""

    def get(self, investigation_id):
        return None


def test_investigation_responses_are_not_cacheable():
    app = create_app()
    app.dependency_overrides[get_store] = lambda: _EmptyStore()
    r = TestClient(app).get("/investigations/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_swagger_docs_are_not_broken_by_the_page_csp():
    r = TestClient(create_app()).get("/docs")
    assert r.status_code == 200 and "content-security-policy" not in r.headers


def test_docs_and_schema_are_hidden_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    client = TestClient(create_app())
    for path in ("/docs", "/openapi.json", "/redoc"):
        assert client.get(path).status_code == 404


# ----------------------------------------------------------------- body limit


def test_declared_oversized_body_is_rejected_before_parsing():
    r = TestClient(create_app()).post("/investigations", json={"question": "x" * 50_000})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_streamed_oversized_body_without_content_length_is_rejected():
    def chunks():
        for _ in range(40):
            yield b'{"question": "' + b"x" * 1000 + b'"}'

    r = TestClient(create_app()).post(
        "/investigations", content=chunks(), headers={"content-type": "application/json"}
    )
    assert r.status_code == 413


def test_bad_content_length_header_is_rejected():
    client = TestClient(create_app())
    r = client.post(
        "/investigations",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": "abc"},
    )
    assert r.status_code in (400, 413)


def test_json_is_required_so_cross_site_form_posts_fail():
    r = TestClient(create_app()).post(
        "/investigations",
        content='{"question": "Why did net revenue drop?"}',
        headers={"content-type": "text/plain"},
    )
    assert r.status_code == 422


# --------------------------------------------------------------- rate limiter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_limiter_allows_up_to_limit_then_blocks_with_retry_after():
    clock = Clock()
    limiter = SlidingWindowLimiter(3, window=60, clock=clock)
    assert [limiter.check("a") for _ in range(3)] == [None, None, None]
    wait = limiter.check("a")
    assert wait is not None and 0 < wait <= 60
    assert limiter.check("b") is None  # other clients are unaffected


def test_limiter_window_slides():
    clock = Clock()
    limiter = SlidingWindowLimiter(2, window=60, clock=clock)
    limiter.check("a")
    clock.t += 30
    limiter.check("a")
    assert limiter.check("a") is not None
    clock.t += 31  # first event is now outside the window
    assert limiter.check("a") is None


def test_limiter_disabled_with_zero_and_memory_bounded():
    assert all(SlidingWindowLimiter(0).check("a") is None for _ in range(100))
    clock = Clock()
    limiter = SlidingWindowLimiter(5, window=60, clock=clock, max_keys=100)
    for i in range(1000):
        limiter.check(f"client-{i}")
    assert len(limiter._events) <= 100


def test_api_returns_429_with_retry_after_when_flooded(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    codes = [client.post("/investigations", json={"question": ""}).status_code for _ in range(5)]
    # Requests count even when invalid, so malformed floods are limited too.
    assert codes[:3] == [422, 422, 422] and codes[3:] == [429, 429]
    blocked = client.post("/investigations", json={"question": "hello there"})
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1
    assert blocked.json()["error"]["code"] == "rate_limited"
    other = TestClient(app, client=("203.0.113.9", 5000))
    assert other.post("/investigations", json={"question": ""}).status_code == 422


# ---------------------------------------------------- hostile model replies


def plan_with(**step_overrides):
    reply = copy.deepcopy(GOOD_PLAN)
    reply["plan"]["steps"][0].update(step_overrides)
    return reply


HOSTILE_STEPS = [
    {"metric": "net_revenue; DROP TABLE retail.orders; --"},
    {"metric": "pg_sleep(60)"},
    {"metric": "1 UNION SELECT password FROM pg_shadow"},
    {"operation": "metric_by_dimension", "dimension": "region); DELETE FROM retail.orders; --"},
    {"operation": "metric_by_dimension", "dimension": "customers.email"},
    {"operation": "raw_sql"},
    {"filters": [{"dimension": "email", "values": ["a@b.c"]}]},
    {"filters": [{"dimension": "month", "values": ["x"]}]},  # not a filterable dimension
    {"period": {"start": "1900-01-01", "end": "2999-01-01"}},  # absurd span
    {"sql": "SELECT * FROM retail.customers"},  # unknown field
    {"id": "S1; DROP TABLE x"},
]


@pytest.mark.parametrize("override", HOSTILE_STEPS)
def test_hostile_plans_are_rejected_and_never_returned(override):
    llm = ScriptedLLM(plan_with(**override), plan_with(**override))
    with pytest.raises(LLMOutputError):
        ask(service(llm))
    assert len(llm.requests) == 2  # one original ask plus one bounded repair, then give up


def test_model_cannot_smuggle_extra_top_level_fields():
    reply = {**GOOD_PLAN, "sql": "DROP TABLE retail.orders"}
    with pytest.raises(LLMOutputError):
        ask(service(ScriptedLLM(reply, reply)))


def test_control_characters_in_model_text_are_stripped():
    reply = {
        "outcome": "plan",
        "plan": {
            "steps": GOOD_PLAN["plan"]["steps"],
            "hypotheses": [{"id": "H1", "statement": "a\x00b\x07c", "step_ids": ["S1"]}],
        },
    }
    parsed = PlannerResponse.model_validate(reply)
    assert parsed.plan.hypotheses[0].statement == "abc"
    msg = PlannerResponse.model_validate({"outcome": "out_of_scope", "message": "no\x00pe"})
    assert msg.message == "nope"


# -------------------------------------------------------------- secret hygiene


def test_settings_repr_and_dump_never_reveal_secrets():
    s = Settings(
        _env_file=None,
        postgres_password="pg-secret-1",
        app_db_password="app-secret-2",
        analyst_db_password="analyst-secret-3",
        llm_provider="gemini",
        llm_api_key="key-secret-4",
    )
    shown = repr(s) + str(s) + str(s.model_dump()) + s.model_dump_json()
    for secret in ("pg-secret-1", "app-secret-2", "analyst-secret-3", "key-secret-4"):
        assert secret not in shown
    assert "pg-secret-1" not in str(s.admin_db_url) or "***" in str(s.admin_db_url)


def test_api_key_is_never_logged_or_in_errors(caplog):
    secret = "AIza-super-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": f"boom {secret}"}})

    client = GeminiClient(
        api_key=SecretStr(secret),
        model="m",
        max_retries=1,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    req = LLMRequest(task="plan", system="s", user="u", schema=PlannerResponse)
    with caplog.at_level(logging.DEBUG), pytest.raises(Exception) as exc:
        client.generate_json(req)
    assert secret not in str(exc.value) and secret not in repr(client)
    assert secret not in caplog.text


def test_env_file_and_secrets_are_not_shipped():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".gitignore").read_text().splitlines()
    assert ".env" in ignored
    dockerignore = (root / ".dockerignore").read_text().splitlines()
    assert ".env" in dockerignore
    compose = (root / "docker-compose.yml").read_text()
    assert '"127.0.0.1:${POSTGRES_PORT' in compose, "database port must bind to loopback only"
    example = (root / ".env.example").read_text()
    assert "sk-" not in example and "AIza" not in example
