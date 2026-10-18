"""Investigations API with an in-memory store and a fake runner (no database, no LLM)."""

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.agent.report import InvestigationReport, RunStats
from app.api.routes_investigations import get_runner, get_store
from app.core.config import Settings, get_settings
from app.main import create_app


class FakeStore:
    def __init__(self):
        self.rows: dict[uuid.UUID, dict] = {}
        self.event_log: dict[uuid.UUID, list[dict]] = {}
        self.active = 0

    def create(self, question, params):
        iid = uuid.uuid4()
        self.rows[iid] = {
            "id": iid, "question": question, "status": "queued", "error_code": None,
            "created_at": datetime.now(UTC), "started_at": None, "finished_at": None,
            "report": None, "llm_calls": 0, "tokens_in": None, "tokens_out": None, "model": None,
        }  # fmt: skip
        return iid

    def mark_running(self, iid):
        self.rows[iid]["status"] = "running"

    def save_result(self, iid, report, events):
        row = self.rows[iid]
        row.update(status=report.status, report=report.model_dump(mode="json"),
                   error_code=report.error_code, finished_at=datetime.now(UTC),
                   llm_calls=report.stats.llm_calls)  # fmt: skip
        self.event_log[iid] = events

    def save_failure(self, iid, code):
        self.rows[iid].update(status="failed", error_code=code)

    def count_active(self, max_age_seconds):
        return self.active

    def get(self, iid):
        return self.rows.get(iid)

    def list_recent(self, *, limit, offset, status):
        rows = [r for r in self.rows.values() if status in (None, r["status"])]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return rows[offset : offset + limit], len(rows)

    def steps(self, iid):
        return [
            {"seq": i, "node": e["node"], "status": e["status"], "duration_ms": 1, "detail": {}}
            for i, e in enumerate(self.event_log.get(iid, []), 1)
        ]


def ok_report(question):
    return InvestigationReport(
        question=question, status="completed", summary="done", stats=RunStats(llm_calls=2)
    )


@pytest.fixture
def env():
    store = FakeStore()
    calls: list[str] = []

    def runner():
        def run(question):
            calls.append(question)
            return ok_report(question), [{"node": "plan", "status": "ok"}]

        return run

    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_runner] = runner
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    return TestClient(app), store, calls, app


def test_post_returns_202_and_the_investigation_completes(env):
    client, store, calls, _ = env
    r = client.post("/investigations", json={"question": "  Why did   net revenue drop? "})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert body["status_url"].endswith(f"/investigations/{body['id']}")
    assert calls == ["Why did net revenue drop?"]  # whitespace normalised before use

    got = client.get(f"/investigations/{body['id']}")
    assert got.status_code == 200
    data = got.json()
    assert data["status"] == "completed" and data["report"]["summary"] == "done"
    assert data["llm_calls"] == 2

    steps = client.get(f"/investigations/{body['id']}/steps").json()
    assert [s["node"] for s in steps] == ["plan"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 501},
        {"question": "x" * 2001},
        {"question": "ok question here", "extra": 1},
        {"question": 5},
    ],
)
def test_invalid_requests_are_rejected_without_running_anything(env, payload):
    client, store, calls, _ = env
    r = client.post("/investigations", json=payload)
    assert r.status_code == 422
    assert calls == [] and store.rows == {}


def test_busy_server_returns_429_with_retry_after(env):
    client, store, calls, _ = env
    store.active = 4
    r = client.post("/investigations", json={"question": "Why did net revenue drop?"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "10"
    assert r.json()["error"]["code"] == "too_many_investigations"
    assert store.rows == {} and calls == []


def test_unknown_and_malformed_ids(env):
    client, *_ = env
    missing = client.get(f"/investigations/{uuid.uuid4()}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    assert client.get(f"/investigations/{uuid.uuid4()}/steps").status_code == 404
    assert client.get("/investigations/not-a-uuid").status_code == 422


def test_runner_crash_is_recorded_as_failed_and_does_not_leak(env):
    client, store, _, app = env

    def broken():
        def run(question):
            raise RuntimeError("secret connection string postgresql://u:p@host")

        return run

    app.dependency_overrides[get_runner] = broken
    iid = client.post("/investigations", json={"question": "Why did net revenue drop?"}).json()[
        "id"
    ]
    data = client.get(f"/investigations/{iid}").json()
    assert data["status"] == "failed" and data["error_code"] == "internal_error"
    assert "secret" not in client.get(f"/investigations/{iid}").text


def test_list_pagination_and_status_filter(env):
    client, store, *_ = env
    for i in range(5):
        client.post("/investigations", json={"question": f"Why did net revenue drop {i}?"})
    first = next(iter(store.rows))
    store.rows[first]["status"] = "failed"
    page = client.get("/investigations", params={"limit": 2, "offset": 1}).json()
    assert page["total"] == 5 and len(page["items"]) == 2 and page["limit"] == 2
    failed = client.get("/investigations", params={"status": "failed"}).json()
    assert failed["total"] == 1
    assert client.get("/investigations", params={"status": "bogus"}).status_code == 422
    assert client.get("/investigations", params={"limit": 0}).status_code == 422
    assert client.get("/investigations", params={"limit": 101}).status_code == 422
    assert client.get("/investigations", params={"offset": -1}).status_code == 422


def test_unexpected_server_error_has_safe_body():
    app = create_app()

    class ExplodingStore(FakeStore):
        def get(self, iid):
            raise RuntimeError("db password is hunter2")

    app.dependency_overrides[get_store] = lambda: ExplodingStore()
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get(f"/investigations/{uuid.uuid4()}")
    assert r.status_code == 500
    assert r.json() == {
        "error": {"code": "internal_error", "message": "An internal error occurred."}
    }
