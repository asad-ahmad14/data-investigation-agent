"""The investigations API and store against a real PostgreSQL (app_rw role) with the mock LLM."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from app.agent.report import InvestigationReport, RunStats
from app.agent.runner import run_investigation
from app.api.routes_investigations import get_runner, get_store
from app.core.config import get_settings
from app.db.models import Evidence, Hypothesis, Investigation, InvestigationStep
from app.main import create_app
from app.repositories.investigations import InvestigationStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store(integration_db):
    return InvestigationStore(integration_db.app)


@pytest.fixture
def client(integration_db, store):
    settings = integration_db.settings
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_runner] = lambda: (
        lambda q: run_investigation(q, settings=settings, analyst_engine=integration_db.analyst)
    )
    return TestClient(app)


def count(integration_db, model, iid):
    with Session(integration_db.app) as s:
        return s.scalar(
            select(func.count()).select_from(model).where(model.investigation_id == iid)
        )


def test_full_flow_saves_report_trace_evidence_and_hypotheses(client, integration_db):
    r = client.post("/investigations", json={"question": "Why did net revenue drop?"})
    assert r.status_code == 202
    iid = uuid.UUID(r.json()["id"])

    data = client.get(f"/investigations/{iid}").json()
    assert data["status"] == "completed"
    assert data["started_at"] and data["finished_at"]
    assert data["llm_calls"] == 2 and data["model"] == "mock-planner-v1"
    report = data["report"]
    assert report["stats"]["iterations"] == 2
    assert {h["status"] for h in report["hypotheses"]} == {"supported", "contradicted"}

    steps = client.get(f"/investigations/{iid}/steps").json()
    assert [s["node"] for s in steps] == [
        "plan", "execute", "evaluate", "plan", "execute", "evaluate", "report",
    ]  # fmt: skip
    assert [s["seq"] for s in steps] == list(range(1, 8))

    assert count(integration_db, Evidence, iid) == len(report["evidence"]) == 5
    assert count(integration_db, Hypothesis, iid) == 3
    assert count(integration_db, InvestigationStep, iid) == 7
    with Session(integration_db.app) as s:
        row = s.scalars(select(Evidence).where(Evidence.investigation_id == iid)).first()
        assert row.evidence_key == "E1" and float(row.value) == pytest.approx(-16583, abs=1)
        assert "description" in row.source_operation


def test_clarification_is_saved_with_message_and_no_evidence(client, integration_db):
    iid = client.post("/investigations", json={"question": "What is the weather in Paris?"}).json()[
        "id"
    ]
    data = client.get(f"/investigations/{iid}").json()
    assert data["status"] == "needs_clarification"
    assert data["report"]["message"]
    assert count(integration_db, Evidence, uuid.UUID(iid)) == 0


def test_database_failure_during_run_is_saved_as_failed_report(client, integration_db, monkeypatch):
    from sqlalchemy import create_engine

    dead = create_engine(
        integration_db.settings.analyst_db_url.set(port=1), connect_args={"connect_timeout": 1}
    )
    settings = integration_db.settings
    client.app.dependency_overrides[get_runner] = lambda: (
        lambda q: run_investigation(q, settings=settings, analyst_engine=dead)
    )
    iid = client.post("/investigations", json={"question": "Why did net revenue drop?"}).json()[
        "id"
    ]
    dead.dispose()
    data = client.get(f"/investigations/{iid}").json()
    assert data["status"] == "failed" and data["error_code"] == "database_unavailable"


def test_save_result_is_atomic(store, integration_db):
    iid = store.create("atomic?", {})
    report = InvestigationReport(
        question="atomic?", status="completed", stats=RunStats(),
        evidence=[{"key": "E1", "metric": "m", "value": 1.0, "unit": "count", "period": {},
                   "filters": {}, "source_operation": {}, "description": "d", "detail": {},
                   "result_status": "ok"}],
    )  # fmt: skip
    bad_events = [{"node": "plan", "status": "ok", "duration_ms": 1, "detail": {"x": object()}}]
    with pytest.raises(Exception):  # noqa: B017 - any serialisation failure
        store.save_result(iid, report, bad_events)
    row = store.get(iid)
    assert row["status"] == "queued" and row["report"] is None
    assert count(integration_db, Evidence, iid) == 0


def test_fail_stale_and_count_active(store, integration_db):
    iid = store.create("stale one", {})
    assert store.count_active(3600) >= 1
    with Session(integration_db.app) as s, s.begin():
        s.execute(
            text("UPDATE app.investigations SET created_at = :t WHERE id = :i"),
            {"t": datetime.now(UTC) - timedelta(hours=2), "i": iid},
        )
    before = store.count_active(3600)
    assert store.fail_stale(3600) >= 1
    row = store.get(iid)
    assert row["status"] == "failed" and row["error_code"] == "interrupted"
    assert before >= 0 and store.count_active(10_000_000) >= 0


def test_application_role_cannot_read_retail_data(integration_db):
    with pytest.raises(ProgrammingError), Session(integration_db.app) as s:
        s.execute(text("SELECT count(*) FROM retail.orders"))


def test_status_check_constraint_rejects_unknown_status(integration_db):
    with (
        pytest.raises(Exception, match="status_valid"),
        Session(integration_db.app) as s,
        s.begin(),
    ):
        s.add(Investigation(question="q", status="exploded", params={}))


def test_steps_come_back_in_sequence_order_whatever_the_insert_order(store, integration_db):
    iid = store.create("ordering", {})
    with Session(integration_db.app) as s, s.begin():
        for seq in (3, 1, 2):
            s.add(
                InvestigationStep(
                    investigation_id=iid, seq=seq, node=f"n{seq}", status="ok", duration_ms=1,
                    detail={},
                )
            )  # fmt: skip
    assert [x["seq"] for x in store.steps(iid)] == [1, 2, 3]
