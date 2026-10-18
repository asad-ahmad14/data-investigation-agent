from fastapi.testclient import TestClient

from app.api.routes_health import get_db_checker
from app.main import create_app


def _client(db_ok: bool) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_db_checker] = lambda: lambda: db_ok
    return TestClient(app)


def test_liveness_ok():
    r = _client(True).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness_ok_when_db_up():
    r = _client(True).get("/health/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "database": "ok"}


def test_readiness_503_when_db_down():
    r = _client(False).get("/health/ready")
    assert r.status_code == 503
    assert r.json()["database"] == "down"


def test_unknown_route_404():
    assert _client(True).get("/nope").status_code == 404
