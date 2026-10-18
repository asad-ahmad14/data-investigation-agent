"""Shared fixtures.

Unit tests need nothing external. Integration tests (marked `integration`) use a dedicated
database named `<POSTGRES_DB>_test` on the server from your .env. It is created fresh (dropped
first if it exists), migrated with Alembic, seeded, and dropped again at the end, so your
development data is never touched. If PostgreSQL is not reachable they are skipped, with a reason.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError

from app.core.config import Settings, get_settings
from app.db.engines import build_admin_engine, build_analyst_engine, build_app_engine
from app.retail.generate import Dataset, generate_dataset
from app.retail.seed import seed_database

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_settings_cache(monkeypatch):
    """Isolate tests from the developer's real .env and from each other."""
    for var in ("LLM_PROVIDER", "LLM_API_KEY", "APP_ENV"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@dataclass
class IntegrationDB:
    settings: Settings
    admin: Engine
    app: Engine
    analyst: Engine
    dataset: Dataset


@pytest.fixture(scope="session")
def integration_db():
    base = Settings()  # reads the environment / .env
    if (
        not base.app_db_password.get_secret_value()
        or not base.analyst_db_password.get_secret_value()
    ):
        pytest.skip("Set APP_DB_PASSWORD and ANALYST_DB_PASSWORD in .env to run integration tests.")

    test_db = f"{base.postgres_db}_test"
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", test_db):
        pytest.skip(f"Unsafe test database name: {test_db!r}")
    settings = base.model_copy(update={"postgres_db": test_db})

    maintenance = create_engine(
        base.admin_db_url.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    try:
        with maintenance.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{test_db}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{test_db}"'))
    except OperationalError as exc:
        maintenance.dispose()
        pytest.skip(
            f"PostgreSQL not reachable (start it with: docker compose up -d db): {exc.orig}"
        )

    admin = build_admin_engine(settings)
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    with admin.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")

    dataset = generate_dataset()
    seed_database(admin, dataset)

    db = IntegrationDB(
        settings=settings,
        admin=admin,
        app=build_app_engine(settings),
        analyst=build_analyst_engine(settings),
        dataset=dataset,
    )
    yield db

    for engine in (db.analyst, db.app, db.admin):
        engine.dispose()
    with maintenance.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{test_db}" WITH (FORCE)'))
    maintenance.dispose()
