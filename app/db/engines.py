"""SQLAlchemy engines.

Three distinct credentials are kept apart on purpose:
- admin:   migrations, role creation and seeding (never used while serving requests)
- app:     read/write on the `app` schema only (investigation history)
- analyst: SELECT-only on approved `retail` tables (the only engine analytical queries touch)

The cached getters create engines lazily, so importing this module never opens a connection.
The build_* functions take explicit Settings so tests can point at a different database.
"""

from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL

from app.core.config import Settings, get_settings


def _build(url: URL, connect_args: dict | None = None) -> Engine:
    # connect_timeout keeps a down or unreachable database from hanging requests and startup.
    args = {"connect_timeout": 5, **(connect_args or {})}
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, connect_args=args)


def build_admin_engine(settings: Settings) -> Engine:
    return _build(settings.admin_db_url)


def build_app_engine(settings: Settings) -> Engine:
    return _build(settings.app_db_url)


def build_analyst_engine(settings: Settings) -> Engine:
    # The role itself is read-only with a timeout ceiling (see migration 0002); the options
    # below apply the configured, usually tighter, per-statement timeout to each connection.
    return _build(
        settings.analyst_db_url,
        connect_args={"options": f"-c statement_timeout={settings.query_timeout_ms}"},
    )


@lru_cache
def get_admin_engine() -> Engine:
    return build_admin_engine(get_settings())


@lru_cache
def get_app_engine() -> Engine:
    return build_app_engine(get_settings())


@lru_cache
def get_analyst_engine() -> Engine:
    return build_analyst_engine(get_settings())


def check_database(engine: Engine) -> bool:
    """Return True if a trivial query succeeds. Never raises; failures are reported as False."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
