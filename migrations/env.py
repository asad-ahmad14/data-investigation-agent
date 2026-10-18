"""Alembic environment.

Migrations run as the admin (owner) role; the URL is built from application settings (.env).
Tests may inject an open connection via `config.attributes["connection"]`.
"""

from alembic import context
from sqlalchemy import create_engine, pool

from app.core.config import get_settings
from app.db.models import APP, RETAIL, Base

config = context.config
target_metadata = Base.metadata


def include_name(name, type_, parent_names):
    """Only look at our two schemas when autogenerating."""
    if type_ == "schema":
        return name in (RETAIL, APP)  # skips the default schema, incl. alembic_version
    return True


def _migrate(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=True,
        include_name=include_name,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _migrate(connection)
        return
    engine = create_engine(get_settings().admin_db_url, poolclass=pool.NullPool)
    with engine.connect() as conn:
        _migrate(conn)


if context.is_offline_mode():
    raise RuntimeError("Offline migrations are not supported in this project.")

run_migrations_online()
