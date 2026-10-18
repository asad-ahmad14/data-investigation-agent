"""database roles and least-privilege grants

Revision ID: 0002
Revises: 0001

Two login roles, deliberately separated:

analyst_ro  used by the analytical query executor
            - SELECT on the seven retail tables, nothing else
            - read-only transactions by default and a statement timeout ceiling
            - no access to the `app` schema

app_rw      used by the API to persist investigation history
            - read/write on the `app` schema only
            - no access to the `retail` schema

Passwords come from ANALYST_DB_PASSWORD / APP_DB_PASSWORD at migration time. Roles are
cluster-wide in PostgreSQL, so re-running the migration against another database on the same
server updates the existing roles instead of failing.

Known limitation: PostgreSQL lets every role read most of pg_catalog and information_schema.
That cannot be revoked safely. The analyst role is protected from misuse by the query compiler
(it never emits catalog queries) and by owning no privileges beyond those listed above.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

from app.core.config import get_settings

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RETAIL_TABLES = (
    "regions",
    "customers",
    "categories",
    "products",
    "orders",
    "order_items",
    "refunds",
)


def _upsert_role(conn, name: str, password: str) -> None:
    if not password:
        raise RuntimeError(f"Password for database role '{name}' is empty; set it in your .env.")
    if "%" in password:
        raise RuntimeError("Database role passwords must not contain the '%' character.")
    exists = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :n"), {"n": name}).scalar()
    verb = "ALTER" if exists else "CREATE"
    # format() with %I / %L quotes the identifier and the literal safely on the server side.
    statement = conn.execute(
        text("SELECT format(:tpl, CAST(:n AS text), CAST(:p AS text))"),
        {
            "tpl": f"{verb} ROLE %I WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L",
            "n": name,
            "p": password,
        },
    ).scalar_one()
    conn.exec_driver_sql(statement)


def upgrade() -> None:
    settings = get_settings()
    conn = op.get_bind()
    analyst = settings.analyst_db_user
    app_user = settings.app_db_user

    _upsert_role(conn, analyst, settings.analyst_db_password.get_secret_value())
    _upsert_role(conn, app_user, settings.app_db_password.get_secret_value())

    # Nobody gets anything by default; grants below are explicit.
    op.execute("REVOKE ALL ON SCHEMA retail FROM PUBLIC")
    op.execute("REVOKE ALL ON SCHEMA app FROM PUBLIC")

    # --- analyst: SELECT on approved retail tables only
    op.execute(f'GRANT USAGE ON SCHEMA retail TO "{analyst}"')
    for table in RETAIL_TABLES:
        op.execute(f'GRANT SELECT ON retail.{table} TO "{analyst}"')
    op.execute(f'ALTER ROLE "{analyst}" SET default_transaction_read_only = on')
    op.execute(f"ALTER ROLE \"{analyst}\" SET statement_timeout = '10s'")
    op.execute(f"ALTER ROLE \"{analyst}\" SET idle_in_transaction_session_timeout = '15s'")
    op.execute(f'ALTER ROLE "{analyst}" CONNECTION LIMIT 20')

    # --- app: read/write on the app schema only
    op.execute(f'GRANT USAGE ON SCHEMA app TO "{app_user}"')
    op.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA app TO "{app_user}"')
    op.execute(f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA app TO "{app_user}"')
    op.execute(f'ALTER ROLE "{app_user}" CONNECTION LIMIT 20')


def downgrade() -> None:
    settings = get_settings()
    for role in (settings.analyst_db_user, settings.app_db_user):
        # Roles are cluster-wide; only drop one if nothing else on the server depends on it.
        op.execute(f'REVOKE ALL ON ALL TABLES IN SCHEMA retail FROM "{role}"')
        op.execute(f'REVOKE ALL ON ALL TABLES IN SCHEMA app FROM "{role}"')
        op.execute(f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA app FROM "{role}"')
        op.execute(f'REVOKE ALL ON SCHEMA retail FROM "{role}"')
        op.execute(f'REVOKE ALL ON SCHEMA app FROM "{role}"')
