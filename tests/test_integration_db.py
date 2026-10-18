"""Integration tests against a real PostgreSQL (see tests/conftest.py)."""

import uuid
from datetime import date

import pandas as pd
import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from app.analytics.compiler import CompiledQuery, compile_aggregate
from app.analytics.errors import (
    DatabaseUnavailableError,
    PlanValidationError,
    QueryExecutionError,
    QueryTimeoutError,
)
from app.analytics.executor import QueryExecutor
from app.analytics.operations import run_step
from app.analytics.validator import validate_plan
from app.db.engines import build_analyst_engine
from app.db.models import APP, RETAIL, RETAIL_TABLES, Base, Investigation, Order
from app.retail.registry import get_tables
from app.retail.seed import seed_database
from app.schemas.plan import AnalysisStep, FilterSpec, InvestigationPlan, Period
from tests import expected as E

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def executor(integration_db):
    s = integration_db.settings
    return QueryExecutor(
        integration_db.analyst, max_rows=s.max_result_rows, timeout_ms=s.query_timeout_ms
    )


@pytest.fixture(scope="module")
def frame(integration_db):
    return E.item_frame(integration_db.dataset)


def P(a, b):
    return Period(start=date.fromisoformat(a), end=date.fromisoformat(b))


def step(operation, **kw):
    base = {
        "id": "S1",
        "operation": operation,
        "metric": "net_revenue",
        "period": P("2024-01-01", "2026-01-01"),
    }
    base.update(kw)
    return AnalysisStep(**base)


# --------------------------------------------------------------------------- schema & data


def test_migrations_match_the_models(integration_db):
    """No drift: the migrated database equals what the ORM models describe."""
    with integration_db.admin.connect() as conn:
        ctx = MigrationContext.configure(
            conn,
            opts={
                "include_schemas": True,
                "compare_type": True,
                "include_name": lambda name, type_, parents: (
                    name in (RETAIL, APP) if type_ == "schema" else True
                ),
            },
        )
        assert compare_metadata(ctx, Base.metadata) == []


def test_registry_matches_live_retail_schema(integration_db):
    with integration_db.admin.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'retail'"
            )
        ).all()
    live: dict[str, set[str]] = {}
    for table, column in rows:
        live.setdefault(table, set()).add(column)
    registry = {name: {c.name for c in t.columns} for name, t in get_tables().items()}
    assert registry == live
    assert set(live) == set(RETAIL_TABLES)


def test_seed_loaded_expected_counts_and_is_rerunnable(integration_db):
    expected = integration_db.dataset.counts()

    def live_counts():
        with integration_db.admin.connect() as conn:
            return {
                t: conn.execute(text(f"SELECT count(*) FROM retail.{t}")).scalar() for t in expected
            }

    assert live_counts() == expected
    assert seed_database(integration_db.admin, integration_db.dataset) == expected
    assert live_counts() == expected  # rerun replaced the data; nothing duplicated


def test_sql_integrity_rules_hold(integration_db):
    checks = {
        "order totals match items": """
            SELECT count(*) FROM retail.orders o JOIN (
              SELECT order_id, sum(quantity*unit_price-discount) t, sum(discount) d
              FROM retail.order_items GROUP BY order_id) i ON i.order_id = o.id
            WHERE o.order_total <> i.t OR o.discount_total <> i.d""",
        "orders without items": """
            SELECT count(*) FROM retail.orders o
            WHERE NOT EXISTS (SELECT 1 FROM retail.order_items i WHERE i.order_id = o.id)""",
        "refund above line value": """
            SELECT count(*) FROM retail.refunds r JOIN retail.order_items i ON i.id = r.order_item_id
            WHERE r.amount > i.quantity*i.unit_price - i.discount""",
        "refunds on cancelled orders": """
            SELECT count(*) FROM retail.refunds r JOIN retail.order_items i ON i.id = r.order_item_id
            JOIN retail.orders o ON o.id = i.order_id WHERE o.status = 'cancelled'""",
        "orders before signup": """
            SELECT count(*) FROM retail.orders o JOIN retail.customers c ON c.id = o.customer_id
            WHERE o.order_date < c.signup_date""",
    }
    with integration_db.admin.connect() as conn:
        bad = {name: conn.execute(text(q)).scalar() for name, q in checks.items()}
    assert bad == {name: 0 for name in checks}


# --------------------------------------------------------------------------- roles & security


def test_analyst_can_read_approved_tables(integration_db):
    with integration_db.analyst.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM retail.orders")).scalar() > 0


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO retail.regions (id, name) VALUES (99, 'Hacked')",
        "UPDATE retail.orders SET order_total = 0",
        "DELETE FROM retail.orders",
        "TRUNCATE retail.orders CASCADE",
        "DROP TABLE retail.orders",
        "ALTER TABLE retail.orders ADD COLUMN x int",
        "CREATE TABLE retail.evil (id int)",
        "CREATE TABLE public.evil (id int)",
        "CREATE SCHEMA evil",
        "GRANT ALL ON retail.orders TO PUBLIC",
        "SELECT * FROM app.investigations",
    ],
)
def test_analyst_role_cannot_write_alter_or_read_app_schema(integration_db, statement):
    with pytest.raises(DBAPIError):
        with integration_db.analyst.begin() as conn:
            conn.execute(text(statement))
    with integration_db.admin.connect() as conn:  # and nothing changed
        assert conn.execute(text("SELECT count(*) FROM retail.orders")).scalar() == len(
            integration_db.dataset.orders
        )


def test_analyst_role_is_read_only_by_default_with_a_timeout(integration_db):
    with integration_db.admin.connect() as conn:
        cfg = (
            conn.execute(
                text("SELECT unnest(rolconfig) FROM pg_roles WHERE rolname = :r"),
                {"r": integration_db.settings.analyst_db_user},
            )
            .scalars()
            .all()
        )
    assert "default_transaction_read_only=on" in cfg
    assert any(c.startswith("statement_timeout=") for c in cfg)


def test_app_role_can_write_app_schema_but_not_touch_retail(integration_db):
    inv_id = uuid.uuid4()
    with integration_db.app.begin() as conn:
        conn.execute(
            Investigation.__table__.insert().values(
                id=inv_id, question="q", status="queued", params={}, llm_calls=0
            )
        )
        own = select(func.count()).select_from(Investigation).where(Investigation.id == inv_id)
        assert conn.execute(own).scalar() == 1
    for stmt in ("SELECT * FROM retail.orders", "DELETE FROM retail.orders"):
        with pytest.raises(DBAPIError):
            with integration_db.app.begin() as conn:
                conn.execute(text(stmt))
    with integration_db.app.begin() as conn:  # tidy up
        conn.execute(Investigation.__table__.delete().where(Investigation.id == inv_id))


def test_sql_injection_in_filter_is_harmless(integration_db, executor):
    evil = "West'; DROP TABLE retail.orders; --"
    out = run_step(
        step(
            "metric_by_dimension",
            dimension="region",
            filters=[FilterSpec(dimension="region", values=[evil])],
        ),
        executor,
    )
    assert out.status == "no_data" and out.result["groups"] == []
    with integration_db.admin.connect() as conn:
        assert conn.execute(select(func.count()).select_from(Order)).scalar() == len(
            integration_db.dataset.orders
        )


# --------------------------------------------------------------------------- executor behaviour


def test_executor_enforces_statement_timeout(integration_db):
    slow = QueryExecutor(integration_db.analyst, max_rows=10, timeout_ms=100)
    query = CompiledQuery(select(func.pg_sleep(2)), "x", None)
    with pytest.raises(QueryTimeoutError):
        slow.execute(query)


def test_executor_truncates_and_reports_it(executor):
    q = compile_aggregate(
        metric="net_revenue",
        dimension="product",
        start=date(2024, 1, 1),
        end=date(2026, 1, 1),
        max_rows=3,
    )
    result = QueryExecutor(executor._engine, max_rows=3, timeout_ms=5000).execute(q)
    assert len(result.rows) == 3 and result.truncated is True


def test_executor_error_messages_do_not_leak_sql(integration_db, executor):
    from sqlalchemy import insert

    from app.db.models import Region

    bad = CompiledQuery(insert(Region).values(id=99, name="x"), "x", None)  # type: ignore[arg-type]
    with pytest.raises(QueryExecutionError) as exc:
        executor.execute(bad)
    message = str(exc.value).lower()
    assert "insert" not in message and "regions" not in message and "permission" not in message


def test_executor_maps_connection_failure_to_retryable_error(integration_db):
    broken = build_analyst_engine(integration_db.settings.model_copy(update={"postgres_port": 1}))
    ex = QueryExecutor(broken, max_rows=10, timeout_ms=1000)
    q = compile_aggregate(
        metric="net_revenue",
        dimension=None,
        start=date(2025, 1, 1),
        end=date(2025, 2, 1),
        max_rows=10,
    )
    with pytest.raises(DatabaseUnavailableError) as exc:
        ex.execute(q)
    assert exc.value.retryable is True
    broken.dispose()


# --------------------------------------------------------------------------- acceptance questions
# Expected values come from pandas over the generated dataset (tests/expected.py), which uses no
# database, compiler or application code, so these are independent cross-checks.


def test_Q1_month_with_largest_revenue_decline(executor, frame):
    out = run_step(step("metric_by_period", dimension=None), executor)
    monthly = E.monthly_net_revenue(frame, "2024-01-01", "2026-01-01")
    got = {p["label"]: p["value"] for p in out.result["series"]}
    assert len(got) == 24
    for month, value in monthly.items():
        assert got[month.date().isoformat()] == pytest.approx(value, abs=0.01)
    worst = monthly.diff().dropna().idxmin()
    decline = out.result["largest_decline"]
    assert decline["label"] == worst.date().isoformat() == "2025-06-01"
    assert decline["absolute_change"] == pytest.approx(monthly.diff().min(), abs=0.01)


def test_Q2_category_with_highest_refund_rate(executor, frame):
    out = run_step(
        step("metric_by_dimension", metric="refund_rate", dimension="category"), executor
    )
    expected = E.rate_by(frame, "category", "refund_amount", "net", "2024-01-01", "2026-01-01")
    got = {g["group"]: g["value"] for g in out.result["groups"]}
    assert set(got) == set(expected.index)
    for category, rate in expected.items():
        assert got[category] == pytest.approx(rate, rel=1e-9)
    assert out.result["groups"][0]["group"] == expected.idxmax()  # sorted highest first


def test_Q3_compare_revenue_between_two_periods(executor, frame):
    s = step(
        "compare_periods",
        period=P("2025-07-01", "2025-08-01"),
        comparison_period=P("2025-05-01", "2025-06-01"),
    )
    overall = run_step(s, executor).result["overall"]
    new = E.completed(E.in_period(frame, "2025-07-01", "2025-08-01"))["net"].sum()
    old = E.completed(E.in_period(frame, "2025-05-01", "2025-06-01"))["net"].sum()
    assert overall["new"] == pytest.approx(new, abs=0.01)
    assert overall["old"] == pytest.approx(old, abs=0.01)
    assert overall["absolute_change"] == pytest.approx(new - old, abs=0.01)
    assert overall["relative_change"] == pytest.approx((new - old) / old, rel=1e-9)
    assert overall["percentage_point_change"] is None  # currency, not a rate


def test_Q4_region_contributing_most_to_the_decline(executor, frame):
    s = step(
        "contribution_to_change",
        dimension="region",
        period=P("2025-06-01", "2025-07-01"),
        comparison_period=P("2025-05-01", "2025-06-01"),
    )
    result = run_step(s, executor).result
    after = (
        E.completed(E.in_period(frame, "2025-06-01", "2025-07-01")).groupby("region")["net"].sum()
    )
    before = (
        E.completed(E.in_period(frame, "2025-05-01", "2025-06-01")).groupby("region")["net"].sum()
    )
    delta = after - before
    assert result["total_change"] == pytest.approx(delta.sum(), abs=0.01)
    top = result["contributions"][0]
    assert top["group"] == delta.abs().idxmax() == "West"
    assert top["change"] == pytest.approx(delta["West"], abs=0.01)
    assert sum(c["change"] for c in result["contributions"]) == pytest.approx(
        result["total_change"], abs=0.01
    )


def test_Q5_unavailable_metric_is_rejected_not_invented(executor):
    plan = InvestigationPlan(
        steps=[step("metric_by_dimension", metric="profit_margin", dimension="region")]
    )
    issues = validate_plan(plan, max_steps=6, max_total_queries=12)
    assert [i.code for i in issues] == ["unknown_metric"]
    with pytest.raises(PlanValidationError):
        run_step(plan.steps[0], executor)


def test_refund_rate_change_is_reported_in_percentage_points(executor, frame):
    s = step(
        "compare_periods",
        metric="refund_rate",
        period=P("2025-09-01", "2025-10-01"),
        comparison_period=P("2025-08-01", "2025-09-01"),
        filters=[FilterSpec(dimension="category", values=["Apparel"])],
    )
    overall = run_step(s, executor).result["overall"]

    def rate(a, b):
        sel = E.completed(E.in_period(frame, a, b))
        sel = sel[sel.category == "Apparel"]
        return sel["refund_amount"].sum() / sel["net"].sum()

    new, old = rate("2025-09-01", "2025-10-01"), rate("2025-08-01", "2025-09-01")
    assert overall["percentage_point_change"] == pytest.approx((new - old) * 100, rel=1e-9)
    assert overall["percentage_point_change"] > 10  # the planted spike
    assert overall["relative_change"] == pytest.approx((new - old) / old, rel=1e-9)


def test_filters_combine_with_and(executor, frame):
    s = step(
        "metric_by_period",
        dimension=None,
        filters=[
            FilterSpec(dimension="region", values=["West"]),
            FilterSpec(dimension="category", values=["Electronics"]),
        ],
    )
    out = run_step(s, executor)
    sel = E.completed(frame)
    sel = sel[(sel.region == "West") & (sel.category == "Electronics")]
    expected = sel.groupby("month")["net"].sum()
    for p in out.result["series"]:
        want = expected.get(pd.Timestamp(p["label"]))
        assert p["value"] == pytest.approx(want, abs=0.01)


def test_order_count_is_distinct_orders_not_line_items(executor, frame):
    out = run_step(step("metric_by_dimension", metric="order_count", dimension="region"), executor)
    sel = E.completed(frame).drop_duplicates("order_id")
    expected = sel.groupby("region").size()
    for g in out.result["groups"]:
        assert g["value"] == expected[g["group"]]


def test_executed_queries_are_recorded_for_the_audit_trail(executor):
    out = run_step(step("metric_by_dimension", dimension="region"), executor)
    q = out.queries[0]
    assert q.sql.lower().startswith("select") and "retail." in q.sql
    assert q.params and q.row_count == 4 and q.truncated is False
