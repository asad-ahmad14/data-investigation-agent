"""SQLAlchemy ORM models.

Two schemas:
- `retail`: the synthetic business dataset the agent analyses. The analyst role can only SELECT.
- `app`:    investigation history written by the API (role `app_rw`).

The query compiler builds its SELECTs from these mapped classes, so table and column names come
from one source of truth rather than from strings in plans.

A naming convention gives every constraint and index a predictable name, which keeps Alembic
migrations stable and readable.
"""

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    UUID,
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

RETAIL = "retail"
APP = "app"

# Names of the retail tables the analyst role may read (and the only ones the registry exposes).
RETAIL_TABLES = (
    "regions",
    "customers",
    "categories",
    "products",
    "orders",
    "order_items",
    "refunds",
)

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# ---------------------------------------------------------------------------
# retail schema (synthetic data)
# ---------------------------------------------------------------------------


class Region(Base):
    __tablename__ = "regions"
    __table_args__ = {"schema": RETAIL}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(50), unique=True)


class Category(Base):
    __tablename__ = "categories"
    __table_args__ = {"schema": RETAIL}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(50), unique=True)


class Customer(Base):
    __tablename__ = "customers"
    __table_args__ = (
        CheckConstraint(
            "segment IN ('Consumer', 'Small Business', 'Corporate')", name="segment_valid"
        ),
        {"schema": RETAIL},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    region_id: Mapped[int] = mapped_column(ForeignKey(f"{RETAIL}.regions.id"), index=True)
    segment: Mapped[str] = mapped_column(String(30))
    signup_date: Mapped[date] = mapped_column(Date)


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("unit_price > 0", name="unit_price_positive"),
        {"schema": RETAIL},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    category_id: Mapped[int] = mapped_column(ForeignKey(f"{RETAIL}.categories.id"), index=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(10, 2))


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint("status IN ('completed', 'cancelled')", name="status_valid"),
        CheckConstraint("discount_total >= 0", name="discount_total_non_negative"),
        CheckConstraint("order_total >= 0", name="order_total_non_negative"),
        {"schema": RETAIL},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    customer_id: Mapped[int] = mapped_column(ForeignKey(f"{RETAIL}.customers.id"), index=True)
    order_date: Mapped[date] = mapped_column(Date, index=True)
    status: Mapped[str] = mapped_column(String(20))
    discount_total: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    order_total: Mapped[Decimal] = mapped_column(Numeric(12, 2))


class OrderItem(Base):
    __tablename__ = "order_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("unit_price > 0", name="unit_price_positive"),
        CheckConstraint("discount >= 0", name="discount_non_negative"),
        CheckConstraint("discount <= quantity * unit_price", name="discount_within_line"),
        {"schema": RETAIL},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    order_id: Mapped[int] = mapped_column(ForeignKey(f"{RETAIL}.orders.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey(f"{RETAIL}.products.id"), index=True)
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    discount: Mapped[Decimal] = mapped_column(Numeric(10, 2))


class Refund(Base):
    """At most one refund per order item (unique order_item_id), so a LEFT JOIN cannot fan out."""

    __tablename__ = "refunds"
    __table_args__ = (
        CheckConstraint("amount > 0", name="amount_positive"),
        {"schema": RETAIL},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    order_item_id: Mapped[int] = mapped_column(ForeignKey(f"{RETAIL}.order_items.id"), unique=True)
    refund_date: Mapped[date] = mapped_column(Date)
    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    reason: Mapped[str] = mapped_column(String(50))


# ---------------------------------------------------------------------------
# app schema (investigation history)
# ---------------------------------------------------------------------------

INVESTIGATION_STATUSES = (
    "queued",
    "running",
    "completed",
    "insufficient_evidence",
    "needs_clarification",
    "failed",
)


class Investigation(Base):
    __tablename__ = "investigations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'insufficient_evidence', "
            "'needs_clarification', 'failed')",
            name="status_valid",
        ),
        {"schema": APP},
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30))
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)


class InvestigationStep(Base):
    __tablename__ = "investigation_steps"
    __table_args__ = (
        UniqueConstraint(
            "investigation_id", "seq", name="uq_investigation_steps_investigation_seq"
        ),
        {"schema": APP},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{APP}.investigations.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    node: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    tool: Mapped[str | None] = mapped_column(String(50), nullable=True)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class Evidence(Base):
    __tablename__ = "evidence"
    __table_args__ = (
        UniqueConstraint("investigation_id", "evidence_key", name="uq_evidence_investigation_key"),
        {"schema": APP},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{APP}.investigations.id", ondelete="CASCADE"), index=True
    )
    evidence_key: Mapped[str] = mapped_column(String(40))
    metric: Mapped[str] = mapped_column(String(60))
    value: Mapped[Decimal | None] = mapped_column(Numeric(24, 6), nullable=True)
    unit: Mapped[str] = mapped_column(String(30))
    period: Mapped[dict] = mapped_column(JSONB, default=dict)
    filters: Mapped[dict] = mapped_column(JSONB, default=dict)
    source_operation: Mapped[dict] = mapped_column(JSONB, default=dict)
    result_status: Mapped[str] = mapped_column(String(30))


class Hypothesis(Base):
    __tablename__ = "hypotheses"
    __table_args__ = (
        CheckConstraint(
            "status IN ('supported', 'contradicted', 'inconclusive', 'not_testable')",
            name="status_valid",
        ),
        {"schema": APP},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    investigation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{APP}.investigations.id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20))
    evidence_keys: Mapped[list] = mapped_column(JSONB, default=list)
