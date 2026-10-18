"""Schema registry: the explicit, closed list of what the agent may analyse.

Everything an investigation can ask for must appear here: tables, metrics, dimensions, and
operations. The planner validator and the query compiler both read this module, so an LLM
inventing a column or metric is rejected rather than trusted.

The registry is checked against the live database by an integration test, so it cannot
silently drift from the real schema.

Analysis model: every query aggregates the fact table `order_items` joined many-to-one to
orders, products, categories, customers, regions, plus an optional one-to-one refunds row.
Because no join can multiply rows, SUM/COUNT results are not inflated by the joins.
"""

from dataclasses import dataclass

from sqlalchemy import ColumnElement, Date, func
from sqlalchemy import distinct as sql_distinct

from app.db.models import (
    RETAIL,
    RETAIL_TABLES,
    Base,
    Category,
    Customer,
    Order,
    OrderItem,
    Product,
    Refund,
    Region,
)

OPERATIONS = {
    "metric_by_period": "One metric over time, grouped by calendar month.",
    "metric_by_dimension": "One metric grouped by a dimension over a period.",
    "compare_periods": (
        "One metric in a focus period versus a baseline period, optionally by dimension."
    ),
    "contribution_to_change": "How much each group contributed to the change between two periods.",
}

OPERATION_QUERY_COST = {
    "metric_by_period": 1,
    "metric_by_dimension": 1,
    "compare_periods": 2,
    "contribution_to_change": 2,
}


@dataclass(frozen=True)
class ColumnDef:
    name: str
    type: str


@dataclass(frozen=True)
class TableDef:
    name: str
    description: str
    columns: tuple[ColumnDef, ...]


@dataclass(frozen=True)
class MetricDef:
    name: str
    description: str
    unit: str  # "currency" | "count" | "fraction"
    numerator: ColumnElement
    denominator: ColumnElement | None = None
    # Dimensions over which group values add up to the overall value. Needed to decide
    # whether a "contribution to change" decomposition is mathematically meaningful.
    additive_dimensions: frozenset[str] = frozenset()

    @property
    def is_ratio(self) -> bool:
        return self.denominator is not None


@dataclass(frozen=True)
class DimensionDef:
    name: str
    description: str
    expression: ColumnElement
    filterable: bool = True


_completed = Order.status == "completed"
_line_gross = OrderItem.quantity * OrderItem.unit_price
_line_net = _line_gross - OrderItem.discount

_net_revenue = func.sum(_line_net).filter(_completed)
_gross_revenue = func.sum(_line_gross).filter(_completed)
_discount_amount = func.sum(OrderItem.discount).filter(_completed)
_orders_completed = func.count(sql_distinct(Order.id)).filter(_completed)
_orders_cancelled = func.count(sql_distinct(Order.id)).filter(Order.status == "cancelled")
_orders_all = func.count(sql_distinct(Order.id))

_ALL_DIMENSIONS = frozenset({"month", "region", "category", "product", "customer_segment"})
# An order can contain items from several categories/products, so order counts only add up
# across dimensions where each order belongs to exactly one group.
_ORDER_DIMENSIONS = frozenset({"month", "region", "customer_segment"})

METRICS: dict[str, MetricDef] = {
    m.name: m
    for m in (
        MetricDef(
            "net_revenue",
            "Sales after discounts, completed orders only (before refunds).",
            "currency",
            _net_revenue,
            additive_dimensions=_ALL_DIMENSIONS,
        ),
        MetricDef(
            "gross_revenue",
            "Sales before discounts, completed orders only.",
            "currency",
            _gross_revenue,
            additive_dimensions=_ALL_DIMENSIONS,
        ),
        MetricDef(
            "discount_amount",
            "Total discount value given, completed orders only.",
            "currency",
            _discount_amount,
            additive_dimensions=_ALL_DIMENSIONS,
        ),
        MetricDef(
            "units_sold",
            "Units sold, completed orders only.",
            "count",
            func.sum(OrderItem.quantity).filter(_completed),
            additive_dimensions=_ALL_DIMENSIONS,
        ),
        MetricDef(
            "order_count",
            "Distinct completed orders.",
            "count",
            _orders_completed,
            additive_dimensions=_ORDER_DIMENSIONS,
        ),
        MetricDef(
            "refund_amount",
            "Value refunded for items in completed orders, attributed to the ORDER date.",
            "currency",
            func.coalesce(func.sum(Refund.amount).filter(_completed), 0),
            additive_dimensions=_ALL_DIMENSIONS,
        ),
        MetricDef(
            "avg_order_value",
            "Net revenue divided by completed orders.",
            "currency",
            _net_revenue,
            _orders_completed,
        ),
        MetricDef(
            "refund_rate",
            "Refunded value divided by net revenue (a fraction, 0.05 = 5%).",
            "fraction",
            func.coalesce(func.sum(Refund.amount).filter(_completed), 0),
            _net_revenue,
        ),
        MetricDef(
            "discount_rate",
            "Discount value divided by gross revenue (a fraction, 0.10 = 10%).",
            "fraction",
            _discount_amount,
            _gross_revenue,
        ),
        MetricDef(
            "cancellation_rate",
            "Cancelled orders divided by all orders (a fraction).",
            "fraction",
            _orders_cancelled,
            _orders_all,
        ),
    )
}

DIMENSIONS: dict[str, DimensionDef] = {
    d.name: d
    for d in (
        DimensionDef(
            "month",
            "Calendar month of the order date.",
            func.date_trunc("month", Order.order_date).cast(Date),
            filterable=False,
        ),
        DimensionDef("region", "Customer region.", Region.name),
        DimensionDef("category", "Product category.", Category.name),
        DimensionDef("product", "Product name.", Product.name),
        DimensionDef("customer_segment", "Customer segment.", Customer.segment),
    )
}

FILTERABLE_DIMENSIONS = frozenset(name for name, d in DIMENSIONS.items() if d.filterable)

APPROVED_TABLES = frozenset(RETAIL_TABLES)


def get_tables() -> dict[str, TableDef]:
    """Table and column definitions for the retail schema, taken from the ORM metadata."""
    tables: dict[str, TableDef] = {}
    for table in Base.metadata.tables.values():
        if table.schema != RETAIL:
            continue
        tables[table.name] = TableDef(
            name=table.name,
            description=(table.comment or ""),
            columns=tuple(ColumnDef(c.name, str(c.type)) for c in table.columns),
        )
    return tables


def schema_summary() -> dict:
    """JSON-serialisable description of what can be analysed (for the schema endpoint)."""
    return {
        "dataset_note": "Synthetic retail data. Metrics cover completed orders unless stated.",
        "tables": {
            name: [{"name": c.name, "type": c.type} for c in t.columns]
            for name, t in sorted(get_tables().items())
        },
        "metrics": [
            {
                "name": m.name,
                "description": m.description,
                "unit": m.unit,
                "ratio": m.is_ratio,
            }
            for m in METRICS.values()
        ],
        "dimensions": [
            {"name": d.name, "description": d.description, "filterable": d.filterable}
            for d in DIMENSIONS.values()
        ],
        "operations": [{"name": n, "description": desc} for n, desc in OPERATIONS.items()],
    }
