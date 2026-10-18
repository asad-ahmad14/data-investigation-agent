"""Query compiler: turns an already-validated analysis request into a parameterized SELECT.

There is no string concatenation and no model-provided SQL anywhere. The compiler:
- builds one fixed join tree over approved retail tables (every join is many-to-one, plus one
  one-to-one LEFT JOIN to refunds, so rows are never multiplied);
- takes metric and dimension expressions only from app.retail.registry;
- passes every user-influenced value (dates, filter values) as a bound parameter;
- adds a LIMIT so a result can never be unbounded.

It re-checks names against the registry even though the validator already did, so a bug or a
bypass upstream cannot reach the database.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import Select, func, null, select
from sqlalchemy import distinct as sql_distinct
from sqlalchemy.dialects import postgresql

from app.analytics.errors import PlanValidationError
from app.db.models import Category, Customer, Order, OrderItem, Product, Refund, Region
from app.retail.registry import DIMENSIONS, FILTERABLE_DIMENSIONS, METRICS

MAX_FILTER_VALUES = 10
MAX_FILTER_VALUE_LENGTH = 60


@dataclass(frozen=True)
class CompiledQuery:
    statement: Select
    metric: str
    dimension: str | None

    def sql_text(self) -> str:
        """SQL with named placeholders, for audit trails. Values are in `params()`."""
        return str(self.statement.compile(dialect=postgresql.dialect()))

    def params(self) -> dict:
        compiled = self.statement.compile(dialect=postgresql.dialect())
        return {
            k: (v.isoformat() if isinstance(v, date) else v) for k, v in compiled.params.items()
        }


def compile_aggregate(
    *,
    metric: str,
    dimension: str | None,
    start: date,
    end: date,
    filters: Mapping[str, Sequence[str]] | None = None,
    max_rows: int,
) -> CompiledQuery:
    metric_def = METRICS.get(metric)
    if metric_def is None:
        raise PlanValidationError(f"Unknown metric '{metric}'.")

    dimension_def = None
    if dimension is not None:
        dimension_def = DIMENSIONS.get(dimension)
        if dimension_def is None:
            raise PlanValidationError(f"Unknown dimension '{dimension}'.")

    if end <= start:
        raise PlanValidationError("Period end must be after start.")
    if max_rows < 1:
        raise PlanValidationError("max_rows must be positive.")

    columns = []
    if dimension_def is not None:
        columns.append(dimension_def.expression.label("dimension"))
    columns.append(metric_def.numerator.label("numerator"))
    columns.append((metric_def.denominator if metric_def.is_ratio else null()).label("denominator"))
    columns.append(func.count(sql_distinct(Order.id)).label("n_orders"))

    stmt = (
        select(*columns)
        .select_from(OrderItem)
        .join(Order, Order.id == OrderItem.order_id)
        .join(Product, Product.id == OrderItem.product_id)
        .join(Category, Category.id == Product.category_id)
        .join(Customer, Customer.id == Order.customer_id)
        .join(Region, Region.id == Customer.region_id)
        .outerjoin(Refund, Refund.order_item_id == OrderItem.id)
        .where(Order.order_date >= start, Order.order_date < end)
    )

    for filter_dimension, values in (filters or {}).items():
        if filter_dimension not in FILTERABLE_DIMENSIONS:
            raise PlanValidationError(f"Cannot filter on '{filter_dimension}'.")
        values = list(values)
        if not values or len(values) > MAX_FILTER_VALUES:
            raise PlanValidationError("A filter needs between 1 and 10 values.")
        if any(not isinstance(v, str) or not v or len(v) > MAX_FILTER_VALUE_LENGTH for v in values):
            raise PlanValidationError("Filter values must be short, non-empty strings.")
        stmt = stmt.where(DIMENSIONS[filter_dimension].expression.in_(values))

    if dimension_def is not None:
        stmt = stmt.group_by(dimension_def.expression).order_by(dimension_def.expression)

    # One extra row lets the executor detect truncation without fetching everything.
    stmt = stmt.limit(max_rows + 1)
    return CompiledQuery(stmt, metric, dimension)
