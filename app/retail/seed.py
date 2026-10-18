"""Load the deterministic synthetic dataset (app.retail.generate) into PostgreSQL.

`seed_database` replaces ALL retail data inside one transaction (TRUNCATE, then bulk insert), so
it is safe to rerun: you always end with exactly one copy of the dataset, never duplicates, and
a failure part-way leaves the previous data untouched.

Use the ADMIN engine: the analyst role is read-only by design and cannot write.
"""

from sqlalchemy import Engine, insert, text

from app.db.models import Category, Customer, Order, OrderItem, Product, Refund, Region
from app.retail.generate import SEED, Dataset, generate_dataset

# Parents before children, so foreign keys are satisfied during insert.
_LOAD_ORDER = (
    ("regions", Region),
    ("categories", Category),
    ("customers", Customer),
    ("products", Product),
    ("orders", Order),
    ("order_items", OrderItem),
    ("refunds", Refund),
)
_CHUNK = 5000


def seed_database(
    engine: Engine, dataset: Dataset | None = None, seed: int = SEED
) -> dict[str, int]:
    """Replace the retail data with a freshly generated (or supplied) dataset."""
    data = dataset or generate_dataset(seed)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE retail.refunds, retail.order_items, retail.orders, retail.products, "
                "retail.customers, retail.categories, retail.regions"
            )
        )
        for attr, model in _LOAD_ORDER:
            rows = getattr(data, attr)
            for i in range(0, len(rows), _CHUNK):
                conn.execute(insert(model), rows[i : i + _CHUNK])
    return data.counts()
