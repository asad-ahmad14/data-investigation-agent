"""Deterministic synthetic retail dataset.

THIS DATA IS SYNTHETIC. It is generated from a fixed random seed, so the same code always
produces the same rows, and expected analytical answers can be tested.

Limitations (documented honestly in docs/DATABASE_DESIGN.md):
- Demand is uniform apart from the planted patterns below; real data has trends, seasonality
  and correlated behaviour that this dataset does not.
- Customers and products are chosen uniformly at random, so there are no loyalty or
  cross-sell effects.
- Prices never change over time.

Planted patterns (the "ground truth" the agent should be able to discover):
1. WEST DISRUPTION (2025-06 .. 2025-08): West-region orders are dropped with probability
   WEST_ORDER_DROP and Electronics items in the West are dropped with probability
   WEST_ELECTRONICS_DROP, so revenue falls, concentrated in West Electronics.
2. APPAREL REFUND SPIKE (2025-09 .. 2025-10): the refund probability for Apparel items rises
   from BASE_REFUND_PROB to APPAREL_SPIKE_REFUND_PROB.
3. HOME DISCOUNT PROMOTION (2025-11): Home items are discounted much more often and deeper.

All random numbers for an item/order are drawn unconditionally, so changing a planted
parameter never shifts the rest of the data.
"""

import calendar
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

SEED = 20260101
FIRST_MONTH = date(2024, 1, 1)
N_MONTHS = 24
N_CUSTOMERS = 800
BASE_ORDERS_PER_MONTH = 150
PRODUCTS_PER_CATEGORY = 6

REGIONS = ["North", "South", "East", "West"]
CATEGORY_PRICE_RANGES = {
    "Electronics": (80, 900),
    "Home": (15, 250),
    "Apparel": (12, 140),
    "Beauty": (6, 70),
    "Sports": (20, 300),
}
SEGMENTS = ["Consumer", "Small Business", "Corporate"]
SEGMENT_WEIGHTS = [60, 30, 10]
REFUND_REASONS = ["damaged", "wrong_size", "not_as_described", "changed_mind", "late_delivery"]

CANCEL_PROB = 0.04
BASE_DISCOUNT_PROB = 0.20
BASE_DISCOUNT_PCT = Decimal("0.10")
BASE_REFUND_PROB = 0.04

WEST_ORDER_DROP = 0.30
WEST_ELECTRONICS_DROP = 0.85
APPAREL_SPIKE_REFUND_PROB = 0.28
HOME_PROMO_DISCOUNT_PROB = 0.80
HOME_PROMO_DISCOUNT_PCT = Decimal("0.25")


def add_months(first_of_month: date, months: int) -> date:
    index = first_of_month.year * 12 + (first_of_month.month - 1) + months
    return date(index // 12, index % 12 + 1, 1)


def month_range(first: date, last: date) -> frozenset[date]:
    months = []
    cursor = first
    while cursor <= last:
        months.append(cursor)
        cursor = add_months(cursor, 1)
    return frozenset(months)


WEST_DISRUPTION_MONTHS = month_range(date(2025, 6, 1), date(2025, 8, 1))
APPAREL_SPIKE_MONTHS = month_range(date(2025, 9, 1), date(2025, 10, 1))
HOME_PROMO_MONTHS = month_range(date(2025, 11, 1), date(2025, 11, 1))


def money(value: Decimal | float) -> Decimal:
    return Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass
class Dataset:
    regions: list[dict]
    categories: list[dict]
    products: list[dict]
    customers: list[dict]
    orders: list[dict]
    order_items: list[dict]
    refunds: list[dict]

    def counts(self) -> dict[str, int]:
        return {
            "regions": len(self.regions),
            "categories": len(self.categories),
            "products": len(self.products),
            "customers": len(self.customers),
            "orders": len(self.orders),
            "order_items": len(self.order_items),
            "refunds": len(self.refunds),
        }


def generate_dataset(seed: int = SEED) -> Dataset:
    rng = random.Random(seed)  # noqa: S311  (synthetic data, not security-sensitive)

    regions = [{"id": i + 1, "name": name} for i, name in enumerate(REGIONS)]
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CATEGORY_PRICE_RANGES)]
    category_name = {c["id"]: c["name"] for c in categories}

    products: list[dict] = []
    for category in categories:
        low, high = CATEGORY_PRICE_RANGES[category["name"]]
        for n in range(1, PRODUCTS_PER_CATEGORY + 1):
            products.append(
                {
                    "id": len(products) + 1,
                    "category_id": category["id"],
                    "name": f"{category['name']} Item {n}",
                    "unit_price": money(rng.uniform(low, high)),
                }
            )

    customers = []
    for i in range(N_CUSTOMERS):
        customers.append(
            {
                "id": i + 1,
                "region_id": regions[i % len(regions)]["id"],
                "segment": rng.choices(SEGMENTS, SEGMENT_WEIGHTS)[0],
                "signup_date": FIRST_MONTH - timedelta(days=rng.randint(1, 700)),
            }
        )
    region_name = {r["id"]: r["name"] for r in regions}

    orders: list[dict] = []
    order_items: list[dict] = []
    refunds: list[dict] = []

    for month_index in range(N_MONTHS):
        month = add_months(FIRST_MONTH, month_index)
        days_in_month = calendar.monthrange(month.year, month.month)[1]
        n_orders = BASE_ORDERS_PER_MONTH + rng.randint(-8, 8)

        for _ in range(n_orders):
            customer = customers[rng.randrange(len(customers))]
            order_date = month.replace(day=rng.randint(1, days_in_month))
            drop_roll = rng.random()
            cancel_roll = rng.random()
            n_items = rng.choices([1, 2, 3, 4], [50, 30, 15, 5])[0]
            specs = []
            for _ in range(n_items):
                specs.append(
                    {
                        "product": products[rng.randrange(len(products))],
                        "quantity": rng.choices([1, 2, 3], [70, 20, 10])[0],
                        "discount_roll": rng.random(),
                        "electronics_roll": rng.random(),
                        "refund_roll": rng.random(),
                        "refund_full_roll": rng.random(),
                        "refund_delay": rng.randint(1, 30),
                        "refund_reason": rng.choice(REFUND_REASONS),
                    }
                )

            in_west_disruption = (
                region_name[customer["region_id"]] == "West" and month in WEST_DISRUPTION_MONTHS
            )
            if in_west_disruption and drop_roll < WEST_ORDER_DROP:
                continue

            kept = []
            for spec in specs:
                cat = category_name[spec["product"]["category_id"]]
                if (
                    in_west_disruption
                    and cat == "Electronics"
                    and spec["electronics_roll"] < WEST_ELECTRONICS_DROP
                ):
                    continue
                kept.append(spec)
            if not kept:
                continue

            order_id = len(orders) + 1
            status = "cancelled" if cancel_roll < CANCEL_PROB else "completed"
            order_discount = Decimal("0.00")
            order_net = Decimal("0.00")

            for spec in kept:
                product = spec["product"]
                cat = category_name[product["category_id"]]
                quantity = spec["quantity"]
                unit_price = product["unit_price"]
                gross = unit_price * quantity

                if cat == "Home" and month in HOME_PROMO_MONTHS:
                    disc_prob, disc_pct = HOME_PROMO_DISCOUNT_PROB, HOME_PROMO_DISCOUNT_PCT
                else:
                    disc_prob, disc_pct = BASE_DISCOUNT_PROB, BASE_DISCOUNT_PCT
                discount = (
                    money(gross * disc_pct) if spec["discount_roll"] < disc_prob else money(0)
                )
                line_net = gross - discount

                item_id = len(order_items) + 1
                order_items.append(
                    {
                        "id": item_id,
                        "order_id": order_id,
                        "product_id": product["id"],
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "discount": discount,
                    }
                )
                order_discount += discount
                order_net += line_net

                if status == "completed":
                    if cat == "Apparel" and month in APPAREL_SPIKE_MONTHS:
                        refund_prob = APPAREL_SPIKE_REFUND_PROB
                    else:
                        refund_prob = BASE_REFUND_PROB
                    if spec["refund_roll"] < refund_prob:
                        amount = line_net if spec["refund_full_roll"] < 0.7 else money(line_net / 2)
                        refunds.append(
                            {
                                "id": len(refunds) + 1,
                                "order_item_id": item_id,
                                "refund_date": order_date + timedelta(days=spec["refund_delay"]),
                                "amount": amount,
                                "reason": spec["refund_reason"],
                            }
                        )

            orders.append(
                {
                    "id": order_id,
                    "customer_id": customer["id"],
                    "order_date": order_date,
                    "status": status,
                    "discount_total": order_discount,
                    "order_total": order_net,
                }
            )

    return Dataset(regions, categories, products, customers, orders, order_items, refunds)
