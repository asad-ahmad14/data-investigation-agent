"""Synthetic dataset: determinism, internal consistency, and the planted patterns.

Pure Python/pandas: no database needed.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from app.retail.generate import FIRST_MONTH, N_MONTHS, SEED, add_months, generate_dataset
from tests import expected as E


@pytest.fixture(scope="module")
def ds():
    return generate_dataset()


@pytest.fixture(scope="module")
def frame(ds):
    return E.item_frame(ds)


def test_generation_is_deterministic():
    a, b = generate_dataset(SEED), generate_dataset(SEED)
    assert a.counts() == b.counts()
    assert a.orders[:50] == b.orders[:50] and a.refunds[:50] == b.refunds[:50]
    assert generate_dataset(SEED + 1).counts() != a.counts()


def test_expected_table_sizes(ds):
    c = ds.counts()
    assert c["regions"] == 4 and c["categories"] == 5 and c["products"] == 30
    assert c["customers"] == 800
    assert 3000 < c["orders"] < 4500 and c["order_items"] > c["orders"] and c["refunds"] > 0


def test_ids_unique_and_foreign_keys_resolve(ds):
    for rows in (
        ds.regions,
        ds.categories,
        ds.products,
        ds.customers,
        ds.orders,
        ds.order_items,
        ds.refunds,
    ):
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))
    region_ids = {r["id"] for r in ds.regions}
    category_ids = {r["id"] for r in ds.categories}
    product_ids = {r["id"] for r in ds.products}
    customer_ids = {r["id"] for r in ds.customers}
    order_ids = {r["id"] for r in ds.orders}
    item_ids = {r["id"] for r in ds.order_items}
    assert all(c["region_id"] in region_ids for c in ds.customers)
    assert all(p["category_id"] in category_ids for p in ds.products)
    assert all(o["customer_id"] in customer_ids for o in ds.orders)
    assert all(
        i["order_id"] in order_ids and i["product_id"] in product_ids for i in ds.order_items
    )
    assert all(r["order_item_id"] in item_ids for r in ds.refunds)


def test_order_totals_equal_sum_of_their_items(ds):
    totals: dict[int, Decimal] = {}
    discounts: dict[int, Decimal] = {}
    for i in ds.order_items:
        totals[i["order_id"]] = (
            totals.get(i["order_id"], Decimal(0)) + i["quantity"] * i["unit_price"] - i["discount"]
        )
        discounts[i["order_id"]] = discounts.get(i["order_id"], Decimal(0)) + i["discount"]
    assert set(totals) == {o["id"] for o in ds.orders}  # no order without items
    for o in ds.orders:
        assert o["order_total"] == totals[o["id"]]
        assert o["discount_total"] == discounts[o["id"]]


def test_money_has_two_decimals_and_discounts_stay_within_the_line(ds):
    for i in ds.order_items:
        assert i["unit_price"] == i["unit_price"].quantize(Decimal("0.01"))
        assert Decimal(0) <= i["discount"] <= i["quantity"] * i["unit_price"]
        assert i["quantity"] > 0


def test_refund_rules(ds):
    items = {i["id"]: i for i in ds.order_items}
    orders = {o["id"]: o for o in ds.orders}
    seen = set()
    for r in ds.refunds:
        assert r["order_item_id"] not in seen  # at most one refund per item
        seen.add(r["order_item_id"])
        item = items[r["order_item_id"]]
        order = orders[item["order_id"]]
        assert order["status"] == "completed"
        assert Decimal(0) < r["amount"] <= item["quantity"] * item["unit_price"] - item["discount"]
        assert r["refund_date"] > order["order_date"]


def test_dates_are_plausible(ds):
    customers = {c["id"]: c for c in ds.customers}
    last = add_months(FIRST_MONTH, N_MONTHS)
    for o in ds.orders:
        assert FIRST_MONTH <= o["order_date"] < last
        assert o["order_date"] > customers[o["customer_id"]]["signup_date"]
    assert all(c["signup_date"] < FIRST_MONTH for c in ds.customers)


def test_planted_west_disruption_is_the_largest_monthly_decline(frame):
    monthly = E.completed(frame).groupby("month")["net"].sum()
    drops = monthly.diff().dropna().sort_values()
    assert drops.index[0] == pd.Timestamp(date(2025, 6, 1))
    assert drops.iloc[0] < 1.5 * drops.iloc[1]  # clearly the biggest, not a near-tie


def test_planted_west_electronics_collapse(frame):
    we = E.completed(frame)
    we = we[(we.region == "West") & (we.category == "Electronics")].groupby("month")["net"].sum()
    baseline = we["2025-03-01":"2025-05-01"].mean()
    disrupted = we["2025-06-01":"2025-08-01"].mean()
    assert disrupted < 0.35 * baseline


def test_planted_apparel_refund_spike(frame):
    c = E.completed(frame)
    ap = c[c.category == "Apparel"].groupby("month")[["refund_amount", "net"]].sum()
    rate = ap["refund_amount"] / ap["net"]
    spike = rate[[pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")]].min()
    normal = rate.drop([pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")]).max()
    assert spike > 2 * normal


def test_planted_home_discount_promotion(frame):
    c = E.completed(frame)
    home = c[c.category == "Home"].groupby("month")[["discount", "gross"]].sum()
    rate = home["discount"] / home["gross"]
    promo = rate[pd.Timestamp("2025-11-01")]
    normal = rate.drop(pd.Timestamp("2025-11-01")).max()
    assert promo > 2 * normal
