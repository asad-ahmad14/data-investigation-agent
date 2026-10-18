"""Independent expected values, computed with pandas straight from the generated dataset.

Nothing here touches the database, the compiler or app.analytics, so comparing database-backed
results with these numbers is a genuine cross-check rather than the code agreeing with itself.
"""

import pandas as pd

from app.retail.generate import Dataset


def item_frame(ds: Dataset) -> pd.DataFrame:
    regions = {r["id"]: r["name"] for r in ds.regions}
    categories = {c["id"]: c["name"] for c in ds.categories}
    products = pd.DataFrame(ds.products).rename(columns={"id": "product_id", "name": "product"})
    products["category"] = products["category_id"].map(categories)
    customers = pd.DataFrame(ds.customers).rename(columns={"id": "customer_id"})
    customers["region"] = customers["region_id"].map(regions)
    orders = pd.DataFrame(ds.orders).rename(columns={"id": "order_id"})
    items = pd.DataFrame(ds.order_items).rename(columns={"id": "item_id"})
    refunds = pd.DataFrame(ds.refunds)[["order_item_id", "amount"]].rename(
        columns={"order_item_id": "item_id", "amount": "refund_amount"}
    )

    df = (
        items.merge(orders[["order_id", "customer_id", "order_date", "status"]], on="order_id")
        .merge(products[["product_id", "product", "category"]], on="product_id")
        .merge(customers[["customer_id", "region", "segment"]], on="customer_id")
        .merge(refunds, on="item_id", how="left")
    )
    df["refund_amount"] = df["refund_amount"].fillna(0).astype(float)
    df["unit_price"] = df["unit_price"].astype(float)
    df["discount"] = df["discount"].astype(float)
    df["gross"] = df["quantity"] * df["unit_price"]
    df["net"] = df["gross"] - df["discount"]
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    return df


def completed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "completed"]


def in_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Half-open [start, end) on order_date, matching the application's convention."""
    return df[(df["order_date"] >= start) & (df["order_date"] < end)]


def monthly_net_revenue(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    sel = completed(in_period(df, start, end))
    return sel.groupby("month")["net"].sum()


def rate_by(df: pd.DataFrame, group: str, numerator: str, denominator: str, start, end):
    sel = completed(in_period(df, start, end))
    g = sel.groupby(group)[[numerator, denominator]].sum()
    return g[numerator] / g[denominator]
