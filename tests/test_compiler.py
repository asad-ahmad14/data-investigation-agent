import re
from datetime import date

import pytest
from sqlalchemy import Select

from app.analytics.compiler import compile_aggregate
from app.analytics.errors import PlanValidationError
from app.db.models import RETAIL_TABLES

START, END = date(2025, 1, 1), date(2025, 2, 1)


def compile_(**kw):
    args = {
        "metric": "net_revenue",
        "dimension": "region",
        "start": START,
        "end": END,
        "filters": None,
        "max_rows": 100,
    }
    args.update(kw)
    return compile_aggregate(**args)


def test_returns_a_select_over_approved_retail_tables_only():
    q = compile_(filters={"region": ["West"]})
    assert isinstance(q.statement, Select)
    sql = q.sql_text().lower()
    tables = set(re.findall(r"retail\.(\w+)", sql))
    assert tables and tables <= set(RETAIL_TABLES)
    assert "app." not in sql and "pg_" not in sql and "information_schema" not in sql


def test_every_metric_and_dimension_combination_compiles():
    from app.retail.registry import DIMENSIONS, METRICS

    for metric in METRICS:
        for dimension in [None, *DIMENSIONS]:
            assert (
                compile_(metric=metric, dimension=dimension).sql_text().lower().startswith("select")
            )


def test_injection_text_is_bound_as_a_parameter_never_in_sql():
    evil = "West'; DROP TABLE retail.orders; --"
    q = compile_(filters={"region": [evil]})
    assert evil not in q.sql_text()
    assert "drop table" not in q.sql_text().lower()
    assert evil in q.params().values() or any(
        evil in v for v in q.params().values() if isinstance(v, list)
    )


def test_dates_and_limit_are_parameters_and_limit_is_max_rows_plus_one():
    q = compile_(max_rows=7)
    params = q.params()
    assert "2025-01-01" in params.values() and "2025-02-01" in params.values()
    assert 8 in params.values()
    assert "limit" in q.sql_text().lower()


def test_unknown_metric_or_dimension_rejected():
    with pytest.raises(PlanValidationError):
        compile_(metric="profit_margin")
    with pytest.raises(PlanValidationError):
        compile_(dimension="shoe_size")


def test_metric_name_cannot_smuggle_sql():
    with pytest.raises(PlanValidationError):
        compile_(metric="net_revenue; DROP TABLE retail.orders")


@pytest.mark.parametrize(
    "filters",
    [
        {"month": ["2025-01-01"]},  # not filterable
        {"password": ["x"]},  # unknown
        {"region": []},  # empty
        {"region": [str(i) for i in range(11)]},  # too many
        {"region": ["x" * 61]},  # too long
        {"region": [""]},  # empty string
    ],
)
def test_invalid_filters_rejected(filters):
    with pytest.raises(PlanValidationError):
        compile_(filters=filters)


def test_bad_period_and_row_limit_rejected():
    with pytest.raises(PlanValidationError):
        compile_(start=END, end=START)
    with pytest.raises(PlanValidationError):
        compile_(max_rows=0)


def test_no_dimension_means_no_group_by():
    q = compile_(dimension=None)
    assert "group by" not in q.sql_text().lower()
