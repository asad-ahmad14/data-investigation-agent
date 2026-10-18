"""Deterministic stand-in for a real model. Used when LLM_PROVIDER=mock (the default).

It lets the whole system run, be demoed and be tested without an API key. It is rule-based
keyword matching, NOT intelligence: it recognises a metric word, an optional month and year, and
builds the same kind of plan a real model would be asked to produce. Its replies go through the
same schema parsing and plan validation as a real model's, so it exercises the real code path.

Known limits: no filters on named values, no quarters or relative dates ("last month"),
English only. Anything it cannot place it answers with needs_clarification or out_of_scope.
"""

import re
from datetime import date
from typing import Any

from app.llm.base import LLMRequest, LLMResponse, LLMUsage
from app.llm.errors import LLMError
from app.retail.registry import METRICS, OPERATION_QUERY_COST
from app.schemas.llm_outputs import PlannerResponse
from app.schemas.plan import AnalysisStep, HypothesisSpec, InvestigationPlan, Period

MODEL_NAME = "mock-planner-v1"

_MONTHS = {
    name: i + 1
    for i, name in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
    )
}
_MONTH_NAME_YEAR = re.compile(
    r"\b("
    + "|".join(_MONTHS)
    + r"|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?,?\s+(\d{4})\b"
)
_ISO_MONTH = re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])\b")
_SHORT = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9}
_SHORT.update({"oct": 10, "nov": 11, "dec": 12})

_INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above)|system prompt|drop table|delete from|"
    r"reveal .*instructions",
    re.IGNORECASE,
)
_VAGUE = re.compile(r"\b(why|what happened|drop|dropped|decline|declined|fell|fall|down|spike)\b")
_DIMENSION_WORDS = {
    "region": "region",
    "category": "category",
    "segment": "customer_segment",
    "product": "product",
}


def _next_month(d: date) -> date:
    return date(d.year + d.month // 12, d.month % 12 + 1, 1)


def _prev_month(d: date) -> date:
    return date(d.year - (d.month == 1), 12 if d.month == 1 else d.month - 1, 1)


def _month_period(first: date) -> Period:
    return Period(start=first, end=_next_month(first))


def detect_metric(q: str) -> str | None:
    if "refund" in q:
        return "refund_rate" if "rate" in q else "refund_amount"
    if "discount" in q:
        return "discount_rate" if "rate" in q else "discount_amount"
    if "cancel" in q:
        return "cancellation_rate"
    if "average order" in q or "aov" in q.split():
        return "avg_order_value"
    if "units" in q:
        return "units_sold"
    if "orders" in q or "order count" in q:
        return "order_count"
    if "gross" in q:
        return "gross_revenue"
    if "revenue" in q or "sales" in q:
        return "net_revenue"
    return None


def detect_months(q: str) -> list[date]:
    found: set[date] = set()
    for name, year in _MONTH_NAME_YEAR.findall(q):
        month = _MONTHS.get(name) or _SHORT[name]
        found.add(date(int(year), month, 1))
    for year, month in _ISO_MONTH.findall(q):
        found.add(date(int(year), int(month), 1))
    return sorted(found)


def _within(first: date, start: date, end: date) -> bool:
    return start <= first < end


class MockLLM:
    provider = "mock"
    model = MODEL_NAME

    def generate_json(self, request: LLMRequest) -> LLMResponse:
        if request.task != "plan":
            raise LLMError(f"MockLLM does not implement task '{request.task}'.")
        reply = self._plan(request.context)
        return LLMResponse(
            data=reply.model_dump(mode="json"), model=self.model, usage=LLMUsage(None, None)
        )

    # ------------------------------------------------------------------ planner
    def _plan(self, ctx: dict[str, Any]) -> PlannerResponse:
        question: str = ctx["question"]
        q = question.lower()
        data_start: date = ctx["data_start"]
        data_end: date = ctx["data_end_exclusive"]
        max_steps: int = ctx["max_steps"]
        max_queries: int = ctx["max_total_queries"]

        if _INJECTION.search(q):
            return PlannerResponse(
                outcome="out_of_scope",
                message="I can only plan analyses of the retail dataset (revenue, refunds, "
                "discounts, orders by month, region, category, product or segment).",
            )

        metric = detect_metric(q)
        if metric is None:
            if _VAGUE.search(q):
                return PlannerResponse(
                    outcome="needs_clarification",
                    message="Which metric do you mean: net revenue, refunds, discounts, "
                    "order count, or cancellations?",
                )
            return PlannerResponse(
                outcome="out_of_scope",
                message="That does not look like a question about this retail dataset.",
            )

        months = detect_months(q)
        for m in months:
            if not _within(m, data_start, data_end):
                return PlannerResponse(
                    outcome="needs_clarification",
                    message=f"The data covers {data_start:%B %Y} to "
                    f"{_prev_month(data_end):%B %Y}. Which month in that range do you mean?",
                )

        focus, baseline = self._choose_periods(months, ctx.get("prior_findings") or [])
        if focus is None:
            plan = InvestigationPlan(
                steps=[
                    AnalysisStep(
                        id="S1",
                        operation="metric_by_period",
                        metric=metric,
                        period=Period(start=data_start, end=data_end),
                        purpose="Find when the metric changed over the full history.",
                    )
                ]
            )
            return PlannerResponse(outcome="plan", plan=plan)
        if baseline.start < data_start:
            return PlannerResponse(
                outcome="needs_clarification",
                message="There is no earlier month in the data to compare against. "
                "Pick a later month or name two months.",
            )

        dims = [d for w, d in _DIMENSION_WORDS.items() if w in q] or [
            "region",
            "category",
            "customer_segment",
        ]
        return PlannerResponse(
            outcome="plan",
            plan=self._comparison_plan(metric, focus, baseline, dims, max_steps, max_queries),
        )

    @staticmethod
    def _choose_periods(
        months: list[date], prior_findings: list[dict]
    ) -> tuple[Period | None, Period | None]:
        if len(months) >= 2:
            return _month_period(months[-1]), _month_period(months[-2])
        if len(months) == 1:
            return _month_period(months[0]), _month_period(_prev_month(months[0]))
        for finding in prior_findings:
            decline = finding.get("largest_decline")
            if decline:
                return Period(**decline["focus"]), Period(**decline["baseline"])
        return None, None

    @staticmethod
    def _comparison_plan(
        metric: str,
        focus: Period,
        baseline: Period,
        dims: list[str],
        max_steps: int,
        max_queries: int,
    ) -> InvestigationPlan:
        defn = METRICS[metric]
        steps = [
            AnalysisStep(
                id="S1",
                operation="compare_periods",
                metric=metric,
                period=focus,
                comparison_period=baseline,
                purpose="Measure the overall change versus the baseline.",
            )
        ]
        hypotheses: list[HypothesisSpec] = []
        cost = OPERATION_QUERY_COST["compare_periods"]
        for dim in dims:
            splittable = not defn.is_ratio and dim in defn.additive_dimensions
            op = "contribution_to_change" if splittable else "compare_periods"
            if len(steps) >= max_steps or cost + OPERATION_QUERY_COST[op] > max_queries:
                break
            sid = f"S{len(steps) + 1}"
            steps.append(
                AnalysisStep(
                    id=sid,
                    operation=op,
                    metric=metric,
                    dimension=dim,
                    period=focus,
                    comparison_period=baseline,
                    purpose=f"Check whether the change is concentrated in particular {dim} groups.",
                )
            )
            cost += OPERATION_QUERY_COST[op]
            hypotheses.append(
                HypothesisSpec(
                    id=f"H{len(hypotheses) + 1}",
                    statement="The change is concentrated in specific "
                    f"{dim.replace('_', ' ')} groups.",
                    step_ids=[sid],
                )
            )
        return InvestigationPlan(steps=steps, hypotheses=hypotheses)
