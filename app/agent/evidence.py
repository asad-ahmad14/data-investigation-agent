"""Turn executed analysis steps into numbered, citeable evidence.

Evidence is the only source of numbers in a report. Each item stores the value, the period and
filters it describes, and which operation produced it. Descriptions are built here from the
computed values with fixed templates; no language model writes them.
"""

from dataclasses import dataclass, field

from app.agent.formatting import fmt_change, fmt_value, month_name, period_text
from app.analytics.operations import StepOutput


@dataclass(frozen=True)
class EvidenceItem:
    key: str
    step_id: str
    metric: str
    unit: str
    value: float | None
    period: dict
    filters: dict
    source_operation: dict
    result_status: str  # ok | no_data | baseline_zero | no_decline
    description: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "step_id": self.step_id,
            "metric": self.metric,
            "unit": self.unit,
            "value": self.value,
            "period": self.period,
            "filters": self.filters,
            "source_operation": self.source_operation,
            "result_status": self.result_status,
            "description": self.description,
            "detail": self.detail,
        }


def _source(out: StepOutput) -> dict:
    return {
        "step_id": out.step_id,
        "operation": out.operation,
        "dimension": out.dimension,
        "queries": [q.label for q in out.queries],
    }


def _label(metric: str) -> str:
    return metric.replace("_", " ")


def extract_evidence(out: StepOutput, first_index: int) -> list[EvidenceItem]:
    """Evidence for one step. Keys are E<first_index>, E<first_index+1>, ..."""
    builders = {
        "metric_by_period": _from_period_series,
        "metric_by_dimension": _from_dimension,
        "compare_periods": _from_comparison,
        "contribution_to_change": _from_contribution,
    }
    items = builders[out.operation](out, first_index)
    return items


def _no_data(out: StepOutput, key: str, period: dict) -> EvidenceItem:
    return EvidenceItem(
        key=key,
        step_id=out.step_id,
        metric=out.metric,
        unit=out.unit,
        value=None,
        period=period,
        filters=out.filters,
        source_operation=_source(out),
        result_status="no_data",
        description=f"No data was returned for {_label(out.metric)} in this analysis.",
    )


def _from_period_series(out: StepOutput, i: int) -> list[EvidenceItem]:
    key = f"E{i}"
    series = out.result.get("series", [])
    period = {"label": f"{series[0]['label']} to {series[-1]['label']}"} if series else {}
    if out.status == "no_data":
        return [_no_data(out, key, period)]
    decline = out.result.get("largest_decline")
    if not decline:
        return [
            EvidenceItem(
                key=key,
                step_id=out.step_id,
                metric=out.metric,
                unit=out.unit,
                value=None,
                period=period,
                filters=out.filters,
                source_operation=_source(out),
                result_status="no_decline",
                description=f"No month-over-month decline in {_label(out.metric)} was found "
                "in this period.",
            )
        ]
    desc = (
        f"The largest month-over-month decline in {_label(out.metric)} was "
        f"{month_name(decline['previous_label'])} to {month_name(decline['label'])}: "
        f"{fmt_value(decline['old'], out.unit)} to {fmt_value(decline['new'], out.unit)} "
        f"({fmt_change(decline['absolute_change'], decline['relative_change'], out.unit)})."
    )
    return [
        EvidenceItem(
            key=key,
            step_id=out.step_id,
            metric=out.metric,
            unit=out.unit,
            value=decline["absolute_change"],
            period={"label": f"{decline['previous_label']} to {decline['label']}"},
            filters=out.filters,
            source_operation=_source(out),
            result_status="ok",
            description=desc,
            detail={
                "kind": "largest_decline",
                "baseline_month": decline["previous_label"],
                "focus_month": decline["label"],
                "old": decline["old"],
                "new": decline["new"],
                "relative_change": decline["relative_change"],
            },
        )
    ]


def _from_dimension(out: StepOutput, i: int) -> list[EvidenceItem]:
    key = f"E{i}"
    period = {"label": out.result.get("period", "")}
    groups = [g for g in out.result.get("groups", []) if g["value"] is not None]
    if out.status == "no_data" or not groups:
        return [_no_data(out, key, period)]
    top = groups[0]
    return [
        EvidenceItem(
            key=key,
            step_id=out.step_id,
            metric=out.metric,
            unit=out.unit,
            value=top["value"],
            period=period,
            filters={**out.filters, out.dimension or "group": [top["group"]]},
            source_operation=_source(out),
            result_status="ok",
            description=f"Highest {_label(out.metric)} by {out.dimension}: {top['group']} at "
            f"{fmt_value(top['value'], out.unit)} ({top['n_orders']} orders).",
            detail={"kind": "top_group", "small_sample": top["small_sample"]},
        )
    ]


def _from_comparison(out: StepOutput, i: int) -> list[EvidenceItem]:
    key = f"E{i}"
    res = out.result
    period = {"focus": res.get("focus_period"), "baseline": res.get("baseline_period")}
    if out.status == "no_data":
        return [_no_data(out, key, period)]

    if out.dimension is None:
        o = res["overall"]
        if o["status"] == "no_data":
            return [_no_data(out, key, period)]
        change = fmt_change(o["absolute_change"], o["relative_change"], out.unit)
        if out.unit == "fraction" and o["percentage_point_change"] is not None:
            change = fmt_change(o["percentage_point_change"] / 100, None, out.unit)
        return [
            EvidenceItem(
                key=key,
                step_id=out.step_id,
                metric=out.metric,
                unit=out.unit,
                value=o["absolute_change"],
                period=period,
                filters=out.filters,
                source_operation=_source(out),
                result_status=o["status"],
                description=f"{_label(out.metric).capitalize()} changed from "
                f"{fmt_value(o['old'], out.unit)} to {fmt_value(o['new'], out.unit)} "
                f"({change}) from {period_text(res.get('baseline_period'))} to "
                f"{period_text(res.get('focus_period'))}.",
                detail={
                    "kind": "overall_change",
                    "old": o["old"],
                    "new": o["new"],
                    "relative_change": o["relative_change"],
                    "percentage_point_change": o["percentage_point_change"],
                    "n_orders_focus": o["n_orders_focus"],
                    "n_orders_baseline": o["n_orders_baseline"],
                },
            )
        ]

    groups = [g for g in res.get("groups", []) if g["absolute_change"] is not None]
    if not groups:
        return [_no_data(out, key, period)]
    top = groups[0]
    second = abs(groups[1]["absolute_change"]) if len(groups) > 1 else 0.0
    change = fmt_change(top["absolute_change"], top["relative_change"], out.unit)
    return [
        EvidenceItem(
            key=key,
            step_id=out.step_id,
            metric=out.metric,
            unit=out.unit,
            value=top["absolute_change"],
            period=period,
            filters={**out.filters, out.dimension: [top["group"]]},
            source_operation=_source(out),
            result_status=top["status"],
            description=f"Largest change in {_label(out.metric)} by {out.dimension} "
            f"({period_text(res.get('baseline_period'))} to "
            f"{period_text(res.get('focus_period'))}): "
            f"{top['group']}, {fmt_value(top['old'], out.unit)} to "
            f"{fmt_value(top['new'], out.unit)} ({change}).",
            detail={
                "kind": "top_group_change",
                "group": top["group"],
                "small_sample": top["small_sample"],
                "second_abs_change": second,
                "n_groups": len(groups),
            },
        )
    ]


def _from_contribution(out: StepOutput, i: int) -> list[EvidenceItem]:
    key = f"E{i}"
    res = out.result
    period = {"focus": res.get("focus_period"), "baseline": res.get("baseline_period")}
    contributions = res.get("contributions", [])
    if out.status == "no_data" or not contributions:
        return [_no_data(out, key, period)]
    top = contributions[0]
    share = top["share_of_total_change"]
    share_text = f"{share * 100:.0f}% of the total change" if share is not None else "no share"
    base_share = top["before"] / res["total_before"] if res["total_before"] else None
    desc = (
        f"By {out.dimension} ({period_text(res.get('baseline_period'))} to "
        f"{period_text(res.get('focus_period'))}), total {_label(out.metric)} moved "
        f"{fmt_value(res['total_before'], out.unit)} to {fmt_value(res['total_after'], out.unit)} "
        f"({fmt_change(res['total_change'], None, out.unit)}). The largest contributor was "
        f"{top['group']} ({fmt_change(top['change'], None, out.unit)}, {share_text})."
    )
    return [
        EvidenceItem(
            key=key,
            step_id=out.step_id,
            metric=out.metric,
            unit=out.unit,
            value=top["change"],
            period=period,
            filters={**out.filters, out.dimension or "group": [top["group"]]},
            source_operation=_source(out),
            result_status="ok",
            description=desc,
            detail={
                "kind": "top_contributor",
                "group": top["group"],
                "share_of_total_change": share,
                "baseline_share": base_share,
                "total_change": res["total_change"],
                "mixed_direction": res["mixed_direction"],
                "dimension": out.dimension,
            },
        )
    ]
