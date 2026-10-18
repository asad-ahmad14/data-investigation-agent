"""Deterministic evaluation of hypotheses and construction of findings.

No language model is involved. A hypothesis is judged only from computed evidence, with fixed,
documented rules, so the same data always gives the same verdict.

Rules (for hypotheses of the form "the change is concentrated in specific groups"):
- Decomposition (additive metrics): supported if the largest contributor accounts for at least
  CONCENTRATION_SHARE (50%) of the total change AND that share is at least DISPROPORTION (1.5x)
  its share of the baseline total (a group that is 60% of sales and 62% of the drop is simply
  proportionate, not a concentration). Contradicted if the share is below 50% or proportionate.
  Inconclusive if the share is undefined, or if groups moved in opposite directions and the
  share is below 50% (shares are then unreliable).
- Group comparison (ratio metrics): supported if the largest change is at least
  DOMINANCE_RATIO (2x) the next largest and is not a small sample; contradicted if not;
  inconclusive when the group is a small sample or there is only one group.
- No data / failed steps: inconclusive / not_testable. A hypothesis with several steps is
  supported if any step supports it, else contradicted if any contradicts, else inconclusive.

These thresholds are heuristics chosen for explainability, not statistical tests.
"""

from dataclasses import dataclass

from app.agent.evidence import EvidenceItem
from app.agent.report import ReportFinding, ReportHypothesis
from app.schemas.plan import HypothesisSpec

CONCENTRATION_SHARE = 0.5
DISPROPORTION = 1.5
DOMINANCE_RATIO = 2.0


@dataclass(frozen=True)
class _Verdict:
    status: str
    reason: str


def _judge(item: EvidenceItem) -> _Verdict:
    d = item.detail
    if item.result_status in ("no_data", "no_decline") or item.value is None:
        return _Verdict("inconclusive", f"No usable data for this check ({item.key}).")
    kind = d.get("kind")
    if kind == "top_contributor":
        share = d.get("share_of_total_change")
        group = d["group"]
        if share is None:
            return _Verdict("inconclusive", "The total change is zero, so shares are undefined.")
        pct = f"{share * 100:.0f}%"
        base = d.get("baseline_share")
        offset_note = " (other groups partly offset it)" if d.get("mixed_direction") else ""
        if share >= CONCENTRATION_SHARE:
            if base is not None and share < DISPROPORTION * base:
                return _Verdict(
                    "contradicted",
                    f"{group} accounts for {pct} of the change but also {base * 100:.0f}% of the "
                    "baseline, so the change is proportionate rather than concentrated.",
                )
            return _Verdict(
                "supported",
                f"{group} accounts for {pct} of the total change in {d['dimension']}"
                + (f", versus {base * 100:.0f}% of the baseline" if base is not None else "")
                + f"{offset_note}.",
            )
        if d.get("mixed_direction"):
            return _Verdict(
                "inconclusive",
                f"Groups moved in opposite directions, so {group}'s {pct} share is unreliable.",
            )
        return _Verdict(
            "contradicted",
            f"The largest contributor, {group}, accounts for only {pct} of the total change.",
        )
    if kind == "top_group_change":
        group = d["group"]
        if d.get("small_sample"):
            return _Verdict("inconclusive", f"{group} has too few orders to be reliable.")
        if d.get("n_groups", 0) < 2:
            return _Verdict("inconclusive", "Only one group could be compared.")
        top, second = abs(item.value), d.get("second_abs_change", 0.0)
        if second == 0 or top >= DOMINANCE_RATIO * second:
            return _Verdict("supported", f"{group} changed at least twice as much as any other.")
        return _Verdict(
            "contradicted", f"{group} did not change markedly more than the other groups."
        )
    return _Verdict("inconclusive", "This evidence does not test a concentration claim.")


def evaluate_hypotheses(
    specs: list[HypothesisSpec],
    evidence_by_step: dict[str, list[EvidenceItem]],
    failed_step_ids: set[str],
    *,
    first_number: int,
) -> list[ReportHypothesis]:
    """Judge each hypothesis. Ids are renumbered H<first_number>... so they stay unique when an
    investigation plans more than once."""
    results: list[ReportHypothesis] = []
    for offset, spec in enumerate(specs):
        verdicts: list[tuple[EvidenceItem, _Verdict]] = []
        for step_id in spec.step_ids:
            for item in evidence_by_step.get(step_id, []):
                verdicts.append((item, _judge(item)))
        keys = [item.key for item, _ in verdicts]
        if not verdicts:
            reason = (
                "The analysis for this hypothesis could not be completed."
                if set(spec.step_ids) & failed_step_ids
                else "No analysis was run for this hypothesis."
            )
            status = "not_testable"
        else:
            statuses = [v.status for _, v in verdicts]
            for wanted in ("supported", "contradicted"):
                if wanted in statuses:
                    status = wanted
                    reason = next(v.reason for _, v in verdicts if v.status == wanted)
                    break
            else:
                status = "inconclusive"
                reason = verdicts[0][1].reason
        results.append(
            ReportHypothesis(
                id=f"H{first_number + offset}",
                statement=spec.statement,
                status=status,
                reason=reason,
                evidence_keys=keys,
            )
        )
    return results


def build_findings(
    evidence: list[EvidenceItem], hypotheses: list[ReportHypothesis]
) -> list[ReportFinding]:
    """Headline findings, each citing the evidence it comes from."""
    by_key = {e.key: e for e in evidence}
    findings: list[ReportFinding] = []
    used: set[str] = set()

    def add(item: EvidenceItem) -> None:
        if item.key not in used and item.result_status not in ("no_data",):
            used.add(item.key)
            findings.append(ReportFinding(text=item.description, evidence_keys=[item.key]))

    for item in evidence:
        if item.detail.get("kind") in ("largest_decline", "overall_change"):
            add(item)
    for h in hypotheses:
        if h.status == "supported":
            for key in h.evidence_keys:
                add(by_key[key])
    for item in evidence:
        if item.result_status == "no_decline":
            add(item)
    return findings
