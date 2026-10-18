"""The structured result of an investigation, and a plain-text rendering of it."""

from typing import Literal

from pydantic import BaseModel, Field

ReportStatus = Literal["completed", "insufficient_evidence", "needs_clarification", "failed"]
HypothesisStatus = Literal["supported", "contradicted", "inconclusive", "not_testable"]


class ReportFinding(BaseModel):
    text: str
    evidence_keys: list[str]


class ReportHypothesis(BaseModel):
    id: str
    statement: str
    status: HypothesisStatus
    reason: str
    evidence_keys: list[str] = Field(default_factory=list)


class ReportStep(BaseModel):
    iteration: int
    id: str
    operation: str
    metric: str
    dimension: str | None
    purpose: str
    status: str  # ok | no_data | failed | skipped
    note: str | None = None


class RunStats(BaseModel):
    iterations: int = 0
    queries_run: int = 0
    llm_calls: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str | None = None
    duration_ms: int = 0


class InvestigationReport(BaseModel):
    question: str
    status: ReportStatus
    message: str | None = None  # clarification / refusal / failure text for the user
    error_code: str | None = None
    summary: str = ""
    findings: list[ReportFinding] = Field(default_factory=list)
    hypotheses: list[ReportHypothesis] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    steps: list[ReportStep] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stats: RunStats = Field(default_factory=RunStats)

    def to_text(self) -> str:
        lines = [f"Question: {self.question}", f"Status:   {self.status}", ""]
        if self.message:
            lines += [self.message, ""]
        if self.summary:
            lines += ["SUMMARY", self.summary, ""]
        if self.findings:
            lines.append("FINDINGS")
            lines += [f"- {f.text} [{', '.join(f.evidence_keys)}]" for f in self.findings]
            lines.append("")
        if self.hypotheses:
            lines.append("HYPOTHESES")
            for h in self.hypotheses:
                keys = f" [{', '.join(h.evidence_keys)}]" if h.evidence_keys else ""
                lines.append(f"- {h.id} {h.status.upper()}: {h.statement}")
                lines.append(f"    {h.reason}{keys}")
            lines.append("")
        if self.evidence:
            lines.append("EVIDENCE")
            lines += [f"- {e['key']}: {e['description']}" for e in self.evidence]
            lines.append("")
        if self.caveats:
            lines.append("CAVEATS")
            lines += [f"- {c}" for c in self.caveats]
            lines.append("")
        if self.limitations:
            lines.append("LIMITATIONS")
            lines += [f"- {c}" for c in self.limitations]
            lines.append("")
        s = self.stats
        lines.append(
            f"Run: {s.iterations} iteration(s), {s.queries_run} queries, {s.llm_calls} LLM "
            f"call(s), {s.duration_ms} ms."
        )
        return "\n".join(lines)
