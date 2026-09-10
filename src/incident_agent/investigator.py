"""LLM-powered investigation hypotheses from collected evidence.

(Incident, ToolResults) -> ranked UNVERIFIED hypotheses. The investigator
reasons over supplied evidence only. It never executes tools, verifies
claims, or performs remediation.
"""

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from incident_agent.executor import ToolResult
from incident_agent.models import Incident

Confidence = Literal["low", "medium", "high"]
# Relative ordering signal only, NOT a calibrated probability:
# high = multiple consistent signals, no contradiction;

INVESTIGATOR_SYSTEM_PROMPT = """You are the investigator for a production-incident agent.
Your ONLY job is to propose ranked, UNVERIFIED hypotheses from supplied evidence.

You receive the incident plus an evidence block. Evidence items are numbered
[evidence 0], [evidence 1], ... Reference them by index only.

Rules:
- Reason ONLY over the supplied evidence. Do not invent metrics, logs,
  versions, timestamps, services, or tool results.
- Separate facts from inference: observed_facts paraphrase the evidence
  block; statement is your inference and must use hedging language
  (may, appears, is consistent with).
- Cite supporting_evidence and contradicting_evidence as evidence indices.
  Referencing negative evidence (an item with no rows) is allowed.
- Rank best-first. Limit 1-3 hypotheses.
- If evidence is insufficient, return one low-confidence hypothesis whose
  statement begins with "Insufficient evidence".
- Output ONLY the structured result. No prose outside the schema.

You MUST NOT: execute tools, access filesystem/network/shell, remediate,
invent observations, or claim a hypothesis is verified or proven.
These hypotheses are UNVERIFIED candidates for a later verifier.
"""


class Hypothesis(BaseModel, extra="forbid"):
    """One unverified candidate explanation grounded in evidence indices."""

    statement: str = Field(min_length=1, max_length=1000)
    observed_facts: list[str] = Field(min_length=1, max_length=10)
    supporting_evidence: list[int] = Field(default_factory=list, max_length=10)
    contradicting_evidence: list[int] = Field(default_factory=list, max_length=10)
    reasoning: str = Field(min_length=1, max_length=2000)
    confidence: Confidence


class InvestigationHypotheses(BaseModel, extra="forbid"):
    """Ranked-by-order hypotheses; index 0 is the top candidate."""

    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=3)


class InvestigatorValidationError(ValueError):
    """Raised when LLM hypotheses fail machine-checkable policy checks."""


def format_evidence_for_llm(
    evidence: list[ToolResult],
    max_metrics: int = 20,
    max_logs: int = 20,
    max_deployments: int = 10,
) -> str:
    """Render evidence deterministically as a bounded text block."""
    if not evidence:
        return "No evidence collected."
    lines: list[str] = []
    for index, result in enumerate(evidence):
        task = result.task
        if task.tool == "get_metrics":
            window = f"{task.start.isoformat()}..{task.end.isoformat()}"
            shown = result.metrics[:max_metrics]
            lines.append(
                f"[evidence {index}] tool=get_metrics service={task.service} "
                f"metric={task.metric} window={window} "
                f"({len(result.metrics)} points, showing {len(shown)}):"
            )
            for point in shown:
                lines.append(f"  {point.timestamp.isoformat()} {point.service} {point.metric}={point.value}")
            if len(result.metrics) > len(shown):
                lines.append(f"  ({len(result.metrics) - len(shown)} omitted)")
        elif task.tool == "search_logs":
            window = f"{task.start.isoformat()}..{task.end.isoformat()}"
            shown = result.logs[:max_logs]
            lines.append(
                f"[evidence {index}] tool=search_logs service={task.service} "
                f"window={window} keyword={task.keyword} level={task.level} "
                f"({len(result.logs)} lines, showing {len(shown)}):"
            )
            for entry in shown:
                lines.append(f"  {entry.timestamp.isoformat()} {entry.level} {entry.message}")
            if len(result.logs) > len(shown):
                lines.append(f"  ({len(result.logs) - len(shown)} omitted)")
        else:
            shown = result.deployments[:max_deployments]
            lines.append(
                f"[evidence {index}] tool=get_recent_deployments service={task.service} "
                f"since={task.since.isoformat() if task.since else None} "
                f"({len(result.deployments)} items, showing {len(shown)}):"
            )
            for item in shown:
                lines.append(
                    f"  {item.timestamp.isoformat()} {item.service} {item.version} "
                    f"id={item.id} status={item.status}"
                )
            if len(result.deployments) > len(shown):
                lines.append(f"  ({len(result.deployments) - len(shown)} omitted)")
    return "\n".join(lines)


def investigator_messages(incident: Incident, evidence: list[ToolResult]) -> list[tuple[str, str]]:
    """Render fixed system prompt plus incident and evidence block."""
    human = (
        f"id: {incident.id}\n"
        f"service: {incident.service}\n"
        f"symptom: {incident.symptom}\n"
        f"started_at: {incident.started_at.isoformat()}\n"
        f"evidence:\n{format_evidence_for_llm(evidence)}\n"
        "Produce InvestigationHypotheses. These are UNVERIFIED candidates."
    )
    return [("system", INVESTIGATOR_SYSTEM_PROMPT), ("human", human)]


def _insufficient_evidence() -> InvestigationHypotheses:
    return InvestigationHypotheses(
        hypotheses=[
            Hypothesis(
                statement="Insufficient evidence to form a hypothesis.",
                observed_facts=["No evidence was collected."],
                supporting_evidence=[],
                contradicting_evidence=[],
                reasoning="No observations are available, so no inference is possible.",
                confidence="low",
            )
        ]
    )


def enforce_hypothesis_policy(
    hypotheses: InvestigationHypotheses, evidence: list[ToolResult]
) -> InvestigationHypotheses:
    """Enforce machine-checkable invariants (indices, shape, verdict fields)."""
    count = len(evidence)
    for item in hypotheses.hypotheses:
        for ref in (*item.supporting_evidence, *item.contradicting_evidence):
            if not 0 <= ref < count:
                raise InvestigatorValidationError(f"evidence reference out of range: {ref}")
        if evidence and not item.observed_facts:
            raise InvestigatorValidationError("observed_facts must not be empty when evidence exists")
    return hypotheses


def run_investigator(
    incident: Incident, evidence: list[ToolResult], _llm: Any | None = None
) -> InvestigationHypotheses:
    """Produce ranked unverified hypotheses via LLM structured output."""
    if not evidence:
        return _insufficient_evidence()
    if _llm is None:
        from langchain_ollama import ChatOllama

        _llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0)
    out = _llm.with_structured_output(InvestigationHypotheses).invoke(
        investigator_messages(incident, evidence)
    )
    if not isinstance(out, InvestigationHypotheses):
        out = InvestigationHypotheses.model_validate(out)
    return enforce_hypothesis_policy(out, evidence)


class InvestigationHypotheses(BaseModel, extra="forbid"):
    """Ranked-by-order hypotheses; index 0 is the top candidate."""

    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=3)

# medium = one strong signal or minor contradiction;
# low = weak/single/contradictory signals or insufficient evidence.
