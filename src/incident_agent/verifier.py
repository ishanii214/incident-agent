"""Structured verifier: judge whether the leading hypothesis is supported.

The verifier is independent from the investigator: it never treats the
investigator's confidence as truth, never executes tools, never remediates,
and never determines a root cause on its own.
"""

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from incident_agent.executor import ToolResult
from incident_agent.investigator import Hypothesis, format_evidence_for_llm
from incident_agent.models import Incident

# Maximum number of replan cycles. Verification attempts == MAX_RETRIES + 1.
MAX_RETRIES = 3

VERIFIER_SYSTEM_PROMPT = """You are the verifier for a production-incident agent.
Your ONLY job is to decide whether the LEADING hypothesis is sufficiently
supported by the supplied evidence.

You receive: the incident, the ranked hypotheses (index 0 is the leading
hypothesis), and a numbered evidence block.

Instructions:
- Judge the leading hypothesis against the evidence block ONLY. Do not invent
  evidence, metrics, logs, versions, timestamps, or services.
- Do NOT treat the investigator's confidence field as truth. Reach your own
  judgment from the evidence.
- PASS only when the evidence substantially supports the leading hypothesis.
- On FAIL, list concrete missing_information that additional evidence could
  address (for example signals, windows, or services).
- supporting_evidence must list evidence indices that back a PASS; when you
  answer FAIL, supporting_evidence must be empty.
- Refer to evidence by integer index only (0 <= index < len(evidence)).

You MUST NOT: execute tools or generate tool calls, remediate, determine a
root cause independently, or claim certainty beyond the evidence.
Output ONLY the structured VerificationResult.
"""


class VerificationResult(BaseModel, extra="forbid"):
    """Independent structured verdict on the leading hypothesis."""

    status: Literal["PASS", "FAIL"]
    rationale: str = Field(min_length=1, max_length=2000)
    supporting_evidence: list[int] = Field(default_factory=list, max_length=10)
    missing_information: list[str] = Field(default_factory=list, max_length=10)


class VerifierValidationError(ValueError):
    """Raised when a verification result fails machine-checkable policy."""


def verifier_messages(
    incident: Incident,
    hypotheses: list[Hypothesis],
    evidence: list[ToolResult],
) -> list[tuple[str, str]]:
    """Render fixed system prompt plus incident, hypotheses, evidence."""
    lines = [
        f"id: {incident.id}",
        f"service: {incident.service}",
        f"symptom: {incident.symptom}",
        f"started_at: {incident.started_at.isoformat()}",
        "hypotheses (index 0 is the LEADING hypothesis):",
    ]
    for i, h in enumerate(hypotheses):
        lines.append(
            f"[hypothesis {i}] statement={h.statement} confidence={h.confidence} "
            f"supporting={h.supporting_evidence} contradicting={h.contradicting_evidence} "
            f"reasoning={h.reasoning}"
        )
    lines.append(f"evidence:\n{format_evidence_for_llm(evidence)}")
    lines.append("Produce the VerificationResult. Judge the LEADING hypothesis only.")
    return [("system", VERIFIER_SYSTEM_PROMPT), ("human", "\n".join(lines))]


def enforce_verification_policy(
    result: VerificationResult,
    hypotheses: list[Hypothesis],
    evidence: list[ToolResult],
) -> VerificationResult:
    """Enforce machine-checkable invariants on a verification result."""
    if result.status == "PASS" and (not evidence or not hypotheses):
        raise VerifierValidationError("PASS requires non-empty evidence and hypotheses")
    count = len(evidence)
    for ref in result.supporting_evidence:
        if not 0 <= ref < count:
            raise VerifierValidationError(f"supporting evidence reference out of range: {ref}")
    return result


def _deterministic_fail(
    evidence: list[ToolResult], hypotheses: list[Hypothesis]
) -> VerificationResult:
    missing: list[str] = []
    if not evidence:
        missing.append("No evidence was collected; additional tool execution is required.")
    if not hypotheses:
        missing.append("No hypotheses were generated; there is nothing to verify.")
    return VerificationResult(
        status="FAIL",
        rationale="Verification cannot pass without evidence and hypotheses.",
        supporting_evidence=[],
        missing_information=missing,
    )


def run_verifier(
    incident: Incident,
    hypotheses: list[Hypothesis],
    evidence: list[ToolResult],
    _llm: Any | None = None,
) -> VerificationResult:
    """Produce an independent verdict via structured output (lazy LLM)."""
    if not evidence or not hypotheses:
        return _deterministic_fail(evidence, hypotheses)
    if _llm is None:
        from langchain_ollama import ChatOllama

        _llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0)
    out = _llm.with_structured_output(VerificationResult).invoke(
        verifier_messages(incident, hypotheses, evidence)
    )
    if not isinstance(out, VerificationResult):
        out = VerificationResult.model_validate(out)
    return enforce_verification_policy(out, hypotheses, evidence)