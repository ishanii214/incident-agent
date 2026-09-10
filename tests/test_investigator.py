"""Phase 5 checks for evidence-grounded hypotheses (fake LLM only)."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from incident_agent.executor import ToolResult, execute_planned_task
from incident_agent.investigator import (
    Hypothesis,
    InvestigationHypotheses,
    InvestigatorValidationError,
    enforce_hypothesis_policy,
    format_evidence_for_llm,
    run_investigator,
)
from incident_agent.models import Incident
from incident_agent.planner import PlannedTask


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _incident() -> Incident:
    return Incident(
        id="inc-001",
        service="checkout",
        symptom="p95 latency elevated",
        started_at=_utc("2026-09-10T14:05:00Z"),
    )


def _evidence() -> list[ToolResult]:
    metrics = execute_planned_task(
        PlannedTask.model_validate(
            {
                "tool": "get_metrics",
                "service": "checkout",
                "metric": "p95_latency_ms",
                "start": "2026-09-10T13:50:00+00:00",
                "end": "2026-09-10T14:05:00+00:00",
            }
        )
    )
    empty = execute_planned_task(
        PlannedTask.model_validate(
            {
                "tool": "search_logs",
                "service": "checkout",
                "start": "2026-09-10T12:00:00+00:00",
                "end": "2026-09-10T12:30:00+00:00",
            }
        )
    )
    return [metrics, empty]


def _hypothesis(**overrides) -> dict:
    base = {
        "statement": "Checkout degradation may be related to the recent change window.",
        "observed_facts": ["Latency samples increased within the investigated window."],
        "supporting_evidence": [0],
        "contradicting_evidence": [],
        "reasoning": "Metrics show higher values later in the window.",
        "confidence": "medium",
    }
    base.update(overrides)
    return base


class _FakeStructured:
    def __init__(self, payload: dict):
        self.payload = payload

    def invoke(self, _messages):  # type: ignore[no-untyped-def]
        return self.payload


class _FakeLlm:
    def __init__(self, payload: dict):
        self.payload = payload

    def with_structured_output(self, _model):  # type: ignore[no-untyped-def]
        return _FakeStructured(self.payload)



def test_valid_single_hypothesis() -> None:
    out = run_investigator(_incident(), _evidence(), _llm=_FakeLlm({"hypotheses": [_hypothesis()]}))

    assert isinstance(out, InvestigationHypotheses)
    assert len(out.hypotheses) == 1
    assert isinstance(out.hypotheses[0], Hypothesis)


def test_multiple_hypotheses_preserve_order() -> None:
    payload = {"hypotheses": [_hypothesis(confidence="high"), _hypothesis(confidence="low")]}
    out = run_investigator(_incident(), _evidence(), _llm=_FakeLlm(payload))

    assert [item.confidence for item in out.hypotheses] == ["high", "low"]


def test_references_support_and_contradict() -> None:
    payload = {"hypotheses": [_hypothesis(supporting_evidence=[0], contradicting_evidence=[1])]}
    out = run_investigator(_incident(), _evidence(), _llm=_FakeLlm(payload))

    assert out.hypotheses[0].supporting_evidence == [0]
    assert out.hypotheses[0].contradicting_evidence == [1]


def test_empty_tool_result_can_be_referenced() -> None:
    payload = {"hypotheses": [_hypothesis(supporting_evidence=[1])]}
    out = run_investigator(_incident(), _evidence(), _llm=_FakeLlm(payload))

    assert out.hypotheses[0].supporting_evidence == [1]


def test_out_of_range_reference_rejected() -> None:
    payload = {"hypotheses": [_hypothesis(supporting_evidence=[9])]}

    with pytest.raises(InvestigatorValidationError):
        run_investigator(_incident(), _evidence(), _llm=_FakeLlm(payload))


def test_empty_evidence_returns_insufficient() -> None:
    out = run_investigator(_incident(), [], _llm=_FakeLlm({"hypotheses": []}))

    assert len(out.hypotheses) == 1
    assert out.hypotheses[0].confidence == "low"
    assert out.hypotheses[0].statement.startswith("Insufficient evidence")


def test_invalid_schema_rejected() -> None:
    with pytest.raises(ValidationError):
        InvestigationHypotheses.model_validate({"hypotheses": []})
    with pytest.raises(ValidationError):
        InvestigationHypotheses.model_validate(
            {"hypotheses": [_hypothesis(confidence="high", verdict="x")]}
        )
    with pytest.raises(ValidationError):
        InvestigationHypotheses.model_validate(
            {"hypotheses": [_hypothesis(confidence="0.97")]}
        )


def test_no_tool_execution() -> None:
    import incident_agent.investigator as investigator_module

    source = open(investigator_module.__file__, encoding="utf-8").read()
    assert "incident_agent.tools" not in source
    assert "execute_planned_task" not in source
    assert format_evidence_for_llm(_evidence()).startswith("[evidence 0]")


def test_no_verdict_fields() -> None:
    assert "verified" not in Hypothesis.model_fields
    assert "root_cause" not in Hypothesis.model_fields
    assert "remediation" not in Hypothesis.model_fields
