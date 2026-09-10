"""Phase 6 checks for the structured verifier (fake LLM only)."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from incident_agent.executor import ToolResult, execute_planned_task
from incident_agent.investigator import Hypothesis, InvestigationHypotheses
from incident_agent.models import Incident
from incident_agent.planner import PlannedTask
from incident_agent.verifier import (
    VerificationResult,
    VerifierValidationError,
    enforce_verification_policy,
    run_verifier,
)


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _incident() -> Incident:
    return Incident(
        id="inc-001",
        service="checkout",
        symptom="p95 latency elevated",
        started_at=_utc("2026-09-10T14:05:00Z"),
    )


def _hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(
            statement="Checkout degradation may be related to the recent change window.",
            observed_facts=["Latency samples increased within the investigated window."],
            supporting_evidence=[0],
            contradicting_evidence=[],
            reasoning="Metrics rise later in the investigated window.",
            confidence="high",
        )
    ]


def _evidence() -> list[ToolResult]:
    return [
        execute_planned_task(
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
    ]


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


class _NeverCalledLlm:
    def with_structured_output(self, _model):  # type: ignore[no-untyped-def]
        raise AssertionError("LLM must not be called for deterministic FAIL paths")


def test_valid_pass() -> None:
    payload = {
        "status": "PASS",
        "rationale": "Evidence substantially supports the leading hypothesis.",
        "supporting_evidence": [0],
        "missing_information": [],
    }
    result = run_verifier(_incident(), _hypotheses(), _evidence(), _llm=_FakeLlm(payload))

    assert isinstance(result, VerificationResult)
    assert result.status == "PASS"
    assert result.supporting_evidence == [0]
    assert result.missing_information == []


def test_valid_fail() -> None:
    payload = {
        "status": "FAIL",
        "rationale": "Evidence is insufficient.",
        "supporting_evidence": [],
        "missing_information": ["error_rate signal missing for the same window"],
    }
    result = run_verifier(_incident(), _hypotheses(), _evidence(), _llm=_FakeLlm(payload))

    assert result.status == "FAIL"
    assert result.missing_information == ["error_rate signal missing for the same window"]


def test_out_of_range_reference_rejected() -> None:
    payload = {
        "status": "PASS",
        "rationale": "Supported.",
        "supporting_evidence": [7],
        "missing_information": [],
    }
    with pytest.raises(VerifierValidationError):
        run_verifier(_incident(), _hypotheses(), _evidence(), _llm=_FakeLlm(payload))


def test_malformed_status_rejected() -> None:
    payload = {
        "status": "MAYBE",
        "rationale": "Weird.",
        "supporting_evidence": [],
        "missing_information": [],
    }
    with pytest.raises(ValidationError):
        VerificationResult.model_validate(payload)


def test_forbidden_extra_fields_rejected() -> None:
    payload = {
        "status": "PASS",
        "rationale": "Supported.",
        "supporting_evidence": [0],
        "missing_information": [],
        "tool_call": "get_metrics",
    }
    with pytest.raises(ValidationError):
        VerificationResult.model_validate(payload)


def test_no_evidence_deterministic_fail_no_llm() -> None:
    result = run_verifier(_incident(), _hypotheses(), [], _llm=_NeverCalledLlm())

    assert isinstance(result, VerificationResult)
    assert result.status == "FAIL"
    assert result.supporting_evidence == []
    assert any("No evidence" in item for item in result.missing_information)


def test_no_hypotheses_deterministic_fail_no_llm() -> None:
    result = run_verifier(_incident(), [], _evidence(), _llm=_NeverCalledLlm())

    assert result.status == "FAIL"
    assert any("No hypotheses" in item for item in result.missing_information)


def test_high_confidence_hypothesis_does_not_force_pass() -> None:
    payload = {
        "status": "FAIL",
        "rationale": "Evidence does not sufficiently support the claim.",
        "supporting_evidence": [],
        "missing_information": ["deployment timeline evidence needed"],
    }
    result = run_verifier(_incident(), _hypotheses(), _evidence(), _llm=_FakeLlm(payload))

    assert result.status == "FAIL"


def test_pass_with_empty_evidence_is_policy_error() -> None:
    result = VerificationResult(
        status="PASS",
        rationale="Supported.",
        supporting_evidence=[],
        missing_information=[],
    )
    with pytest.raises(VerifierValidationError):
        enforce_verification_policy(result, _hypotheses(), [])