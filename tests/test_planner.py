"""Phase 3 checks for the LLM-powered structured planner (fake LLM only)."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from incident_agent.models import Incident
from incident_agent.planner import (
    InvestigationPlan,
    PlannedTask,
    PlannerValidationError,
    enforce_plan_policy,
)



def _incident() -> Incident:
    return Incident(
        id="inc-001",
        service="checkout",
        symptom="p95 latency elevated",
        started_at=datetime.fromisoformat("2026-09-10T14:05:00+00:00"),
    )


def _task(**overrides) -> dict:
    base = {
        "tool": "get_metrics",
        "service": "checkout",
        "metric": "p95_latency_ms",
        "start": "2026-09-10T13:50:00+00:00",
        "end": "2026-09-10T14:05:00+00:00",
    }
    base.update(overrides)
    return base


def _plan_data(tasks) -> dict:
    return {"tasks": tasks}


def test_valid_structured_plan() -> None:
    plan = InvestigationPlan.model_validate(
        _plan_data(
            [
                _task(),
                {
                    "tool": "search_logs",
                    "service": "checkout",
                    "start": "2026-09-10T13:50:00+00:00",
                    "end": "2026-09-10T14:05:00+00:00",
                    "keyword": "timeout",
                    "level": "ERROR",
                    "limit": 20,
                },
                {
                    "tool": "get_recent_deployments",
                    "service": "checkout",
                    "since": "2026-09-10T12:05:00+00:00",
                    "limit": 10,
                },
            ]
        )
    )
    assert isinstance(enforce_plan_policy(plan, _incident()), InvestigationPlan)


def test_invalid_tool_rejected() -> None:
    with pytest.raises(ValidationError):
        PlannedTask.model_validate(_task(tool="run_shell"))


def test_invalid_metric_rejected() -> None:
    with pytest.raises(ValidationError):
        PlannedTask.model_validate(_task(metric="p95_latency"))


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValidationError):
        PlannedTask.model_validate(
            {
                "tool": "search_logs",
                "service": "checkout",
                "start": "2026-09-10T13:50:00+00:00",
                "end": "2026-09-10T14:05:00+00:00",
                "level": "CRITICAL",
            }
        )


def test_malformed_parameters_rejected() -> None:
    bad = dict(_task())
    del bad["end"]
    with pytest.raises(ValidationError):
        PlannedTask.model_validate(bad)
    with pytest.raises(ValidationError):
        PlannedTask.model_validate(_task(limit=500))


def test_invalid_timestamps_rejected() -> None:
    with pytest.raises(ValidationError):
        PlannedTask.model_validate(_task(start="2026-09-10T14:05:00", end="2026-09-10T14:05:00+00:00"))
    with pytest.raises(ValidationError):
        PlannedTask.model_validate(
            _task(start="2026-09-10T14:05:00+00:00", end="2026-09-10T13:50:00+00:00")
        )


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        PlannedTask.model_validate(_task(command="rm -rf /"))
    with pytest.raises(ValidationError):
        InvestigationPlan.model_validate(_plan_data([_task(), _task()]) | {"answer": "x"})


def test_invalid_task_count_rejected() -> None:
    with pytest.raises(ValidationError):
        InvestigationPlan.model_validate(_plan_data([]))
    with pytest.raises(ValidationError):
        InvestigationPlan.model_validate(_plan_data([_task()] * 6))


def test_wrong_service_rejected() -> None:
    plan = InvestigationPlan.model_validate(_plan_data([_task(service="search")]))
    with pytest.raises(PlannerValidationError):
        enforce_plan_policy(plan, _incident())


def test_invalid_window_rejected() -> None:
    plan = InvestigationPlan.model_validate(
        _plan_data([_task(start="2026-09-10T10:00:00+00:00", end="2026-09-10T10:30:00+00:00")])
    )
    with pytest.raises(PlannerValidationError):
        enforce_plan_policy(plan, _incident())
    late = InvestigationPlan.model_validate(
        _plan_data(
            [
                {
                    "tool": "get_recent_deployments",
                    "service": "checkout",
                    "since": "2026-09-10T15:00:00+00:00",
                }
            ]
        )
    )
    with pytest.raises(PlannerValidationError):
        enforce_plan_policy(late, _incident())


def test_suspicious_values_rejected() -> None:
    with pytest.raises(ValidationError):
        PlannedTask.model_validate(_task(service="checkout; rm -rf /"))
    with pytest.raises(ValidationError):
        PlannedTask.model_validate(
            {
                "tool": "search_logs",
                "service": "checkout",
                "start": "2026-09-10T13:50:00+00:00",
                "end": "2026-09-10T14:05:00+00:00",
                "keyword": "$(whoami)",
            }
        )
