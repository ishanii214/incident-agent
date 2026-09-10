"""Phase 4 checks for deterministic PlannedTask execution (no LLM)."""

from datetime import datetime

import pytest

from incident_agent.executor import (
    ToolResult,
    UnsupportedToolError,
    execute_planned_task,
)
from incident_agent.graph import execute_task
from incident_agent.models import Deployment, LogEntry, MetricPoint
from incident_agent.planner import PlannedTask


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _metrics_task(**overrides) -> PlannedTask:
    data = {
        "tool": "get_metrics",
        "service": "checkout",
        "metric": "p95_latency_ms",
        "start": _utc("2026-09-10T13:50:00Z"),
        "end": _utc("2026-09-10T14:05:00Z"),
    }
    data.update(overrides)
    return PlannedTask.model_validate(data)



def test_execute_get_metrics() -> None:
    result = execute_planned_task(_metrics_task())

    assert isinstance(result, ToolResult)
    assert result.task.tool == "get_metrics"
    assert len(result.metrics) >= 5
    assert all(isinstance(point, MetricPoint) for point in result.metrics)
    assert result.logs == [] and result.deployments == []


def test_execute_search_logs() -> None:
    task = PlannedTask.model_validate(
        {
            "tool": "search_logs",
            "service": "checkout",
            "start": "2026-09-10T13:50:00+00:00",
            "end": "2026-09-10T14:15:00+00:00",
            "keyword": "timeout",
            "level": "ERROR",
        }
    )
    result = execute_planned_task(task)

    assert result.logs
    assert all(isinstance(entry, LogEntry) for entry in result.logs)
    assert all("timeout" in entry.message.lower() for entry in result.logs)
    assert result.metrics == [] and result.deployments == []


def test_execute_deployments() -> None:
    task = PlannedTask.model_validate(
        {
            "tool": "get_recent_deployments",
            "service": "checkout",
            "since": "2026-09-10T12:05:00+00:00",
            "limit": 10,
        }
    )
    result = execute_planned_task(task)

    assert result.deployments
    assert all(isinstance(item, Deployment) for item in result.deployments)
    assert result.deployments[0].version == "v2.4.1"
    assert result.metrics == [] and result.logs == []


def test_parameter_passing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict = {}

    def fake_metrics(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return []

    monkeypatch.setattr("incident_agent.executor.get_metrics", fake_metrics)
    task = _metrics_task()
    execute_planned_task(task)

    assert seen == {
        "service": "checkout",
        "metric": "p95_latency_ms",
        "start": task.start,
        "end": task.end,
    }


def test_unsupported_tool_rejected() -> None:
    task = _metrics_task().model_construct(tool="run_shell")

    with pytest.raises((UnsupportedToolError, ValueError)):
        execute_planned_task(task)


def test_empty_result_preserved() -> None:
    result = execute_planned_task(_metrics_task(start=_utc("2026-09-10T12:00:00Z"), end=_utc("2026-09-10T12:30:00Z")))

    assert isinstance(result, ToolResult)
    assert result.metrics == []


def test_multiple_tasks_execute_in_order() -> None:
    logs_task = PlannedTask.model_validate(
        {
            "tool": "search_logs",
            "service": "checkout",
            "start": "2026-09-10T13:50:00+00:00",
            "end": "2026-09-10T14:15:00+00:00",
        }
    )
    deploy_task = PlannedTask.model_validate(
        {"tool": "get_recent_deployments", "service": "checkout"}
    )
    state = {
        "pending_tasks": [_metrics_task(), logs_task, deploy_task],
        "completed_tasks": [],
        "evidence": [],
    }

    for _ in range(3):
        update = execute_task(state)  # type: ignore[arg-type]
        state = {**state, **update}

    assert [task.tool for task in state["completed_tasks"]] == [
        "get_metrics",
        "search_logs",
        "get_recent_deployments",
    ]
    assert [item.task.tool for item in state["evidence"]] == [
        "get_metrics",
        "search_logs",
        "get_recent_deployments",
    ]


def test_no_pending_task_executes_nothing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = []
    monkeypatch.setattr(
        "incident_agent.executor.execute_planned_task",
        lambda task: calls.append(task),
    )
    state = {"pending_tasks": [], "completed_tasks": [], "evidence": []}

    update = execute_task(state)  # type: ignore[arg-type]

    assert update["completed_tasks"] == [] and calls == []
    assert "evidence" not in update


def test_executor_uses_allow_list_only() -> None:
    import incident_agent.executor as executor_module

    source = open(executor_module.__file__, encoding="utf-8").read()
    assert "getattr" not in source and "eval(" not in source
    assert "exec(" not in source and "subprocess" not in source
