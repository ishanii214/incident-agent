"""Deterministic dispatcher from validated plans to operational tools.

LLM proposes, Pydantic validates, this module dispatches. No dynamic lookup,
no shell, no network beyond local fixture reads inside tools.py.
"""

from pydantic import BaseModel, Field

from incident_agent.models import Deployment, LogEntry, MetricPoint
from incident_agent.planner import PlannedTask
from incident_agent.tools import get_metrics, get_recent_deployments, search_logs


class UnsupportedToolError(ValueError):
    """Raised when a task names a tool outside the explicit allow-list."""


class ToolResult(BaseModel):
    """Typed evidence envelope linking one task to its tool output."""

    task: PlannedTask
    metrics: list[MetricPoint] = Field(default_factory=list)
    logs: list[LogEntry] = Field(default_factory=list)
    deployments: list[Deployment] = Field(default_factory=list)


def execute_planned_task(task: PlannedTask) -> ToolResult:
    """Execute one validated task via explicit allow-list dispatch."""
    if task.tool == "get_metrics":
        assert task.metric is not None and task.start is not None
        assert task.end is not None
        points = get_metrics(
            service=task.service,
            metric=task.metric,
            start=task.start,
            end=task.end,
        )
        return ToolResult(task=task, metrics=points)
    if task.tool == "search_logs":
        assert task.start is not None and task.end is not None
        entries = search_logs(
            service=task.service,
            start=task.start,
            end=task.end,
            keyword=task.keyword,
            level=task.level,
            limit=task.limit if task.limit is not None else 50,
        )
        return ToolResult(task=task, logs=entries)
    if task.tool == "get_recent_deployments":
        found = get_recent_deployments(
            service=task.service,
            since=task.since,
            limit=task.limit if task.limit is not None else 10,
        )
        return ToolResult(task=task, deployments=found)
    raise UnsupportedToolError(f"unsupported tool: {task.tool}")
