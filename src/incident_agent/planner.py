"""LLM-powered investigation planner.

Incident -> validated InvestigationPlan only. The planner selects and orders
read-only investigation tools. It never executes tools, diagnoses incidents,
or accesses filesystem/network itself.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from incident_agent.models import Incident

ToolName = Literal["get_metrics", "search_logs", "get_recent_deployments"]

KNOWN_METRICS = {"p95_latency_ms", "error_rate"}
LOG_LEVELS = {"DEBUG", "INFO", "WARN", "ERROR"}
PLANNER_SYSTEM_PROMPT = """You are the investigation planner for a production-incident agent.
Your ONLY job is to produce an executable investigation plan.

You receive: incident id, service, symptom, started_at (UTC ISO-8601).

Available read-only tools (choose the smallest relevant subset, in the order
you judge most informative for this incident):
- get_metrics(service, metric in [p95_latency_ms, error_rate], start, end)
- search_logs(service, start, end, keyword?, level in [DEBUG, INFO, WARN, ERROR]?, limit 1-50)
- get_recent_deployments(service?, since?, limit 1-10)

Rules:
- Select and order tools based on the incident symptom and context. Do not
  follow a fixed metrics-then-logs-then-deployments workflow. Prioritize the
  most informative signals first, but the choice is yours per incident.
- Scope every task to the incident's service.
- Choose windows covering roughly 15-60 minutes before started_at through
  started_at. All timestamps must be UTC ISO-8601 with timezone.
- For deployments, use since (about 2 hours before started_at).
- Output ONLY the structured plan. No prose, no conclusions.

You MUST NOT: diagnose the incident, name a root cause or version, predict
what tools will return, execute tools, emit shell or Python code, access the
filesystem or network, invent services/metrics/tools, or include extra fields.
"""

_FORBIDDEN_CHARS = (";", "`", "$", "(", ")", "{", "}", "|", "&")


class PlannedTask(BaseModel, extra="forbid"):
    """One executable investigation step naming a single allowed tool."""

    tool: ToolName
    service: str = Field(min_length=1, max_length=64)
    metric: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    keyword: str | None = Field(default=None, max_length=64)
    level: str | None = None
    limit: int | None = None
    since: datetime | None = None

    @model_validator(mode="after")
    def _check_per_tool(self) -> "PlannedTask":
        if self.tool == "get_metrics":
            if self.metric not in KNOWN_METRICS:
                raise ValueError("get_metrics requires a known metric")
            if self.start is None or self.end is None:
                raise ValueError("get_metrics requires start <= end")
            if self.start.tzinfo is None or self.end.tzinfo is None:
                raise ValueError("timestamps must be timezone-aware")
            if self.start > self.end:
                raise ValueError("get_metrics requires start <= end")
            if self.keyword is not None or self.since is not None:
                raise ValueError("get_metrics accepts only service/metric/start/end")
            if self.limit is not None:
                raise ValueError("get_metrics accepts only service/metric/start/end")
        elif self.tool == "search_logs":
            if self.start is None or self.end is None:
                raise ValueError("search_logs requires start <= end")
            if self.start.tzinfo is None or self.end.tzinfo is None:
                raise ValueError("timestamps must be timezone-aware")
            if self.start > self.end:
                raise ValueError("search_logs requires start <= end")
            if self.metric is not None or self.since is not None:
                raise ValueError("search_logs accepts only service/start/end/keyword/level/limit")
            if self.level is not None and self.level not in LOG_LEVELS:
                raise ValueError("search_logs level is not allowed")
            if self.limit is not None and not 1 <= self.limit <= 50:
                raise ValueError("search_logs limit must be within 1-50")
        else:
            if self.metric is not None or self.start is not None or self.end is not None:
                raise ValueError("get_recent_deployments uses since, not start/end")
            if self.keyword is not None or self.level is not None:
                raise ValueError("get_recent_deployments accepts only service/since/limit")
            if self.limit is not None and not 1 <= self.limit <= 10:
                raise ValueError("deployments limit must be within 1-10")
            if self.since is not None and self.since.tzinfo is None:
                raise ValueError("since must be timezone-aware")
        for value in (self.service, self.metric, self.keyword):
            if value is not None and any(c in value for c in _FORBIDDEN_CHARS):
                raise ValueError("executable-command-like content is not allowed")
        return self


class InvestigationPlan(BaseModel, extra="forbid"):
    """Ordered executable investigation plan produced by the planner."""

    tasks: list[PlannedTask] = Field(min_length=1, max_length=5)


class PlannerValidationError(ValueError):
    """Raised when an LLM-produced plan fails incident policy checks."""


def planner_messages(incident: Incident) -> list[tuple[str, str]]:
    """Render fixed system prompt plus mechanical incident message."""
    human = (
        f"id: {incident.id}\n"
        f"service: {incident.service}\n"
        f"symptom: {incident.symptom}\n"
        f"started_at: {incident.started_at.isoformat()}\n"
        "Produce the InvestigationPlan."
    )
    return [("system", PLANNER_SYSTEM_PROMPT), ("human", human)]


def enforce_plan_policy(plan: InvestigationPlan, incident: Incident) -> InvestigationPlan:
    """Enforce incident-scoped policy beyond per-task schema validation."""
    window_start = incident.started_at - timedelta(hours=2)
    window_end = incident.started_at
    for task in plan.tasks:
        if task.service != incident.service:
            raise PlannerValidationError("tasks must target the incident service")
        if task.tool in ("get_metrics", "search_logs"):
            assert task.start is not None and task.end is not None
            if task.end < window_start or task.start > window_end:
                raise PlannerValidationError("window must overlap investigation window")
        if task.tool == "get_recent_deployments" and task.since is not None:
            if task.since > incident.started_at:
                raise PlannerValidationError("since must not be after incident start")
    return plan


def run_planner(incident: Incident, _llm: Any | None = None) -> InvestigationPlan:
    """Produce validated plan via LLM structured output (lazy construction)."""
    if _llm is None:
        from langchain_ollama import ChatOllama

        _llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0)
    plan = _llm.with_structured_output(InvestigationPlan).invoke(planner_messages(incident))
    if not isinstance(plan, InvestigationPlan):
        plan = InvestigationPlan.model_validate(plan)
    return enforce_plan_policy(plan, incident)

