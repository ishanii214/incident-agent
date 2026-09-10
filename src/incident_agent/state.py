"""Investigation state contract for the LangGraph workflow."""

from typing import TypedDict

from incident_agent.executor import ToolResult
from incident_agent.models import Incident
from incident_agent.planner import InvestigationPlan, PlannedTask


class InvestigationState(TypedDict):
    """Single source of truth for one investigation run."""

    incident: Incident
    plan: InvestigationPlan | None
    pending_tasks: list[PlannedTask]
    completed_tasks: list[PlannedTask]
    evidence: list[ToolResult]
    hypotheses: list[str]
    verification: str | None
    retry_count: int
    final_result: str | None
