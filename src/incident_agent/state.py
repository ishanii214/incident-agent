"""Investigation state contract for the LangGraph orchestration skeleton."""

from typing import TypedDict

from incident_agent.models import Incident


class InvestigationState(TypedDict):
    """Single source of truth for one investigation run."""

    incident: Incident
    plan: list[str]
    pending_tasks: list[str]
    completed_tasks: list[str]
    evidence: list[str]
    hypotheses: list[str]
    verification: str | None
    retry_count: int
    final_result: str | None
