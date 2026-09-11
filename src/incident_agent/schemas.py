"""Typed API record schemas shared by the API layer and persistence.

Moved out of app.py in Phase 9 so the repository can construct records at
runtime without a circular import (app.py imports repository.py).
"""

from __future__ import annotations

from pydantic import BaseModel

from incident_agent.executor import ToolResult
from incident_agent.investigator import Hypothesis
from incident_agent.models import Incident
from incident_agent.verifier import VerificationResult


class InvestigationResponse(BaseModel):
    """Typed, externally consumable view of the final investigation state.

    Exposes the incident, ranked hypotheses, verification verdict, outcome,
    retry count, and collected evidence. Deliberately omits orchestration
    internals (plan, pending_tasks).
    """

    incident: Incident
    hypotheses: list[Hypothesis]
    verification: VerificationResult | None
    final_result: str | None
    retry_count: int
    evidence: list[ToolResult]


class InvestigationSummary(BaseModel):
    """Listing projection of a stored investigation (no evidence/hypotheses)."""

    incident: Incident
    final_result: str | None
    verification_status: str | None
