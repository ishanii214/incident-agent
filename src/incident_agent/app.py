"""FastAPI transport layer for the incident investigation agent.

Thin boundary only: validate HTTP input, construct the internal Incident,
invoke the existing LangGraph workflow with a fresh per-request state, and map
the final graph state into a typed response. No agent logic lives here.
"""

from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from incident_agent.executor import ToolResult, UnsupportedToolError
from incident_agent.graph import graph
from incident_agent.investigator import Hypothesis, InvestigatorValidationError
from incident_agent.models import Incident
from incident_agent.planner import PlannerValidationError
from incident_agent.verifier import VerifierValidationError, VerificationResult

app = FastAPI(title="incident-agent")

_AGENT_BOUNDARY_ERRORS = (
    PlannerValidationError,
    InvestigatorValidationError,
    VerifierValidationError,
    UnsupportedToolError,
)


class InvestigateRequest(BaseModel):
    """HTTP request contract for starting an investigation.

    Deliberately not the internal Incident model: identity is server-owned
    (no client-supplied id) and the API-level ``description`` maps to the
    internal ``symptom`` at this boundary.
    """

    service: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=2000)
    started_at: datetime


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


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint used for local checks and future readiness probes."""
    return {"status": "ok"}


@app.post("/incidents/investigate", response_model=InvestigationResponse)
def investigate_incident(request: InvestigateRequest) -> InvestigationResponse:
    """Run one bounded investigation for the supplied incident description."""
    incident = Incident(
        id=f"inc-{uuid4().hex[:8]}",
        service=request.service,
        symptom=request.description,
        started_at=request.started_at,
    )
    state = {
        "incident": incident,
        "plan": None,
        "pending_tasks": [],
        "completed_tasks": [],
        "evidence": [],
        "hypotheses": [],
        "verification": None,
        "retry_count": 0,
        "final_result": None,
    }
    try:
        result = graph.invoke(state)
    except _AGENT_BOUNDARY_ERRORS as exc:
        raise HTTPException(
            status_code=502,
            detail="investigation reasoning component produced invalid output",
        ) from exc
    return InvestigationResponse(
        incident=result["incident"],
        hypotheses=result["hypotheses"],
        verification=result["verification"],
        final_result=result["final_result"],
        retry_count=result["retry_count"],
        evidence=result["evidence"],
    )
