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
from incident_agent.repository import build_repository, StorageUnavailableError
from incident_agent.investigator import Hypothesis, InvestigatorValidationError
from incident_agent.models import Incident
from incident_agent.planner import PlannerValidationError
from incident_agent.schemas import InvestigationResponse, InvestigationSummary
from incident_agent.verifier import VerifierValidationError, VerificationResult

app = FastAPI(title="incident-agent")

# Single construction point for storage. Default is the in-memory store so
# the app (and the whole hermetic test suite) runs with zero infrastructure;
# INCIDENT_AGENT_STORE=postgres + DATABASE_URL selects PostgreSQL.
repository = build_repository()

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
    response = InvestigationResponse(
        incident=result["incident"],
        hypotheses=result["hypotheses"],
        verification=result["verification"],
        final_result=result["final_result"],
        retry_count=result["retry_count"],
        evidence=result["evidence"],
    )
    # Persist the terminal result. Reached only after the try/except above,
    # so agent-boundary failures (502) never persist a record. The caller
    # gets the built response; the repository keeps its own deep copy.
    try:
        repository.save(response)
    except StorageUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="investigation storage unavailable",
        ) from exc
    return response


@app.get("/incidents", response_model=list[InvestigationSummary])
def list_investigations() -> list[InvestigationSummary]:
    """List previously investigated incidents, newest first."""
    try:
        records = repository.list()
    except StorageUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="investigation storage unavailable",
        ) from exc
    return [
        InvestigationSummary(
            incident=record.incident,
            final_result=record.final_result,
            verification_status=record.verification.status if record.verification else None,
        )
        for record in records
    ]


@app.get("/incidents/{incident_id}", response_model=InvestigationResponse)
def get_investigation(incident_id: str) -> InvestigationResponse:
    """Retrieve one previously investigated incident by its incident id."""
    try:
        record = repository.get(incident_id)
    except StorageUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="investigation storage unavailable",
        ) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="investigation not found")
    return record
