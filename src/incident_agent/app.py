"""Minimal FastAPI application for incident-agent Phase 0."""

from fastapi import FastAPI

app = FastAPI(title="incident-agent")


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint used for local checks and future readiness probes."""
    return {"status": "ok"}
