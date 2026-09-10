"""Phase 0 checks for the minimal FastAPI health endpoint."""

from fastapi.testclient import TestClient

from incident_agent.app import app


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
