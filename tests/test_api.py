"""Hermetic API tests for Phase 7.

No Ollama, no network: the module-level graph symbol in
incident_agent.app is monkeypatched with a recording fake.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import incident_agent.app as app_module
from incident_agent.executor import ToolResult
from incident_agent.investigator import Hypothesis
from incident_agent.models import MetricPoint
from incident_agent.planner import PlannerValidationError
from incident_agent.repository import InMemoryInvestigationRepository
from incident_agent.verifier import VerificationResult


def _fake_final_state(incident):
    point = MetricPoint(
        timestamp=incident.started_at,
        service=incident.service,
        metric="p95_latency_ms",
        value=820.0,
    )
    tool_result = ToolResult(
        task={
            "tool": "get_metrics",
            "service": incident.service,
            "metric": "p95_latency_ms",
            "start": "2026-09-10T13:50:00Z",
            "end": "2026-09-10T14:05:00Z",
        },
        metrics=[point],
    )
    hypothesis = Hypothesis(
        statement="Checkout degradation may be related to a recent deployment.",
        observed_facts=["checkout p95 latency increased after 14:00Z."],
        supporting_evidence=[0],
        contradicting_evidence=[],
        reasoning="Latency rose immediately after the deployment window.",
        confidence="medium",
    )
    verification = VerificationResult(
        status="PASS",
        rationale="Latency increase is directly supported by metrics evidence.",
        supporting_evidence=[0],
        missing_information=[],
    )
    return {
        "incident": incident,
        "plan": None,
        "pending_tasks": [],
        "completed_tasks": [],
        "evidence": [tool_result],
        "hypotheses": [hypothesis],
        "verification": verification,
        "retry_count": 0,
        "final_result": "verified",
    }


class FakeGraph:
    """Recording fake replacing incident_agent.app.graph in tests."""

    def __init__(self, final_state_fn):
        self.final_state_fn = final_state_fn
        self.calls: list[dict] = []

    def invoke(self, state):
        self.calls.append(state)
        return self.final_state_fn(state["incident"])

@pytest.fixture()
def fake_graph(monkeypatch):
    graph = FakeGraph(_fake_final_state)
    monkeypatch.setattr(app_module, "graph", graph)
    return graph


@pytest.fixture()
def repo(monkeypatch):
    """Fresh repository per test so API tests never share stored records."""
    repository = InMemoryInvestigationRepository()
    monkeypatch.setattr(app_module, "repository", repository)
    return repository


@pytest.fixture()
def client(repo):
    return TestClient(app_module.app)


def _post(client, **overrides):
    body = {
        "service": "checkout",
        "description": "Checkout latency increased significantly",
        "started_at": "2026-09-10T14:05:00Z",
    }
    body.update(overrides)
    return client.post("/incidents/investigate", json=body)


def test_health_still_works(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_valid_post_returns_200(client, fake_graph):
    resp = _post(client)
    assert resp.status_code == 200
    assert len(fake_graph.calls) == 1


def test_response_contains_structured_fields(client, fake_graph):
    data = _post(client).json()
    assert data["incident"]["service"] == "checkout"
    assert data["final_result"] == "verified"
    assert data["retry_count"] == 0
    assert data["verification"]["status"] == "PASS"
    assert data["hypotheses"][0]["confidence"] == "medium"
    assert data["evidence"][0]["task"]["tool"] == "get_metrics"
    assert data["evidence"][0]["metrics"][0]["value"] == 820.0


def test_missing_required_fields_return_422(client, fake_graph):
    for missing in ("service", "description", "started_at"):
        body = {
            "service": "checkout",
            "description": "Checkout latency increased",
            "started_at": "2026-09-10T14:05:00Z",
        }
        body.pop(missing)
        resp = client.post("/incidents/investigate", json=body)
        assert resp.status_code == 422, f"missing {missing} should 422"
    assert fake_graph.calls == []


def test_invalid_started_at_returns_422(client, fake_graph):
    resp = _post(client, started_at="not-a-date")
    assert resp.status_code == 422
    assert fake_graph.calls == []


def test_fake_graph_receives_mapped_incident_and_fresh_state(client, fake_graph):
    _post(client)
    state = fake_graph.calls[0]
    incident = state["incident"]
    assert incident.service == "checkout"
    assert incident.symptom == "Checkout latency increased significantly"
    assert incident.id.startswith("inc-")
    assert incident.started_at == datetime(2026, 9, 10, 14, 5, tzinfo=timezone.utc)
    assert state["plan"] is None
    assert state["pending_tasks"] == []
    assert state["completed_tasks"] == []
    assert state["evidence"] == []
    assert state["hypotheses"] == []
    assert state["verification"] is None
    assert state["retry_count"] == 0
    assert state["final_result"] is None


def test_response_does_not_expose_plan_or_pending(client, fake_graph):
    data = _post(client).json()
    assert "plan" not in data
    assert "pending_tasks" not in data
    assert "completed_tasks" not in data


def test_planner_validation_error_maps_to_502(client, monkeypatch):
    def boom(_state):
        raise PlannerValidationError("planner produced invalid plan")

    monkeypatch.setattr(app_module, "graph", FakeGraph(None))
    monkeypatch.setattr(app_module.graph, "invoke", boom)
    client = TestClient(app_module.app)
    resp = _post(client)
    assert resp.status_code == 502
    assert "detail" in resp.json()


def test_sequential_requests_have_independent_state(client, fake_graph):
    r1 = _post(client, service="checkout").json()
    r2 = _post(client, service="search").json()
    assert r1["incident"]["service"] == "checkout"
    assert r2["incident"]["service"] == "search"
    assert r1["incident"]["id"] != r2["incident"]["id"]
    assert len(fake_graph.calls) == 2
    assert fake_graph.calls[0]["incident"] is not fake_graph.calls[1]["incident"]


def test_post_persists_result(client, repo, fake_graph):
    resp = _post(client)
    assert resp.status_code == 200
    incident_id = resp.json()["incident"]["id"]
    stored = repo.get(incident_id)
    assert stored is not None
    assert stored.incident.id == incident_id
    assert stored.final_result == "verified"
    assert len(stored.evidence) == 1


def test_post_response_id_matches_get_id(client, fake_graph):
    post_id = _post(client).json()["incident"]["id"]
    detail = client.get(f"/incidents/{post_id}")
    assert detail.status_code == 200
    assert detail.json()["incident"]["id"] == post_id


def test_get_detail_round_trip(client, fake_graph):
    data = _post(client).json()
    detail = client.get(f"/incidents/{data['incident']['id']}").json()
    assert detail == data


def test_get_list_contains_investigated_incident(client, fake_graph):
    data = _post(client).json()
    listing = client.get("/incidents")
    assert listing.status_code == 200
    items = listing.json()
    assert len(items) == 1
    assert items[0]["incident"]["id"] == data["incident"]["id"]
    assert items[0]["incident"]["service"] == "checkout"
    assert items[0]["final_result"] == "verified"
    assert items[0]["verification_status"] == "PASS"


def test_get_list_newest_first(client, fake_graph):
    r1 = _post(client, service="checkout").json()
    r2 = _post(client, service="search").json()
    items = client.get("/incidents").json()
    assert [item["incident"]["id"] for item in items] == [
        r2["incident"]["id"],
        r1["incident"]["id"],
    ]


def test_list_returns_summaries_only(client, fake_graph):
    _post(client)
    item = client.get("/incidents").json()[0]
    assert set(item.keys()) == {"incident", "final_result", "verification_status"}


def test_list_does_not_expose_internals(client, fake_graph):
    _post(client)
    item = client.get("/incidents").json()[0]
    for forbidden in ("evidence", "hypotheses", "plan", "pending_tasks", "retry_count"):
        assert forbidden not in item


def test_unknown_id_returns_404(client):
    resp = client.get("/incidents/inc-doesnotexist")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "investigation not found"}


def test_repository_isolation_between_requests(client, repo, fake_graph):
    r1 = _post(client, service="checkout").json()
    r2 = _post(client, service="search").json()
    assert len(repo.list()) == 2
    assert repo.get(r1["incident"]["id"]).incident.service == "checkout"
    assert repo.get(r2["incident"]["id"]).incident.service == "search"


def test_repository_deep_copy_semantics(client, repo, fake_graph):
    data = _post(client).json()
    fetched = repo.get(data["incident"]["id"])
    fetched.final_result = "tampered"
    fetched.incident.service = "tampered"
    again = repo.get(data["incident"]["id"])
    assert again.final_result == "verified"
    assert again.incident.service == "checkout"