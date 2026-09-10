"""Phase 2 checks for the deterministic LangGraph skeleton."""

from datetime import datetime

from incident_agent.graph import build_graph
from incident_agent.models import Incident


def _incident() -> Incident:
    return Incident(
        id="inc-001",
        service="checkout",
        symptom="p95 latency elevated",
        started_at=datetime.fromisoformat("2026-09-10T14:05:00+00:00"),
    )


def _initial_state() -> dict:
    return {
        "incident": _incident(),
        "plan": [],
        "pending_tasks": [],
        "completed_tasks": [],
        "evidence": [],
        "hypotheses": [],
        "verification": None,
        "retry_count": 0,
        "final_result": None,
    }


def _serializable(result: dict) -> dict:
    return {
        **result,
        "incident": result["incident"].model_dump(mode="json")
        if isinstance(result["incident"], Incident)
        else result["incident"],
    }


def test_graph_compiles() -> None:
    assert build_graph() is not None


def test_graph_runs_incident_to_terminal_state() -> None:
    result = build_graph().invoke(_initial_state())

    assert set(result) == {
        "incident",
        "plan",
        "pending_tasks",
        "completed_tasks",
        "evidence",
        "hypotheses",
        "verification",
        "retry_count",
        "final_result",
    }
    assert result["incident"] == _incident()
    assert result["completed_tasks"] == ["check-metrics:checkout"]
    assert result["pending_tasks"] == ["check-logs:checkout", "check-deployments:checkout"]
    assert result["evidence"] == ["evidence:check-metrics:checkout"]
    assert result["hypotheses"] == ["placeholder-hypothesis-1"]
    assert result["verification"] == "passed"
    assert result["final_result"] == "skeleton-complete"
    assert result["retry_count"] == 0


def test_graph_is_deterministic() -> None:
    graph = build_graph()

    assert _serializable(graph.invoke(_initial_state())) == _serializable(graph.invoke(_initial_state()))


def test_graph_leaks_no_verdict_or_remote_calls() -> None:
    result = build_graph().invoke(_initial_state())
    payload = str(_serializable(result)).lower()

    assert "root_cause" not in payload
    assert "rootcause" not in payload
    assert "culprit" not in payload
    assert "diagnosis" not in payload
    assert "answer" not in payload.replace("placeholder-hypothesis-1", "")

    import incident_agent.graph as graph_module

    source = open(graph_module.__file__, encoding="utf-8").read()
    assert "incident_agent.tools" not in source
    assert "ToolNode" not in source
    assert "urllib" not in source and "socket" not in source and "requests" not in source
