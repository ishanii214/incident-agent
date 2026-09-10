"""Phase 5 graph checks with injected fake planner and investigator."""

from datetime import datetime

from incident_agent.executor import ToolResult
from incident_agent.graph import build_graph
from incident_agent.investigator import Hypothesis, InvestigationHypotheses
from incident_agent.models import Incident
from incident_agent.planner import InvestigationPlan, PlannedTask


def _incident() -> Incident:
    return Incident(
        id="inc-001",
        service="checkout",
        symptom="p95 latency elevated",
        started_at=datetime.fromisoformat("2026-09-10T14:05:00+00:00"),
    )


def _fake_plan() -> InvestigationPlan:
    return InvestigationPlan(
        tasks=[
            PlannedTask(
                tool="get_metrics",
                service="checkout",
                metric="p95_latency_ms",
                start=datetime.fromisoformat("2026-09-10T13:50:00+00:00"),
                end=datetime.fromisoformat("2026-09-10T14:05:00+00:00"),
            ),
            PlannedTask(
                tool="search_logs",
                service="checkout",
                start=datetime.fromisoformat("2026-09-10T13:50:00+00:00"),
                end=datetime.fromisoformat("2026-09-10T14:05:00+00:00"),
                keyword="timeout",
                level="ERROR",
            ),
        ]
    )


def _fake_planner(incident: Incident) -> InvestigationPlan:
    assert incident.id == "inc-001"
    return _fake_plan()


def _fake_investigator(incident: Incident, evidence: list[ToolResult]) -> InvestigationHypotheses:
    assert incident.id == "inc-001"
    assert evidence
    return InvestigationHypotheses(
        hypotheses=[
            Hypothesis(
                statement="Checkout degradation may be related to the recent change window.",
                observed_facts=["Latency samples increased within the investigated window."],
                supporting_evidence=[0],
                contradicting_evidence=[],
                reasoning="The collected metrics show higher values later in the window.",
                confidence="medium",
            )
        ]
    )


def _initial_state() -> dict:
    return {
        "incident": _incident(),
        "plan": None,
        "pending_tasks": [],
        "completed_tasks": [],
        "evidence": [],
        "hypotheses": [],
        "verification": None,
        "retry_count": 0,
        "final_result": None,
    }


def _serializable(result: dict) -> dict:
    from incident_agent.executor import ToolResult

    dumped = dict(result)
    if isinstance(result["incident"], Incident):
        dumped["incident"] = result["incident"].model_dump(mode="json")
    if isinstance(result.get("plan"), InvestigationPlan):
        dumped["plan"] = result["plan"].model_dump(mode="json")
    for key in ("pending_tasks", "completed_tasks"):
        dumped[key] = [
            item.model_dump(mode="json") if isinstance(item, PlannedTask) else item
            for item in result[key]
        ]
    dumped["evidence"] = [
        item.model_dump(mode="json") if isinstance(item, ToolResult) else item
        for item in result["evidence"]
    ]
    dumped["hypotheses"] = [
        item.model_dump(mode="json") if isinstance(item, Hypothesis) else item
        for item in result["hypotheses"]
    ]
    return dumped


def test_graph_compiles() -> None:
    assert build_graph() is not None


def test_graph_runs_incident_to_terminal_state() -> None:
    result = build_graph(
        _planner_fn=_fake_planner, _investigator_fn=_fake_investigator
    ).invoke(_initial_state())

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
    assert isinstance(result["plan"], InvestigationPlan)
    assert [task.tool for task in result["plan"].tasks] == ["get_metrics", "search_logs"]
    assert all(isinstance(task, PlannedTask) for task in result["pending_tasks"])
    assert all(isinstance(task, PlannedTask) for task in result["completed_tasks"])
    assert result["completed_tasks"][0].tool == "get_metrics"
    assert result["completed_tasks"][0].service == "checkout"
    assert len(result["pending_tasks"]) == len(result["plan"].tasks) - 1
    assert len(result["evidence"]) == 1
    assert isinstance(result["evidence"][0], ToolResult)
    assert result["evidence"][0].task.tool == "get_metrics"
    assert len(result["evidence"][0].metrics) >= 5
    assert len(result["hypotheses"]) == 1
    assert isinstance(result["hypotheses"][0], Hypothesis)
    assert result["hypotheses"][0].supporting_evidence == [0]
    assert result["hypotheses"][0].confidence == "medium"
    assert "verified" not in Hypothesis.model_fields
    assert result["verification"] == "passed"
    assert result["final_result"] == "skeleton-complete"
    assert result["retry_count"] == 0


def test_graph_is_deterministic() -> None:
    graph = build_graph(_planner_fn=_fake_planner, _investigator_fn=_fake_investigator)

    assert _serializable(graph.invoke(_initial_state())) == _serializable(graph.invoke(_initial_state()))


def test_graph_leaks_no_verdict_or_remote_calls() -> None:
    result = build_graph(
        _planner_fn=_fake_planner, _investigator_fn=_fake_investigator
    ).invoke(_initial_state())
    payload = str(_serializable(result)).lower()

    assert "root_cause" not in payload
    assert "rootcause" not in payload
    assert "culprit" not in payload

    import incident_agent.graph as graph_module
    import incident_agent.planner as planner_module

    graph_source = open(graph_module.__file__, encoding="utf-8").read()
    planner_source = open(planner_module.__file__, encoding="utf-8").read()
    import incident_agent.executor as executor_module

    executor_source = open(executor_module.__file__, encoding="utf-8").read()
    assert "incident_agent.tools" not in graph_source
    assert "incident_agent.tools" not in planner_source
    assert "incident_agent.executor" in graph_source
    assert "getattr" not in executor_source and "subprocess" not in executor_source
    assert "ToolNode" not in graph_source
    assert "urllib" not in graph_source and "socket" not in graph_source
    assert "requests" not in graph_source and "subprocess" not in graph_source
    assert "v2.4.1" not in planner_source and "payments" not in planner_source.lower()
    assert "incident_agent.executor" in graph_source

