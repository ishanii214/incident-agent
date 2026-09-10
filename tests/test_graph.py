"""Phase 6 graph checks with fake planner, investigator, and verifier."""

from datetime import datetime

from incident_agent.executor import ToolResult
from incident_agent.graph import build_graph, execute_task
from incident_agent.investigator import Hypothesis, InvestigationHypotheses
from incident_agent.models import Incident
from incident_agent.planner import InvestigationPlan, PlannedTask, dedupe_tasks
from incident_agent.verifier import MAX_RETRIES, VerificationResult


def _incident() -> Incident:
    return Incident(
        id="inc-001",
        service="checkout",
        symptom="p95 latency elevated",
        started_at=datetime.fromisoformat("2026-09-10T14:05:00+00:00"),
    )


def _metrics_task() -> PlannedTask:
    return PlannedTask(
        tool="get_metrics",
        service="checkout",
        metric="p95_latency_ms",
        start=datetime.fromisoformat("2026-09-10T13:50:00+00:00"),
        end=datetime.fromisoformat("2026-09-10T14:05:00+00:00"),
    )


def _logs_task() -> PlannedTask:
    return PlannedTask(
        tool="search_logs",
        service="checkout",
        start=datetime.fromisoformat("2026-09-10T13:50:00+00:00"),
        end=datetime.fromisoformat("2026-09-10T14:05:00+00:00"),
        keyword="timeout",
        level="ERROR",
    )


def _deps_task() -> PlannedTask:
    return PlannedTask(
        tool="get_recent_deployments",
        service="checkout",
        since=datetime.fromisoformat("2026-09-10T12:05:00+00:00"),
        limit=10,
    )


def _plan(tools: list[str]) -> InvestigationPlan:
    mapper = {
        "get_metrics": _metrics_task,
        "search_logs": _logs_task,
        "get_recent_deployments": _deps_task,
    }
    return InvestigationPlan(tasks=[mapper[tool]() for tool in tools])


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        statement="Checkout degradation may be related to the recent change window.",
        observed_facts=["Latency samples increased within the investigated window."],
        supporting_evidence=[0],
        contradicting_evidence=[],
        reasoning="Metrics rise later in the investigated window.",
        confidence="medium",
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


class _RecordingPlanner:
    def __init__(self, plan_provider):  # type: ignore[no-untyped-def]
        self.calls: list[dict] = []
        self.plan_provider = plan_provider

    def __call__(self, incident, completed_tasks, replan=None):  # type: ignore[no-untyped-def]
        self.calls.append(
            {
                "incident": incident,
                "completed_tasks": list(completed_tasks),
                "replan": replan,
            }
        )
        return self.plan_provider(len(self.calls))


class _RecordingInvestigator:
    def __call__(self, incident, evidence):  # type: ignore[no-untyped-def]
        assert incident.id == "inc-001"
        return InvestigationHypotheses(hypotheses=[_hypothesis()])


class _RecordingVerifier:
    def __init__(self, verdict_provider):  # type: ignore[no-untyped-def]
        self.calls: list = []
        self.verdict_provider = verdict_provider

    def __call__(self, incident, hypotheses, evidence):  # type: ignore[no-untyped-def]
        self.calls.append((incident, list(hypotheses), list(evidence)))
        return self.verdict_provider(len(self.calls))


def _pass_verdict(_attempt: int) -> VerificationResult:
    return VerificationResult(
        status="PASS",
        rationale="Evidence substantially supports the leading hypothesis.",
        supporting_evidence=[0],
        missing_information=[],
    )


def _fail_verdict(_attempt: int) -> VerificationResult:
    return VerificationResult(
        status="FAIL",
        rationale="Evidence does not sufficiently support the leading hypothesis.",
        supporting_evidence=[],
        missing_information=["error_rate signal missing for the same window"],
    )


def _serializable(result: dict) -> dict:
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
    dumped["verification"] = (
        result["verification"].model_dump(mode="json")
        if isinstance(result["verification"], VerificationResult)
        else None
    )
    return dumped


def test_graph_compiles() -> None:
    assert build_graph() is not None


def test_graph_pass_path() -> None:
    planner = _RecordingPlanner(lambda _i: _plan(["get_metrics", "search_logs"]))
    verifier = _RecordingVerifier(_pass_verdict)
    result = build_graph(
        _planner_fn=planner,
        _investigator_fn=_RecordingInvestigator(),
        _verifier_fn=verifier,
    ).invoke(_initial_state())

    assert len(planner.calls) == 1
    assert len(verifier.calls) == 1
    assert result["retry_count"] == 0
    assert result["pending_tasks"] == []
    assert [task.tool for task in result["completed_tasks"]] == ["get_metrics", "search_logs"]
    assert len(result["evidence"]) == 2
    assert isinstance(result["verification"], VerificationResult)
    assert result["verification"].status == "PASS"
    assert result["final_result"] == "verified"


def test_graph_all_fail_reaches_budget() -> None:
    planner = _RecordingPlanner(lambda _i: _plan(["get_metrics"]))
    verifier = _RecordingVerifier(_fail_verdict)
    result = build_graph(
        _planner_fn=planner,
        _investigator_fn=_RecordingInvestigator(),
        _verifier_fn=verifier,
    ).invoke(_initial_state())

    assert len(verifier.calls) == MAX_RETRIES + 1
    assert len(planner.calls) == MAX_RETRIES + 1
    replans = [call for call in planner.calls[1:] if call["replan"] is not None]
    assert len(replans) == MAX_RETRIES
    assert result["retry_count"] == MAX_RETRIES
    assert result["verification"].status == "FAIL"
    assert result["final_result"] == "investigation-unverified"


def test_execute_drain_invariant() -> None:
    tasks = [_metrics_task(), _logs_task(), _deps_task()]
    st = {"pending_tasks": list(tasks), "completed_tasks": [], "evidence": []}
    for expected in tasks:
        before_p = len(st["pending_tasks"])
        before_c = len(st["completed_tasks"])
        before_e = len(st["evidence"])
        update = execute_task(st)
        st = {**st, **update}
        assert len(st["pending_tasks"]) == before_p - 1
        assert len(st["completed_tasks"]) == before_c + 1
        assert len(st["evidence"]) == before_e + 1
        assert st["completed_tasks"][-1] == expected
        assert st["evidence"][-1].task.tool == expected.tool
    assert st["pending_tasks"] == []

    before_c = len(st["completed_tasks"])
    before_e = len(st["evidence"])
    update = execute_task(st)
    st = {**st, **update}
    assert len(st["completed_tasks"]) == before_c
    assert len(st["evidence"]) == before_e


def test_graph_executes_all_planned_tasks_in_one_cycle() -> None:
    planner = _RecordingPlanner(
        lambda _i: _plan(["get_metrics", "search_logs", "get_recent_deployments"])
    )
    result = build_graph(
        _planner_fn=planner,
        _investigator_fn=_RecordingInvestigator(),
        _verifier_fn=_RecordingVerifier(_pass_verdict),
    ).invoke(_initial_state())

    assert result["pending_tasks"] == []
    assert [task.tool for task in result["completed_tasks"]] == [
        "get_metrics",
        "search_logs",
        "get_recent_deployments",
    ]
    assert [item.task.tool for item in result["evidence"]] == [
        "get_metrics",
        "search_logs",
        "get_recent_deployments",
    ]
    assert len(planner.calls) == 1


def test_dedupe_within_plan_and_against_completed() -> None:
    metrics = _metrics_task()
    logs = _logs_task()
    deps = _deps_task()

    out = dedupe_tasks([logs, logs, metrics, deps], completed=[metrics])
    assert out == [logs, deps]


def test_replanning_executes_new_task_and_preserves_evidence() -> None:
    planner = _RecordingPlanner(
        lambda i: _plan(["get_metrics"]) if i == 1 else _plan(["search_logs"])
    )
    verifier = _RecordingVerifier(lambda i: _pass_verdict(i) if i == 2 else _fail_verdict(i))
    result = build_graph(
        _planner_fn=planner,
        _investigator_fn=_RecordingInvestigator(),
        _verifier_fn=verifier,
    ).invoke(_initial_state())

    assert len(verifier.calls) == 2
    assert result["retry_count"] == 1
    assert result["final_result"] == "verified"
    assert [task.tool for task in result["completed_tasks"]] == ["get_metrics", "search_logs"]
    assert [item.task.tool for item in result["evidence"]] == ["get_metrics", "search_logs"]


def test_planner_receives_replan_context() -> None:
    planner = _RecordingPlanner(lambda _i: _plan(["get_metrics"]))
    verifier = _RecordingVerifier(_fail_verdict)
    build_graph(
        _planner_fn=planner,
        _investigator_fn=_RecordingInvestigator(),
        _verifier_fn=verifier,
    ).invoke(_initial_state())

    replan = planner.calls[1]["replan"]
    assert replan is not None
    assert replan.verification_status == "FAIL"
    assert replan.missing_information == ["error_rate signal missing for the same window"]
    assert [task.tool for task in planner.calls[1]["completed_tasks"]] == ["get_metrics"]


def test_graph_is_deterministic() -> None:
    planner = _RecordingPlanner(lambda _i: _plan(["get_metrics", "search_logs"]))
    graph = build_graph(
        _planner_fn=planner,
        _investigator_fn=_RecordingInvestigator(),
        _verifier_fn=_RecordingVerifier(_pass_verdict),
    )

    assert _serializable(graph.invoke(_initial_state())) == _serializable(graph.invoke(_initial_state()))


def test_graph_leaks_no_verdict_or_remote_calls() -> None:
    planner = _RecordingPlanner(lambda _i: _plan(["get_metrics"]))
    result = build_graph(
        _planner_fn=planner,
        _investigator_fn=_RecordingInvestigator(),
        _verifier_fn=_RecordingVerifier(_pass_verdict),
    ).invoke(_initial_state())
    payload = str(_serializable(result)).lower()
    assert "root_cause" not in payload
    assert "rootcause" not in payload
    assert "culprit" not in payload

    import incident_agent.graph as graph_module
    import incident_agent.planner as planner_module
    import incident_agent.verifier as verifier_module

    graph_source = open(graph_module.__file__, encoding="utf-8").read()
    planner_source = open(planner_module.__file__, encoding="utf-8").read()
    verifier_source = open(verifier_module.__file__, encoding="utf-8").read()
    assert "incident_agent.tools" not in graph_source
    assert "incident_agent.tools" not in planner_source
    assert "incident_agent.tools" not in verifier_source
    assert "incident_agent.executor" in graph_source
    assert "getattr" not in verifier_source and "subprocess" not in verifier_source
    assert "ToolNode" not in graph_source
    assert "v2.4.1" not in planner_source and "payments" not in planner_source.lower()
    assert "incident_agent.executor" in graph_source

