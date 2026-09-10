"""LangGraph investigation workflow with an LLM-powered planner.

Only the planner uses an LLM. All other nodes remain deterministic
placeholders and perform no tool execution or diagnosis.
"""

from langgraph.graph import END, START, StateGraph

from incident_agent.planner import run_planner
from incident_agent.state import InvestigationState


def planner(state: InvestigationState, _planner_fn=None) -> dict:  # type: ignore[no-untyped-def]
    """Produce a validated structured plan from the incident via LLM."""
    plan_fn = _planner_fn or run_planner
    plan = plan_fn(state["incident"])
    return {"plan": plan, "pending_tasks": list(plan.tasks)}


def execute_task(state: InvestigationState) -> dict:
    """Consume exactly one pending task, FIFO."""
    pending = list(state["pending_tasks"])
    completed = list(state["completed_tasks"])
    if pending:
        completed.append(pending.pop(0))
    return {"pending_tasks": pending, "completed_tasks": completed}


def collect_evidence(state: InvestigationState) -> dict:
    """Record one generic marker per completed task."""
    seen = set(state["evidence"])
    markers = []
    for task in state["completed_tasks"]:
        marker = f"evidence:{task.tool}:{task.service}"
        if marker not in seen:
            seen.add(marker)
            markers.append(marker)
    return {"evidence": [*state["evidence"], *markers]}


def investigate(state: InvestigationState) -> dict:
    """Create a generic placeholder hypothesis when evidence exists."""
    if state["evidence"] and not state["hypotheses"]:
        return {"hypotheses": ["placeholder-hypothesis-1"]}
    return {}


def verify(state: InvestigationState) -> dict:
    """Produce a deterministic placeholder verification and result."""
    if state["hypotheses"] and state["evidence"]:
        return {"verification": "passed", "final_result": "skeleton-complete"}
    return {"verification": "failed", "final_result": "skeleton-incomplete"}


def build_graph(_planner_fn=None):  # type: ignore[no-untyped-def]
    """Build and compile the linear investigation graph."""
    from functools import partial

    builder = StateGraph(InvestigationState)
    planner_node = partial(planner, _planner_fn=_planner_fn) if _planner_fn else planner
    builder.add_node("planner", planner_node)
    builder.add_node("execute_task", execute_task)
    builder.add_node("collect_evidence", collect_evidence)
    builder.add_node("investigate", investigate)
    builder.add_node("verify", verify)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "execute_task")
    builder.add_edge("execute_task", "collect_evidence")
    builder.add_edge("collect_evidence", "investigate")
    builder.add_edge("investigate", "verify")
    builder.add_edge("verify", END)
    return builder.compile()


graph = build_graph()
