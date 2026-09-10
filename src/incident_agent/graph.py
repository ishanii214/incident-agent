"""LangGraph investigation workflow with planner, execution, hypotheses.

The planner and investigator use an LLM. execute_task deterministically
dispatches validated tasks. verify remains a placeholder.
"""

from langgraph.graph import END, START, StateGraph

from incident_agent.executor import execute_planned_task
from incident_agent.investigator import run_investigator
from incident_agent.planner import run_planner
from incident_agent.state import InvestigationState


def planner(state: InvestigationState, _planner_fn=None) -> dict:  # type: ignore[no-untyped-def]
    """Produce a validated structured plan from the incident via LLM."""
    plan_fn = _planner_fn or run_planner
    plan = plan_fn(state["incident"])
    return {"plan": plan, "pending_tasks": list(plan.tasks)}


def execute_task(state: InvestigationState) -> dict:
    """Execute exactly one pending task and store its typed result."""
    pending = list(state["pending_tasks"])
    completed = list(state["completed_tasks"])
    evidence = list(state["evidence"])
    if not pending:
        return {"pending_tasks": pending, "completed_tasks": completed}
    task = pending.pop(0)
    result = execute_planned_task(task)
    return {
        "pending_tasks": pending,
        "completed_tasks": [*completed, task],
        "evidence": [*evidence, result],
    }


def collect_evidence(state: InvestigationState) -> dict:
    """Explicit stage kept for Phase 5; execute_task already stores results."""
    return {}


def investigate(state: InvestigationState, _investigator_fn=None) -> dict:  # type: ignore[no-untyped-def]
    """Generate ranked unverified hypotheses from collected evidence."""
    fn = _investigator_fn or run_investigator
    out = fn(state["incident"], state["evidence"])
    return {"hypotheses": list(out.hypotheses)}


def verify(state: InvestigationState) -> dict:
    """Produce a deterministic placeholder verification and result."""
    if state["hypotheses"] and state["evidence"]:
        return {"verification": "passed", "final_result": "skeleton-complete"}
    return {"verification": "failed", "final_result": "skeleton-incomplete"}


def build_graph(_planner_fn=None, _investigator_fn=None):  # type: ignore[no-untyped-def]
    """Build and compile the linear investigation graph."""
    from functools import partial

    builder = StateGraph(InvestigationState)
    planner_node = partial(planner, _planner_fn=_planner_fn) if _planner_fn else planner
    builder.add_node("planner", planner_node)
    builder.add_node("execute_task", execute_task)
    builder.add_node("collect_evidence", collect_evidence)
    investigator_node = (
        partial(investigate, _investigator_fn=_investigator_fn) if _investigator_fn else investigate
    )
    builder.add_node("investigate", investigator_node)
    builder.add_node("verify", verify)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "execute_task")
    builder.add_edge("execute_task", "collect_evidence")
    builder.add_edge("collect_evidence", "investigate")
    builder.add_edge("investigate", "verify")
    builder.add_edge("verify", END)
    return builder.compile()


graph = build_graph()
