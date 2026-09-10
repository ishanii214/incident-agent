"""LangGraph investigation workflow: plan, execute, investigate, verify, replan.

Planner, investigator, and verifier are logical LLM roles behind structured
Pydantic contracts. execute_task deterministically dispatches validated tasks
and never loops internally. The verifier gates a bounded replan loop.
"""

from langgraph.graph import END, START, StateGraph

from incident_agent.executor import execute_planned_task
from incident_agent.investigator import run_investigator
from incident_agent.planner import ReplanContext, dedupe_tasks, run_planner
from incident_agent.state import InvestigationState
from incident_agent.verifier import MAX_RETRIES, run_verifier


def planner(state: InvestigationState, _planner_fn=None) -> dict:  # type: ignore[no-untyped-def]
    """Produce a plan; on replan cycles count the replan and dedupe tasks."""
    plan_fn = _planner_fn or run_planner
    is_replan = state["plan"] is not None
    replan = None
    if is_replan:
        verification = state["verification"]
        replan = ReplanContext(
            retry_count=state["retry_count"],
            verification_status=verification.status if verification is not None else None,
            missing_information=(
                list(verification.missing_information) if verification is not None else []
            ),
        )
    plan = plan_fn(state["incident"], list(state["completed_tasks"]), replan)
    pending = dedupe_tasks(list(plan.tasks), state["completed_tasks"])
    retry_count = state["retry_count"] + 1 if is_replan else state["retry_count"]
    return {"plan": plan, "pending_tasks": pending, "retry_count": retry_count}


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
    """Explicit stage kept for consistency; results are already in state."""
    return {}


def investigate(state: InvestigationState, _investigator_fn=None) -> dict:  # type: ignore[no-untyped-def]
    """Generate ranked unverified hypotheses from collected evidence."""
    fn = _investigator_fn or run_investigator
    out = fn(state["incident"], state["evidence"])
    return {"hypotheses": list(out.hypotheses)}


def verify(state: InvestigationState, _verifier_fn=None) -> dict:  # type: ignore[no-untyped-def]
    """Judge the leading hypothesis; set final result; never touch retry_count."""
    fn = _verifier_fn or run_verifier
    result = fn(state["incident"], state["hypotheses"], state["evidence"])
    update = {"verification": result}
    if result.status == "PASS":
        update["final_result"] = "verified"
    elif state["retry_count"] >= MAX_RETRIES:
        update["final_result"] = "investigation-unverified"
    return update


def _route_after_execute(state: InvestigationState) -> str:
    return "execute_task" if state["pending_tasks"] else "collect_evidence"


def _route_after_verify(state: InvestigationState) -> str:
    verification = state["verification"]
    if verification is not None and verification.status == "PASS":
        return "pass"
    return "replan" if state["retry_count"] < MAX_RETRIES else "fail"


def build_graph(_planner_fn=None, _investigator_fn=None, _verifier_fn=None):  # type: ignore[no-untyped-def]
    """Build and compile the bounded agentic investigation graph."""
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
    verifier_node = partial(verify, _verifier_fn=_verifier_fn) if _verifier_fn else verify
    builder.add_node("verify", verifier_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "execute_task")
    builder.add_conditional_edges(
        "execute_task",
        _route_after_execute,
        {"execute_task": "execute_task", "collect_evidence": "collect_evidence"},
    )
    builder.add_edge("collect_evidence", "investigate")
    builder.add_edge("investigate", "verify")
    builder.add_conditional_edges(
        "verify",
        _route_after_verify,
        {"pass": END, "replan": "planner", "fail": END},
    )
    return builder.compile()


graph = build_graph()
