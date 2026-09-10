# incident-agent

AI Incident Resolution Agent for investigating production/service incidents.

## Problem

When a production incident happens, engineers manually inspect metrics, logs, recent deployments, service health, and other operational evidence to determine the likely root cause. This is slow, inconsistent, and difficult to explain under pressure.

## Solution

`incident-agent` performs a structured investigation instead of generating a single LLM answer:

```text
Incident
→ Plan
→ Execute investigation tasks using typed tools
→ Collect evidence
→ Form hypothesis
→ Verify hypothesis
→ If verification fails, re-plan and investigate again
→ Final recommendation
```

The governing principle is:

```text
PLAN → EXECUTE → VERIFY → RECOVER/REPLAN
```

## Architecture

- LangGraph is the workflow/orchestration layer.
- LangChain is used for LLM/tool integration.
- FastAPI is the API layer, not the agent logic.
- Tools provide deterministic facts/data.
- Pydantic defines structured interfaces/state.
- The verifier is logically separate from the investigator/hypothesis generator.
- Potentially destructive actions are approval-gated or simulated.
- No unrestricted shell/Python execution as an agent tool.

## Workflow

1. Receive an incident.
2. Produce an investigation plan.
3. Execute investigation tasks through typed tools.
4. Collect evidence from those tools.
5. Form a hypothesis.
6. Independently verify the hypothesis.
7. Re-plan within bounded retries when verification fails.
8. Return an evidence-based final recommendation.

## Key features

- Planning before tool use.
- Explicit tool selection and multi-step investigation.
- Evidence-based reasoning.
- Hypothesis generation and independent verification.
- Bounded retries and re-planning.
- Final recommendation grounded in collected evidence.

## Technology stack

- Python
- LangGraph
- LangChain
- Pydantic
- FastAPI
- Ollama/local LLM initially where practical
- JSON/CSV for initial synthetic operational data
- SQLite initially, PostgreSQL later
- pytest
- Git/GitHub

## Example investigation

```text
Incident: checkout latency is elevated.

Plan: inspect latency metrics, review recent deployments, and check error logs.

Evidence:
- Latency increased after the latest checkout deployment.
- Error logs show repeated downstream timeouts.
- No comparable change in unrelated services.

Hypothesis: the latest checkout deployment introduced a downstream timeout.

Verification: deployment timestamp, affected service scope, and error pattern align; alternative explanations were checked and rejected.

Recommendation: roll back or fix the checkout deployment after approval, then monitor latency and timeouts.
```

## Setup

Requires Python 3.11 or later.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[test]
pytest
uvicorn incident_agent.app:app --reload
```

Check:

```text
GET http://127.0.0.1:8000/health
```

Expected response:

```json
{"status": "ok"}
```

## Evaluation

Future evaluation should check:

- whether plans cover the reported incident;
- whether tool use is appropriate and multi-step;
- whether conclusions cite collected evidence;
- whether verification is independent of hypothesis generation;
- whether retries and re-planning stay bounded;
- whether recommendations are safe and approval-aware.

## Limitations and future improvements

- No investigation workflow is implemented yet.
- No synthetic operational data or tools are implemented yet.
- No planner, investigator, or verifier is implemented yet.
- Future work may add a dashboard, PostgreSQL, observability, Docker, and CI/CD.
- Kafka, Redis, Kubernetes, Neo4j, FAISS, heavy RAG, authentication, and unrelated infrastructure are intentionally out of scope unless explicitly required later.
