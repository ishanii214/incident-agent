"""Persistence for investigation results.

Two interchangeable implementations behind one three-method contract
(save/get/list):

- InMemoryInvestigationRepository: dict-backed, deep-copy semantics, used
  by default and by the hermetic test suite.
- PostgresInvestigationRepository: psycopg 3 with hand-written SQL, one
  ``investigations`` table, connect-per-operation, lazy schema creation.

build_repository() is the single construction point, selected via
INCIDENT_AGENT_STORE=memory|postgres and DATABASE_URL.
"""

from __future__ import annotations

import os
import psycopg
from psycopg.rows import dict_row

from incident_agent.executor import ToolResult
from incident_agent.investigator import Hypothesis
from incident_agent.models import Incident
from incident_agent.schemas import InvestigationResponse
from incident_agent.verifier import VerificationResult

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS investigations (
    incident_id text PRIMARY KEY,
    service text NOT NULL,
    symptom text NOT NULL,
    started_at timestamptz NOT NULL,
    hypotheses jsonb NOT NULL DEFAULT '[]'::jsonb,
    verification jsonb,
    final_result text,
    retry_count integer NOT NULL DEFAULT 0,
    evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
)
"""


class StorageUnavailableError(RuntimeError):
    """Raised when the storage backend is unreachable or an operation fails."""


class InMemoryInvestigationRepository:
    """Dict-backed store for investigation records.

    Every record crossing this boundary is a deep Pydantic copy, so callers
    can never mutate stored state through a returned reference. Listing is
    newest-first (reverse insertion order).
    """

    def __init__(self) -> None:
        self._records: dict[str, InvestigationResponse] = {}

    def save(self, record: InvestigationResponse) -> InvestigationResponse:
        """Store a record under its incident id; returns a stored copy."""
        stored = record.model_copy(deep=True)
        self._records[record.incident.id] = stored
        return stored.model_copy(deep=True)

    def get(self, incident_id: str) -> InvestigationResponse | None:
        """Return a deep copy of the record, or None when unknown."""
        record = self._records.get(incident_id)
        return record.model_copy(deep=True) if record is not None else None

    def list(self) -> list[InvestigationResponse]:
        """Return deep copies of all records, newest first."""
        return [
            record.model_copy(deep=True)
            for record in reversed(list(self._records.values()))
        ]

_SAVE_SQL = """
INSERT INTO investigations (
    incident_id, service, symptom, started_at, hypotheses,
    verification, final_result, retry_count, evidence
) VALUES (
    %(incident_id)s, %(service)s, %(symptom)s, %(started_at)s, %(hypotheses)s,
    %(verification)s, %(final_result)s, %(retry_count)s, %(evidence)s
)
ON CONFLICT (incident_id) DO UPDATE SET
    service = EXCLUDED.service,
    symptom = EXCLUDED.symptom,
    started_at = EXCLUDED.started_at,
    hypotheses = EXCLUDED.hypotheses,
    verification = EXCLUDED.verification,
    final_result = EXCLUDED.final_result,
    retry_count = EXCLUDED.retry_count,
    evidence = EXCLUDED.evidence
"""

_SELECT_COLUMNS = (
    "incident_id, service, symptom, started_at, hypotheses, "
    "verification, final_result, retry_count, evidence"
)


class PostgresInvestigationRepository:
    """psycopg 3 store backed by the ``investigations`` table.

    Connections are opened per operation (no pooling this phase) and the
    constructor holds only the DSN — no connection is attempted until the
    first operation. The table is created lazily on first use.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._schema_ready = False

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self._dsn)
        except psycopg.OperationalError as exc:
            raise StorageUnavailableError("could not connect to storage") from exc

    def _ensure_schema(self, conn: psycopg.Connection) -> None:
        if self._schema_ready:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_DDL)
            conn.commit()
        except psycopg.OperationalError as exc:
            raise StorageUnavailableError("could not prepare storage schema") from exc
        self._schema_ready = True

    def save(self, record: InvestigationResponse) -> InvestigationResponse:
        """Upsert one record; duplicate incident ids overwrite."""
        verification = (
            record.verification.model_dump() if record.verification else None
        )
        values = {
            "incident_id": record.incident.id,
            "service": record.incident.service,
            "symptom": record.incident.symptom,
            "started_at": record.incident.started_at,
            "hypotheses": [h.model_dump() for h in record.hypotheses],
            "verification": verification,
            "final_result": record.final_result,
            "retry_count": record.retry_count,
            "evidence": [e.model_dump() for e in record.evidence],
        }
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                with conn.cursor() as cur:
                    cur.execute(_SAVE_SQL, values)
        except psycopg.OperationalError as exc:
            raise StorageUnavailableError("could not save investigation") from exc
        return record

    def get(self, incident_id: str) -> InvestigationResponse | None:
        """Fetch one record by incident id, or None when unknown."""
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        f"SELECT {_SELECT_COLUMNS} FROM investigations "
                        "WHERE incident_id = %s",
                        (incident_id,),
                    )
                    row = cur.fetchone()
        except psycopg.OperationalError as exc:
            raise StorageUnavailableError("could not read investigation") from exc
        return self._row_to_record(row) if row is not None else None

    def list(self) -> list[InvestigationResponse]:
        """Return all records, newest first (by insertion time)."""
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        f"SELECT {_SELECT_COLUMNS} FROM investigations "
                        "ORDER BY created_at DESC"
                    )
                    rows = cur.fetchall()
        except psycopg.OperationalError as exc:
            raise StorageUnavailableError("could not list investigations") from exc
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: dict) -> InvestigationResponse:
        """Reconstruct a validated record from one dict-shaped table row."""
        return InvestigationResponse(
            incident=Incident(
                id=row["incident_id"],
                service=row["service"],
                symptom=row["symptom"],
                started_at=row["started_at"],
            ),
            hypotheses=[Hypothesis.model_validate(h) for h in row["hypotheses"]],
            verification=(
                VerificationResult.model_validate(row["verification"])
                if row["verification"] is not None
                else None
            ),
            final_result=row["final_result"],
            retry_count=row["retry_count"],
            evidence=[ToolResult.model_validate(e) for e in row["evidence"]],
        )


def build_repository() -> (
    InMemoryInvestigationRepository | PostgresInvestigationRepository
):
    """Single construction point for the configured storage backend.

    INCIDENT_AGENT_STORE=memory (default) or postgres; postgres mode
    requires DATABASE_URL. Constructing either repository never opens a
    connection.
    """
    store = os.getenv("INCIDENT_AGENT_STORE", "memory").strip().lower()
    if store == "memory":
        return InMemoryInvestigationRepository()
    if store == "postgres":
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            raise RuntimeError(
                "INCIDENT_AGENT_STORE=postgres requires DATABASE_URL to be set"
            )
        return PostgresInvestigationRepository(dsn)
    raise RuntimeError(f"unsupported INCIDENT_AGENT_STORE value: {store!r}")