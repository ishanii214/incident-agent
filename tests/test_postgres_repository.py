"""Hermetic tests for the PostgreSQL-backed repository.

Hermetic tests cover the repository factory, configuration behavior, and the
focused storage exception contract without requiring a real PostgreSQL instance.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from incident_agent.repository import (
    StorageUnavailableError,
    build_repository,
    InMemoryInvestigationRepository,
    PostgresInvestigationRepository,
)


# ---------------------------------------------------------------------------
# Hermetic tests (no PostgreSQL required)
# ---------------------------------------------------------------------------


class TestBuildRepositoryDefaultsToMemory:
    """build_repository() defaults to the in-memory store."""

    def test_default_store_is_memory(self):
        with patch.dict(os.environ, {}, clear=True):
            repo = build_repository()
        assert isinstance(repo, InMemoryInvestigationRepository)


class TestFactoryMemoryMode:
    """INCIDENT_AGENT_STORE=memory selects the in-memory repository."""

    def test_memory_store_selected(self):
        with patch.dict(os.environ, {"INCIDENT_AGENT_STORE": "memory"}, clear=True):
            repo = build_repository()
        assert isinstance(repo, InMemoryInvestigationRepository)


class TestFactoryPostgresMode:
    """INCIDENT_AGENT_STORE=postgres selects PostgresInvestigationRepository."""

    def test_postgres_store_selected(self):
        dsn = "postgresql://incident_agent:password@localhost:5432/incident_agent"
        with patch.dict(
            os.environ,
            {"INCIDENT_AGENT_STORE": "postgres", "DATABASE_URL": dsn},
            clear=True,
        ):
            repo = build_repository()
        assert isinstance(repo, PostgresInvestigationRepository)
        assert repo._dsn == dsn


class TestFactoryRequiresDsnForPostgres:
    """Postgres mode raises RuntimeError without DATABASE_URL."""

    def test_raises_without_database_url(self):
        with patch.dict(os.environ, {"INCIDENT_AGENT_STORE": "postgres"}, clear=True):
            with pytest.raises(RuntimeError, match="DATABASE_URL"):
                build_repository()


class TestPostgresRepositoryIsLazy:
    """Constructing PostgresInvestigationRepository never opens a connection."""

    def test_no_connection_on_construction(self):
        with patch("psycopg.connect") as mock_connect:
            PostgresInvestigationRepository("postgresql://localhost/test")
        mock_connect.assert_not_called()


class TestStorageFailureConvertsToStorageUnavailableError:
    """psycopg.OperationalError from save() becomes StorageUnavailableError."""

    def test_save_operational_error_is_wrapped(self):
        import psycopg

        repo = PostgresInvestigationRepository("postgresql://localhost/test")
        mock_connect = MagicMock()
        mock_connect.__enter__ = MagicMock(return_value=mock_connect)
        mock_connect.__exit__ = MagicMock(return_value=False)
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.execute.side_effect = psycopg.OperationalError("connection refused")
        mock_connect.cursor.return_value = mock_cursor

        with patch("psycopg.connect", return_value=mock_connect):
            with pytest.raises(StorageUnavailableError, match="save investigation"):
                repo.save(MagicMock())


# ---------------------------------------------------------------------------
# Integration tests (require DATABASE_URL)
# ---------------------------------------------------------------------------


def _has_postgres():
    """Return True when DATABASE_URL is set and PostgreSQL is reachable."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return False
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_postgres(),
    reason="DATABASE_URL not set or PostgreSQL not reachable",
)


@pytest.mark.integration
class TestPostgresRepositoryIntegration:
    """Integration tests against a real PostgreSQL instance."""

    @pytest.fixture(autouse=True)
    def _ensure_postgres(self):
        assert os.environ.get("DATABASE_URL"), "DATABASE_URL not set"

    def test_save_and_get(self):
        repo = PostgresInvestigationRepository(os.environ["DATABASE_URL"])
        record = InvestigationResponse(
            incident=Incident(
                id="inc-integration-001",
                service="checkout",
                symptom="Checkout latency increased",
                started_at="2026-09-11T10:00:00Z",
            ),
            hypotheses=[],
            verification=None,
            final_result=None,
            retry_count=0,
            evidence=[],
        )
        repo.save(record)

        got = repo.get("inc-integration-001")
        assert got is not None
        assert got.incident.id == record.incident.id
        assert got.incident.service == record.incident.service
        assert got.hypotheses == record.hypotheses
        assert got.verification == record.verification

    def test_list_newest_first(self):
        repo = PostgresInvestigationRepository(os.environ["DATABASE_URL"])
        r1 = InvestigationResponse(
            incident=Incident(id="inc-list-001", service="svc", symptom="s1", started_at="2026-09-11T10:00:00Z"),
            hypotheses=[], verification=None, final_result=None, retry_count=0, evidence=[],
        )
        r2 = InvestigationResponse(
            incident=Incident(id="inc-list-002", service="svc", symptom="s2", started_at="2026-09-11T11:00:00Z"),
            hypotheses=[], verification=None, final_result=None, retry_count=0, evidence=[],
        )
        repo.save(r1)
        repo.save(r2)

        records = repo.list()
        assert len(records) >= 2
        assert records[0].incident.id == "inc-list-002"
        assert records[1].incident.id == "inc-list-001"

    def test_get_unknown_returns_none(self):
        repo = PostgresInvestigationRepository(os.environ["DATABASE_URL"])
        assert repo.get("inc-nonexistent-001") is None

    def test_on_conflict_upsert(self):
        repo = PostgresInvestigationRepository(os.environ["DATABASE_URL"])
        record = InvestigationResponse(
            incident=Incident(id="inc-upsert-001", service="svc", symptom="s", started_at="2026-09-11T10:00:00Z"),
            hypotheses=[], verification=None, final_result="verified", retry_count=0, evidence=[],
        )
        repo.save(record)

        record.final_result = "investigation-unverified"
        record.retry_count = 3
        repo.save(record)

        got = repo.get("inc-upsert-001")
        assert got is not None
        assert got.final_result == "investigation-unverified"
        assert got.retry_count == 3

    def test_nested_jsonb_roundtrip(self):
        repo = PostgresInvestigationRepository(os.environ["DATABASE_URL"])
        record = InvestigationResponse(
            incident=Incident(
                id="inc-nested-001",
                service="payments",
                symptom="Payment timeouts",
                started_at="2026-09-11T11:00:00Z",
            ),
            hypotheses=[
                Hypothesis(
                    statement="deployment related",
                    observed_facts=["fact 1", "fact 2"],
                    supporting_evidence=[0],
                    contradicting_evidence=[],
                    reasoning="deploy caused it",
                    confidence=Confidence.high,
                )
            ],
            verification=VerificationResult(status="PASS", rationale="enough evidence"),
            final_result="verified",
            retry_count=3,
            evidence=[
                ToolResult(
                    task=MagicMock(
                        tool="get_metrics",
                        service="payments",
                        metric="p95_latency_ms",
                        start="2026-09-11T10:30:00Z",
                        end="2026-09-11T11:00:00Z",
                    ),
                    metrics=[],
                    logs=[],
                    deployments=[],
                )
            ],
        )
        repo.save(record)

        got = repo.get("inc-nested-001")
        assert got is not None
        assert len(got.hypotheses) == 1
        assert got.hypotheses[0].statement == "deployment related"
        assert got.verification is not None
        assert got.verification.status == "PASS"
        assert len(got.evidence) == 1
        assert got.final_result == "verified"
        assert got.retry_count == 3