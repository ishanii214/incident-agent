"""In-process persistence for investigation results.

Phase 8 storage: a dict-backed repository keeping terminal investigation
results addressable by incident id. The three-method contract (save/get/list)
is the seam a future PostgreSQL implementation will satisfy without changing
the API contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only back-reference; avoids a runtime import cycle
    from incident_agent.app import InvestigationResponse


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
