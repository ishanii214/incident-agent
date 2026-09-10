"""Phase 1 checks for deterministic synthetic operational tools."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from incident_agent.models import Deployment, LogEntry, MetricPoint
from incident_agent.tools import get_metrics, get_recent_deployments, search_logs

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_metrics_filter_by_service_metric_and_window() -> None:
    points = get_metrics(
        service="checkout",
        metric="p95_latency_ms",
        start=_utc("2026-09-10T13:50:00Z"),
        end=_utc("2026-09-10T14:01:00Z"),
    )

    assert points
    assert all(isinstance(point, MetricPoint) for point in points)
    assert all(point.service == "checkout" and point.metric == "p95_latency_ms" for point in points)


def test_metrics_show_baseline_before_degraded_after() -> None:
    before = get_metrics(
        service="checkout",
        metric="p95_latency_ms",
        start=_utc("2026-09-10T13:50:00Z"),
        end=_utc("2026-09-10T14:01:00Z"),
    )
    after = get_metrics(
        service="checkout",
        metric="p95_latency_ms",
        start=_utc("2026-09-10T14:03:00Z"),
        end=_utc("2026-09-10T14:15:00Z"),
    )

    assert max(point.value for point in before) < min(point.value for point in after)


def test_logs_filter_by_service_window_keyword_and_level() -> None:
    entries = search_logs(
        service="checkout",
        start=_utc("2026-09-10T13:50:00Z"),
        end=_utc("2026-09-10T14:15:00Z"),
        keyword="timeout",
        level="ERROR",
    )

    assert entries
    assert all(isinstance(entry, LogEntry) for entry in entries)
    assert all(entry.level == "ERROR" and "timeout" in entry.message.lower() for entry in entries)


def test_timeout_evidence_only_after_deployment() -> None:
    before = search_logs(
        service="checkout",
        start=_utc("2026-09-10T13:50:00Z"),
        end=_utc("2026-09-10T14:01:00Z"),
        keyword="timeout",
        level="ERROR",
    )
    after = search_logs(
        service="checkout",
        start=_utc("2026-09-10T14:02:00Z"),
        end=_utc("2026-09-10T14:15:00Z"),
        keyword="timeout",
        level="ERROR",
    )

    assert before == []
    assert len(after) >= 3


def test_deployments_filtered_and_newest_first() -> None:
    deployments = get_recent_deployments(service="checkout")

    assert deployments
    assert all(isinstance(deployment, Deployment) for deployment in deployments)
    assert all(deployment.service == "checkout" for deployment in deployments)
    assert deployments[0].version == "v2.4.1"
    timestamps = [deployment.timestamp for deployment in deployments]
    assert timestamps == sorted(timestamps, reverse=True)


def test_empty_queries_return_empty_lists() -> None:
    assert (
        get_metrics(
            service="checkout",
            metric="p95_latency_ms",
            start=_utc("2026-09-10T12:00:00Z"),
            end=_utc("2026-09-10T12:30:00Z"),
        )
        == []
    )
    assert (
        search_logs(
            service="unknown-service",
            start=_utc("2026-09-10T13:50:00Z"),
            end=_utc("2026-09-10T14:15:00Z"),
        )
        == []
    )
    assert get_recent_deployments(service="unknown-service") == []


def test_metric_window_start_after_end_raises() -> None:
    with pytest.raises(ValueError):
        get_metrics(
            service="checkout",
            metric="p95_latency_ms",
            start=_utc("2026-09-10T14:15:00Z"),
            end=_utc("2026-09-10T13:50:00Z"),
        )


def test_results_have_ascending_timestamps() -> None:
    points = get_metrics(
        service="checkout",
        metric="p95_latency_ms",
        start=_utc("2026-09-10T13:50:00Z"),
        end=_utc("2026-09-10T14:15:00Z"),
    )
    entries = search_logs(
        service="checkout",
        start=_utc("2026-09-10T13:50:00Z"),
        end=_utc("2026-09-10T14:15:00Z"),
    )

    assert [point.timestamp for point in points] == sorted(point.timestamp for point in points)
    assert [entry.timestamp for entry in entries] == sorted(entry.timestamp for entry in entries)


def test_deployment_precedes_metric_and_log_evidence() -> None:
    deployment = get_recent_deployments(service="checkout")[0]
    degraded = get_metrics(
        service="checkout",
        metric="p95_latency_ms",
        start=_utc("2026-09-10T14:03:00Z"),
        end=_utc("2026-09-10T14:15:00Z"),
    )
    timeouts = search_logs(
        service="checkout",
        start=_utc("2026-09-10T14:02:00Z"),
        end=_utc("2026-09-10T14:15:00Z"),
        keyword="timeout",
        level="ERROR",
    )

    assert deployment.timestamp == _utc("2026-09-10T14:02:00Z")
    assert deployment.timestamp < degraded[0].timestamp
    assert deployment.timestamp < timeouts[0].timestamp
    assert timeouts[0].timestamp.replace(tzinfo=timezone.utc) >= _utc("2026-09-10T14:02:00Z")


def test_no_answer_fields_leaked() -> None:
    forbidden = {"root_cause", "rootcause", "culprit", "answer", "diagnosis", "diagnose"}

    for filename in ("metrics.json", "logs.json", "deployments.json"):
        payload = json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        for record in payload:
            assert forbidden.isdisjoint({key.lower() for key in record})

    metrics = get_metrics(
        service="checkout",
        metric="p95_latency_ms",
        start=_utc("2026-09-10T13:50:00Z"),
        end=_utc("2026-09-10T14:15:00Z"),
    )
    logs = search_logs(
        service="checkout",
        start=_utc("2026-09-10T13:50:00Z"),
        end=_utc("2026-09-10T14:15:00Z"),
    )
    deployments = get_recent_deployments()

    for model in (*metrics, *logs, *deployments):
        assert forbidden.isdisjoint(set(model.model_dump()))
