"""Read-only deterministic tools over synthetic operational JSON fixtures.

These helpers only filter and return observed facts. They intentionally do no
reasoning, anomaly detection, scoring, or root-cause analysis.
"""

from datetime import datetime
from functools import lru_cache
from json import loads
from pathlib import Path

from incident_agent.models import Deployment, LogEntry, MetricPoint

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _read_json(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    return loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _metric_points() -> list[MetricPoint]:
    return [MetricPoint.model_validate(item) for item in _read_json("metrics.json")]


@lru_cache(maxsize=1)
def _log_entries() -> list[LogEntry]:
    return [LogEntry.model_validate(item) for item in _read_json("logs.json")]


@lru_cache(maxsize=1)
def _deployments() -> list[Deployment]:
    return [Deployment.model_validate(item) for item in _read_json("deployments.json")]


def get_metrics(service: str, metric: str, start: datetime, end: datetime) -> list[MetricPoint]:
    """Return metric points for one service and metric inside [start, end]."""
    if start > end:
        raise ValueError("start must be less than or equal to end")

    selected = [
        point
        for point in _metric_points()
        if point.service == service and point.metric == metric and start <= point.timestamp <= end
    ]
    selected.sort(key=lambda item: item.timestamp)
    return selected


def search_logs(
    service: str,
    start: datetime,
    end: datetime,
    keyword: str | None = None,
    level: str | None = None,
    limit: int = 50,
) -> list[LogEntry]:
    """Return log entries for one service inside [start, end], newest last."""
    if start > end:
        raise ValueError("start must be less than or equal to end")
    if limit < 0:
        raise ValueError("limit must be greater than or equal to zero")

    wanted_keyword = keyword.lower() if keyword is not None else None
    selected = []
    for entry in _log_entries():
        if entry.service != service:
            continue
        if not start <= entry.timestamp <= end:
            continue
        if level is not None and entry.level != level:
            continue
        if wanted_keyword is not None and wanted_keyword not in entry.message.lower():
            continue
        selected.append(entry)

    selected.sort(key=lambda item: item.timestamp)
    return selected[:limit]


def get_recent_deployments(
    service: str | None = None,
    since: datetime | None = None,
    limit: int = 10,
) -> list[Deployment]:
    """Return deployments, newest first, optionally filtered by service/since."""
    if limit < 0:
        raise ValueError("limit must be greater than or equal to zero")

    selected = [
        deployment
        for deployment in _deployments()
        if (service is None or deployment.service == service)
        and (since is None or deployment.timestamp >= since)
    ]
    selected.sort(key=lambda item: item.timestamp, reverse=True)
    return selected[:limit]
