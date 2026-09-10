"""Pydantic schemas for incidents and operational evidence."""

from datetime import datetime

from pydantic import BaseModel, Field


class Incident(BaseModel):
    """Incoming incident report provided by a caller or test."""

    id: str
    service: str
    symptom: str
    started_at: datetime


class MetricPoint(BaseModel):
    """One observed metric sample for a service."""

    timestamp: datetime
    service: str
    metric: str
    value: float


class LogEntry(BaseModel):
    """One observed log line for a service."""

    timestamp: datetime
    service: str
    level: str
    message: str


class Deployment(BaseModel):
    """One observed deployment event for a service."""

    id: str
    service: str
    version: str
    timestamp: datetime
    author: str = "unknown"
    status: str = Field(default="success")
