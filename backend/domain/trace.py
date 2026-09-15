"""Bounded, request-local execution traces for the QA agent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable
from uuid import uuid4

TRACE_SCHEMA_VERSION = "v1"

# These are real application workflow boundaries, not autonomous LLM tool calls.
TRACE_OPERATIONS = frozenset(
    {
        "safety.validate_question",
        "cache.lookup",
        "query.classify",
        "query.rewrite",
        "retrieval.vector",
        "retrieval.bm25",
        "retrieval.graph",
        "safety.filter_contexts",
        "ranking.rerank",
        "qa.rerank_completed",
        "qa.evidence_qualified",
        "qa.structured_generated",
        "qa.grounding_verified",
        "generation.answer",
        "safety.sanitize_answer",
        "qa.completed",
        "qa.failed",
    }
)

_ALLOWED_METADATA_KEYS = frozenset(
    {
        "retrieval_mode",
        "contexts_count",
        "available",
        "unavailable",
        "status",
        "vector_count",
        "bm25_count",
        "graph_count",
        "intent",
        "cache_hit",
        "degradation_code",
        "security_action_count",
        "status_code",
        "reason_code",
        "result_count",
        "mode",
        "scored_count",
        "evaluated_count",
        "accepted_count",
        "rejected_count",
        "evidence_state",
        "response_status",
        "policy_version",
        "calibration_version",
        "schema_version",
        "grounding_passed",
    }
)
_MAX_STRING_LENGTH = 96


def _safe_scalar(value: Any) -> str | int | float | bool | None:
    """Return a bounded scalar suitable for a trace artifact."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or len(normalized) > _MAX_STRING_LENGTH:
            return normalized[:_MAX_STRING_LENGTH] if normalized else None
        return normalized
    return None


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, str | int | float | bool | None]:
    """Keep only the explicit, bounded metadata vocabulary."""
    if not metadata:
        return {}
    sanitized: dict[str, str | int | float | bool | None] = {}
    for key, value in metadata.items():
        if key not in _ALLOWED_METADATA_KEYS:
            continue
        safe_value = _safe_scalar(value)
        if safe_value is not None:
            sanitized[key] = safe_value
    return sanitized


@dataclass(frozen=True)
class TraceEvent:
    """One completed workflow operation."""

    event_id: str
    operation: str
    status: str
    timestamp: str
    duration_ms: float
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    exception_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event without unsafe exception details."""
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "operation": self.operation,
            "status": self.status,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }
        if self.exception_type:
            payload["exception_type"] = self.exception_type
        return payload


@dataclass(frozen=True)
class AgentTrace:
    """Request-local structured execution evidence."""

    trace_id: str
    events: list[TraceEvent] = field(default_factory=list)
    schema_version: str = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace as JSON-safe data."""
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "events": [event.to_dict() for event in self.events],
        }


class TraceRecorder:
    """Record bounded workflow events for one request."""

    def __init__(self, *, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or f"trace_{uuid4().hex}"
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> list[TraceEvent]:
        """Return a snapshot of recorded events."""
        return list(self._events)

    def record(
        self,
        operation: str,
        *,
        status: str = "succeeded",
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
        exception_type: str | None = None,
    ) -> TraceEvent:
        """Record a completed operation using only safe bounded fields."""
        if operation not in TRACE_OPERATIONS:
            raise ValueError(f"unknown trace operation: {operation}")
        if status not in {"succeeded", "failed", "skipped"}:
            raise ValueError(f"unsupported trace status: {status}")
        try:
            bounded_duration = max(0.0, float(duration_ms))
        except (TypeError, ValueError):
            bounded_duration = 0.0
        event = TraceEvent(
            event_id=f"event_{uuid4().hex}",
            operation=operation,
            status=status,
            timestamp=datetime.now(UTC).isoformat(),
            duration_ms=round(bounded_duration, 3),
            metadata=_safe_metadata(metadata),
            exception_type=(exception_type or "").split(".")[-1] or None,
        )
        self._events.append(event)
        return event

    def capture_sync(
        self,
        operation: str,
        callback: Callable[[], Any],
        *,
        metadata: dict[str, Any] | Callable[[Any], dict[str, Any]] | None = None,
    ) -> Any:
        """Run and record a synchronous operation."""
        started = time.perf_counter()
        try:
            value = callback()
        except Exception as error:
            self.record(
                operation,
                status="failed",
                duration_ms=(time.perf_counter() - started) * 1000,
                metadata=metadata if isinstance(metadata, dict) else None,
                exception_type=type(error).__name__,
            )
            raise
        self.record(
            operation,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata=metadata(value) if callable(metadata) else metadata,
        )
        return value

    async def capture_async(
        self,
        operation: str,
        callback: Callable[[], Awaitable[Any]],
        *,
        metadata: dict[str, Any] | Callable[[Any], dict[str, Any]] | None = None,
    ) -> Any:
        """Run and record an asynchronous operation."""
        started = time.perf_counter()
        try:
            value = await callback()
        except Exception as error:
            self.record(
                operation,
                status="failed",
                duration_ms=(time.perf_counter() - started) * 1000,
                metadata=metadata if isinstance(metadata, dict) else None,
                exception_type=type(error).__name__,
            )
            raise
        self.record(
            operation,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata=metadata(value) if callable(metadata) else metadata,
        )
        return value

    def build(self) -> AgentTrace:
        """Build an immutable trace snapshot."""
        return AgentTrace(trace_id=self.trace_id, events=list(self._events))


__all__ = [
    "AgentTrace",
    "TRACE_OPERATIONS",
    "TRACE_SCHEMA_VERSION",
    "TraceEvent",
    "TraceRecorder",
]
