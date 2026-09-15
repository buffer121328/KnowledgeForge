"""Bounded, secret-safe runtime dependency readiness probes."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from infrastructure.celery_app import celery_app
from shared.utils.logging import get_logger

logger = get_logger(__name__)

def _probe_task_broker(timeout_seconds: float) -> bool:
    """Open and close one bounded broker connection without inspecting workers."""
    with celery_app.connection_for_read() as connection:
        connection.ensure_connection(max_retries=0, timeout=timeout_seconds)
    return True


class ReadinessService:
    """Check required API dependencies without leaking connection details."""

    def __init__(
        self,
        *,
        vector_store: Any,
        knowledge_graph: Any,
        security_state: Any,
        database: Any | None = None,
        timeout_seconds: float,
        broker_probe: Callable[[], Any] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.security_state = security_state
        self.database = database
        self.timeout_seconds = timeout_seconds
        self.broker_probe = broker_probe or (
            lambda: _probe_task_broker(timeout_seconds)
        )

    async def _bounded(self, component: str, callback: Callable[[], Any]) -> str:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                if inspect.iscoroutinefunction(callback):
                    value = await callback()
                else:
                    value = await asyncio.to_thread(callback)
                    if inspect.isawaitable(value):
                        value = await value
            if value is False:
                raise RuntimeError("probe returned unavailable")
            return "ready"
        except Exception as error:
            logger.warning(
                "readiness_probe_failed",
                component=component,
                error_type=type(error).__name__,
            )
            return "unavailable"

    async def check(self) -> dict[str, Any]:
        """Return stable component states and an aggregate readiness decision."""
        probes: list[tuple[str, Callable[[], Any]]] = [
            ("vector_store", self.vector_store.health_check),
            ("knowledge_graph", self.knowledge_graph.health_check),
            ("security_state", self.security_state.ping),
            ("task_broker", self.broker_probe),
        ]
        if self.database is not None:
            probes.append(("postgresql", self.database.ping))
        results = await asyncio.gather(
            *(self._bounded(component, callback) for component, callback in probes)
        )
        states = dict(zip((component for component, _ in probes), results, strict=True))
        ready = all(state == "ready" for state in states.values())
        return {
            "status": "ready" if ready else "not_ready",
            "components": states,
        }
