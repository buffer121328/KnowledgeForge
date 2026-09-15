"""Explicit FastAPI composition for QA HTTP collaborators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service
from infrastructure.webhooks.service import WebhookEvent, get_webhook_service


@dataclass(frozen=True, slots=True)
class QARouteDependencies:
    """Named runtime collaborators used by the QA HTTP boundary."""

    workflow: Any = None
    history: Any = None
    cache: Any = None
    knowledge_revision: Any = None
    semantic_confirmation: Any = None
    audit_service_factory: Callable[[], Any] = get_audit_service
    webhook_service_factory: Callable[[], Any] = get_webhook_service

    def audit_service(self) -> Any:
        """Return the configured audit service without eager route assembly."""
        return self.audit_service_factory()

    def webhook_service(self) -> Any:
        """Return the configured Webhook service without eager route assembly."""
        return self.webhook_service_factory()


def get_qa_route_dependencies(request: Request) -> QARouteDependencies:
    """Compose QA HTTP collaborators from application-owned runtime state."""

    state = request.app.state
    workflows = getattr(state, "workflows", {})
    return QARouteDependencies(
        workflow=workflows.get("qa") if workflows is not None else None,
        history=getattr(state, "qa_history", None),
        cache=getattr(state, "qa_cache", None),
        knowledge_revision=getattr(state, "knowledge_revision", None),
        semantic_confirmation=getattr(state, "semantic_confirmation", None),
    )


__all__ = [
    "AuditAction",
    "AuditResult",
    "QARouteDependencies",
    "WebhookEvent",
    "get_audit_service",
    "get_qa_route_dependencies",
    "get_webhook_service",
]
