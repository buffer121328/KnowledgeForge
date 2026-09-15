"""Tenant-scoped domain records for durable question-answering history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class QAFeedbackRating(str, Enum):
    """Represent the bounded user feedback values accepted by the API."""

    UP = "up"
    DOWN = "down"
    ISSUE = "issue"


@dataclass(frozen=True)
class QAConversation:
    """Represent one durable user-owned QA conversation."""

    id: str
    tenant_id: str
    user_id: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class QAMessage:
    """Represent one ordered user-visible conversation message."""

    id: str
    sequence: int
    role: str
    content: str
    created_at: datetime
    qa_run_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class QAConversationPage:
    """Represent one bounded cursor page of conversations."""

    items: list[QAConversation]
    next_cursor: str | None


__all__ = ["QAConversation", "QAConversationPage", "QAFeedbackRating", "QAMessage"]
