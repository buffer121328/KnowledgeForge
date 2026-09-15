"""Internal Webhook data contracts re-exported by the public facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WebhookEvent(str, Enum):
    """Represent a webhook event."""
    DOC_INGESTED = "doc.ingested"
    DOC_UPDATED = "doc.updated"
    DOC_DELETED = "doc.deleted"
    QA_COMPLETED = "qa.completed"
    QA_FEEDBACK = "qa.feedback"
    SYSTEM_ERROR = "system.error"


@dataclass
class Webhook:
    """Represent webhook."""
    id: str
    url: str
    events: list[WebhookEvent]
    org_id: str = ""
    secret: str = ""
    is_active: bool = True
    created_at: str = ""
    failure_count: int = 0
    max_failures: int = 5
    last_triggered_at: str | None = None
    last_response_code: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the webhook to dict."""
        return {
            "id": self.id,
            "org_id": self.org_id,
            "url": self.url,
            "events": [event.value for event in self.events],
            "secret": self.secret,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "failure_count": self.failure_count,
            "max_failures": self.max_failures,
            "last_triggered_at": self.last_triggered_at,
            "last_response_code": self.last_response_code,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Webhook":
        """Create the webhook from dict."""
        events: list[WebhookEvent] = []
        for event in data.get("events") or []:
            try:
                events.append(WebhookEvent(event))
            except ValueError:
                continue
        return cls(
            id=data["id"],
            org_id=str(data.get("org_id") or ""),
            url=data["url"],
            events=events,
            secret=data.get("secret") or "",
            is_active=bool(data.get("is_active", True)),
            created_at=data.get("created_at") or "",
            failure_count=int(data.get("failure_count") or 0),
            max_failures=int(data.get("max_failures") or 5),
            last_triggered_at=data.get("last_triggered_at"),
            last_response_code=data.get("last_response_code"),
            metadata=data.get("metadata") or {},
        )
