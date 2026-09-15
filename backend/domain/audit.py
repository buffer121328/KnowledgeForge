"""Neutral audit records, storage protocol, and persistence errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class AuditStoreError(RuntimeError):
    """Base class for controlled audit persistence failures."""


class AuditWriteError(AuditStoreError):
    """Raised when an audit record cannot be durably appended."""


class AuditIntegrityError(AuditStoreError):
    """Raised when persisted audit history fails integrity validation."""


@dataclass
class AuditLog:
    """Represent one mutable append-only audit record."""

    audit_id: str
    timestamp: str
    user_id: str
    action: str = ""
    username: str = ""
    resource: str = ""
    result: str = "success"
    ip: str = ""
    user_agent: str = ""
    org_id: str = ""
    metadata: dict = field(default_factory=dict)
    prev_hash: str = ""
    hash: str = ""


class AuditStore(Protocol):
    """Define persistence operations required by the audit service."""

    def append(self, log: AuditLog) -> None:
        """Append one audit record."""
        ...

    def query(
        self,
        user_id: str | None = None,
        action: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        org_id: str | None = None,
        offset: int = 0,
    ) -> list[AuditLog]:
        """Return matching records in the store-defined stable order."""
        ...


__all__ = [
    "AuditIntegrityError",
    "AuditLog",
    "AuditStore",
    "AuditStoreError",
    "AuditWriteError",
]
