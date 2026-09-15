"""Webhook persistence boundaries and JSON serialization."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from sqlalchemy import delete, insert, select, update

from infrastructure.postgres.database import DatabaseConflictError, DatabaseService
from infrastructure.postgres.models import webhook_delivery_attempts, webhooks
from infrastructure.webhooks.models import Webhook, WebhookEvent
from infrastructure.security.security_state import CorruptSecurityStateError, SecurityStateBackend
from shared.config import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class WebhookSecretProtectionError(RuntimeError):
    """Signal that durable webhook secrets cannot be safely protected."""


class WebhookSecretProtector(Protocol):
    """Protect and reveal webhook signing secrets with an approved mechanism."""

    @property
    def key_reference(self) -> str: ...

    def protect(self, value: str) -> bytes: ...

    def reveal(self, value: bytes) -> str: ...


class FernetWebhookSecretProtector:
    """Encrypt webhook secrets using a configured Fernet key reference."""

    def __init__(self, key: str, *, key_reference: str) -> None:
        if not key or not key_reference:
            raise WebhookSecretProtectionError("webhook secret protection is required")
        try:
            from cryptography.fernet import Fernet

            self._fernet = Fernet(key.encode("ascii"))
        except Exception as error:
            raise WebhookSecretProtectionError("invalid webhook secret protection configuration") from error
        self._key_reference = key_reference

    @property
    def key_reference(self) -> str:
        return self._key_reference

    def protect(self, value: str) -> bytes:
        return self._fernet.encrypt(value.encode("utf-8"))

    def reveal(self, value: bytes) -> str:
        try:
            return self._fernet.decrypt(value).decode("utf-8")
        except Exception as error:
            raise WebhookSecretProtectionError("webhook secret cannot be decrypted") from error


class WebhookStore(Protocol):
    """Define the contract for webhook store operations."""
    def save(self, webhook: Webhook) -> None:
        """Persist a record through the webhook store."""
        ...
    def get(self, webhook_id: str, org_id: str | None = None) -> Webhook | None:
        """Return the requested value from the webhook store."""
        ...
    def list_all(self, org_id: str | None = None) -> list[Webhook]:
        """List the all."""
        ...
    def list_by_event(self, event: WebhookEvent, org_id: str | None = None) -> list[Webhook]:
        """List the by event."""
        ...
    def delete(self, webhook_id: str, org_id: str | None = None) -> bool:
        """Delete a record through the webhook store."""
        ...
    def update(self, webhook: Webhook) -> None:
        """Update a record through the webhook store."""
        ...


class MemoryWebhookStore:
    """Persist and retrieve memory webhook data."""
    def __init__(self) -> None:
        """Initialize the memory webhook store."""
        self._store: dict[str, Webhook] = {}

    def save(self, webhook: Webhook) -> None:
        """Persist a record through the memory webhook store."""
        self._store[webhook.id] = webhook

    def get(self, webhook_id: str, org_id: str | None = None) -> Webhook | None:
        """Return the requested value from the memory webhook store."""
        webhook = self._store.get(webhook_id)
        if webhook and (org_id is None or webhook.org_id == org_id):
            return webhook
        return None

    def get_any(self, webhook_id: str) -> Webhook | None:
        """Return the any."""
        return self._store.get(webhook_id)

    def list_all(self, org_id: str | None = None) -> list[Webhook]:
        """List the all."""
        return [
            webhook
            for webhook in self._store.values()
            if org_id is None or webhook.org_id == org_id
        ]

    def list_by_event(self, event: WebhookEvent, org_id: str | None = None) -> list[Webhook]:
        """List the by event."""
        return [
            webhook
            for webhook in self._store.values()
            if webhook.is_active
            and event in webhook.events
            and (org_id is None or webhook.org_id == org_id)
        ]

    def delete(self, webhook_id: str, org_id: str | None = None) -> bool:
        """Delete a record through the memory webhook store."""
        if not self.get(webhook_id, org_id):
            return False
        return self._store.pop(webhook_id, None) is not None

    def update(self, webhook: Webhook) -> None:
        """Update a record through the memory webhook store."""
        self._store[webhook.id] = webhook


class FileWebhookStore:
    """JSON persistence that retains the existing on-disk Webhook shape."""

    def __init__(self, path: Path | None = None) -> None:
        """Initialize the file webhook store."""
        base = Path(settings.upload_dir).resolve().parent / "logs"
        self._path = path or (base / "webhooks.json")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._store: dict[str, Webhook] = {}
        self._load()

    def _load(self) -> None:
        """Load the file webhook store."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else raw.get("webhooks", [])
            for item in items:
                webhook = Webhook.from_dict(item)
                self._store[webhook.id] = webhook
        except Exception as error:
            logger.warning(
                "webhook_store_load_failed",
                error_type=type(error).__name__,
            )

    def _persist(self) -> None:
        """Persist the file webhook store."""
        payload = [webhook.to_dict() for webhook in self._store.values()]
        temp_path = self._path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self._path)

    def save(self, webhook: Webhook) -> None:
        """Persist a record through the file webhook store."""
        with self._lock:
            self._store[webhook.id] = webhook
            self._persist()

    def get(self, webhook_id: str, org_id: str | None = None) -> Webhook | None:
        """Return the requested value from the file webhook store."""
        webhook = self._store.get(webhook_id)
        if webhook and (org_id is None or webhook.org_id == org_id):
            return webhook
        return None

    def get_any(self, webhook_id: str) -> Webhook | None:
        """Return the any."""
        return self._store.get(webhook_id)

    def list_all(self, org_id: str | None = None) -> list[Webhook]:
        """List the all."""
        return [
            webhook
            for webhook in self._store.values()
            if org_id is None or webhook.org_id == org_id
        ]

    def list_by_event(self, event: WebhookEvent, org_id: str | None = None) -> list[Webhook]:
        """List the by event."""
        return [
            webhook
            for webhook in self._store.values()
            if webhook.is_active
            and event in webhook.events
            and (org_id is None or webhook.org_id == org_id)
        ]

    def delete(self, webhook_id: str, org_id: str | None = None) -> bool:
        """Delete a record through the file webhook store."""
        with self._lock:
            if not self.get(webhook_id, org_id):
                return False
            removed = self._store.pop(webhook_id, None) is not None
            if removed:
                self._persist()
            return removed

    def update(self, webhook: Webhook) -> None:
        """Update a record through the file webhook store."""
        with self._lock:
            self._store[webhook.id] = webhook
            self._persist()


def _webhook_envelope(webhook: Webhook) -> dict:
    """Return the webhook envelope."""
    return {
        "version": 1,
        "kind": "webhook",
        "data": webhook.to_dict(),
    }


def _load_webhook(value: dict) -> Webhook:
    """Load the webhook."""
    if value.get("version") != 1 or value.get("kind") != "webhook":
        raise CorruptSecurityStateError()
    data = value.get("data")
    if not isinstance(data, dict):
        raise CorruptSecurityStateError()
    try:
        return Webhook.from_dict(data)
    except (KeyError, TypeError, ValueError) as error:
        raise CorruptSecurityStateError() from error


class RedisWebhookStore:
    """Shared Webhook store with organization/event indexes."""

    _PREFIX = "security:v1:webhook:"
    _LEGACY_PREFIX = "security:v1:webhook-legacy:"

    def __init__(self, backend: SecurityStateBackend) -> None:
        """Initialize the Redis webhook store."""
        self.backend = backend

    @classmethod
    def _key(cls, webhook_id: str) -> str:
        """Build the Redis key for a webhook record."""
        return f"{cls._PREFIX}{webhook_id}"

    @classmethod
    def _legacy_key(cls, webhook_id: str) -> str:
        """Build the legacy Redis key used during migration."""
        return f"{cls._LEGACY_PREFIX}{webhook_id}"

    @staticmethod
    def _org_index(org_id: str) -> str:
        """Build the Redis webhook index for an organization."""
        return f"security:v1:webhook-org:{org_id}"

    @staticmethod
    def _event_index(org_id: str, event: WebhookEvent) -> str:
        """Build the Redis webhook index for an event type."""
        return f"security:v1:webhook-event:{org_id}:{event.value}"

    @classmethod
    def _indexes(cls, webhook: Webhook) -> tuple[str, ...]:
        """Return all Redis indexes associated with a webhook."""
        if not webhook.org_id:
            return ()
        return (
            cls._org_index(webhook.org_id),
            *(
                cls._event_index(webhook.org_id, event)
                for event in webhook.events
            ),
        )

    def save(self, webhook: Webhook) -> None:
        """Persist a record through the Redis webhook store."""
        if not webhook.org_id:
            raise CorruptSecurityStateError()
        key = self._key(webhook.id)
        existing = self.backend.read(key)
        remove_indexes: tuple[str, ...] = ()
        if existing is not None:
            remove_indexes = self._indexes(_load_webhook(existing))
        self.backend.write(
            key,
            _webhook_envelope(webhook),
            add_indexes=self._indexes(webhook),
            remove_indexes=remove_indexes,
        )

    def get(
        self,
        webhook_id: str,
        org_id: str | None = None,
    ) -> Webhook | None:
        """Return the requested value from the Redis webhook store."""
        if not org_id:
            return None
        value = self.backend.read(self._key(webhook_id))
        if value is None:
            return None
        webhook = _load_webhook(value)
        return webhook if webhook.org_id == org_id else None

    def get_any(self, webhook_id: str) -> Webhook | None:
        """Return a record for migration conflict checks, without tenant filtering."""
        value = self.backend.read(self._key(webhook_id))
        return None if value is None else _load_webhook(value)

    def list_all(self, org_id: str | None = None) -> list[Webhook]:
        """List the all."""
        if not org_id:
            return []
        records: list[Webhook] = []
        for key in self.backend.members(self._org_index(org_id)):
            value = self.backend.read(key)
            if value is not None:
                webhook = _load_webhook(value)
                if webhook.org_id == org_id:
                    records.append(webhook)
        return sorted(records, key=lambda item: item.id)

    def list_by_event(
        self,
        event: WebhookEvent,
        org_id: str | None = None,
    ) -> list[Webhook]:
        """List the by event."""
        if not org_id:
            return []
        records: list[Webhook] = []
        for key in self.backend.members(self._event_index(org_id, event)):
            value = self.backend.read(key)
            if value is None:
                continue
            webhook = _load_webhook(value)
            if (
                webhook.org_id == org_id
                and webhook.is_active
                and event in webhook.events
            ):
                records.append(webhook)
        return sorted(records, key=lambda item: item.id)

    def delete(
        self,
        webhook_id: str,
        org_id: str | None = None,
    ) -> bool:
        """Delete a record through the Redis webhook store."""
        webhook = self.get(webhook_id, org_id)
        if webhook is None:
            return False
        return self.backend.delete(
            self._key(webhook_id),
            remove_indexes=self._indexes(webhook),
        )

    def update(self, webhook: Webhook) -> None:
        """Update a record through the Redis webhook store."""
        self.save(webhook)

    def import_legacy_unowned(self, webhook: Webhook) -> None:
        """Test/migration staging only; never adds a delivery index."""
        if webhook.org_id:
            raise ValueError("legacy record must be unowned")
        self.backend.write(
            self._legacy_key(webhook.id),
            _webhook_envelope(webhook),
        )

    def legacy_inventory(self) -> list[dict[str, str]]:
        """Handle legacy inventory for the Redis webhook store."""
        return [
            {"webhook_id": key.removeprefix(self._LEGACY_PREFIX)}
            for key in self.backend.keys(self._LEGACY_PREFIX)
        ]


class PostgreSQLWebhookStore:
    """Persist tenant-scoped webhook facts with encrypted signing secrets."""

    def __init__(
        self,
        database: DatabaseService,
        protector: WebhookSecretProtector,
    ) -> None:
        self.database = database
        self.protector = protector

    @staticmethod
    def _timestamp(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    def _values(self, webhook: Webhook) -> dict:
        if not webhook.org_id:
            raise ValueError("PostgreSQL webhook requires tenant ownership")
        return {
            "id": webhook.id,
            "tenant_id": webhook.org_id,
            "url": webhook.url,
            "events": [event.value for event in webhook.events],
            "secret_ciphertext": self.protector.protect(webhook.secret),
            "secret_key_ref": self.protector.key_reference,
            "is_active": webhook.is_active,
            "failure_count": webhook.failure_count,
            "max_failures": webhook.max_failures,
            "last_triggered_at": self._timestamp(webhook.last_triggered_at),
            "last_response_category": (
                f"{webhook.last_response_code // 100}xx"
                if webhook.last_response_code is not None
                else None
            ),
            "created_at": self._timestamp(webhook.created_at) or datetime.now(UTC),
            "metadata_json": webhook.metadata,
        }

    def _from_row(self, row) -> Webhook:
        values = dict(row)
        ciphertext = values.get("secret_ciphertext")
        if not ciphertext or values.get("secret_key_ref") != self.protector.key_reference:
            raise WebhookSecretProtectionError("webhook secret key reference is unavailable")
        created_at = values["created_at"]
        last_triggered_at = values.get("last_triggered_at")
        return Webhook(
            id=values["id"],
            org_id=values["tenant_id"],
            url=values["url"],
            events=[WebhookEvent(item) for item in (values.get("events") or [])],
            secret=self.protector.reveal(bytes(ciphertext)),
            is_active=bool(values["is_active"]),
            created_at=created_at.isoformat(),
            failure_count=int(values["failure_count"]),
            max_failures=int(values["max_failures"]),
            last_triggered_at=last_triggered_at.isoformat() if last_triggered_at else None,
            last_response_code=None,
            metadata=values.get("metadata_json") or {},
        )

    def save(self, webhook: Webhook) -> None:
        try:
            with self.database.session() as session:
                session.execute(insert(webhooks).values(**self._values(webhook)))
        except DatabaseConflictError as error:
            raise ValueError("webhook persistence conflict") from error

    def get(self, webhook_id: str, org_id: str | None = None) -> Webhook | None:
        if not org_id:
            return None
        with self.database.session() as session:
            row = session.execute(
                select(webhooks).where(
                    webhooks.c.id == webhook_id,
                    webhooks.c.tenant_id == org_id,
                )
            ).mappings().one_or_none()
        return self._from_row(row) if row else None

    def get_any(self, webhook_id: str) -> Webhook | None:
        with self.database.session() as session:
            row = session.execute(
                select(webhooks).where(webhooks.c.id == webhook_id)
            ).mappings().one_or_none()
        return self._from_row(row) if row else None

    def list_all(self, org_id: str | None = None) -> list[Webhook]:
        if not org_id:
            return []
        with self.database.session() as session:
            rows = session.execute(
                select(webhooks)
                .where(webhooks.c.tenant_id == org_id)
                .order_by(webhooks.c.id)
            ).mappings().all()
        return [self._from_row(row) for row in rows]

    def list_by_event(self, event: WebhookEvent, org_id: str | None = None) -> list[Webhook]:
        return [
            item
            for item in self.list_all(org_id)
            if item.is_active and event in item.events
        ]

    def delete(self, webhook_id: str, org_id: str | None = None) -> bool:
        if not org_id:
            return False
        with self.database.session() as session:
            result = session.execute(
                delete(webhooks).where(
                    webhooks.c.id == webhook_id,
                    webhooks.c.tenant_id == org_id,
                )
            )
        return result.rowcount == 1

    def update(self, webhook: Webhook) -> None:
        values = self._values(webhook)
        values.pop("id")
        values.pop("tenant_id")
        with self.database.session() as session:
            result = session.execute(
                update(webhooks)
                .where(
                    webhooks.c.id == webhook.id,
                    webhooks.c.tenant_id == webhook.org_id,
                )
                .values(**values)
            )
            if result.rowcount != 1:
                raise LookupError("webhook not found")

    def record_delivery_attempt(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        *,
        attempt: int,
        status_category: str,
        final_status: str,
    ) -> None:
        """Persist bounded delivery facts without payload or response body."""
        with self.database.session() as session:
            session.execute(
                insert(webhook_delivery_attempts).values(
                    id=str(uuid4()),
                    tenant_id=webhook.org_id,
                    webhook_id=webhook.id,
                    event=event.value,
                    attempt=max(1, attempt),
                    status_category=status_category[:64],
                    final_status=final_status[:64],
                    created_at=datetime.now(UTC),
                )
            )


__all__ = [
    "FernetWebhookSecretProtector",
    "FileWebhookStore",
    "MemoryWebhookStore",
    "PostgreSQLWebhookStore",
    "RedisWebhookStore",
    "WebhookSecretProtectionError",
    "WebhookSecretProtector",
    "WebhookStore",
]
