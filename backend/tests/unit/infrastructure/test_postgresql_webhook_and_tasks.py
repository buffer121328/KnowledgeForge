"""PostgreSQL webhook security and task-history contract tests."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine, select

from infrastructure.postgres.database import DatabaseService
from infrastructure.postgres.models import (
    metadata,
    webhook_delivery_attempts,
    webhooks,
)
from infrastructure.security.security_state import MemorySecurityStateBackend
from infrastructure.tasks.task_registry import (
    PersistentTaskRegistry,
    PostgreSQLTaskRunRepository,
    TaskRegistry,
)
from infrastructure.webhooks.models import Webhook, WebhookEvent
from infrastructure.webhooks.stores import PostgreSQLWebhookStore


class ReversingProtector:
    """Deterministic test protector proving plaintext is not persisted."""

    key_reference = "test-key-v1"

    def protect(self, value: str) -> bytes:
        return value[::-1].encode("utf-8")

    def reveal(self, value: bytes) -> str:
        return value.decode("utf-8")[::-1]


def _database() -> DatabaseService:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    return DatabaseService.from_engine(engine)


def test_postgresql_webhook_is_tenant_scoped_and_secret_is_protected() -> None:
    database = _database()
    store = PostgreSQLWebhookStore(database, ReversingProtector())
    webhook = Webhook(
        id="wh-1",
        org_id="org-a",
        url="https://example.com/hook",
        events=[WebhookEvent.DOC_INGESTED],
        secret="plain-secret",
        created_at=datetime.now(UTC).isoformat(),
    )
    store.save(webhook)

    assert store.get("wh-1", "org-a").secret == "plain-secret"
    assert store.get("wh-1", "org-b") is None
    with database.session() as session:
        row = session.execute(select(webhooks)).mappings().one()
    assert bytes(row["secret_ciphertext"]) != b"plain-secret"
    assert row["secret_key_ref"] == "test-key-v1"


def test_postgresql_webhook_records_only_controlled_delivery_facts() -> None:
    database = _database()
    store = PostgreSQLWebhookStore(database, ReversingProtector())
    webhook = Webhook(
        id="wh-1",
        org_id="org-a",
        url="https://example.com/hook",
        events=[WebhookEvent.QA_COMPLETED],
        secret="secret",
        created_at=datetime.now(UTC).isoformat(),
    )
    store.save(webhook)
    store.record_delivery_attempt(
        webhook,
        WebhookEvent.QA_COMPLETED,
        attempt=2,
        status_category="5xx",
        final_status="failed",
    )
    with database.session() as session:
        row = session.execute(select(webhook_delivery_attempts)).mappings().one()
    assert row["tenant_id"] == "org-a"
    assert row["attempt"] == 2
    assert row["status_category"] == "5xx"
    assert "payload" not in row
    assert "response_body" not in row


def test_persistent_task_registry_survives_active_record_expiry_and_scopes_tenant() -> None:
    database = _database()
    active = TaskRegistry(MemorySecurityStateBackend(), ttl_seconds=700_000)
    registry = PersistentTaskRegistry(active, PostgreSQLTaskRunRepository(database))
    registry.reserve(
        "task-1",
        org_id="org-a",
        actor_id="user-a",
        kind="ingest",
        file_reference="sha256:file",
    )
    registry.update_state("task-1", "succeeded")
    active.release("task-1")

    assert registry.get_owned("task-1", "org-a").state == "succeeded"
    assert registry.get_owned("task-1", "org-b") is None
    assert registry.list_owned("org-a")[0].task_id == "task-1"


def test_publish_failure_removes_active_record_but_retains_terminal_history() -> None:
    database = _database()
    active = TaskRegistry(MemorySecurityStateBackend(), ttl_seconds=700_000)
    registry = PersistentTaskRegistry(active, PostgreSQLTaskRunRepository(database))
    registry.reserve(
        "task-1",
        org_id="org-a",
        actor_id="user-a",
        kind="ingest",
    )
    assert registry.release("task-1") is True
    assert active.get_owned("task-1", "org-a") is None
    assert registry.get_owned("task-1", "org-a").state == "publish_failed"
