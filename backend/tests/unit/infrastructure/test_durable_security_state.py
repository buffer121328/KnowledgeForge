"""ATDD for shared, tenant-scoped security state."""

from __future__ import annotations

from pathlib import Path

import pytest

from auth.apikey_service import APIKeyService
from auth.apikey_store import RedisKeyStore
from auth.role_service import RoleService
from auth.user_service import UserService
from domain.identity import Permission, UserRole
from infrastructure.security.security_state import (
    CorruptSecurityStateError,
    MemorySecurityStateBackend,
)
from infrastructure.cache.identity_store import RedisIdentityStore
from infrastructure.webhooks.models import Webhook, WebhookEvent
from infrastructure.webhooks.stores import RedisWebhookStore
from shared.config.settings import Settings


def _user_service(store: RedisIdentityStore) -> UserService:
    return UserService(store=store)


def test_runtime_security_validation_has_no_durable_backend_selector() -> None:
    candidate = Settings(_env_file=None, security_state_backend="memory")

    candidate.validate_security_state_for_runtime("test")
    assert not hasattr(candidate, "security_state_backend")


def test_users_roles_and_token_versions_are_visible_across_store_clients() -> None:
    backend = MemorySecurityStateBackend()
    store_a = RedisIdentityStore(backend)
    store_b = RedisIdentityStore(backend)
    users_a = _user_service(store_a)
    users_b = _user_service(store_b)
    roles_a = RoleService(store=store_a)
    roles_b = RoleService(store=store_b)

    created = users_a.create_user(
        username="alice",
        password="not-used-by-store",
        display_name="Alice",
        email="alice@example.test",
        role=UserRole.VIEWER,
        org_id="org-a",
    )
    roles_a.create_role(
        name="auditor",
        display_name="Auditor",
        description="read only",
        permissions=[Permission.ADMIN_AUDIT.value],
        org_id="org-a",
    )
    updated = users_a.update_user(
        created["user_id"],
        role=UserRole.EDITOR,
        scope_org_id="org-a",
    )

    assert users_b.find_by_id(created["user_id"], org_id="org-a")["role"] == UserRole.EDITOR
    assert users_b.find_by_id(created["user_id"], org_id="org-a")["token_version"] == 1
    assert roles_b.get_permissions("auditor", org_id="org-a") == [
        Permission.ADMIN_AUDIT.value
    ]
    assert updated["token_version"] == 1


def test_api_key_rotation_and_revocation_are_visible_across_clients() -> None:
    backend = MemorySecurityStateBackend()
    service_a = APIKeyService(store=RedisKeyStore(backend))
    service_b = APIKeyService(store=RedisKeyStore(backend))

    key, raw = service_a.create_key(
        "automation",
        "user-a",
        "org-a",
        UserRole.ADMIN,
        [Permission.QA_QUERY],
        actor_permissions=list(Permission),
    )
    assert service_b.verify(raw, record_use=False).id == key.id

    rotated, new_raw = service_a.rotate(key.id, "user-a", "org-a")
    assert rotated.rotation_count == 1
    assert service_b.verify(raw, record_use=False) is None
    assert service_b.verify(new_raw, record_use=False).id == key.id

    service_b.revoke(key.id, "user-a", "org-a")
    assert service_a.verify(new_raw, record_use=False) is None


def test_corrupt_or_oversized_shared_record_fails_closed_without_content() -> None:
    backend = MemorySecurityStateBackend(max_record_bytes=128)
    backend.set_raw("security:v1:apikey:key-bad", "{not-json")
    store = RedisKeyStore(backend)

    with pytest.raises(CorruptSecurityStateError) as captured:
        store.get_by_id("key-bad")

    assert "not-json" not in str(captured.value)


def test_shared_webhook_store_is_coherent_and_excludes_unowned_legacy() -> None:
    backend = MemorySecurityStateBackend()
    store_a = RedisWebhookStore(backend)
    store_b = RedisWebhookStore(backend)
    owned = Webhook(
        id="wh-owned",
        org_id="org-a",
        url="https://example.test/hook",
        events=[WebhookEvent.QA_COMPLETED],
        secret="never-print-this",
        is_active=True,
    )
    unowned = Webhook(
        id="wh-legacy",
        org_id="",
        url="https://legacy.test/hook",
        events=[WebhookEvent.QA_COMPLETED],
        secret="legacy-secret",
    )

    store_a.save(owned)
    store_a.import_legacy_unowned(unowned)
    owned.is_active = False
    store_a.update(owned)

    assert store_b.get("wh-owned", "org-a").is_active is False
    assert store_b.get("wh-owned", "org-b") is None
    assert store_b.list_by_event(WebhookEvent.QA_COMPLETED, "org-a") == []
    assert store_b.get("wh-legacy", "org-a") is None
    assert store_b.list_all("org-a") == [owned]
    inventory = store_b.legacy_inventory()
    assert inventory == [{"webhook_id": "wh-legacy"}]
    assert "legacy.test" not in repr(inventory)
    assert "legacy-secret" not in repr(inventory)


def test_templates_declare_shared_state_and_never_enable_local_bootstrap_in_production() -> None:
    root = Path(__file__).resolve().parents[4]
    env_template = (root / "config" / ".env.example").read_text(encoding="utf-8")
    k8s_config = (root / "deploy" / "k8s" / "config.yaml").read_text(encoding="utf-8")

    assert "SECURITY_STATE_BACKEND" not in env_template
    assert "DOCUMENT_CATALOG_BACKEND" not in env_template
    assert "QA_HISTORY_BACKEND" not in env_template
    assert "WEBHOOK_FACT_BACKEND" not in env_template
    assert "TASK_HISTORY_BACKEND" not in env_template
    assert "SECURITY_STATE_ALLOW_LOCAL_BOOTSTRAP=false" in env_template
    assert "SECURITY_STATE_BACKEND" not in k8s_config
    assert "DOCUMENT_CATALOG_BACKEND" not in k8s_config
    assert "QA_HISTORY_BACKEND" not in k8s_config
    assert "WEBHOOK_FACT_BACKEND" not in k8s_config
    assert "TASK_HISTORY_BACKEND" not in k8s_config
    assert 'SECURITY_STATE_ALLOW_LOCAL_BOOTSTRAP: "false"' in k8s_config
