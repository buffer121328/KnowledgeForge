"""ATDD coverage for explicit-owner migration and deployment preflight."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from auth.apikey_service import APIKeyPolicy, APIKeyService
from auth.apikey_store import MemoryKeyStore
from domain.identity import UserRole
from infrastructure.security.security_state_migration import (
    apply_legacy_migration,
    apply_legacy_uploads,
    build_legacy_migration_plan,
    plan_legacy_uploads,
)
from infrastructure.deployment_preflight import main as deployment_preflight_main
from infrastructure.deployment_preflight import run_deployment_preflight


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _webhook(webhook_id: str, *, org_id: str = "", url: str = "https://hooks.example.test/a") -> dict:
    return {
        "id": webhook_id,
        "org_id": org_id,
        "url": url,
        "events": ["qa.completed"],
        "secret": "super-secret",
        "is_active": True,
    }


def test_legacy_plan_is_secret_free_and_requires_explicit_owner(tmp_path: Path) -> None:
    source = tmp_path / "webhooks.json"
    source.write_text(json.dumps([_webhook("wh-mapped"), _webhook("wh-unowned", url="https://other.example.test")]))

    plan = build_legacy_migration_plan(source, owner_mapping={"wh-mapped": "org-a"})

    assert plan["mode"] == "dry-run"
    assert plan["actions"] == [
        {"record_id": "wh-mapped", "org_id": "org-a", "action": "migrate"},
        {"record_id": "wh-unowned", "action": "owner_required"},
    ]
    rendered = json.dumps(plan, ensure_ascii=False)
    assert "super-secret" not in rendered
    assert "hooks.example.test" not in rendered


def test_apply_requires_confirmation_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "webhooks.json"
    source.write_text(json.dumps([_webhook("wh-1")]))
    store = _RecordingWebhookStore()

    refused = apply_legacy_migration(source, owner_mapping={"wh-1": "org-a"}, store=store)
    assert refused["status"] == "confirmation_required"
    assert source.exists()
    assert store.saved == []

    applied = apply_legacy_migration(
        source,
        owner_mapping={"wh-1": "org-a"},
        store=store,
        allow_apply=True,
    )
    assert applied["status"] == "applied"
    assert store.saved[0].org_id == "org-a"
    assert json.loads(source.read_text()) == []

    repeated = apply_legacy_migration(
        source,
        owner_mapping={"wh-1": "org-a"},
        store=store,
        allow_apply=True,
    )
    assert repeated["status"] == "noop"
    assert len(store.saved) == 1


def test_destination_failure_leaves_legacy_source_untouched(tmp_path: Path) -> None:
    source = tmp_path / "webhooks.json"
    source.write_text(json.dumps([_webhook("wh-fail")]))

    result = apply_legacy_migration(
        source,
        owner_mapping={"wh-fail": "org-a"},
        store=_FailingWebhookStore(),
        allow_apply=True,
    )

    assert result["status"] == "failed"
    assert result["actions"] == [{"record_id": "wh-fail", "action": "destination_write_failed"}]
    assert json.loads(source.read_text())[0]["id"] == "wh-fail"


def test_shared_root_upload_plan_requires_mapping_and_moves_only_approved_file(tmp_path: Path) -> None:
    legacy_root = tmp_path / "uploads"
    legacy_root.mkdir()
    approved = legacy_root / "approved.txt"
    unresolved = legacy_root / "unresolved.txt"
    approved.write_text("approved")
    unresolved.write_text("unresolved")

    plan = plan_legacy_uploads(legacy_root, owner_mapping={"approved.txt": "org-a"})
    assert plan["actions"] == [
        {"source_id": "approved.txt", "org_id": "org-a", "action": "migrate"},
        {"source_id": "unresolved.txt", "action": "owner_required"},
    ]
    rendered = json.dumps(plan)
    assert "approved" in rendered  # stable source ID is allowed
    assert "unresolved" in rendered  # unresolved ID is allowed
    assert "approved\"" not in rendered  # file contents are never included

    from infrastructure.documents.local_uploads import LocalUploadStorage

    result = apply_legacy_uploads(
        legacy_root,
        owner_mapping={"approved.txt": "org-a"},
        storage=LocalUploadStorage(str(legacy_root)),
        allow_apply=True,
    )
    assert result["status"] == "applied"
    assert not approved.exists()
    assert unresolved.exists()
    assert list((legacy_root / "tenants").rglob("approved.txt")) == []
    assert list((legacy_root / "tenants").rglob("*"))


def test_preflight_reports_static_only_and_fails_closed_on_policy_drift() -> None:
    deployment = """
apiVersion: apps/v1
kind: Deployment
metadata: {name: parser}
spec:
  selector: {matchLabels: {app: agenthub-parser}}
  template:
    metadata: {labels: {app: agenthub-parser}}
"""
    policy = """
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: deny}
spec:
  podSelector: {matchLabels: {app: agenthub-parser}}
  policyTypes: [Egress]
  egress: []
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: allow}
spec:
  podSelector: {matchLabels: {app: agenthub-parser}}
  policyTypes: [Egress]
  egress:
    - to: [{namespaceSelector: {matchLabels: {kubernetes.io/metadata.name: kube-system}}, podSelector: {matchLabels: {k8s-app: kube-dns}}}]
      ports:
        - {protocol: UDP, port: 53}
        - {protocol: TCP, port: 53}
    - to: [{podSelector: {matchLabels: {app: redis-agenthub}}}]
      ports: [{protocol: TCP, port: 6379}]
    - to: [{podSelector: {matchLabels: {app: neo4j-agenthub}}}]
      ports: [{protocol: TCP, port: 7687}]
    - to: [{podSelector: {matchLabels: {app: chroma-agenthub}}}]
      ports: [{protocol: TCP, port: 8000}]
"""
    auth = {"app_environment": "production", "allow_insecure_local_development": False}

    result = run_deployment_preflight(deployment, policy, auth)
    assert result["status"] == "pass"
    assert result["evidence"] == "static_only"

    drifted = policy.replace("port: 6379", "port: 443")
    failed = run_deployment_preflight(deployment, drifted, auth)
    assert failed["status"] == "fail"
    assert failed["evidence"] == "static_only"
    assert "undeclared_egress" in failed["failures"]

    selector_drift = deployment.replace(
        "labels: {app: agenthub-parser}",
        "labels: {app: parser-drifted}",
    )
    selector_failed = run_deployment_preflight(selector_drift, policy, auth)
    assert selector_failed["status"] == "fail"
    assert "parser_selector_mismatch" in selector_failed["failures"]

    dns_missing = policy.replace(
        "    - to: [{namespaceSelector: {matchLabels: {kubernetes.io/metadata.name: kube-system}}, podSelector: {matchLabels: {k8s-app: kube-dns}}}]\n      ports:\n        - {protocol: UDP, port: 53}\n        - {protocol: TCP, port: 53}\n",
        "",
    )
    dns_failed = run_deployment_preflight(deployment, dns_missing, auth)
    assert dns_failed["status"] == "fail"
    assert "required_dns_egress_missing" in dns_failed["failures"]

    insecure = run_deployment_preflight(deployment, policy, {"app_environment": "production", "allow_insecure_local_development": True})
    assert insecure["status"] == "fail"
    assert "production_insecure_auth" in insecure["failures"]


def test_api_key_warning_projection_is_scoped_and_secret_free() -> None:
    service = APIKeyService(
        store=MemoryKeyStore(),
        policy=APIKeyPolicy(stale_warning_days=14, expiry_warning_days=7),
        clock=lambda: NOW,
    )
    key, _ = service.create_key(
        name="automation",
        user_id="user-a",
        org_id="org-a",
        role=UserRole.ADMIN,
        scopes=[],
        expires_days=3,
        actor_permissions=[],
    )
    key.created_at = (NOW - timedelta(days=20)).isoformat()
    key.last_used_at = (NOW - timedelta(days=20)).isoformat()
    service.store.save(key)

    other, _ = service.create_key(
        name="other",
        user_id="user-b",
        org_id="org-b",
        role=UserRole.ADMIN,
        scopes=[],
        expires_days=3,
        actor_permissions=[],
    )

    projection = service.warning_projection("user-a", "org-a")
    assert projection == [
        {
            "key_id": key.id,
            "user_id": "user-a",
            "org_id": "org-a",
            "warnings": ["stale", "expires_soon"],
            "expires_at": key.expires_at,
            "created_at": key.created_at,
            "last_used_at": key.last_used_at,
            "rotation_count": 0,
        }
    ]
    assert other.id not in json.dumps(projection)
    assert key.key_hash not in json.dumps(projection)

    inactive, _ = service.create_key(
        name="inactive",
        user_id="user-a",
        org_id="org-a",
        role=UserRole.ADMIN,
        scopes=[],
        expires_days=3,
        actor_permissions=[],
    )
    inactive.is_active = False
    service.store.save(inactive)

    expired, _ = service.create_key(
        name="expired",
        user_id="user-a",
        org_id="org-a",
        role=UserRole.ADMIN,
        scopes=[],
        expires_days=3,
        actor_permissions=[],
    )
    expired.expires_at = (NOW - timedelta(minutes=1)).isoformat()
    service.store.save(expired)

    malformed, _ = service.create_key(
        name="malformed",
        user_id="user-a",
        org_id="org-a",
        role=UserRole.ADMIN,
        scopes=[],
        expires_days=3,
        actor_permissions=[],
    )
    malformed.expires_at = "not-a-timestamp"
    service.store.save(malformed)

    projection = service.warning_projection("user-a", "org-a")
    projected_ids = {item["key_id"] for item in projection}
    assert inactive.id not in projected_ids
    assert expired.id not in projected_ids
    assert malformed.id not in projected_ids


def test_preflight_cli_invalid_input_is_stable_and_secret_free(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deployment = tmp_path / "deployment.yaml"
    policy = tmp_path / "policy.yaml"
    auth = tmp_path / "auth.json"
    deployment.write_text("not: [valid")
    policy.write_text("kind: NetworkPolicy\n")
    auth.write_text("{not-json")

    status = deployment_preflight_main(
        [
            "--deployment",
            str(deployment),
            "--network-policy",
            str(policy),
            "--auth-config",
            str(auth),
        ]
    )

    assert status == 2
    output = capsys.readouterr().out
    assert output.strip() == '{"error": "invalid_preflight_input", "status": "fail"}'
    assert str(deployment) not in output


class _RecordingWebhookStore:
    def __init__(self) -> None:
        self.saved = []

    def save(self, webhook) -> None:
        self.saved.append(webhook)


class _FailingWebhookStore:
    def save(self, webhook) -> None:
        raise RuntimeError("backend details must not leak")
