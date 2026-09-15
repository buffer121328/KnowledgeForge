"""Read-only inventory for legacy process/file security state.

The command intentionally reports identifiers and one-way fingerprints only. It
does not assign owners, write Redis, or print URLs, secrets, password hashes,
filesystem paths, task arguments, or exception text.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from auth.memory_accounts import ROLE_DB, USER_DB
from infrastructure.webhooks.models import Webhook
from infrastructure.documents.local_uploads import LocalUploadStorage, UploadPolicyError
from infrastructure.webhooks.stores import FileWebhookStore
from shared.config import settings
from shared.utils.logging import safe_fingerprint


def _fingerprint(value: object, namespace: str) -> str:
    """Return the fingerprint."""
    return safe_fingerprint(str(value), namespace=namespace)


def inventory_legacy_state(webhook_path: str | Path | None = None) -> dict[str, Any]:
    """Return a stable, secret-free inventory without mutating any state."""
    path = Path(webhook_path) if webhook_path is not None else (
        Path(settings.upload_dir).resolve().parent / "logs" / "webhooks.json"
    )
    raw_items: list[dict[str, Any]] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw_items = raw if isinstance(raw, list) else raw.get("webhooks", [])
        except (OSError, TypeError, ValueError):
            raw_items = []
    webhooks: list[Webhook] = []
    for item in raw_items:
        if isinstance(item, dict):
            try:
                webhooks.append(Webhook.from_dict(item))
            except (KeyError, TypeError, ValueError):
                continue
    legacy_webhooks = [
        {"webhook_id": webhook.id}
        for webhook in webhooks
        if not webhook.org_id
    ]
    users = [
        {
            "user_id": str(record.get("user_id", "")),
            "username_fingerprint": _fingerprint(record.get("username", key), "legacy-user"),
            "org_id": str(record.get("org_id", "")),
        }
        for key, record in sorted(USER_DB.items())
    ]
    roles = [
        {
            "role_id": str(record.get("role_id", "")),
            "name_fingerprint": _fingerprint(record.get("name", key), "legacy-role"),
            "org_id": str(record.get("org_id", "")),
        }
        for key, record in sorted(ROLE_DB.items())
        if not record.get("is_builtin", False)
    ]
    return {
        "mode": "dry-run",
        "legacy_webhooks": legacy_webhooks,
        "users": users,
        "custom_roles": roles,
        "task_records": [],
        "owner_assignment_required": bool(legacy_webhooks),
    }


def _load_webhook_items(path: str | Path) -> list[dict[str, Any]]:
    """Load the webhook items."""
    source = Path(path)
    if not source.exists():
        return []
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return []
    items = raw if isinstance(raw, list) else raw.get("webhooks", []) if isinstance(raw, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _validated_mapping(owner_mapping: dict[str, str] | None) -> dict[str, str]:
    """Return the validated mapping."""
    mapping = owner_mapping or {}
    if not isinstance(mapping, dict):
        raise ValueError("owner mapping must be an object")
    result: dict[str, str] = {}
    for key, value in mapping.items():
        identifier = str(key).strip()
        owner = str(value).strip()
        if not identifier or not owner:
            raise ValueError("owner mapping identifiers and organizations are required")
        result[identifier] = owner
    return result


def _stable_fingerprint(value: object, namespace: str) -> str:
    """Return the stable fingerprint."""
    return safe_fingerprint(str(value), namespace=namespace)


def build_legacy_migration_plan(
    webhook_path: str | Path,
    *,
    owner_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, secret-free migration plan (dry-run by default)."""
    mapping = _validated_mapping(owner_mapping)
    actions: list[dict[str, str]] = []
    source_items = _load_webhook_items(webhook_path)
    for item in source_items:
        try:
            webhook = Webhook.from_dict(item)
        except (KeyError, TypeError, ValueError):
            continue
        if webhook.org_id:
            continue
        owner = mapping.get(webhook.id)
        if owner:
            actions.append({"record_id": webhook.id, "org_id": owner, "action": "migrate"})
        else:
            actions.append({"record_id": webhook.id, "action": "owner_required"})
    source_fingerprint = _stable_fingerprint(
        "|".join(sorted(item.get("id", "") for item in source_items)),
        "legacy-migration-source",
    )
    mapping_fingerprint = _stable_fingerprint(
        json.dumps(mapping, sort_keys=True),
        "legacy-migration-mapping",
    )
    return {
        "mode": "dry-run",
        "evidence": "static_only",
        "source_fingerprint": source_fingerprint,
        "mapping_fingerprint": mapping_fingerprint,
        "actions": actions,
        "counts": {
            "migrate": sum(action["action"] == "migrate" for action in actions),
            "owner_required": sum(action["action"] == "owner_required" for action in actions),
        },
    }


def _write_webhook_items(path: Path, items: list[dict[str, Any]]) -> None:
    """Write the webhook items."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(items, target, ensure_ascii=False, indent=2)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def apply_legacy_migration(
    webhook_path: str | Path,
    *,
    owner_mapping: dict[str, str] | None,
    store: Any,
    allow_apply: bool = False,
) -> dict[str, Any]:
    """Apply only explicitly mapped legacy Webhooks after destination success."""
    plan = build_legacy_migration_plan(webhook_path, owner_mapping=owner_mapping)
    if not allow_apply:
        return {"status": "confirmation_required", "actions": plan["actions"]}
    mapping = _validated_mapping(owner_mapping)
    source = Path(webhook_path)
    items = _load_webhook_items(source)
    remaining: list[dict[str, Any]] = []
    results: list[dict[str, str]] = []
    changed = False
    for item in items:
        try:
            webhook = Webhook.from_dict(item)
        except (KeyError, TypeError, ValueError):
            remaining.append(item)
            continue
        if webhook.org_id or webhook.id not in mapping:
            remaining.append(item)
            continue
        owner = mapping[webhook.id]
        existing = None
        try:
            existing = store.get(webhook.id, owner)
        except (AttributeError, TypeError):
            existing = None
        if existing is not None:
            changed = True
            results.append({"record_id": webhook.id, "action": "already_migrated"})
            continue
        try:
            get_any = getattr(store, "get_any", None)
            conflicting = get_any(webhook.id) if get_any is not None else store.get(webhook.id)
        except (AttributeError, TypeError):
            conflicting = None
        if conflicting is not None and getattr(conflicting, "org_id", "") != owner:
            remaining.append(item)
            results.append({"record_id": webhook.id, "action": "destination_owner_conflict"})
            continue
        migrated = Webhook.from_dict({**item, "org_id": owner})
        try:
            store.save(migrated)
        except Exception:
            remaining.append(item)
            results.append({"record_id": webhook.id, "action": "destination_write_failed"})
            continue
        changed = True
        results.append({"record_id": webhook.id, "action": "migrated"})
    if changed:
        _write_webhook_items(source, remaining)
    status = "failed" if any(item["action"] == "destination_write_failed" for item in results) else (
        "applied" if results else "noop"
    )
    return {"status": status, "actions": results}


def plan_legacy_uploads(
    legacy_root: str | Path,
    *,
    owner_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Plan direct shared-root files without exposing content or absolute paths."""
    mapping = _validated_mapping(owner_mapping)
    root = Path(legacy_root).resolve()
    actions: list[dict[str, str]] = []
    if root.exists() and root.is_dir():
        for candidate in sorted(item for item in root.iterdir() if item.is_file()):
            source_id = candidate.name
            owner = mapping.get(source_id)
            action = {"source_id": source_id}
            if owner:
                action.update({"org_id": owner, "action": "migrate"})
            else:
                action["action"] = "owner_required"
            actions.append(action)
    return {
        "mode": "dry-run",
        "evidence": "static_only",
        "actions": actions,
        "counts": {
            "migrate": sum(action["action"] == "migrate" for action in actions),
            "owner_required": sum(action["action"] == "owner_required" for action in actions),
        },
    }


def apply_legacy_uploads(
    legacy_root: str | Path,
    *,
    owner_mapping: dict[str, str] | None,
    storage: LocalUploadStorage,
    allow_apply: bool = False,
) -> dict[str, Any]:
    """Apply the legacy uploads."""
    if not allow_apply:
        return {"status": "confirmation_required", "actions": plan_legacy_uploads(legacy_root, owner_mapping=owner_mapping)["actions"]}
    mapping = _validated_mapping(owner_mapping)
    root = Path(legacy_root).resolve()
    results: list[dict[str, str]] = []
    if not root.exists() or not root.is_dir():
        return {"status": "noop", "actions": []}
    for candidate in sorted(item for item in root.iterdir() if item.is_file()):
        owner = mapping.get(candidate.name)
        if not owner:
            continue
        try:
            storage.migrate_legacy_file(candidate, owner)
        except (OSError, UploadPolicyError):
            results.append({"source_id": candidate.name, "action": "destination_write_failed"})
        else:
            results.append({"source_id": candidate.name, "action": "migrated"})
    return {"status": "applied" if results else "noop", "actions": results}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webhook-file", default=None, help="legacy webhook JSON file to inspect")
    parser.add_argument("--owner-mapping", default=None, help="JSON file mapping legacy IDs/names to organization IDs")
    parser.add_argument("--destination-webhook-file", default=None, help="destination JSON file used only with --apply")
    parser.add_argument("--uploads-root", default=None, help="shared-root upload directory to plan or migrate")
    parser.add_argument("--apply", action="store_true", help="apply only explicitly mapped records; otherwise dry-run")
    args = parser.parse_args(argv)
    if not args.owner_mapping:
        print(json.dumps(inventory_legacy_state(args.webhook_file), ensure_ascii=False, sort_keys=True))
        return 0
    try:
        mapping = json.loads(Path(args.owner_mapping).read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("owner mapping must be an object")
        result: dict[str, Any] = {}
        if args.webhook_file:
            if args.apply:
                if not args.destination_webhook_file:
                    raise ValueError("--destination-webhook-file is required with --apply and --webhook-file")
                result["webhooks"] = apply_legacy_migration(
                    args.webhook_file,
                    owner_mapping=mapping,
                    store=FileWebhookStore(Path(args.destination_webhook_file)),
                    allow_apply=True,
                )
            else:
                result["webhooks"] = build_legacy_migration_plan(args.webhook_file, owner_mapping=mapping)
        if args.uploads_root:
            if args.apply:
                storage = LocalUploadStorage(args.uploads_root)
                result["uploads"] = apply_legacy_uploads(
                    args.uploads_root,
                    owner_mapping=mapping,
                    storage=storage,
                    allow_apply=True,
                )
            else:
                result["uploads"] = plan_legacy_uploads(args.uploads_root, owner_mapping=mapping)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except (OSError, TypeError, ValueError) as error:
        print(json.dumps({"status": "fail", "error": "invalid_migration_input"}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
