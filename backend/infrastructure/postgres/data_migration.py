"""Secret-safe, non-destructive offline PostgreSQL migration runner.

The JSON manifest is an offline interchange boundary for legacy adapters.  It
may contain password/API-key digests or encrypted webhook ciphertext, so this
module never logs or echoes record values.  Reports contain only stable IDs,
counts, conflict categories and keyed checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from sqlalchemy import Table, insert, select

from infrastructure.postgres.database import DatabaseService, get_database_service
from infrastructure.postgres.models import (
    api_keys,
    departments,
    documents,
    organizations,
    permissions,
    role_permissions,
    roles,
    task_runs,
    user_roles,
    users,
    webhooks,
)


_TABLES: dict[str, Table] = {
    "organization": organizations,
    "permission": permissions,
    "role": roles,
    "role_permission": role_permissions,
    "user": users,
    "user_role": user_roles,
    "api_key": api_keys,
    "department": departments,
    "document": documents,
    "webhook": webhooks,
    "task_run": task_runs,
}
_ORDER = {name: index for index, name in enumerate(_TABLES)}
_CHECKSUM_FIELDS: dict[str, tuple[str, ...]] = {
    "organization": ("id", "status"),
    "permission": ("code",),
    "role": ("id", "tenant_id", "name", "is_builtin"),
    "role_permission": ("tenant_id", "role_id", "permission_code"),
    "user": ("id", "tenant_id", "username", "email", "role", "is_active", "token_version"),
    "user_role": ("tenant_id", "user_id", "role_id"),
    "api_key": ("id", "tenant_id", "user_id", "key_prefix", "role", "is_active", "rotation_count"),
    "department": ("tenant_id", "department_id", "company_id", "normalized_key"),
    "document": ("tenant_id", "doc_id", "department_id", "normalized_relative_path", "version", "ingest_status"),
    "webhook": ("id", "tenant_id", "url", "events", "is_active", "failure_count"),
    "task_run": ("id", "tenant_id", "user_id", "task_type", "status"),
}


@dataclass(frozen=True)
class MigrationRecord:
    """One stable legacy fact and its private target values."""

    record_class: str
    stable_id: str
    tenant_id: str
    owner_id: str
    values: dict[str, Any]


@dataclass(frozen=True)
class MigrationReport:
    """Safe report that deliberately excludes raw migrated values."""

    mode: str
    counts: dict[str, int]
    created: int
    unchanged: int
    conflicts: tuple[dict[str, str], ...]
    checksums: dict[str, str]
    ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "counts": self.counts,
            "created": self.created,
            "unchanged": self.unchanged,
            "conflicts": list(self.conflicts),
            "checksums": self.checksums,
            "ready": self.ready,
        }


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return value


def _checksum(record: MigrationRecord, key: bytes) -> str:
    allowlisted = {
        name: _canonical(record.values.get(name))
        for name in _CHECKSUM_FIELDS[record.record_class]
    }
    body = json.dumps(
        {
            "class": record.record_class,
            "id": record.stable_id,
            "tenant": record.tenant_id,
            "owner": record.owner_id,
            "fields": allowlisted,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, body, hashlib.sha256).hexdigest()


def load_manifest(path: str | Path) -> list[MigrationRecord]:
    """Load a bounded allowlisted manifest without exposing its values."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) > 1_000_000:
        raise ValueError("invalid migration manifest")
    records: list[MigrationRecord] = []
    for item in items:
        if not isinstance(item, dict) or item.get("record_class") not in _TABLES:
            raise ValueError("unsupported migration record")
        record_class = str(item["record_class"])
        values = item.get("values")
        if not isinstance(values, dict):
            raise ValueError("migration values must be an object")
        allowed_columns = {column.name for column in _TABLES[record_class].columns}
        if not values or set(values) - allowed_columns:
            raise ValueError("migration values contain unsupported columns")
        if record_class == "webhook" and "secret" in values:
            raise ValueError("plaintext webhook secrets are forbidden")
        stable_id = str(item.get("stable_id") or "").strip()
        if not stable_id:
            raise ValueError("stable migration identity is required")
        records.append(
            MigrationRecord(
                record_class=record_class,
                stable_id=stable_id,
                tenant_id=str(item.get("tenant_id") or ""),
                owner_id=str(item.get("owner_id") or ""),
                values=values,
            )
        )
    return sorted(records, key=lambda item: (_ORDER[item.record_class], item.stable_id))


class PostgreSQLMigrationRunner:
    """Plan, apply and verify deterministic manifest batches."""

    def __init__(
        self,
        database: DatabaseService,
        *,
        checksum_key: bytes,
        batch_size: int = 200,
    ) -> None:
        if not checksum_key:
            raise ValueError("migration checksum key is required")
        self.database = database
        self.checksum_key = checksum_key
        self.batch_size = max(1, min(batch_size, 1000))

    @staticmethod
    def _primary_filter(table: Table, values: dict[str, Any]):
        primary = list(table.primary_key.columns)
        if any(column.name not in values for column in primary):
            raise ValueError("migration record is missing a primary key")
        return [column == values[column.name] for column in primary]

    def _existing(self, session, record: MigrationRecord) -> dict[str, Any] | None:
        table = _TABLES[record.record_class]
        return session.execute(
            select(table).where(*self._primary_filter(table, record.values))
        ).mappings().one_or_none()

    def _source_conflicts(self, records: list[MigrationRecord]) -> list[dict[str, str]]:
        conflicts: list[dict[str, str]] = []
        stable_seen: set[tuple[str, str]] = set()
        unique_seen: dict[tuple[str, str], str] = {}
        unique_fields = {
            "user": ("username", "email"),
            "api_key": ("key_hash",),
        }
        for record in records:
            identity = (record.record_class, record.stable_id)
            if identity in stable_seen:
                conflicts.append({"category": "duplicate_stable_id", "stable_id": record.stable_id})
            stable_seen.add(identity)
            for field in unique_fields.get(record.record_class, ()):
                value = record.values.get(field)
                if value in {None, ""}:
                    continue
                fingerprint = hmac.new(
                    self.checksum_key,
                    str(value).casefold().encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                unique = (record.record_class, f"{field}:{fingerprint}")
                previous = unique_seen.get(unique)
                if previous and previous != record.stable_id:
                    conflicts.append({"category": f"duplicate_{field}", "stable_id": record.stable_id})
                unique_seen[unique] = record.stable_id
            if record.record_class not in {"organization", "permission"} and not record.tenant_id:
                conflicts.append({"category": "orphan_tenant", "stable_id": record.stable_id})
        return conflicts

    def run(self, records: Iterable[MigrationRecord], *, mode: str) -> MigrationReport:
        """Run inventory, dry-run, apply or verify without deleting either source."""
        if mode not in {"inventory", "dry-run", "apply", "verify"}:
            raise ValueError("unsupported migration mode")
        ordered = sorted(records, key=lambda item: (_ORDER[item.record_class], item.stable_id))
        counts: dict[str, int] = {}
        checksums: dict[str, str] = {}
        for record in ordered:
            counts[record.record_class] = counts.get(record.record_class, 0) + 1
            checksums[record.stable_id] = _checksum(record, self.checksum_key)
        conflicts = self._source_conflicts(ordered)
        created = unchanged = 0

        if mode != "inventory":
            with self.database.session() as session:
                for record in ordered:
                    existing = self._existing(session, record)
                    if existing is None:
                        continue
                    comparable = {name: _canonical(existing[name]) for name in record.values}
                    expected = {name: _canonical(value) for name, value in record.values.items()}
                    if comparable == expected:
                        unchanged += 1
                    else:
                        conflicts.append({"category": "target_mismatch", "stable_id": record.stable_id})

        if mode == "apply" and not conflicts:
            missing = []
            with self.database.session() as session:
                missing = [record for record in ordered if self._existing(session, record) is None]
            for offset in range(0, len(missing), self.batch_size):
                batch = missing[offset : offset + self.batch_size]
                # One DatabaseService session is one transaction: any row failure rolls back the batch.
                with self.database.session() as session:
                    for record in batch:
                        session.execute(insert(_TABLES[record.record_class]).values(**record.values))
                        created += 1

        if mode == "verify":
            with self.database.session() as session:
                for record in ordered:
                    existing = self._existing(session, record)
                    if existing is None:
                        conflicts.append({"category": "target_missing", "stable_id": record.stable_id})
                        continue
                    target_record = MigrationRecord(
                        record_class=record.record_class,
                        stable_id=record.stable_id,
                        tenant_id=record.tenant_id,
                        owner_id=record.owner_id,
                        values=dict(existing),
                    )
                    if _checksum(target_record, self.checksum_key) != checksums[record.stable_id]:
                        conflicts.append({"category": "ownership_or_checksum_mismatch", "stable_id": record.stable_id})

        return MigrationReport(
            mode=mode,
            counts=counts,
            created=created,
            unchanged=unchanged,
            conflicts=tuple(conflicts),
            checksums=checksums,
            ready=not conflicts and mode in {"dry-run", "apply", "verify"},
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline manifest migration CLI with secret-safe JSON output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "dry-run", "apply", "verify"))
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--checksum-key-env", default="MIGRATION_CHECKSUM_KEY")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        key = os.environ.get(args.checksum_key_env, "").encode("utf-8")
        runner = PostgreSQLMigrationRunner(
            get_database_service(), checksum_key=key, batch_size=args.batch_size
        )
        report = runner.run(load_manifest(args.source_manifest), mode=args.mode)
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0 if report.ready or args.mode == "inventory" else 2
    except Exception:
        print(json.dumps({"status": "failed", "category": "safe_migration_failure"}))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MigrationRecord",
    "MigrationReport",
    "PostgreSQLMigrationRunner",
    "load_manifest",
    "main",
]
