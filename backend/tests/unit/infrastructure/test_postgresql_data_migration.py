"""Tests for secret-safe, idempotent and transaction-bounded migration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select

from infrastructure.postgres.database import DatabaseConflictError, DatabaseService
from infrastructure.postgres.data_migration import (
    MigrationRecord,
    PostgreSQLMigrationRunner,
    load_manifest,
)
from infrastructure.postgres.models import documents, metadata, organizations


def _runner() -> PostgreSQLMigrationRunner:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    return PostgreSQLMigrationRunner(
        DatabaseService.from_engine(engine),
        checksum_key=b"test-only-key",
        batch_size=20,
    )


def test_manifest_inventory_never_reports_password_or_api_key_digests(tmp_path: Path) -> None:
    manifest = tmp_path / "migration.json"
    manifest.write_text(
        json.dumps({
            "records": [{
                "record_class": "user",
                "stable_id": "user-1",
                "tenant_id": "org-1",
                "owner_id": "user-1",
                "values": {
                    "id": "user-1",
                    "tenant_id": "org-1",
                    "username": "alice",
                    "email": "alice@example.com",
                    "display_name": "Alice",
                    "password_hash": "super-secret-password-digest",
                    "role": "admin",
                    "is_active": True,
                    "token_version": 3,
                    "created_at": "2026-08-01T00:00:00+00:00",
                    "updated_at": "2026-08-01T00:00:00+00:00"
                }
            }]
        }),
        encoding="utf-8",
    )
    report = _runner().run(load_manifest(manifest), mode="inventory").to_dict()
    rendered = json.dumps(report)
    assert report["counts"] == {"user": 1}
    assert "user-1" in report["checksums"]
    assert "super-secret-password-digest" not in rendered
    assert "alice@example.com" not in rendered


def test_apply_is_idempotent_and_verify_uses_keyed_checksums() -> None:
    runner = _runner()
    now = datetime.now(UTC)
    record = MigrationRecord(
        "organization",
        "org-1",
        "org-1",
        "",
        {"id": "org-1", "name": "Tenant", "status": "active", "created_at": now, "updated_at": now},
    )
    first = runner.run([record], mode="apply")
    second = runner.run([record], mode="apply")
    verified = runner.run([record], mode="verify")
    assert first.created == 1
    assert second.created == 0
    assert second.unchanged == 1
    assert verified.ready is True
    assert len(verified.checksums["org-1"]) == 64


def test_dry_run_groups_duplicate_username_without_target_mutation() -> None:
    runner = _runner()
    records = [
        MigrationRecord(
            "user",
            f"user-{index}",
            "org-1",
            f"user-{index}",
            {"id": f"user-{index}", "tenant_id": "org-1", "username": "alice"},
        )
        for index in (1, 2)
    ]
    report = runner.run(records, mode="dry-run")
    assert {item["category"] for item in report.conflicts} == {"duplicate_username"}
    with runner.database.session() as session:
        assert session.execute(select(func.count()).select_from(organizations)).scalar_one() == 0


def test_constraint_failure_rolls_back_the_whole_batch() -> None:
    runner = _runner()
    now = datetime.now(UTC)
    department = MigrationRecord(
        "department",
        "org-1:hr",
        "org-1",
        "",
        {"tenant_id": "org-1", "department_id": "hr", "company_id": "org-1", "name": "HR", "normalized_key": "hr", "status": "active"},
    )
    base = {
        "tenant_id": "org-1", "company_id": "org-1", "department_id": "hr",
        "folder_path": "hr", "relative_path": "hr/a.txt", "normalized_relative_path": "hr/a.txt",
        "uploaded_filename": "a.txt", "display_name": "a", "provenance_source_filename": "a.txt",
        "storage_reference": "opaque", "content_sha256": "a" * 64, "size_bytes": 1,
        "mime_type": "text/plain", "document_type": "text", "source_format": "txt",
        "ingest_status": "accepted", "version": 1, "authority": "", "review_status": "",
        "sensitivity": "", "external_source_id": "", "created_by": "", "created_at": now,
        "updated_at": now, "chunks_count": 0, "entities_count": 0, "relations_count": 0,
        "error_code": "", "legacy": False, "metadata_json": {},
    }
    records = [department] + [
        MigrationRecord("document", f"doc-{index}", "org-1", "", {**base, "doc_id": f"doc-{index}"})
        for index in (1, 2)
    ]
    with pytest.raises(DatabaseConflictError):
        runner.run(records, mode="apply")
    with runner.database.session() as session:
        assert session.execute(select(func.count()).select_from(documents)).scalar_one() == 0
