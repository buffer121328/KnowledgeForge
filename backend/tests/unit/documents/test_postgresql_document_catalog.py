"""Contract tests for PostgreSQL document metadata persistence and migration."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from domain.documents import CatalogConflictError, DepartmentRecord, DocumentRecord
from infrastructure.documents.catalog import (
    PostgreSQLDocumentCatalogRepository,
    SQLiteDocumentCatalogRepository,
)
from infrastructure.documents.catalog_migration import migrate_sqlite_catalog
from infrastructure.postgres.database import DatabaseService
from infrastructure.postgres.models import metadata


def _record(**overrides) -> DocumentRecord:
    values = {
        "doc_id": str(uuid4()),
        "tenant_id": "tenant-a",
        "company_id": "tenant-a",
        "department_id": "hr",
        "folder_path": "hr",
        "relative_path": "hr/policy.txt",
        "normalized_relative_path": "hr/policy.txt",
        "uploaded_filename": "policy.txt",
        "display_name": "Policy",
        "provenance_source_filename": "policy.docx",
        "storage_reference": "accepted/opaque-reference",
        "content_sha256": "a" * 64,
        "size_bytes": 100,
        "mime_type": "text/plain",
        "document_type": "policy",
        "source_format": "txt",
        "ingest_status": "accepted",
        "version": 1,
        "metadata": {"safe": True},
    }
    values.update(overrides)
    return DocumentRecord(**values)


@pytest.fixture
def pg_catalog() -> PostgreSQLDocumentCatalogRepository:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    repository = PostgreSQLDocumentCatalogRepository(DatabaseService.from_engine(engine))
    repository.upsert_department(
        DepartmentRecord("hr", "tenant-a", "tenant-a", "Human Resources", "hr")
    )
    return repository


def test_postgresql_catalog_is_tenant_scoped_metadata_only(pg_catalog) -> None:
    record = pg_catalog.create_document(_record())
    assert pg_catalog.get_document(record.doc_id, tenant_id="tenant-a") == record
    assert pg_catalog.get_document(record.doc_id, tenant_id="tenant-b") is None
    columns = {item["name"] for item in pg_catalog.table_columns("documents")}
    assert "storage_reference" in columns
    assert "content" not in columns
    assert "blob" not in columns


def test_postgresql_catalog_rejects_same_scoped_path_version(pg_catalog) -> None:
    pg_catalog.create_document(_record())
    with pytest.raises(CatalogConflictError):
        pg_catalog.create_document(_record())
    assert pg_catalog.create_document(_record(version=2)).version == 2


def test_sqlite_catalog_migration_dry_run_apply_and_verify(
    tmp_path: Path, pg_catalog
) -> None:
    source = SQLiteDocumentCatalogRepository(tmp_path / "catalog.sqlite3")
    source.upsert_department(
        DepartmentRecord("hr", "tenant-a", "tenant-a", "Human Resources", "hr")
    )
    record = source.create_document(_record())

    dry_run = migrate_sqlite_catalog(
        source, pg_catalog, tenant_ids=["tenant-a"], mode="dry-run"
    )
    assert dry_run.documents == 1
    assert dry_run.created == 0
    assert pg_catalog.get_document(record.doc_id, tenant_id="tenant-a") is None

    applied = migrate_sqlite_catalog(
        source, pg_catalog, tenant_ids=["tenant-a"], mode="apply"
    )
    assert applied.created == 1
    repeated = migrate_sqlite_catalog(
        source, pg_catalog, tenant_ids=["tenant-a"], mode="apply"
    )
    assert repeated.created == 0
    assert repeated.skipped == 2
    verified = migrate_sqlite_catalog(
        source, pg_catalog, tenant_ids=["tenant-a"], mode="verify"
    )
    assert verified.conflicts == ()
    assert verified.source_checksum == verified.target_checksum
