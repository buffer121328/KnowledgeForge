"""ATDD coverage for stable document catalog identity and provenance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from domain.documents import (
    CatalogConflictError,
    DepartmentRecord,
    DocumentIngestStatus,
    DocumentRecord,
    normalize_relative_path,
    normalize_uploaded_filename,
)
from infrastructure.documents.catalog import SQLiteDocumentCatalogRepository


def _catalog(tmp_path: Path) -> SQLiteDocumentCatalogRepository:
    """Create one isolated SQLite catalog with an authorized department."""

    catalog = SQLiteDocumentCatalogRepository(tmp_path / "catalog.sqlite3")
    catalog.upsert_department(
        DepartmentRecord(
            department_id="human_resources",
            tenant_id="tenant-a",
            company_id="tenant-a",
            name="人力资源部",
            normalized_key="human_resources",
        )
    )
    return catalog


def _record(**overrides: object) -> DocumentRecord:
    """Build one complete metadata-only document record for tests."""

    values: dict[str, object] = {
        "doc_id": str(uuid4()),
        "tenant_id": "tenant-a",
        "company_id": "tenant-a",
        "department_id": "human_resources",
        "folder_path": "human_resources",
        "relative_path": "human_resources/招聘管理制度.txt",
        "normalized_relative_path": "human_resources/招聘管理制度.txt",
        "uploaded_filename": "招聘管理制度.txt",
        "display_name": "招聘管理制度",
        "provenance_source_filename": "招聘管理制度.docx",
        "storage_reference": "/uploads/tenant/doc/file.txt",
        "content_sha256": "a" * 64,
        "size_bytes": 128,
        "mime_type": "text/plain",
        "document_type": "policy",
        "source_format": "txt",
        "ingest_status": DocumentIngestStatus.ACCEPTED.value,
        "version": 1,
        "authority": "formal_candidate",
        "review_status": "needs_review",
        "sensitivity": "internal_demo",
        "created_by": "user-a",
        "metadata": {"normalized_runtime_input": True},
    }
    values.update(overrides)
    return DocumentRecord(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "unsafe",
    [
        "../secret.txt",
        "human_resources/../../secret.txt",
        "/absolute.txt",
        r"C:\\secret.txt",
        "human_resources//file.txt",
        "human_resources/./file.txt",
        "human_resources/",
        "",
    ],
)
def test_relative_path_normalization_rejects_unsafe_paths(unsafe: str) -> None:
    """A client path cannot escape or ambiguously select server storage."""

    with pytest.raises(ValueError):
        normalize_relative_path(unsafe)


def test_relative_path_normalization_uses_nfc_and_posix_separators() -> None:
    """Equivalent Unicode and Windows-style logical paths normalize consistently."""

    path = "human_resources/招聘/制度.txt"
    assert normalize_relative_path(path.replace("/", "\\")) == path


def test_uploaded_filename_rejects_path_segments() -> None:
    """An original filename is allowed only as a final basename."""

    assert normalize_uploaded_filename("制度.txt") == "制度.txt"
    with pytest.raises(ValueError):
        normalize_uploaded_filename("folder/制度.txt")


def test_catalog_persists_provenance_and_contains_no_blob_column(tmp_path: Path) -> None:
    """Catalog rows retain names and governance while storing metadata only."""

    catalog = _catalog(tmp_path)
    record = catalog.create_document(_record())

    loaded = catalog.get_document(record.doc_id, tenant_id="tenant-a")

    assert loaded == record
    assert loaded.uploaded_filename.endswith(".txt")
    assert loaded.provenance_source_filename.endswith(".docx")
    columns = {column["name"] for column in catalog.table_columns("documents")}
    assert "content" not in columns
    assert "blob" not in columns
    assert "storage_reference" in columns


def test_catalog_enforces_relative_path_version_uniqueness(tmp_path: Path) -> None:
    """Database uniqueness rejects an implicit overwrite while allowing versions."""

    catalog = _catalog(tmp_path)
    first = catalog.create_document(_record())

    with pytest.raises(CatalogConflictError):
        catalog.create_document(_record(doc_id=str(uuid4())))

    second_version = catalog.create_document(
        _record(doc_id=str(uuid4()), version=2, content_sha256="b" * 64)
    )
    assert second_version.version == 2
    assert len(catalog.list_documents(tenant_id="tenant-a")) == 2
    assert catalog.get_document(first.doc_id, tenant_id="tenant-b") is None


def test_catalog_filters_tenant_and_department_and_updates_status(tmp_path: Path) -> None:
    """Catalog reads and lifecycle updates never cross tenant scope."""

    catalog = _catalog(tmp_path)
    record = catalog.create_document(_record())
    updated = replace(
        record,
        ingest_status=DocumentIngestStatus.INGESTED.value,
        chunks_count=4,
        entities_count=3,
        relations_count=2,
    ).with_updates()
    catalog.update_document(updated)

    assert catalog.list_documents(tenant_id="tenant-b") == []
    rows = catalog.list_documents(
        tenant_id="tenant-a",
        company_id="tenant-a",
        department_id="human_resources",
    )
    assert rows == [updated]
    assert rows[0].chunks_count == 4
