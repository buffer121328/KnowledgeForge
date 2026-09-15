"""Acceptance tests for conservative legacy document catalog backfill."""

from pathlib import Path

from infrastructure.documents.catalog import SQLiteDocumentCatalogRepository
from infrastructure.documents.catalog_migration import backfill_legacy_documents


def test_legacy_backfill_marks_unresolved_name_and_never_overwrites_catalog(tmp_path: Path) -> None:
    """Opaque accepted basenames stay unresolved and a second run skips the catalog record."""

    catalog = SQLiteDocumentCatalogRepository(tmp_path / "catalog.sqlite3")
    vectors = [{
        "doc_id": "legacy-doc-1",
        "file_name": "a35f9e8b4f524bcf.txt",
        "source": "/accepted/a35f9e8b4f524bcf.txt",
        "doc_type": "txt",
        "chunks_count": 4,
    }]

    first = backfill_legacy_documents(catalog, vectors, tenant_id="tenant-a")
    record = catalog.get_document("legacy-doc-1", tenant_id="tenant-a")
    second = backfill_legacy_documents(catalog, [{**vectors[0], "uploaded_filename": "guessed.txt"}], tenant_id="tenant-a")

    assert first.created == 1 and first.unresolved_names == 1
    assert record is not None
    assert record.display_name == ""
    assert record.metadata["legacy_name_unresolved"] is True
    assert second.skipped == 1
    assert catalog.get_document("legacy-doc-1", tenant_id="tenant-a") == record
