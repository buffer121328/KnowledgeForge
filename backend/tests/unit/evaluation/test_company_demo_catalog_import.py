"""Acceptance tests for controlled company-demo normalized catalog import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from evaluation.company_demo.corpus_import import import_company_demo_catalog
from infrastructure.documents.catalog import SQLiteDocumentCatalogRepository


DEPARTMENT_COUNTS = {
    "administration": 10,
    "finance": 5,
    "human_resources": 21,
    "procurement_warehouse": 5,
}


def _write_company_demo_fixture(root: Path) -> Path:
    """Create a deterministic 41-file normalized import fixture with source provenance."""

    documents: list[dict[str, object]] = []
    for department, count in DEPARTMENT_COUNTS.items():
        for index in range(count):
            basename = f"{department}-{index + 1:02d}"
            relative_path = Path("normalized") / department / f"{basename}.txt"
            runtime_path = root / relative_path
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            content = f"{department} demo policy {index + 1}\n".encode()
            runtime_path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            documents.append(
                {
                    "source_document_id": str(uuid5(NAMESPACE_URL, relative_path.as_posix())),
                    "department": department,
                    "department_label": department,
                    "document_type": "policy",
                    "title": basename,
                    "source_filename": f"{basename}.docx",
                    "source_format": "docx",
                    "sha256": digest,
                    "normalized_path": relative_path.as_posix(),
                    "normalized_sha256": digest,
                    "authority": "unverified_reference",
                    "status": "needs_review",
                    "sensitivity": "internal_demo",
                }
            )
    (root / "corpus_manifest.json").write_text(
        json.dumps({"documents": documents}, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def test_company_demo_import_is_idempotent_and_uses_only_normalized_txt(tmp_path: Path) -> None:
    """The controlled import creates 41 filesystem/catalog records once with 21/5/5/10 counts."""

    corpus_root = _write_company_demo_fixture(tmp_path / "company-demo")
    catalog = SQLiteDocumentCatalogRepository(tmp_path / "catalog.sqlite3")
    upload_root = tmp_path / "uploads"

    first = import_company_demo_catalog(
        corpus_root,
        catalog=catalog,
        upload_root=upload_root,
        tenant_id="company-demo-tenant",
    )
    second = import_company_demo_catalog(
        corpus_root,
        catalog=catalog,
        upload_root=upload_root,
        tenant_id="company-demo-tenant",
    )

    assert first.document_count == 41
    assert first.imported_count == 41
    assert first.department_counts == {
        "administration": 10,
        "finance": 5,
        "human_resources": 21,
        "procurement_warehouse": 5,
    }
    assert second.imported_count == 0
    assert second.existing_count == 41
    records = catalog.list_documents(tenant_id="company-demo-tenant")
    assert len(records) == 41
    assert len(list((upload_root / "tenants").rglob("*.txt"))) == 41
    assert not list((upload_root / "tenants").rglob("*.docx"))
    assert all("/normalized/" not in record.storage_reference for record in records)


def test_company_demo_import_preserves_runtime_and_source_provenance(tmp_path: Path) -> None:
    """TXT runtime filenames stay distinct from DOCX provenance and governance fields."""

    corpus_root = _write_company_demo_fixture(tmp_path / "company-demo")
    catalog = SQLiteDocumentCatalogRepository(tmp_path / "catalog.sqlite3")
    report = import_company_demo_catalog(
        corpus_root,
        catalog=catalog,
        upload_root=tmp_path / "uploads",
        tenant_id="company-demo-tenant",
    )

    record = catalog.get_document(report.ingest_inputs[0].doc_id, tenant_id="company-demo-tenant")
    assert record is not None
    assert record.uploaded_filename.endswith(".txt")
    assert record.provenance_source_filename.endswith(".docx")
    assert record.source_format == "txt"
    assert record.authority == "unverified_reference"
    assert record.review_status == "needs_review"
    assert record.sensitivity == "internal_demo"
    assert record.metadata["original_source_format"] == "docx"
