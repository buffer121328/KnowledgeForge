"""Idempotent metadata/filesystem import for the normalized company Demo corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain.documents import (
    DepartmentRecord,
    DocumentIngestStatus,
    DocumentRecord,
    IngestDocumentInput,
    normalize_relative_path,
)
from infrastructure.documents.local_uploads import LocalUploadStorage

_EXPECTED_DEPARTMENT_COUNTS = {
    "administration": 10,
    "finance": 5,
    "human_resources": 21,
    "procurement_warehouse": 5,
}
_ALLOWED_DEPARTMENTS = set(_EXPECTED_DEPARTMENT_COUNTS)


class CompanyDemoImportError(ValueError):
    """Raised when the controlled normalized import manifest is incomplete or unsafe."""


@dataclass(frozen=True)
class CompanyDemoImportReport:
    """Summarize catalog records created or reused by one idempotent import."""

    document_count: int
    imported_count: int
    existing_count: int
    department_counts: dict[str, int]
    ingest_inputs: tuple[IngestDocumentInput, ...]


def _sha256(path: Path) -> str:
    """Hash one normalized runtime file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(record: dict[str, Any], field: str) -> str:
    """Return one required bounded manifest string."""

    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CompanyDemoImportError(f"company-demo manifest field is missing: {field}")
    return value.strip()


def _ingest_input(record: DocumentRecord) -> IngestDocumentInput:
    """Build a typed ingestion input from one accepted catalog record."""

    return IngestDocumentInput(
        doc_id=record.doc_id,
        tenant_id=record.tenant_id,
        company_id=record.company_id,
        department_id=record.department_id,
        file_path=record.storage_reference,
        uploaded_filename=record.uploaded_filename,
        display_name=record.display_name,
        provenance_source_filename=record.provenance_source_filename,
        relative_path=record.relative_path,
        folder_path=record.folder_path,
        content_sha256=record.content_sha256,
        version=record.version,
        authority=record.authority,
        review_status=record.review_status,
        sensitivity=record.sensitivity,
        external_source_id=record.external_source_id,
        metadata=record.metadata,
    )


def import_company_demo_catalog(
    root: str | Path,
    *,
    catalog: Any,
    upload_root: str | Path,
    tenant_id: str,
    company_id: str | None = None,
    created_by: str = "company-demo-import",
) -> CompanyDemoImportReport:
    """Import only manifest-declared normalized TXT files into filesystem/catalog storage."""

    corpus_root = Path(root).resolve()
    try:
        manifest = json.loads((corpus_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompanyDemoImportError("company-demo manifest is unavailable or invalid") from error
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(documents, list) or len(documents) != sum(_EXPECTED_DEPARTMENT_COUNTS.values()):
        raise CompanyDemoImportError(
            "company-demo import manifest document count does not match the frozen baseline"
        )
    effective_company = company_id or tenant_id
    storage = LocalUploadStorage(upload_root)
    imported = 0
    existing = 0
    counts: dict[str, int] = {}
    inputs: list[IngestDocumentInput] = []
    seen_ids: set[str] = set()

    for raw in documents:
        if not isinstance(raw, dict):
            raise CompanyDemoImportError("company-demo document entry must be an object")
        doc_id = _required(raw, "source_document_id")
        department_id = _required(raw, "department")
        if doc_id in seen_ids or department_id not in _ALLOWED_DEPARTMENTS:
            raise CompanyDemoImportError("company-demo contains a duplicate ID or unsupported department")
        seen_ids.add(doc_id)
        normalized_manifest_path = _required(raw, "normalized_path")
        normalized_path = normalize_relative_path(normalized_manifest_path)
        expected_prefix = f"normalized/{department_id}/"
        if not normalized_path.startswith(expected_prefix) or not normalized_path.endswith(".txt"):
            raise CompanyDemoImportError("company-demo import may read only normalized department TXT files")
        runtime_path = (corpus_root / normalized_path).resolve()
        normalized_root = (corpus_root / "normalized").resolve()
        if normalized_root not in runtime_path.parents or not runtime_path.is_file():
            raise CompanyDemoImportError("company-demo normalized runtime file is missing")
        expected_digest = _required(raw, "normalized_sha256")
        digest = _sha256(runtime_path)
        if digest != expected_digest:
            raise CompanyDemoImportError("company-demo normalized SHA-256 mismatch")
        for governance_field in ("authority", "status", "sensitivity"):
            _required(raw, governance_field)

        catalog.upsert_department(
            DepartmentRecord(
                department_id=department_id,
                tenant_id=tenant_id,
                company_id=effective_company,
                name=str(raw.get("department_label") or department_id),
                normalized_key=department_id,
            )
        )
        existing_record = catalog.get_document(doc_id, tenant_id=tenant_id)
        logical_path = normalize_relative_path(normalized_path.removeprefix("normalized/"))
        if existing_record is not None:
            if (
                existing_record.department_id != department_id
                or existing_record.normalized_relative_path != logical_path
                or existing_record.content_sha256 != digest
                or not Path(existing_record.storage_reference).is_file()
            ):
                raise CompanyDemoImportError("existing company-demo catalog identity conflicts with the manifest")
            existing += 1
            counts[department_id] = counts.get(department_id, 0) + 1
            inputs.append(_ingest_input(existing_record))
            continue

        staged = None
        record = None
        try:
            with runtime_path.open("rb") as source:
                staged = storage.stage(
                    tenant_id=tenant_id,
                    original_filename=runtime_path.name,
                    source=source,
                    max_file_bytes=50 * 1024 * 1024,
                    tenant_quota_bytes=2 * 1024 * 1024 * 1024,
                    chunk_bytes=1024 * 1024,
                )
            metadata = {
                "company_demo": True,
                "title": _required(raw, "title"),
                "original_source_format": _required(raw, "source_format"),
                "source_sha256": _required(raw, "sha256"),
                "normalized_path": normalized_path,
            }
            record = DocumentRecord(
                doc_id=doc_id,
                tenant_id=tenant_id,
                company_id=effective_company,
                department_id=department_id,
                folder_path=department_id,
                relative_path=logical_path,
                normalized_relative_path=logical_path,
                uploaded_filename=runtime_path.name,
                display_name=_required(raw, "title"),
                provenance_source_filename=_required(raw, "source_filename"),
                storage_reference=staged.path,
                content_sha256=digest,
                size_bytes=runtime_path.stat().st_size,
                mime_type="text/plain",
                document_type=_required(raw, "document_type"),
                source_format="txt",
                ingest_status=DocumentIngestStatus.STAGING.value,
                version=1,
                authority=_required(raw, "authority"),
                review_status=_required(raw, "status"),
                sensitivity=_required(raw, "sensitivity"),
                external_source_id=doc_id,
                created_by=created_by,
                metadata=metadata,
            )
            catalog.create_document(record)
            accepted = storage.promote(
                staged,
                department_id=department_id,
                doc_id=doc_id,
                original_filename=runtime_path.name,
            )
            record = record.with_updates(
                storage_reference=accepted,
                ingest_status=DocumentIngestStatus.ACCEPTED.value,
            )
            catalog.update_document(record)
        except Exception:
            if staged is not None and Path(staged.path).is_file():
                storage.discard(staged)
            if record is not None:
                catalog.remove_document(doc_id, tenant_id=tenant_id)
            raise
        imported += 1
        counts[department_id] = counts.get(department_id, 0) + 1
        inputs.append(_ingest_input(record))

    if counts != _EXPECTED_DEPARTMENT_COUNTS:
        raise CompanyDemoImportError(
            "company-demo department counts do not match the frozen baseline"
        )
    return CompanyDemoImportReport(
        document_count=len(inputs),
        imported_count=imported,
        existing_count=existing,
        department_counts=counts,
        ingest_inputs=tuple(inputs),
    )


__all__ = ["CompanyDemoImportError", "CompanyDemoImportReport", "import_company_demo_catalog"]
