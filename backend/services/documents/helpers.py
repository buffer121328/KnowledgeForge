"""Shared record construction and cleanup helpers for document submission."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from domain.documents import CatalogConflictError, DocumentRecord, IngestDocumentInput
from infrastructure.documents.local_uploads import UploadPolicyError
from services.documents.contracts import (
    DocumentSubmissionDependencies,
    DocumentSubmissionRequest,
    FolderSubmissionRequest,
    FolderSubmissionResult,
    UploadStorage,
)


def progress_metadata(
    metadata: dict[str, Any] | None,
    *,
    upload_id: str,
    client_file_id: str,
    processing_step: str,
) -> dict[str, Any]:
    """Add the established upload correlation and processing-state metadata."""

    next_metadata = dict(metadata or {})
    if upload_id:
        next_metadata["_upload_batch_id"] = upload_id
    if client_file_id:
        next_metadata["_upload_client_file_id"] = client_file_id
    next_metadata["_processing_step"] = processing_step
    return next_metadata


def document_input_from_record(record: DocumentRecord) -> IngestDocumentInput:
    """Build the durable worker payload for one catalog document generation."""

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


def cleanup_rejection(
    dependencies: DocumentSubmissionDependencies,
    storage: UploadStorage,
    staged: object | None,
    accepted_path: str | None,
    catalog_record: DocumentRecord | None,
    request: DocumentSubmissionRequest | FolderSubmissionRequest,
    error: UploadPolicyError | CatalogConflictError,
) -> None:
    """Undo rejected staging, promotion, and catalog work in the original order."""

    staged_path = str(getattr(staged, "path", "")) if staged is not None else ""
    if staged_path and Path(staged_path).is_file():
        if (
            isinstance(error, UploadPolicyError)
            and error.code == "upload_malicious"
            and dependencies.settings.quarantine_retention_hours > 0
        ):
            storage.quarantine(staged)
        else:
            storage.discard(staged)
    if accepted_path:
        storage.delete(accepted_path, tenant_id=request.tenant_id)
    if dependencies.catalog is not None and catalog_record is not None:
        dependencies.catalog.remove_document(catalog_record.doc_id, tenant_id=request.tenant_id)


def folder_result(
    request: FolderSubmissionRequest,
    *,
    record: DocumentRecord | None = None,
    document_input: IngestDocumentInput | None = None,
    error_code: str = "",
    message: str = "",
) -> FolderSubmissionResult:
    """Build a transport-neutral folder-member result."""

    display_name = request.display_name or Path(request.original_filename).stem
    return FolderSubmissionResult(
        record=record,
        document_input=document_input,
        file_name=request.original_filename,
        client_file_id=request.client_file_id,
        relative_path=request.relative_path,
        department_id=request.department_id,
        display_name=display_name,
        provenance_source_filename=request.provenance_source_filename,
        error_code=error_code,
        message=message,
    )


def folder_document_id(dependencies: DocumentSubmissionDependencies, external_source_id: str) -> str:
    """Reuse a valid external UUID or allocate the existing server-owned ID."""

    candidate = (external_source_id or "").strip()
    if candidate:
        try:
            return str(UUID(candidate))
        except ValueError:
            pass
    return dependencies.allocate_document_id()


def metadata_text(metadata: dict[str, Any], key: str) -> str:
    """Return one bounded string field from trusted manifest metadata."""

    value = metadata.get(key, "")
    return str(value).strip()[:255] if value is not None else ""
