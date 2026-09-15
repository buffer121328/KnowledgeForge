"""Asynchronous folder-member preparation for the document submission workflow."""

from __future__ import annotations

import hashlib
from pathlib import Path

from domain.documents import CatalogConflictError, DocumentIngestProcessingStep, DocumentIngestStatus, DocumentRecord
from infrastructure.documents.local_uploads import UploadPolicyError
from services.documents.contracts import (
    DocumentSubmissionDependencies,
    DocumentSubmissionError,
    FolderSubmissionRequest,
    FolderSubmissionResult,
)
from services.documents.helpers import (
    cleanup_rejection,
    document_input_from_record,
    folder_document_id,
    folder_result,
    metadata_text,
    progress_metadata,
)


class FolderSubmissionPreparer:
    """Stage, validate, catalog, and promote a prevalidated folder member."""

    def __init__(self, dependencies: DocumentSubmissionDependencies) -> None:
        self._dependencies = dependencies

    async def prepare(self, request: FolderSubmissionRequest) -> FolderSubmissionResult:
        """Prepare one accepted folder item without executing extraction or indexing."""

        if not request.tenant_id or request.tenant_id != request.org_id:
            raise DocumentSubmissionError(400, "upload_tenant_required", "上传需要有效租户")
        catalog = self._dependencies.catalog
        if catalog is None:
            raise DocumentSubmissionError(503, "document_catalog_unavailable", "文档目录服务暂不可用，请稍后重试。")

        storage = self._dependencies.storage_factory()
        original_name = Path(request.original_filename or "unknown").name
        doc_id = folder_document_id(self._dependencies, request.external_source_id)
        folder_path = Path(request.normalized_relative_path).parent.as_posix()
        if folder_path == ".":
            folder_path = ""
        metadata = dict(request.metadata)
        display_name = request.display_name or str(metadata.get("title") or Path(original_name).stem)
        provenance_name = request.provenance_source_filename or metadata_text(metadata, "source_filename")
        staged: object | None = None
        accepted_path: str | None = None
        record: DocumentRecord | None = None
        try:
            staged = storage.stage(
                tenant_id=request.tenant_id,
                original_filename=original_name,
                source=request.source,
                max_file_bytes=self._dependencies.settings.max_file_bytes,
                tenant_quota_bytes=self._dependencies.settings.tenant_quota_bytes,
                chunk_bytes=self._dependencies.settings.stream_chunk_bytes,
            )
            inspection = self._dependencies.upload_policy.validate(
                staged.path,
                original_name,
                request.content_type,
            )
            record = DocumentRecord(
                doc_id=doc_id,
                tenant_id=request.tenant_id,
                company_id=request.company_id,
                department_id=request.department_id,
                folder_path=folder_path,
                relative_path=request.relative_path,
                normalized_relative_path=request.normalized_relative_path,
                uploaded_filename=original_name,
                display_name=str(display_name)[:255],
                provenance_source_filename=provenance_name,
                storage_reference=staged.path,
                content_sha256=hashlib.sha256(Path(staged.path).read_bytes()).hexdigest(),
                size_bytes=inspection.size_bytes,
                mime_type=request.content_type or "application/octet-stream",
                document_type=metadata_text(metadata, "document_type") or inspection.parser_type,
                source_format=metadata_text(metadata, "source_format") or inspection.extension.lstrip("."),
                ingest_status=DocumentIngestStatus.STAGING.value,
                version=max(1, int(metadata.get("version", 1) or 1)),
                authority=metadata_text(metadata, "authority"),
                review_status=metadata_text(metadata, "status") or metadata_text(metadata, "review_status"),
                sensitivity=metadata_text(metadata, "sensitivity"),
                external_source_id=request.external_source_id,
                created_by=request.user_id,
                metadata=progress_metadata(
                    metadata,
                    upload_id=request.upload_id,
                    client_file_id=request.client_file_id,
                    processing_step=DocumentIngestProcessingStep.PARSE.value,
                ),
            )
            catalog.create_document(record)
            accepted_path = storage.promote(
                staged,
                department_id=request.department_id,
                doc_id=doc_id,
                original_filename=original_name,
            )
            record = record.with_updates(
                storage_reference=accepted_path,
                ingest_status=DocumentIngestStatus.ACCEPTED.value,
                metadata=progress_metadata(
                    record.metadata,
                    upload_id=request.upload_id,
                    client_file_id=request.client_file_id,
                    processing_step=DocumentIngestProcessingStep.PARSE.value,
                ),
            )
            catalog.update_document(record)
        except UploadPolicyError as error:
            cleanup_rejection(self._dependencies, storage, staged, accepted_path, record, request, error)
            try:
                self._dependencies.audit_denial(request, error.code, int(getattr(staged, "size_bytes", 0)))
            except Exception as audit_error:
                raise DocumentSubmissionError(503, "audit_unavailable", "安全审计服务暂不可用，请稍后重试。") from audit_error
            return folder_result(request, error_code=error.code, message=error.message)
        except (CatalogConflictError, ValueError) as error:
            cleanup_rejection(
                self._dependencies,
                storage,
                staged,
                accepted_path,
                record,
                request,
                error if isinstance(error, CatalogConflictError) else CatalogConflictError(),
            )
            conflict = isinstance(error, CatalogConflictError)
            return folder_result(
                request,
                error_code="document_catalog_conflict" if conflict else "upload_manifest_invalid",
                message="文档目录冲突" if conflict else "上传元数据无效",
            )

        assert record is not None
        return folder_result(
            request,
            record=record,
            document_input=document_input_from_record(record),
        )
