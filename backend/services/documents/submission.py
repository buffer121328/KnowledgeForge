"""HTTP-independent coordination for one synchronous document submission."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from domain.documents import (
    CatalogConflictError,
    DepartmentRecord,
    DocumentIngestProcessingStep,
    DocumentIngestStatus,
    DocumentRecord,
    normalize_relative_path,
)
from infrastructure.documents.local_uploads import UploadPolicyError
from shared.utils.logging import get_logger
from services.documents.folder_submission import FolderSubmissionPreparer
from services.documents.contracts import (
    DocumentSubmissionDenialRequest,
    DocumentSubmissionDependencies,
    DocumentSubmissionError,
    DocumentSubmissionRequest,
    DocumentSubmissionResult,
    DocumentSubmissionSettings,
    FolderSubmissionRequest,
    FolderSubmissionResult,
)
from services.documents.helpers import cleanup_rejection, document_input_from_record, progress_metadata

logger = get_logger(__name__)


class DocumentSubmissionCoordinator:
    """Coordinate one submission without importing FastAPI or application state."""

    def __init__(self, dependencies: DocumentSubmissionDependencies) -> None:
        self._dependencies = dependencies
        self._folder_preparer = FolderSubmissionPreparer(dependencies)

    def record_denial(
        self,
        request: DocumentSubmissionDenialRequest,
        *,
        code: str,
        size_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an upload denial through the provider-owned audit adapter."""

        if self._dependencies.audit_upload_denial is not None:
            self._dependencies.audit_upload_denial(request, code, size_bytes, dict(metadata or {}))

    async def prepare_folder_submission(self, request: FolderSubmissionRequest) -> FolderSubmissionResult:
        """Prepare one folder member using the shared submission dependencies."""

        return await self._folder_preparer.prepare(request)

    async def submit(self, request: DocumentSubmissionRequest) -> DocumentSubmissionResult:
        """Stage, validate, ingest, and observe one tenant-owned document."""

        if not request.tenant_id or request.tenant_id != request.org_id:
            raise DocumentSubmissionError(400, "upload_tenant_required", "上传需要有效租户")

        started = self._dependencies.clock()
        original_name = Path(request.original_filename or "unknown").name
        storage = self._dependencies.storage_factory()
        catalog = self._dependencies.catalog
        catalog_record: DocumentRecord | None = None
        allocated_doc_id = self._dependencies.allocate_document_id() if catalog is not None else ""
        staged: object | None = None
        accepted_path: str | None = None
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
            if catalog is not None:
                catalog.upsert_department(
                    DepartmentRecord(
                        department_id="general",
                        tenant_id=request.tenant_id,
                        company_id=request.tenant_id,
                        name="默认部门",
                        normalized_key="general",
                    )
                )
                catalog_record = DocumentRecord(
                    doc_id=allocated_doc_id,
                    tenant_id=request.tenant_id,
                    company_id=request.tenant_id,
                    department_id="general",
                    folder_path="",
                    relative_path=original_name,
                    normalized_relative_path=normalize_relative_path(original_name),
                    uploaded_filename=original_name,
                    display_name=Path(original_name).stem,
                    provenance_source_filename=original_name,
                    storage_reference=staged.path,
                    content_sha256=hashlib.sha256(Path(staged.path).read_bytes()).hexdigest(),
                    size_bytes=inspection.size_bytes,
                    mime_type=request.content_type or "application/octet-stream",
                    document_type=inspection.parser_type,
                    source_format=inspection.extension.lstrip("."),
                    ingest_status=DocumentIngestStatus.STAGING.value,
                    version=1,
                    created_by=request.user_id,
                    metadata=progress_metadata(
                        {},
                        upload_id=request.upload_id,
                        client_file_id=request.client_file_id,
                        processing_step=DocumentIngestProcessingStep.PARSE.value,
                    ),
                )
                catalog.create_document(catalog_record)
                accepted_path = storage.promote(
                    staged,
                    department_id="general",
                    doc_id=allocated_doc_id,
                    original_filename=original_name,
                )
                catalog_record = catalog_record.with_updates(
                    storage_reference=accepted_path,
                    ingest_status=DocumentIngestStatus.ACCEPTED.value,
                )
                catalog.update_document(catalog_record)
            else:
                accepted_path = storage.promote(staged)
        except (UploadPolicyError, CatalogConflictError) as error:
            cleanup_rejection(self._dependencies, storage, staged, accepted_path, catalog_record, request, error)
            policy_error = (
                error
                if isinstance(error, UploadPolicyError)
                else UploadPolicyError(
                    "document_catalog_conflict",
                    "文档目录路径或版本冲突",
                    status_code=409,
                )
            )
            try:
                self._dependencies.audit_denial(
                    request,
                    policy_error.code,
                    staged.size_bytes if staged else 0,
                )
            except Exception as audit_error:
                logger.error("doc_upload_denial_audit_failed", error_type=type(audit_error).__name__)
                raise DocumentSubmissionError(503, "audit_unavailable", "安全审计服务暂不可用，请稍后重试。") from audit_error
            raise DocumentSubmissionError(policy_error.status_code, policy_error.code, policy_error.message) from error

        if self._dependencies.ingest_workflow is None:
            storage.delete(accepted_path, tenant_id=request.tenant_id)
            if catalog is not None and catalog_record is not None:
                catalog.remove_document(catalog_record.doc_id, tenant_id=request.tenant_id)
            raise DocumentSubmissionError(503, "document_ingest_unavailable", "文档入库服务暂不可用，请稍后重试。")

        workflow_payload: dict[str, Any] = {"file_paths": [accepted_path], "tenant_id": request.tenant_id}
        if catalog_record is not None:
            workflow_payload["document_inputs"] = [document_input_from_record(catalog_record)]
        try:
            if catalog is not None and catalog_record is not None:
                catalog.update_document(
                    catalog_record.with_updates(
                        ingest_status=DocumentIngestStatus.PROCESSING.value,
                        metadata=progress_metadata(
                            catalog_record.metadata,
                            upload_id=request.upload_id,
                            client_file_id=request.client_file_id,
                            processing_step=DocumentIngestProcessingStep.PARSE.value,
                        ),
                    )
                )

            async def progress_callback(
                processing_step: str,
                step_index: int,
                step_total: int,
                step_label: str,
            ) -> None:
                if catalog is None or catalog_record is None:
                    return
                current = catalog.get_document(catalog_record.doc_id, tenant_id=request.tenant_id)
                if current is None:
                    return
                metadata = progress_metadata(
                    current.metadata,
                    upload_id=request.upload_id,
                    client_file_id=request.client_file_id,
                    processing_step=processing_step,
                )
                metadata["_processing_step_index"] = step_index
                metadata["_processing_step_total"] = step_total
                metadata["_processing_step_label"] = step_label
                catalog.update_document(current.with_updates(metadata=metadata))

            workflow_payload["progress_callback"] = progress_callback
            workflow_result = await self._dependencies.ingest_workflow.ainvoke(workflow_payload)
        except Exception as error:
            if catalog is not None and catalog_record is not None:
                current_record = catalog.get_document(catalog_record.doc_id, tenant_id=request.tenant_id) or catalog_record
                metadata = dict(current_record.metadata)
                metadata.setdefault("_ingest_stage_index", 3)
                catalog.update_document(
                    current_record.with_updates(
                        ingest_status=DocumentIngestStatus.FAILED.value,
                        error_code="document_ingest_failed",
                        metadata=metadata,
                    )
                )
            else:
                storage.delete(accepted_path, tenant_id=request.tenant_id)
            logger.error("doc_upload_failed", error_type=type(error).__name__, tenant_id=request.tenant_id)
            try:
                self._dependencies.audit_failure(request, "document_ingest_failed")
            except Exception as audit_error:
                raise DocumentSubmissionError(503, "audit_unavailable", "安全审计服务暂不可用，请稍后重试。") from audit_error
            raise DocumentSubmissionError(500, "document_ingest_failed", "文档入库失败，请稍后重试。") from error

        chunks = workflow_result.get("chunks", [])
        total_entities = int(workflow_result.get("entities_stored", 0))
        total_relations = int(workflow_result.get("relations_stored", 0))
        doc_type = chunks[0].doc_type.value if chunks else inspection.parser_type
        document_id = (
            catalog_record.doc_id
            if catalog_record is not None
            else chunks[0].doc_id
            if chunks
            else hashlib.sha256(accepted_path.encode()).hexdigest()[:16]
        )
        result = DocumentSubmissionResult(
            file_name=original_name,
            chunks_count=len(chunks),
            entities_count=total_entities,
            relations_count=total_relations,
            doc_id=document_id,
            client_file_id=request.client_file_id,
        )
        self._dependencies.record_metrics(doc_type, self._dependencies.clock() - started)
        if catalog is not None and catalog_record is not None:
            current_record = catalog.get_document(catalog_record.doc_id, tenant_id=request.tenant_id) or catalog_record
            catalog.update_document(
                current_record.with_updates(
                    ingest_status=DocumentIngestStatus.INGESTED.value,
                    chunks_count=result.chunks_count,
                    entities_count=result.entities_count,
                    relations_count=result.relations_count,
                    error_code="",
                )
            )

        try:
            self._dependencies.audit_success(request, result, inspection)
        except Exception as audit_error:
            if catalog is not None and catalog_record is not None:
                current_record = catalog.get_document(catalog_record.doc_id, tenant_id=request.tenant_id) or catalog_record
                catalog.update_document(
                    current_record.with_updates(
                        ingest_status=DocumentIngestStatus.FAILED.value,
                        error_code="audit_unavailable",
                    )
                )
            else:
                storage.delete(accepted_path, tenant_id=request.tenant_id)
            raise DocumentSubmissionError(503, "audit_unavailable", "安全审计服务暂不可用，请稍后重试。") from audit_error
        await self._dependencies.trigger_webhook(request, result)
        logger.info(
            "doc_uploaded",
            document_id=result.doc_id,
            chunks_count=result.chunks_count,
            entities_count=result.entities_count,
            tenant_id=request.tenant_id,
        )
        return result

__all__ = [
    "DocumentSubmissionCoordinator",
    "DocumentSubmissionDenialRequest",
    "DocumentSubmissionDependencies",
    "DocumentSubmissionError",
    "DocumentSubmissionRequest",
    "DocumentSubmissionResult",
    "DocumentSubmissionSettings",
    "FolderSubmissionRequest",
    "FolderSubmissionResult",
]
