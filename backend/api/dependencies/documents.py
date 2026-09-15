"""Dependency provider for the HTTP-independent document submission coordinator."""

from __future__ import annotations

from uuid import uuid4

from fastapi import Request

from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service
from infrastructure.documents.local_uploads import LocalUploadStorage, UploadInspection, UploadValidationPolicy
from infrastructure.webhooks.service import WebhookEvent, get_webhook_service
from shared.config import settings
from shared.utils.metrics import doc_ingest_duration, doc_ingest_total
from infrastructure.tasks.task_registry import get_task_registry
from shared.utils.task_ids import new_task_id
from services.documents.lifecycle import DocumentLifecycleCoordinator, DocumentLifecycleDependencies
from services.documents.submission import (
    DocumentSubmissionCoordinator,
    DocumentSubmissionDenialRequest,
    DocumentSubmissionDependencies,
    DocumentSubmissionRequest,
    DocumentSubmissionResult,
    DocumentSubmissionSettings,
)


def _upload_policy() -> UploadValidationPolicy:
    """Build the existing bounded upload-policy adapter for a request."""

    return UploadValidationPolicy(
        max_archive_entries=settings.upload_max_archive_entries,
        max_archive_uncompressed_bytes=settings.upload_max_archive_uncompressed_bytes,
        max_archive_compression_ratio=settings.upload_max_archive_compression_ratio,
        max_pdf_pages=settings.upload_max_pdf_pages,
        max_image_pixels=settings.upload_max_image_pixels,
        max_spreadsheet_rows=settings.upload_max_spreadsheet_rows,
        max_text_characters=settings.upload_max_text_characters,
        require_external_scanner=settings.upload_require_external_scanner,
    )


def get_document_submission(request: Request) -> DocumentSubmissionCoordinator:
    """Compose one submission coordinator from established application facades."""

    audit_service = get_audit_service()
    webhook_service = get_webhook_service()
    catalog = getattr(request.app.state, "document_catalog", None)
    workflows = getattr(request.app.state, "workflows", {})

    def audit_denial(submission: DocumentSubmissionRequest, code: str, size_bytes: int) -> None:
        audit_service.log(
            user_id=submission.user_id,
            action=AuditAction.DOC_UPLOAD,
            resource="doc/upload",
            result=AuditResult.DENIED,
            username=submission.username,
            org_id=submission.org_id,
            metadata={"code": code, "size_bytes": size_bytes},
        )

    def audit_upload_denial(
        submission: DocumentSubmissionDenialRequest,
        code: str,
        size_bytes: int,
        metadata: dict,
    ) -> None:
        audit_service.log(
            user_id=submission.user_id,
            action=AuditAction.DOC_UPLOAD,
            resource="doc/upload",
            result=AuditResult.DENIED,
            username=submission.username,
            org_id=submission.org_id,
            metadata=metadata,
        )

    def audit_failure(submission: DocumentSubmissionRequest, code: str) -> None:
        audit_service.log(
            user_id=submission.user_id,
            action=AuditAction.DOC_UPLOAD,
            resource="doc/upload",
            result=AuditResult.FAILURE,
            username=submission.username,
            org_id=submission.org_id,
            metadata={"code": code},
        )

    def audit_success(
        submission: DocumentSubmissionRequest,
        result: DocumentSubmissionResult,
        inspection: UploadInspection,
    ) -> None:
        audit_service.log(
            user_id=submission.user_id,
            action=AuditAction.DOC_UPLOAD,
            resource=f"doc/{result.doc_id}",
            result=AuditResult.SUCCESS,
            username=submission.username,
            org_id=submission.org_id,
            metadata={
                "chunks_count": result.chunks_count,
                "entities_count": result.entities_count,
                "size_bytes": inspection.size_bytes,
                "parser_type": inspection.parser_type,
            },
        )

    async def trigger_webhook(
        submission: DocumentSubmissionRequest,
        result: DocumentSubmissionResult,
    ) -> None:
        await webhook_service.trigger(
            WebhookEvent.DOC_INGESTED,
            {
                "doc_id": result.doc_id,
                "file_name": result.file_name,
                "tenant_id": submission.tenant_id,
                "chunks_count": result.chunks_count,
                "entities_count": result.entities_count,
                "user_id": submission.user_id,
            },
            org_id=submission.org_id,
        )

    def record_metrics(doc_type: str, duration_seconds: float) -> None:
        doc_ingest_total.labels(doc_type=doc_type).inc()
        doc_ingest_duration.labels(doc_type=doc_type).observe(duration_seconds)

    return DocumentSubmissionCoordinator(
        DocumentSubmissionDependencies(
            storage_factory=lambda: LocalUploadStorage(settings.upload_dir),
            upload_policy=_upload_policy(),
            catalog=catalog,
            ingest_workflow=workflows.get("ingest"),
            settings=DocumentSubmissionSettings(
                max_file_bytes=settings.upload_max_file_bytes,
                tenant_quota_bytes=settings.upload_tenant_quota_bytes,
                stream_chunk_bytes=settings.upload_stream_chunk_bytes,
                quarantine_retention_hours=settings.upload_quarantine_retention_hours,
            ),
            audit_denial=audit_denial,
            audit_failure=audit_failure,
            audit_upload_denial=audit_upload_denial,
            audit_success=audit_success,
            trigger_webhook=trigger_webhook,
            record_metrics=record_metrics,
            allocate_document_id=lambda: str(uuid4()),
        )
    )


def get_document_lifecycle(request: Request):
    """Compose the write-side document lifecycle coordinator for HTTP routes."""

    from infrastructure.tasks.celery_ingest_tasks import batch_ingest_task
    from infrastructure.documents.local_uploads import LocalUploadStorage

    audit_service = get_audit_service()
    webhook_service = get_webhook_service()
    catalog = getattr(request.app.state, "document_catalog", None)
    workflows = getattr(request.app.state, "workflows", {})

    def publish_folder_task(
        *,
        file_paths: list[str],
        tenant_id: str,
        user_id: str,
        document_inputs: list[dict],
        actor_username: str,
        task_id: str,
    ) -> None:
        batch_ingest_task.apply_async(
            args=[file_paths, tenant_id, user_id, document_inputs, actor_username],
            task_id=task_id,
            queue="parser",
        )

    def record_retry_failure(actor, doc_id: str, code: str) -> None:
        audit_service.log(
            user_id=actor.user_id,
            action=AuditAction.DOC_UPLOAD,
            resource=f"doc/{doc_id}",
            result=AuditResult.FAILURE,
            username=actor.username,
            org_id=actor.org_id,
            metadata={"retry": True, "code": code},
        )

    def record_retry_success(actor, result) -> None:
        audit_service.log(
            user_id=actor.user_id,
            action=AuditAction.DOC_UPLOAD,
            resource=f"doc/{result.record.doc_id}",
            result=AuditResult.SUCCESS,
            username=actor.username,
            org_id=actor.org_id,
            metadata={
                "retry": True,
                "chunks_count": result.chunks_count,
                "entities_count": result.entities_count,
                "relations_count": result.relations_count,
            },
        )

    async def trigger_retry_webhook(actor, result) -> None:
        await webhook_service.trigger(
            WebhookEvent.DOC_INGESTED,
            {
                "doc_id": result.record.doc_id,
                "file_name": result.record.uploaded_filename,
                "tenant_id": result.record.tenant_id,
                "department_id": result.record.department_id,
                "chunks_count": result.chunks_count,
                "entities_count": result.entities_count,
                "user_id": actor.user_id,
                "retry": True,
            },
            org_id=actor.org_id,
        )

    def record_delete_audit(actor, result, ip: str, user_agent: str) -> None:
        audit_service.log(
            user_id=actor.user_id,
            action=AuditAction.DOC_DELETE,
            resource=f"doc/{result.doc_id}",
            result=AuditResult.SUCCESS,
            username=actor.username,
            ip=ip,
            user_agent=user_agent,
            org_id=actor.org_id,
            metadata={
                "vectors_deleted": result.vectors_deleted,
                "sparse_deleted": result.sparse_deleted,
                "entities_deleted": result.entities_deleted,
                "file_deleted": result.file_deleted,
            },
        )

    async def trigger_delete_webhook(actor, result) -> None:
        await webhook_service.trigger(
            WebhookEvent.DOC_DELETED,
            {
                "doc_id": result.doc_id,
                "vectors_deleted": result.vectors_deleted,
                "entities_deleted": result.entities_deleted,
                "user_id": actor.user_id,
                "tenant_id": actor.org_id,
            },
            org_id=actor.org_id,
        )

    async def advance_revision(tenant_id: str) -> None:
        await request.app.state.knowledge_revision.advance(tenant_id)

    def record_metrics(doc_type: str, duration_seconds: float) -> None:
        doc_ingest_total.labels(doc_type=doc_type).inc()
        doc_ingest_duration.labels(doc_type=doc_type).observe(duration_seconds)

    return DocumentLifecycleCoordinator(
        DocumentLifecycleDependencies(
            catalog=catalog,
            storage_factory=lambda: LocalUploadStorage(settings.upload_dir),
            ingest_workflow=workflows.get("ingest"),
            task_registry_factory=get_task_registry,
            new_task_id=new_task_id,
            publish_folder_task=publish_folder_task,
            vector_store=getattr(request.app.state, "vector_store", None),
            knowledge_graph=getattr(request.app.state, "knowledge_graph", None),
            sparse_index=getattr(request.app.state, "sparse_index", None),
            record_retry_failure=record_retry_failure,
            record_retry_success=record_retry_success,
            trigger_retry_webhook=trigger_retry_webhook,
            record_delete_audit=record_delete_audit,
            trigger_delete_webhook=trigger_delete_webhook,
            advance_revision=advance_revision,
            record_metrics=record_metrics,
        )
    )
