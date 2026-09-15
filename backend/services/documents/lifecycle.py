"""HTTP- and Celery-independent coordination for accepted document lifecycles."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Protocol

from domain.documents import DocumentIngestStatus, DocumentRecord, IngestDocumentInput
from domain.identity import UserContext, UserRole
from shared.utils.logging import get_logger

logger = get_logger(__name__)


from services.documents.lifecycle_completion import (
    ArtifactCleanup,
    LifecycleCatalog,
    finalize_document_failure,
    finalize_document_success,
)


class LifecycleStorage(Protocol):
    """Tenant-safe source-file operations used by retry and delete coordination."""

    def accepted_reference(self, path: str, tenant_id: str) -> str: ...

    def resolve_accepted(self, reference: str, tenant_id: str) -> str: ...

    def delete(self, path: str, tenant_id: str | None = None) -> bool: ...


class VectorStore(Protocol):
    """The document-scoped vector operations used by lifecycle coordination."""

    async def get_chunks_by_doc_id(self, doc_id: str, *, tenant_id: str | None = None) -> list[dict[str, Any]]: ...

    async def delete_by_doc_id(self, doc_id: str, *, tenant_id: str | None = None) -> int: ...


class SparseIndex(Protocol):
    """The document-scoped sparse-index operation used by delete coordination."""

    async def delete_by_doc_id(self, doc_id: str, *, tenant_id: str | None = None) -> int: ...


class KnowledgeGraph(Protocol):
    """The graph cleanup operations used by delete coordination."""

    async def delete_document_evidence(self, doc_id: str, tenant_id: str) -> int: ...

    async def delete_by_source(self, source: str, *, tenant_id: str | None = None) -> int: ...


class IngestWorkflow(Protocol):
    """The existing in-process workflow used by explicit document retry."""

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class IngestRunner(Protocol):
    """The worker-owned parser/extraction runner for an accepted source path."""

    async def __call__(
        self,
        file_path: str,
        *,
        tenant_id: str,
        document_input: IngestDocumentInput | None = None,
        progress_callback: Callable[[str, int, int, str], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]: ...


class TaskRegistry(Protocol):
    """Task reservation state needed before publishing one folder batch."""

    def reserve(
        self,
        task_id: str,
        *,
        org_id: str,
        actor_id: str,
        kind: str,
        file_reference: str = "",
    ) -> object: ...

    def update_state(self, task_id: str, state: str) -> object: ...


class FolderTaskPublisher(Protocol):
    """A framework-neutral publisher for the existing batch Celery task."""

    def __call__(
        self,
        *,
        file_paths: list[str],
        tenant_id: str,
        user_id: str,
        document_inputs: list[dict[str, Any]],
        actor_username: str,
        task_id: str,
    ) -> None: ...


CompletionSideEffects = Callable[[IngestDocumentInput, dict[str, Any], str, str], Awaitable[None]]
RetryAudit = Callable[[UserContext, str, str], None]
RetrySuccessAudit = Callable[[UserContext, "DocumentRetryResult"], None]
RetryWebhook = Callable[[UserContext, "DocumentRetryResult"], Awaitable[None]]
DeleteAudit = Callable[[UserContext, "DocumentDeleteResult", str, str], None]
DeleteWebhook = Callable[[UserContext, "DocumentDeleteResult"], Awaitable[None]]
RevisionAdvance = Callable[[str], Awaitable[None]]
MetricsRecorder = Callable[[str, float], None]


@dataclass(frozen=True)
class AcceptedDocument:
    """One catalog-owned accepted document ready for a background task."""

    record: DocumentRecord
    document_input: IngestDocumentInput
    client_file_id: str = ""


@dataclass(frozen=True)
class FolderPublicationRequest:
    """Trusted folder batch prepared before a durable task is published."""

    tenant_id: str
    actor: UserContext
    documents: list[AcceptedDocument]


@dataclass(frozen=True)
class FolderPublicationResult:
    """The active document IDs and durable task ID after publication."""

    task_id: str
    active_document_ids: frozenset[str]


@dataclass(frozen=True)
class AcceptedDocumentProcessingRequest:
    """Serialized worker input plus the trusted actor that submitted it."""

    file_path: str
    tenant_id: str
    user_id: str
    username: str = ""
    document_input: IngestDocumentInput | None = None


@dataclass(frozen=True)
class AcceptedDocumentProcessingResult:
    """One batch item outcome without exposing a worker exception to callers."""

    result: dict[str, Any] | None
    status: str
    error: str | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class DocumentRetryRequest:
    """Trusted actor context for retrying one durable failed document."""

    doc_id: str
    tenant_id: str
    actor: UserContext


@dataclass(frozen=True)
class DocumentRetryResult:
    """HTTP-neutral representation of a successful synchronous retry."""

    record: DocumentRecord
    chunks_count: int
    entities_count: int
    relations_count: int


@dataclass(frozen=True)
class DocumentDeleteRequest:
    """Trusted actor and request metadata for deleting one tenant document."""

    doc_id: str
    tenant_id: str
    actor: UserContext
    ip: str = ""
    user_agent: str = ""


@dataclass(frozen=True)
class DocumentDeleteResult:
    """HTTP-neutral result of the established document deletion order."""

    doc_id: str
    vectors_deleted: int
    sparse_deleted: int
    entities_deleted: int
    file_deleted: bool


class DocumentLifecycleError(RuntimeError):
    """A safe lifecycle failure that the HTTP boundary maps to its existing envelope."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True)
class DocumentLifecycleDependencies:
    """Explicit collaborators for task publication, completion, retry, and deletion."""

    catalog: LifecycleCatalog | None = None
    storage_factory: Callable[[], LifecycleStorage] | None = None
    ingest_workflow: IngestWorkflow | None = None
    task_registry_factory: Callable[[], TaskRegistry] | None = None
    new_task_id: Callable[[str], str] | None = None
    publish_folder_task: FolderTaskPublisher | None = None
    cleanup_artifacts: ArtifactCleanup | None = None
    completion_side_effects: CompletionSideEffects | None = None
    vector_store: VectorStore | None = None
    knowledge_graph: KnowledgeGraph | None = None
    sparse_index: SparseIndex | None = None
    record_retry_failure: RetryAudit | None = None
    record_retry_success: RetrySuccessAudit | None = None
    trigger_retry_webhook: RetryWebhook | None = None
    record_delete_audit: DeleteAudit | None = None
    trigger_delete_webhook: DeleteWebhook | None = None
    advance_revision: RevisionAdvance | None = None
    record_metrics: MetricsRecorder | None = None
    clock: Callable[[], float] = time.time


def document_input_from_record(record: DocumentRecord) -> IngestDocumentInput:
    """Build the stable worker/workflow payload for one catalog generation."""

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


class DocumentLifecycleCoordinator:
    """Coordinate lifecycle decisions without importing FastAPI or Celery."""

    def __init__(self, dependencies: DocumentLifecycleDependencies) -> None:
        self._dependencies = dependencies

    @staticmethod
    def can_manage_document(user: UserContext, department_id: str | None) -> bool:
        """Return whether one actor may mutate the supplied department document."""

        if user.role == UserRole.ORGANIZATION_ADMIN:
            return True
        if user.role == UserRole.ADMIN or user.is_department_manager:
            return bool(user.department_id and department_id and user.department_id == department_id)
        return False

    def _catalog(self) -> LifecycleCatalog:
        if self._dependencies.catalog is None:
            raise DocumentLifecycleError(404, "document_not_found", "文档不存在或不可访问")
        return self._dependencies.catalog

    def _cleanup(self) -> ArtifactCleanup:
        if self._dependencies.cleanup_artifacts is None:
            async def no_cleanup(_doc_id: str, _tenant_id: str) -> None:
                return None
            return no_cleanup
        return self._dependencies.cleanup_artifacts

    async def publish_folder(self, request: FolderPublicationRequest) -> FolderPublicationResult:
        """Reserve and publish one prepared folder batch while preserving existing states."""

        if not request.documents:
            raise DocumentLifecycleError(503, "folder_ingest_publish_failed", "文件已接收，但后台入库任务暂无法发布，请稍后重试。")
        if (
            self._dependencies.task_registry_factory is None
            or self._dependencies.new_task_id is None
            or self._dependencies.publish_folder_task is None
        ):
            raise DocumentLifecycleError(503, "folder_ingest_publish_failed", "文件已接收，但后台入库任务暂无法发布，请稍后重试。")

        catalog = self._catalog()
        task_id = self._dependencies.new_task_id(request.actor.org_id)
        registry: TaskRegistry | None = None
        try:
            registry = self._dependencies.task_registry_factory()
            registry.reserve(
                task_id,
                org_id=request.actor.org_id,
                actor_id=request.actor.user_id,
                kind="folder_ingest",
                file_reference=f"folder:{len(request.documents)}",
            )
            active: list[AcceptedDocument] = []
            for item in request.documents:
                processing = item.record.with_updates(
                    ingest_status=DocumentIngestStatus.PROCESSING.value,
                    error_code="",
                )
                if catalog.update_document_if_status(
                    processing,
                    allowed_statuses={DocumentIngestStatus.ACCEPTED.value},
                ):
                    active.append(
                        AcceptedDocument(
                            record=processing,
                            document_input=item.document_input,
                            client_file_id=item.client_file_id,
                        )
                    )
            if not active:
                registry.update_state(task_id, "publish_failed")
                raise RuntimeError("no active folder members remain")
            self._dependencies.publish_folder_task(
                file_paths=[item.record.storage_reference for item in active],
                tenant_id=request.tenant_id,
                user_id=request.actor.user_id,
                document_inputs=[asdict(item.document_input) for item in active],
                actor_username=request.actor.username,
                task_id=task_id,
            )
            registry.update_state(task_id, "queued")
            return FolderPublicationResult(
                task_id=task_id,
                active_document_ids=frozenset(item.record.doc_id for item in active),
            )
        except Exception as error:
            for item in request.documents:
                failed = item.record.with_updates(
                    ingest_status=DocumentIngestStatus.FAILED.value,
                    error_code="folder_ingest_publish_failed",
                )
                catalog.update_document_if_status(
                    failed,
                    allowed_statuses={
                        DocumentIngestStatus.ACCEPTED.value,
                        DocumentIngestStatus.PROCESSING.value,
                    },
                )
            if registry is not None:
                try:
                    registry.update_state(task_id, "publish_failed")
                except (ValueError, RuntimeError):
                    pass
            logger.warning(
                "folder_ingest_publish_failed",
                task_id=task_id,
                tenant_id=request.tenant_id,
                error_type=type(error).__name__,
            )
            raise DocumentLifecycleError(
                503,
                "folder_ingest_publish_failed",
                "文件已接收，但后台入库任务暂无法发布，请稍后重试。",
            ) from error

    async def process_accepted_document(
        self,
        request: AcceptedDocumentProcessingRequest,
        *,
        runner: IngestRunner,
    ) -> AcceptedDocumentProcessingResult:
        """Run one worker item, maintaining progress and active-generation semantics."""

        catalog = self._dependencies.catalog if request.document_input is not None else None
        cleanup = self._cleanup()

        async def progress_callback(step: str, step_index: int, step_total: int, step_label: str) -> None:
            if catalog is None or request.document_input is None:
                return
            current = catalog.get_document(request.document_input.doc_id, tenant_id=request.tenant_id)
            if current is None:
                return
            metadata = dict(current.metadata)
            metadata["_processing_step"] = step
            metadata["_processing_step_index"] = step_index
            metadata["_processing_step_total"] = step_total
            metadata["_processing_step_label"] = step_label
            catalog.update_document(current.with_updates(metadata=metadata))

        try:
            result = await runner(
                request.file_path,
                tenant_id=request.tenant_id,
                document_input=request.document_input,
                progress_callback=progress_callback if request.document_input is not None else None,
            )
            if self._dependencies.advance_revision is not None:
                await self._dependencies.advance_revision(request.tenant_id)
        except Exception as error:
            if catalog is not None and request.document_input is not None:
                try:
                    await finalize_document_failure(catalog, request.document_input, cleanup=cleanup)
                except Exception as finalize_error:
                    logger.error(
                        "batch_ingest_failure_compensation_failed",
                        error_type=type(finalize_error).__name__,
                    )
            return AcceptedDocumentProcessingResult(
                result=None,
                status="failed",
                error="task_execution_failed",
                error_type=type(error).__name__,
            )

        committed = True
        side_effect_error: str | None = None
        if catalog is not None and request.document_input is not None:
            committed = await finalize_document_success(catalog, request.document_input, result, cleanup=cleanup)
            if committed and self._dependencies.completion_side_effects is not None:
                try:
                    await self._dependencies.completion_side_effects(
                        request.document_input,
                        result,
                        request.user_id,
                        request.username,
                    )
                except Exception as error:
                    side_effect_error = "completion_side_effect_failed"
                    logger.error(
                        "batch_ingest_completion_side_effect_failed",
                        error_type=type(error).__name__,
                    )
        return AcceptedDocumentProcessingResult(
            result=result,
            status="completed" if committed else "discarded",
            error=side_effect_error,
        )

    async def recover_interrupted_document(
        self,
        document_input: IngestDocumentInput,
        *,
        cleanup_partial_artifacts: bool,
    ) -> bool:
        """Make one orphaned processing generation safely retryable.

        A serial batch knows whether its member entered the runner. Only that
        interrupted member may have partial artifacts; members that never
        started must receive a guarded catalog transition without cleanup.
        """

        catalog = self._catalog()
        current = catalog.get_document(document_input.doc_id, tenant_id=document_input.tenant_id)
        if (
            current is None
            or current.version != document_input.version
            or current.ingest_status != DocumentIngestStatus.PROCESSING.value
        ):
            return False
        if cleanup_partial_artifacts:
            return await finalize_document_failure(
                catalog,
                document_input,
                cleanup=self._cleanup(),
            )
        failed = current.with_updates(
            ingest_status=DocumentIngestStatus.FAILED.value,
            error_code="task_execution_failed",
        )
        return catalog.update_document_if_status(
            failed,
            allowed_statuses={DocumentIngestStatus.PROCESSING.value},
        )

    async def retry_document(self, request: DocumentRetryRequest) -> DocumentRetryResult:
        """Retry a failed tenant-owned catalog document through the existing workflow."""

        catalog = self._catalog()
        if self._dependencies.storage_factory is None or self._dependencies.ingest_workflow is None:
            raise DocumentLifecycleError(503, "document_retry_unavailable", "文档入库服务暂不可用，请稍后重试")

        record = catalog.get_document(request.doc_id, tenant_id=request.tenant_id)
        if record is None:
            raise DocumentLifecycleError(404, "document_not_found", "文档不存在或不可访问")
        if record.legacy or record.ingest_status != DocumentIngestStatus.FAILED.value:
            raise DocumentLifecycleError(409, "document_retry_ineligible", "当前文档状态不可重试")
        storage = self._dependencies.storage_factory()
        try:
            reference = storage.accepted_reference(record.storage_reference, request.tenant_id)
            accepted_path = storage.resolve_accepted(reference, request.tenant_id)
        except Exception as error:
            raise DocumentLifecycleError(409, "document_retry_source_unavailable", "文档源文件已不可用，请重新上传") from error

        started = self._dependencies.clock()
        processing = record.with_updates(
            ingest_status=DocumentIngestStatus.PROCESSING.value,
            error_code="",
        )
        catalog.update_document(processing)
        document_input = document_input_from_record(processing)
        try:
            result = await self._dependencies.ingest_workflow.ainvoke(
                {
                    "file_paths": [accepted_path],
                    "document_inputs": [document_input],
                    "tenant_id": request.tenant_id,
                }
            )
        except Exception as error:
            catalog.update_document(
                processing.with_updates(
                    ingest_status=DocumentIngestStatus.FAILED.value,
                    error_code="document_retry_failed",
                )
            )
            logger.error(
                "document_retry_failed",
                doc_id=request.doc_id,
                tenant_id=request.tenant_id,
                error_type=type(error).__name__,
            )
            if self._dependencies.record_retry_failure is not None:
                try:
                    self._dependencies.record_retry_failure(request.actor, request.doc_id, "document_retry_failed")
                except Exception as audit_error:
                    logger.error(
                        "document_retry_failure_audit_failed",
                        doc_id=request.doc_id,
                        error_type=type(audit_error).__name__,
                    )
            raise DocumentLifecycleError(500, "document_retry_failed", "文档重试失败，请稍后再试") from error

        chunks = result.get("chunks", [])
        retry_result = DocumentRetryResult(
            record=processing.with_updates(
                ingest_status=DocumentIngestStatus.INGESTED.value,
                chunks_count=len(chunks),
                entities_count=int(result.get("entities_stored", 0)),
                relations_count=int(result.get("relations_stored", 0)),
                error_code="",
            ),
            chunks_count=len(chunks),
            entities_count=int(result.get("entities_stored", 0)),
            relations_count=int(result.get("relations_stored", 0)),
        )
        catalog.update_document(retry_result.record)
        doc_type = chunks[0].doc_type.value if chunks else retry_result.record.document_type
        if self._dependencies.record_metrics is not None:
            self._dependencies.record_metrics(doc_type, self._dependencies.clock() - started)
        if self._dependencies.record_retry_success is not None:
            try:
                self._dependencies.record_retry_success(request.actor, retry_result)
            except Exception as error:
                catalog.update_document(
                    retry_result.record.with_updates(
                        ingest_status=DocumentIngestStatus.FAILED.value,
                        error_code="audit_unavailable",
                    )
                )
                raise DocumentLifecycleError(503, "audit_unavailable", "安全审计服务暂不可用，请稍后重试。") from error
        if self._dependencies.trigger_retry_webhook is not None:
            await self._dependencies.trigger_retry_webhook(request.actor, retry_result)
        return retry_result

    async def delete_document(self, request: DocumentDeleteRequest) -> DocumentDeleteResult:
        """Delete one document in graph-first order, preserving partial-failure safety."""

        catalog = self._dependencies.catalog
        vector_store = self._dependencies.vector_store
        knowledge_graph = self._dependencies.knowledge_graph
        if vector_store is None or knowledge_graph is None or self._dependencies.storage_factory is None:
            raise DocumentLifecycleError(503, "document_delete_unavailable", "文档删除服务暂不可用，请稍后重试。")

        catalog_record = catalog.get_document(request.doc_id, tenant_id=request.tenant_id) if catalog else None
        chunks = await vector_store.get_chunks_by_doc_id(request.doc_id, tenant_id=request.tenant_id)
        if catalog_record is None and not chunks:
            raise DocumentLifecycleError(404, "document_not_found", f"文档不存在: {request.doc_id}")

        department_id = (
            getattr(catalog_record, "department_id", None)
            if catalog_record is not None
            else str(chunks[0].get("department_id") or "")
        )
        if not self.can_manage_document(request.actor, department_id):
            raise DocumentLifecycleError(403, "document_department_forbidden", "部门负责人只能管理本部门文档")

        source = catalog_record.storage_reference if catalog_record is not None else str(chunks[0].get("source") or "")
        try:
            if hasattr(knowledge_graph, "delete_document_evidence"):
                entities_deleted = await knowledge_graph.delete_document_evidence(
                    request.doc_id,
                    tenant_id=request.tenant_id,
                )
            elif source:
                entities_deleted = await knowledge_graph.delete_by_source(source, tenant_id=request.tenant_id)
            else:
                entities_deleted = 0
        except Exception as error:
            logger.error(
                "doc_delete_kg_failed",
                doc_id=request.doc_id,
                tenant_id=request.tenant_id,
                error_type=type(error).__name__,
            )
            raise DocumentLifecycleError(
                503,
                "document_graph_cleanup_failed",
                "文档图谱清理失败，删除操作未完成，请稍后重试。",
            ) from error

        if catalog is not None and catalog_record is not None:
            catalog_record = catalog_record.with_updates(ingest_status=DocumentIngestStatus.DELETED.value)
            catalog.update_document(catalog_record)

        sparse_deleted = (
            await self._dependencies.sparse_index.delete_by_doc_id(request.doc_id, tenant_id=request.tenant_id)
            if self._dependencies.sparse_index is not None
            else 0
        )
        vectors_deleted = await vector_store.delete_by_doc_id(request.doc_id, tenant_id=request.tenant_id)
        file_deleted = self._dependencies.storage_factory().delete(source, tenant_id=request.tenant_id) if source else False
        if catalog is not None and catalog_record is not None and file_deleted:
            catalog.update_document(catalog_record.with_updates(storage_reference=""))

        result = DocumentDeleteResult(
            doc_id=request.doc_id,
            vectors_deleted=vectors_deleted,
            sparse_deleted=sparse_deleted,
            entities_deleted=entities_deleted,
            file_deleted=file_deleted,
        )
        if self._dependencies.record_delete_audit is not None:
            self._dependencies.record_delete_audit(request.actor, result, request.ip, request.user_agent)
        if self._dependencies.trigger_delete_webhook is not None:
            await self._dependencies.trigger_delete_webhook(request.actor, result)
        if self._dependencies.advance_revision is not None:
            await self._dependencies.advance_revision(request.tenant_id)
        logger.info(
            "doc_deleted",
            doc_id=result.doc_id,
            vectors_deleted=result.vectors_deleted,
            entities_deleted=result.entities_deleted,
            file_deleted=result.file_deleted,
        )
        return result


__all__ = [
    "AcceptedDocument",
    "AcceptedDocumentProcessingRequest",
    "AcceptedDocumentProcessingResult",
    "ArtifactCleanup",
    "DocumentDeleteRequest",
    "DocumentDeleteResult",
    "DocumentLifecycleCoordinator",
    "DocumentLifecycleDependencies",
    "DocumentLifecycleError",
    "DocumentRetryRequest",
    "DocumentRetryResult",
    "FolderPublicationRequest",
    "FolderPublicationResult",
    "document_input_from_record",
    "finalize_document_failure",
    "finalize_document_success",
]
