"""Celery ingestion task wrappers and their infrastructure composition."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from billiard.exceptions import SoftTimeLimitExceeded

from domain.documents import IngestDocumentInput
from infrastructure.celery_app import celery_app
from infrastructure.cache.redis import KnowledgeRevisionStore, get_cache_service
from infrastructure.tasks.task_registry import update_task_state_safely
from shared.utils.logging import get_logger, safe_file_reference
from services.documents.lifecycle import (
    AcceptedDocumentProcessingRequest,
    DocumentLifecycleCoordinator,
    DocumentLifecycleDependencies,
)

logger = get_logger(__name__)


async def _emit_progress(
    progress_callback: Callable[[str, int, int, str], Awaitable[None] | None] | None,
    step: str,
    index: int,
    total: int,
    label: str,
) -> None:
    """Invoke an optional ingest progress callback."""

    if progress_callback is None:
        return
    result = progress_callback(step, index, total, label)
    if inspect.isawaitable(result):
        await result


def _worker_catalog():
    """Build the process-local PostgreSQL catalog used by background workers."""

    from infrastructure.documents.catalog import PostgreSQLDocumentCatalogRepository
    from infrastructure.postgres.database import get_database_service

    return PostgreSQLDocumentCatalogRepository(get_database_service())


async def _cleanup_document_artifacts(doc_id: str, tenant_id: str) -> None:
    """Best-effort cleanup for partial or late artifacts scoped to one document."""

    from infrastructure.retrieval.bm25_index import NativeBM25Index
    from infrastructure.graph.neo4j_graph import KnowledgeGraphService
    from infrastructure.retrieval.vector_store import VectorStoreService
    from shared.config import settings

    vector_store = VectorStoreService()
    sparse_index = NativeBM25Index(
        settings.qa_bm25_index_path,
        k1=settings.qa_bm25_k1,
        b=settings.qa_bm25_b,
        schema_version=settings.qa_bm25_schema_version,
        tokenizer_version=settings.qa_bm25_tokenizer_version,
    )
    graph = KnowledgeGraphService()
    try:
        await vector_store.init()
        await vector_store.delete_by_doc_id(doc_id, tenant_id=tenant_id)
    except Exception as error:
        logger.warning(
            "ingest_late_vector_cleanup_failed",
            doc_id=doc_id,
            tenant_id=tenant_id,
            error_type=type(error).__name__,
        )
    try:
        await sparse_index.delete_by_doc_id(doc_id, tenant_id=tenant_id)
    except Exception as error:
        logger.warning(
            "ingest_late_sparse_cleanup_failed",
            doc_id=doc_id,
            tenant_id=tenant_id,
            error_type=type(error).__name__,
        )
    try:
        await graph.init()
        await graph.delete_document_evidence(doc_id, tenant_id)
    except Exception as error:
        logger.warning(
            "ingest_late_graph_cleanup_failed",
            doc_id=doc_id,
            tenant_id=tenant_id,
            error_type=type(error).__name__,
        )
    finally:
        try:
            await graph.close()
        except Exception:
            pass


async def _finalize_document_success(
    catalog: Any,
    document_input: dict[str, Any],
    result: dict[str, Any],
    *,
    cleanup: Callable[[str, str], Awaitable[None]] = _cleanup_document_artifacts,
) -> bool:
    """Compatibility import for tests and the legacy task facade."""

    from services.documents.lifecycle_completion import finalize_document_success

    return await finalize_document_success(
        catalog,
        IngestDocumentInput(**document_input),
        result,
        cleanup=cleanup,
    )


async def _finalize_document_failure(
    catalog: Any,
    document_input: dict[str, Any],
    *,
    cleanup: Callable[[str, str], Awaitable[None]] = _cleanup_document_artifacts,
) -> bool:
    """Compatibility import for tests and the legacy task facade."""

    from services.documents.lifecycle_completion import finalize_document_failure

    return await finalize_document_failure(
        catalog,
        IngestDocumentInput(**document_input),
        cleanup=cleanup,
    )


async def _run_document_completion_side_effects(
    document_input: IngestDocumentInput,
    result: dict[str, Any],
    user_id: str,
    username: str = "",
) -> None:
    """Write the upload audit fact and publish the tenant-scoped ingest Webhook."""

    from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service
    from infrastructure.webhooks.service import WebhookEvent, get_webhook_service

    get_audit_service().log(
        user_id=user_id,
        action=AuditAction.DOC_UPLOAD,
        resource=f"doc/{document_input.doc_id}",
        result=AuditResult.SUCCESS,
        username=username,
        org_id=document_input.tenant_id,
        metadata={
            "chunks_count": int(result.get("chunks_count", 0)),
            "entities_count": int(result.get("entities_count", 0)),
            "relations_count": int(result.get("relations_count", 0)),
            "department_id": document_input.department_id,
            "async": True,
        },
    )
    await get_webhook_service().trigger(
        WebhookEvent.DOC_INGESTED,
        {
            "doc_id": document_input.doc_id,
            "file_name": document_input.uploaded_filename,
            "tenant_id": document_input.tenant_id,
            "department_id": document_input.department_id,
            "chunks_count": int(result.get("chunks_count", 0)),
            "entities_count": int(result.get("entities_count", 0)),
            "user_id": user_id,
        },
        org_id=document_input.tenant_id,
    )


async def _advance_worker_revision(tenant_id: str) -> None:
    """Advance a tenant revision at the same completion point as the former runner."""

    await KnowledgeRevisionStore(get_cache_service()).advance(tenant_id)


def _worker_lifecycle(*, catalog: object | None) -> DocumentLifecycleCoordinator:
    """Compose worker-owned adapters for the HTTP- and Celery-neutral coordinator."""

    return DocumentLifecycleCoordinator(
        DocumentLifecycleDependencies(
            catalog=catalog,  # type: ignore[arg-type]
            cleanup_artifacts=_cleanup_document_artifacts,
            completion_side_effects=_run_document_completion_side_effects,
            advance_revision=_advance_worker_revision,
        )
    )


async def _run_ingest_for_lifecycle(
    file_path: str,
    *,
    tenant_id: str,
    document_input: IngestDocumentInput | None = None,
    progress_callback: Callable[[str, int, int, str], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    """Adapt the worker runner while keeping narrow test doubles compatible."""

    kwargs: dict[str, Any] = {"tenant_id": tenant_id}
    try:
        parameters = inspect.signature(_run_ingest).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "document_input" in parameters:
        kwargs["document_input"] = document_input
    if "progress_callback" in parameters:
        kwargs["progress_callback"] = progress_callback
    return await _run_ingest(file_path, **kwargs)


@celery_app.task(
    name="infrastructure.celery_tasks.ingest_document_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def ingest_document_task(
    self,
    file_path: str,
    tenant_id: str,
    user_id: str | None = None,
    document_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the established single-file task name through shared lifecycle completion."""

    task_id = self.request.id
    file_reference = safe_file_reference(file_path)
    update_task_state_safely(task_id, "started")
    logger.info(
        "ingest_task_started",
        task_id=task_id,
        file_reference=file_reference,
        user_id=user_id,
    )
    typed_input = IngestDocumentInput(**document_input) if document_input is not None else None
    lifecycle = _worker_lifecycle(catalog=_worker_catalog() if typed_input is not None else None)
    try:
        outcome = asyncio.run(
            lifecycle.process_accepted_document(
                AcceptedDocumentProcessingRequest(
                    file_path=file_path,
                    tenant_id=tenant_id,
                    user_id=user_id or "system",
                    document_input=typed_input,
                ),
                runner=_run_ingest_for_lifecycle,
            )
        )
        if outcome.status == "failed":
            raise RuntimeError(f"task_execution_failed:{outcome.error_type or 'RuntimeError'}")
        result = outcome.result or {}
        logger.info(
            "ingest_task_completed",
            task_id=task_id,
            file_reference=file_reference,
            chunks_count=result.get("chunks_count", 0),
        )
        update_task_state_safely(task_id, "succeeded")
        return {**result, "status": outcome.status, "error": outcome.error}
    except Exception as exc:
        logger.error(
            "ingest_task_failed",
            task_id=task_id,
            file_reference=file_reference,
            error_type=type(exc).__name__,
        )
        update_task_state_safely(task_id, "failed")
        safe_error = RuntimeError(f"task_execution_failed:{type(exc).__name__}")
        raise self.retry(exc=safe_error, countdown=60 * (2 ** self.request.retries))


async def _run_ingest(
    file_path: str,
    tenant_id: str,
    document_input: IngestDocumentInput | dict[str, Any] | None = None,
    progress_callback: Callable[[str, int, int, str], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    """Run parser, extractors, stores, and progress callbacks for one source path."""

    from agents.document_parser import DocParserAgent
    from agents.knowledge_extractor import KnowledgeExtractAgent
    from infrastructure.retrieval.bm25_index import NativeBM25Index
    from infrastructure.graph.neo4j_graph import KnowledgeGraphService
    from infrastructure.retrieval.vector_store import VectorStoreService
    from shared.config import settings
    from workflows.ingestion_persistence import persist_extractions_to_graph

    parser = DocParserAgent()
    typed_input = (
        document_input
        if isinstance(document_input, IngestDocumentInput)
        else IngestDocumentInput(**document_input)
        if document_input
        else None
    )
    parse_kwargs: dict[str, Any] = {"tenant_id": tenant_id}
    if typed_input is not None:
        parse_kwargs["document_input"] = typed_input
    await _emit_progress(progress_callback, "parse", 1, 6, "解析文件")
    chunks = await parser.parse(file_path, **parse_kwargs)
    extractor = KnowledgeExtractAgent()
    await _emit_progress(progress_callback, "extract", 2, 6, "知识抽取")
    extractions = await extractor.extract(chunks)
    vector_store = VectorStoreService()
    knowledge_graph = KnowledgeGraphService()
    sparse_index = NativeBM25Index(
        settings.qa_bm25_index_path,
        k1=settings.qa_bm25_k1,
        b=settings.qa_bm25_b,
        schema_version=settings.qa_bm25_schema_version,
        tokenizer_version=settings.qa_bm25_tokenizer_version,
    )
    graph_initialized = False
    try:
        await _emit_progress(progress_callback, "store_vectors", 3, 6, "写向量库")
        await vector_store.init()
        vectors_stored = await vector_store.add_chunks(chunks)
        await _emit_progress(progress_callback, "store_graph", 4, 6, "写知识图谱")
        await knowledge_graph.init()
        graph_initialized = True
        entities_stored, relations_stored = await persist_extractions_to_graph(
            chunks=chunks,
            extractions=extractions,
            knowledge_graph=knowledge_graph,
            tenant_id=tenant_id,
        )
        await _emit_progress(progress_callback, "store_sparse", 5, 6, "写稀疏索引")
        sparse_stored = await sparse_index.add_chunks(chunks)
        await _emit_progress(progress_callback, "advance_revision", 6, 6, "收尾提交")
        return {
            "file_reference": safe_file_reference(file_path),
            "chunks_count": len(chunks),
            "vectors_stored": vectors_stored,
            "sparse_stored": sparse_stored,
            "entities_count": entities_stored,
            "relations_count": relations_stored,
        }
    finally:
        if graph_initialized:
            try:
                await knowledge_graph.close()
            except Exception as error:
                logger.warning("ingest_graph_close_failed", error_type=type(error).__name__)


async def _compensate_batch_soft_timeout(
    lifecycle: DocumentLifecycleCoordinator,
    document_inputs: list[IngestDocumentInput] | None,
    *,
    interrupted_index: int,
) -> None:
    """Recover only the generations stranded by one serial parser batch timeout."""

    if document_inputs is None:
        return
    for index, document_input in enumerate(document_inputs[interrupted_index:], start=interrupted_index):
        try:
            await lifecycle.recover_interrupted_document(
                document_input,
                cleanup_partial_artifacts=index == interrupted_index,
            )
        except Exception as error:
            logger.error(
                "batch_ingest_timeout_compensation_failed",
                error_type=type(error).__name__,
            )


@celery_app.task(name="infrastructure.celery_tasks.batch_ingest_task", bind=True, max_retries=2)
def batch_ingest_task(
    self,
    file_paths: list[str],
    tenant_id: str,
    user_id: str | None = None,
    document_inputs: list[dict[str, Any]] | None = None,
    actor_username: str = "",
) -> dict[str, Any]:
    """Run every batch member through the same completion and failure semantics."""

    task_id = self.request.id
    update_task_state_safely(task_id, "started")
    logger.info("batch_ingest_started", task_id=task_id, file_count=len(file_paths))
    if document_inputs is not None and len(document_inputs) != len(file_paths):
        raise ValueError("document input count must match file path count")

    typed_inputs = [IngestDocumentInput(**item) for item in document_inputs] if document_inputs else None
    lifecycle = _worker_lifecycle(catalog=_worker_catalog() if typed_inputs is not None else None)
    results: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    for index, file_path in enumerate(file_paths):
        file_reference = safe_file_reference(file_path)
        document_input = typed_inputs[index] if typed_inputs is not None else None
        try:
            outcome = asyncio.run(
                lifecycle.process_accepted_document(
                    AcceptedDocumentProcessingRequest(
                        file_path=file_path,
                        tenant_id=tenant_id,
                        user_id=user_id or "system",
                        username=actor_username,
                        document_input=document_input,
                    ),
                    runner=_run_ingest_for_lifecycle,
                )
            )
        except SoftTimeLimitExceeded:
            asyncio.run(
                _compensate_batch_soft_timeout(
                    lifecycle,
                    typed_inputs,
                    interrupted_index=index,
                )
            )
            update_task_state_safely(task_id, "failed")
            logger.error(
                "batch_ingest_soft_time_limit_exceeded",
                task_id=task_id,
                interrupted_index=index,
                remaining_count=len(file_paths) - index,
            )
            raise
        if outcome.status == "failed":
            results.append(
                {
                    "file_reference": file_reference,
                    "status": "failed",
                    "error": outcome.error or "task_execution_failed",
                    "error_type": outcome.error_type or "RuntimeError",
                }
            )
            failure_count += 1
            continue

        results.append(
            {
                **(outcome.result or {}),
                "status": outcome.status,
                "error": outcome.error,
            }
        )
        success_count += 1

    logger.info("batch_ingest_completed", task_id=task_id, success=success_count, failure=failure_count)
    update_task_state_safely(task_id, "succeeded" if failure_count == 0 else "failed")
    return {
        "status": "completed" if failure_count == 0 else "partial",
        "success_count": success_count,
        "failure_count": failure_count,
        "results": results,
    }
