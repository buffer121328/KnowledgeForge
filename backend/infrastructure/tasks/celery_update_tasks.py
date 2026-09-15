"""Celery incremental knowledge-update task implementation with legacy name."""

from __future__ import annotations

import asyncio
from typing import Any

from infrastructure.celery_app import celery_app
from infrastructure.cache.redis import CacheService, KnowledgeRevisionStore
from shared.config import settings
from infrastructure.tasks.task_registry import update_task_state_safely
from shared.utils.logging import get_logger, safe_file_reference

logger = get_logger(__name__)


def _has_retry_remaining(*, retries: int, max_retries: int | None) -> bool:
    """Return whether Celery can schedule one more controlled retry."""
    return max_retries is None or retries < max_retries


async def _advance_knowledge_revision(tenant_id: str) -> int:
    """Advance the revision using a Redis client owned by this task event loop."""
    cache = CacheService(
        settings.redis_url,
        socket_connect_timeout=settings.readiness_probe_timeout_seconds,
        socket_timeout=settings.readiness_probe_timeout_seconds,
        retry_on_timeout=False,
    )
    try:
        return await KnowledgeRevisionStore(cache).advance(tenant_id)
    finally:
        await cache.close()


@celery_app.task(name="infrastructure.celery_tasks.knowledge_update_task", bind=True, max_retries=3)
def knowledge_update_task(
    self,
    file_path: str,
    change_type: str,
    tenant_id: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Handle knowledge update task for the module."""
    task_id = self.request.id
    file_reference = safe_file_reference(file_path)
    update_task_state_safely(task_id, "started")
    logger.info(
        "knowledge_update_started",
        task_id=task_id,
        file_reference=file_reference,
        change_type=change_type,
    )
    try:
        result = asyncio.run(_run_update(file_path, change_type, tenant_id=tenant_id))
        logger.info("knowledge_update_completed", task_id=task_id, **result)
        update_task_state_safely(task_id, "succeeded")
        return {**result, "status": "completed", "error": None}
    except Exception as exc:
        logger.error(
            "knowledge_update_failed",
            task_id=task_id,
            file_reference=file_reference,
            error_type=type(exc).__name__,
        )
        if not _has_retry_remaining(
            retries=self.request.retries,
            max_retries=self.max_retries,
        ):
            update_task_state_safely(task_id, "failed")
            raise
        update_task_state_safely(task_id, "retrying")
        raise self.retry(
            exc=RuntimeError(f"task_execution_failed:{type(exc).__name__}")
        )


async def _run_update(file_path: str, change_type: str, tenant_id: str) -> dict[str, Any]:
    """Run one tenant-scoped incremental update across dense, sparse, and graph stores."""
    from agents.document_parser import DocParserAgent
    from agents.knowledge_extractor import KnowledgeExtractAgent
    from agents.knowledge_updater import KnowledgeUpdateAgent
    from domain.tasks import ChangeType, DocumentChange
    from infrastructure.retrieval.bm25_index import NativeBM25Index
    from infrastructure.graph.neo4j_graph import KnowledgeGraphService
    from infrastructure.retrieval.vector_store import VectorStoreService
    from shared.config import settings

    parser = DocParserAgent()
    extractor = KnowledgeExtractAgent()
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
        await vector_store.init()
        await knowledge_graph.init()
        graph_initialized = True
        agent = KnowledgeUpdateAgent(
            doc_parser=parser,
            knowledge_extractor=extractor,
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            sparse_index=sparse_index,
        )
        change = DocumentChange(file_path=file_path, change_type=ChangeType(change_type))
        results = await agent.process_batch([change], tenant_id=tenant_id)
        if any(not result.success for result in results):
            raise RuntimeError("knowledge_update_failed")
        await _advance_knowledge_revision(tenant_id)
        if not results:
            return {
                "file_reference": safe_file_reference(file_path),
                "vectors_added": 0,
                "vectors_deleted": 0,
            }
        result = results[0]
        return {
            "file_reference": safe_file_reference(result.change.file_path),
            "vectors_added": result.vectors_added,
            "vectors_deleted": result.vectors_deleted,
            "entities_added": result.entities_added,
            "relations_added": result.relations_added,
        }
    finally:
        if graph_initialized:
            try:
                await knowledge_graph.close()
            except Exception as error:
                logger.warning(
                    "knowledge_update_graph_close_failed",
                    error_type=type(error).__name__,
                )
