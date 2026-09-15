"""Generation-safe completion and failure finalization for accepted documents."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from domain.documents import DocumentIngestStatus, DocumentRecord, IngestDocumentInput


class LifecycleCatalog(Protocol):
    """Catalog operations required to finalize one accepted generation."""

    def get_document(self, doc_id: str, *, tenant_id: str) -> DocumentRecord | None: ...

    def update_document(self, record: DocumentRecord) -> object: ...

    def update_document_if_status(
        self,
        record: DocumentRecord,
        *,
        allowed_statuses: set[str],
    ) -> bool: ...


ArtifactCleanup = Callable[[str, str], Awaitable[None]]


async def finalize_document_success(
    catalog: LifecycleCatalog,
    document_input: IngestDocumentInput,
    result: dict[str, Any],
    *,
    cleanup: ArtifactCleanup,
) -> bool:
    """Commit success only for the catalog generation still active at completion."""

    current = catalog.get_document(document_input.doc_id, tenant_id=document_input.tenant_id)
    if current is None or current.version != document_input.version:
        await cleanup(document_input.doc_id, document_input.tenant_id)
        return False
    completed = current.with_updates(
        ingest_status=DocumentIngestStatus.INGESTED.value,
        chunks_count=int(result.get("chunks_count", 0)),
        entities_count=int(result.get("entities_count", result.get("entities_stored", 0))),
        relations_count=int(result.get("relations_count", result.get("relations_stored", 0))),
        error_code="",
    )
    accepted = catalog.update_document_if_status(
        completed,
        allowed_statuses={
            DocumentIngestStatus.ACCEPTED.value,
            DocumentIngestStatus.PROCESSING.value,
        },
    )
    if not accepted and current.ingest_status in {
        DocumentIngestStatus.DELETED.value,
        DocumentIngestStatus.FAILED.value,
    }:
        await cleanup(document_input.doc_id, document_input.tenant_id)
    return accepted


async def finalize_document_failure(
    catalog: LifecycleCatalog,
    document_input: IngestDocumentInput,
    *,
    cleanup: ArtifactCleanup,
) -> bool:
    """Clean partial artifacts and fail only the catalog generation still active."""

    await cleanup(document_input.doc_id, document_input.tenant_id)
    current = catalog.get_document(document_input.doc_id, tenant_id=document_input.tenant_id)
    if current is None or current.version != document_input.version:
        return False
    failed = current.with_updates(
        ingest_status=DocumentIngestStatus.FAILED.value,
        error_code="task_execution_failed",
    )
    return catalog.update_document_if_status(
        failed,
        allowed_statuses={
            DocumentIngestStatus.ACCEPTED.value,
            DocumentIngestStatus.PROCESSING.value,
        },
    )


__all__ = ["ArtifactCleanup", "LifecycleCatalog", "finalize_document_failure", "finalize_document_success"]
