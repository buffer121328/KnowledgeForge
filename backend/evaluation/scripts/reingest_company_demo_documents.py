"""Replace one tenant's company-demo DOCX corpus through the normal lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from domain.documents import DepartmentRecord
from domain.identity import ROLE_PERMISSIONS, UserContext, UserRole
from infrastructure.retrieval.bm25_index import NativeBM25Index
from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service
from infrastructure.tasks.celery_ingest_tasks import batch_ingest_task
from infrastructure.documents.catalog import PostgreSQLDocumentCatalogRepository
from infrastructure.documents.local_uploads import LocalUploadStorage, UploadValidationPolicy
from infrastructure.documents.local_uploads import UploadPolicyError
from infrastructure.graph.neo4j_graph import KnowledgeGraphService
from infrastructure.postgres.database import get_database_service
from infrastructure.cache.redis import KnowledgeRevisionStore, get_cache_service
from infrastructure.tasks.task_registry import get_task_registry
from infrastructure.retrieval.vector_store import VectorStoreService
from shared.config import settings
from shared.utils.task_ids import new_task_id
from services.documents.folder_submission import FolderSubmissionPreparer
from services.documents.lifecycle import (
    AcceptedDocument,
    DocumentDeleteRequest,
    DocumentLifecycleCoordinator,
    DocumentLifecycleDependencies,
    FolderPublicationRequest,
)
from services.documents.helpers import document_input_from_record
from services.documents.contracts import (
    DocumentSubmissionDependencies,
    DocumentSubmissionSettings,
    FolderSubmissionRequest,
)


class ReingestError(RuntimeError):
    """Fail closed when the controlled corpus is incomplete or unsafe to replace."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(item: dict[str, Any], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ReingestError(f"manifest field is missing: {field}")
    return value.strip()


def _load_manifest(root: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReingestError("company-demo manifest is unavailable or invalid") from error
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list) or len(documents) != 41:
        raise ReingestError("company-demo manifest must contain exactly 41 documents")
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    for item in documents:
        if not isinstance(item, dict):
            raise ReingestError("company-demo manifest contains a non-object document")
        department = _required(item, "department")
        filename = _required(item, "source_filename")
        source_id = _required(item, "source_document_id")
        if "/" in filename or "\\" in filename:
            raise ReingestError("company-demo source filename must not include a path")
        logical_path = f"{department}/{filename}"
        if logical_path in seen_paths or source_id in seen_ids:
            raise ReingestError("company-demo manifest contains duplicate paths or document IDs")
        seen_paths.add(logical_path)
        seen_ids.add(source_id)
        source = root / "documents" / department / filename
        if not source.is_file() or _sha256(source) != _required(item, "sha256"):
            raise ReingestError("company-demo DOCX source is missing or has a SHA-256 mismatch")
    return documents


def _upload_policy() -> UploadValidationPolicy:
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


def _runtime_source(root: Path, item: dict[str, Any]) -> tuple[Path, str, str, str]:
    """Return the raw DOCX unless its established upload policy rejects embedded content."""

    department = _required(item, "department")
    source_filename = _required(item, "source_filename")
    source = root / "documents" / department / source_filename
    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    try:
        _upload_policy().validate(str(source), source_filename, mime_type)
        return source, source_filename, f"{department}/{source_filename}", mime_type
    except UploadPolicyError as error:
        if error.code != "upload_malicious":
            raise ReingestError("company-demo DOCX cannot pass the upload safety policy") from error
    normalized_path = Path(_required(item, "normalized_path"))
    if normalized_path.is_absolute() or normalized_path.parts[:1] != ("normalized",):
        raise ReingestError("company-demo normalized fallback path is invalid")
    normalized = root / normalized_path
    if not normalized.is_file() or _sha256(normalized) != _required(item, "normalized_sha256"):
        raise ReingestError("company-demo normalized fallback is missing or has a SHA-256 mismatch")
    return (
        normalized,
        normalized.name,
        normalized_path.as_posix().removeprefix("normalized/"),
        "text/plain",
    )


def _actor(user_id: str, username: str, tenant_id: str) -> UserContext:
    return UserContext(
        user_id=user_id,
        username=username,
        role=UserRole.ORGANIZATION_ADMIN,
        org_id=tenant_id,
        permissions=list(ROLE_PERMISSIONS[UserRole.ORGANIZATION_ADMIN]),
    )


def _submission_dependencies(catalog: PostgreSQLDocumentCatalogRepository) -> DocumentSubmissionDependencies:
    """Compose the folder preparer without bypassing its validation or catalog boundary."""

    def ignored_audit(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def ignored_webhook(*_args: Any, **_kwargs: Any) -> None:
        return None

    return DocumentSubmissionDependencies(
        storage_factory=lambda: LocalUploadStorage(settings.upload_dir),
        upload_policy=_upload_policy(),
        catalog=catalog,
        ingest_workflow=None,
        settings=DocumentSubmissionSettings(
            max_file_bytes=settings.upload_max_file_bytes,
            tenant_quota_bytes=settings.upload_tenant_quota_bytes,
            stream_chunk_bytes=settings.upload_stream_chunk_bytes,
            quarantine_retention_hours=settings.upload_quarantine_retention_hours,
        ),
        audit_denial=ignored_audit,
        audit_failure=ignored_audit,
        audit_success=ignored_audit,
        trigger_webhook=ignored_webhook,
        record_metrics=ignored_audit,
        allocate_document_id=lambda: str(uuid4()),
    )


async def _replace(
    root: Path,
    *,
    tenant_id: str,
    actor: UserContext,
    replace_existing: bool,
) -> dict[str, Any]:
    manifest = _load_manifest(root)
    catalog = PostgreSQLDocumentCatalogRepository(get_database_service())
    expected = {
        f"{_required(item, 'department')}/{_required(item, 'source_filename')}": item
        for item in manifest
    }
    active = catalog.list_documents(tenant_id=tenant_id, include_deleted=False)
    stable_ids = {_required(item, "source_document_id") for item in manifest}
    candidates = [
        record
        for record in active
        if record.normalized_relative_path in expected and record.doc_id not in stable_ids
    ]
    for record in candidates:
        item = expected[record.normalized_relative_path]
        if record.content_sha256 != _required(item, "sha256"):
            raise ReingestError("existing document content does not match the controlled manifest")
    prepared_by_id = {}
    for item in manifest:
        source_id = _required(item, "source_document_id")
        existing = catalog.get_document(source_id, tenant_id=tenant_id)
        if existing is None:
            continue
        expected_path = f"{_required(item, 'department')}/{_required(item, 'source_filename')}"
        normalized_logical_path = _required(item, "normalized_path").removeprefix("normalized/")
        expected_digest = (
            _required(item, "sha256")
            if existing.normalized_relative_path == expected_path
            else _required(item, "normalized_sha256")
        )
        if (
            existing.ingest_status != "accepted"
            or existing.normalized_relative_path not in {expected_path, normalized_logical_path}
            or existing.content_sha256 != expected_digest
            or existing.external_source_id != source_id
        ):
            raise ReingestError("stable company-demo document already exists in an unsafe state")
        prepared_by_id[source_id] = existing
    if candidates and not replace_existing:
        raise ReingestError("existing company-demo documents require --replace-existing")

    vector_store = VectorStoreService()
    graph = KnowledgeGraphService()
    sparse_index = NativeBM25Index(
        settings.qa_bm25_index_path,
        k1=settings.qa_bm25_k1,
        b=settings.qa_bm25_b,
        schema_version=settings.qa_bm25_schema_version,
        tokenizer_version=settings.qa_bm25_tokenizer_version,
    )
    await vector_store.init()
    await graph.init()
    lifecycle = DocumentLifecycleCoordinator(
        DocumentLifecycleDependencies(
            catalog=catalog,
            storage_factory=lambda: LocalUploadStorage(settings.upload_dir),
            vector_store=vector_store,
            knowledge_graph=graph,
            sparse_index=sparse_index,
            record_delete_audit=lambda current_actor, result, _ip, _user_agent: get_audit_service().log(
                user_id=current_actor.user_id,
                action=AuditAction.DOC_DELETE,
                resource=f"doc/{result.doc_id}",
                result=AuditResult.SUCCESS,
                username=current_actor.username,
                org_id=current_actor.org_id,
                metadata={
                    "vectors_deleted": result.vectors_deleted,
                    "sparse_deleted": result.sparse_deleted,
                    "entities_deleted": result.entities_deleted,
                    "file_deleted": result.file_deleted,
                    "company_demo_reingest": True,
                },
            ),
            advance_revision=lambda current_tenant: KnowledgeRevisionStore(
                get_cache_service()
            ).advance(current_tenant),
        )
    )
    deleted: list[str] = []
    try:
        for record in candidates:
            result = await lifecycle.delete_document(
                DocumentDeleteRequest(doc_id=record.doc_id, tenant_id=tenant_id, actor=actor)
            )
            deleted.append(result.doc_id)

        submission = FolderSubmissionPreparer(_submission_dependencies(catalog))
        accepted: list[AcceptedDocument] = []
        upload_id = f"company-demo-reingest-{new_task_id(tenant_id).removeprefix('tsk_')[:24]}"
        for item in manifest:
            department = _required(item, "department")
            source, filename, logical_path, content_type = _runtime_source(root, item)
            source_id = _required(item, "source_document_id")
            existing_record = prepared_by_id.get(source_id)
            if existing_record is not None:
                accepted.append(
                    AcceptedDocument(
                        record=existing_record,
                        document_input=document_input_from_record(existing_record),
                        client_file_id=source_id,
                    )
                )
                continue
            existing_versions = [
                record.version
                for record in catalog.list_documents(tenant_id=tenant_id, include_deleted=True)
                if record.department_id == department and record.normalized_relative_path == logical_path
            ]
            next_version = max(existing_versions, default=0) + 1
            catalog.upsert_department(
                DepartmentRecord(
                    department_id=department,
                    tenant_id=tenant_id,
                    company_id=tenant_id,
                    name=str(item.get("department_label") or department),
                    normalized_key=department,
                )
            )
            metadata = {
                "company_demo": True,
                "title": _required(item, "title"),
                "source_filename": filename,
                "source_sha256": _required(item, "sha256"),
                "source_format": Path(filename).suffix.lstrip("."),
                "original_source_format": "docx",
                "document_type": _required(item, "document_type"),
                "authority": _required(item, "authority"),
                "status": _required(item, "status"),
                "sensitivity": _required(item, "sensitivity"),
                "version": next_version,
            }
            with source.open("rb") as handle:
                prepared = await submission.prepare(
                    FolderSubmissionRequest(
                        tenant_id=tenant_id,
                        user_id=actor.user_id,
                        username=actor.username,
                        org_id=tenant_id,
                        source=handle,
                        original_filename=filename,
                        content_type=content_type,
                        company_id=tenant_id,
                        department_id=department,
                        relative_path=logical_path,
                        normalized_relative_path=logical_path,
                        metadata=metadata,
                        display_name=_required(item, "title"),
                        provenance_source_filename=filename,
                        external_source_id=source_id,
                        upload_id=upload_id,
                        client_file_id=source_id,
                    )
                )
            if not prepared.accepted or prepared.record is None or prepared.document_input is None:
                raise ReingestError(f"company-demo document could not be prepared: {filename}")
            accepted.append(
                AcceptedDocument(
                    record=prepared.record,
                    document_input=prepared.document_input,
                    client_file_id=prepared.client_file_id,
                )
            )

        publisher = DocumentLifecycleCoordinator(
            DocumentLifecycleDependencies(
                catalog=catalog,
                task_registry_factory=get_task_registry,
                new_task_id=new_task_id,
                publish_folder_task=lambda **kwargs: batch_ingest_task.apply_async(
                    args=[
                        kwargs["file_paths"],
                        kwargs["tenant_id"],
                        kwargs["user_id"],
                        kwargs["document_inputs"],
                        kwargs["actor_username"],
                    ],
                    task_id=kwargs["task_id"],
                    queue="parser",
                ),
            )
        )
        publication = await publisher.publish_folder(
            FolderPublicationRequest(tenant_id=tenant_id, actor=actor, documents=accepted)
        )
        return {
            "deleted_count": len(deleted),
            "submitted_count": len(publication.active_document_ids),
            "task_id": publication.task_id,
            "upload_id": upload_id,
        }
    finally:
        await graph.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--actor-username", required=True)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _load_manifest(root)
    if not args.apply:
        print(json.dumps({"status": "validated", "document_count": len(manifest)}, ensure_ascii=False))
        return 0
    if not args.replace_existing:
        raise ReingestError("--apply requires --replace-existing")
    report = asyncio.run(
        _replace(
            root,
            tenant_id=args.tenant,
            actor=_actor(args.actor_id, args.actor_username, args.tenant),
            replace_existing=True,
        )
    )
    print(json.dumps({"status": "submitted", **report}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReingestError as error:
        print(f"company-demo reingest failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
