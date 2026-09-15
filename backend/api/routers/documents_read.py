"""文档入库与文档查阅 HTTP 路由（读侧）。"""

# 注意：本模块的部分导入（settings 等）看似未直接使用，但被测试用作
# monkeypatch 锚点，且与 documents_ingest 共享运行时状态；不要按 F401 清理。

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from domain.departments import department_display_name
from domain.documents import (
    DepartmentRecord,
    DocumentIngestProcessingStep,
    DocumentIngestStatus,
    DocumentRecord,
    IngestDocumentInput,
    document_ingest_processing_step,
    document_ingest_stage,
    normalize_relative_path,
    normalize_uploaded_filename,
)
from domain.identity import Permission, UserContext
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import ValidationError
from shared.config import settings
from shared.utils.logging import bind_context, get_logger
from shared.utils.ratelimit import RATE_LIMITS, authenticated_composite_key, limiter
from services.documents.lifecycle import (
    AcceptedDocument,
    DocumentDeleteRequest,
    DocumentLifecycleCoordinator,
    DocumentLifecycleError,
    DocumentRetryRequest,
    FolderPublicationRequest,
)
from services.documents.submission import (
    DocumentSubmissionCoordinator,
    DocumentSubmissionDenialRequest,
    DocumentSubmissionError,
    DocumentSubmissionRequest,
    FolderSubmissionRequest,
)

from api.contracts import ResourceId
from api.dependencies import (
    DocumentReadError,
    DocumentReadProvider,
    get_document_lifecycle,
    get_document_read,
    get_document_submission,
    require_permission,
)
from api.schemas import (
    ChunkItem,
    DepartmentItem,
    DocDeleteResponse,
    DocListItem,
    FolderManifestFileRequest,
    FolderUploadManifestRequest,
    IngestProgressItem,
    IngestProgressResponse,
    IngestResponse,
)


from api.routers.documents_ingest import _ingest_error, _resolved_document_read_provider, _tenant_from_request

docs_router = APIRouter(prefix="/docs", tags=["文档查阅"])  # 文档查阅路由


@docs_router.get("/departments", response_model=list[DepartmentItem])
async def list_departments(
    request: Request,
    company_id: ResourceId | None = None,
    user: UserContext = Depends(require_permission(Permission.DOC_READ)),
    read_provider: DocumentReadProvider = Depends(get_document_read),
):
    """列出可用于文件夹映射的租户部门。

    Args:
        request: 当前 FastAPI 请求。
        company_id: 公司 ID，缺省使用租户 ID。
        user: 当前认证用户上下文。
        read_provider: 注入的文档读取提供者。
    """

    tenant_id = _tenant_from_request(request, user) or ""
    effective_company = company_id or tenant_id
    if effective_company != tenant_id:
        raise _ingest_error(403, "document_company_forbidden", "公司不属于当前租户")
    read_provider = _resolved_document_read_provider(request, read_provider)
    return [
        DepartmentItem(
            department_id=department.department_id,
            company_id=department.company_id,
            name=department_display_name(
                department.department_id,
                department.name,
            ),
            normalized_key=department.normalized_key,
            status=department.status,
        )
        for department in read_provider.list_departments(
            tenant_id=tenant_id,
            company_id=effective_company,
        )
    ]


@docs_router.get("", response_model=list[DocListItem])
async def list_documents(
    request: Request,
    user: UserContext = Depends(require_permission(Permission.DOC_READ)),
    department_id: ResourceId | None = None,
    read_provider: DocumentReadProvider = Depends(get_document_read),
):
    """先列出目录记录，再仅补充显式标记的旧向量数据。

    Args:
        request: 当前 FastAPI 请求。
        user: 当前认证用户上下文。
        department_id: 过滤条件：部门 ID，可空。
        read_provider: 注入的文档读取提供者。
    """

    tenant_id = _tenant_from_request(request, user) or ""
    read_provider = _resolved_document_read_provider(request, read_provider)
    response_items: list[DocListItem] = []
    catalog_doc_ids: set[str] = set()
    # ① 目录记录优先，同时判定失败且仍保留源文件的非旧记录是否可重试
    records = read_provider.list_catalog_documents(
        tenant_id=tenant_id,
        department_id=department_id,
    )
    for record in records:
        catalog_doc_ids.add(record.doc_id)
        file_reference = read_provider.accepted_reference(
            record.storage_reference,
            tenant_id=tenant_id,
        )
        retryable = (
            record.ingest_status == DocumentIngestStatus.FAILED.value
            and not record.legacy
            and bool(file_reference)
        )
        stage, stage_index, stage_total, stage_label = document_ingest_stage(
            record.ingest_status,
            record.metadata,
        )
        (
            processing_step,
            processing_step_index,
            processing_step_total,
            processing_step_label,
        ) = document_ingest_processing_step(record.metadata)
        response_items.append(
            DocListItem(
                doc_id=record.doc_id,
                file_name=record.display_name or record.uploaded_filename,
                source="",
                file_reference=file_reference,
                doc_type=record.document_type or record.source_format,
                chunks_count=record.chunks_count,
                tenant_id=record.tenant_id,
                company_id=record.company_id,
                department_id=record.department_id,
                folder_path=record.folder_path,
                relative_path=record.relative_path,
                uploaded_filename=record.uploaded_filename,
                display_name=record.display_name,
                provenance_source_filename=record.provenance_source_filename,
                content_sha256=record.content_sha256,
                version=record.version,
                ingest_status=record.ingest_status,
                authority=record.authority,
                review_status=record.review_status,
                sensitivity=record.sensitivity,
                entities_count=record.entities_count,
                relations_count=record.relations_count,
                created_at=record.created_at,
                updated_at=record.updated_at,
                error_code=record.error_code,
                retryable=retryable,
                legacy=record.legacy,
                ingest_stage=stage,
                ingest_stage_index=stage_index,
                ingest_stage_total=stage_total,
                ingest_stage_label=stage_label,
                processing_step=processing_step,
                processing_step_index=processing_step_index,
                processing_step_total=processing_step_total,
                processing_step_label=processing_step_label,
                legacy_name_unresolved=bool(
                    record.metadata.get("legacy_name_unresolved", False)
                ),
            )
        )

    # ② 旧向量数据仅补充目录中不存在的文档，并沿用同一部门过滤
    vector_items = await read_provider.list_legacy_documents(tenant_id=tenant_id)
    for item in vector_items:
        if item.get("doc_id") in catalog_doc_ids:
            continue
        if department_id and item.get("department_id") != department_id:
            continue
        file_reference = read_provider.accepted_reference(
            item.get("source", ""),
            tenant_id=tenant_id,
        )
        # ③ 旧数据文件名优先使用显式来源名，取不到时回退不透明名并标记为未解析
        explicit_name = str(
            item.get("original_filename")
            or item.get("uploaded_filename")
            or item.get("provenance_source_filename")
            or ""
        ).strip()
        opaque_name = str(item.get("file_name") or item.get("doc_id") or "legacy-document")
        response_items.append(
            DocListItem(
                **{
                    **item,
                    "file_name": explicit_name or opaque_name,
                    "file_reference": file_reference,
                    "legacy": True,
                    "legacy_name_unresolved": not bool(explicit_name),
                    "ingest_status": item.get("ingest_status", "legacy"),
                    "display_name": explicit_name,
                    "uploaded_filename": explicit_name,
                    "provenance_source_filename": explicit_name,
                }
            )
        )
    return response_items


@docs_router.post("/{doc_id}/retry", response_model=IngestResponse)
@limiter.limit(RATE_LIMITS["doc_upload"], key_func=authenticated_composite_key)
async def retry_document(
    doc_id: ResourceId,
    request: Request,
    response: Response,
    user: UserContext = Depends(require_permission(Permission.DOC_WRITE)),
    lifecycle: DocumentLifecycleCoordinator = Depends(get_document_lifecycle),
):
    """通过生命周期协调器重试一份保留的失败文档。

    Args:
        doc_id: 文档 ID。
        request: 当前 FastAPI 请求（限流使用）。
        response: 当前响应对象。
        user: 当前认证用户上下文。
        lifecycle: 文档生命周期协调器。
    """

    tenant_id = _tenant_from_request(request, user) or ""
    try:
        result = await lifecycle.retry_document(
            DocumentRetryRequest(doc_id=doc_id, tenant_id=tenant_id, actor=user)
        )
    except DocumentLifecycleError as error:
        raise _ingest_error(error.status_code, error.code, error.message) from error

    return IngestResponse(
        file_name=result.record.uploaded_filename,
        chunks_count=result.chunks_count,
        entities_count=result.entities_count,
        relations_count=result.relations_count,
        status="success",
        doc_id=result.record.doc_id,
        relative_path=result.record.relative_path,
        department_id=result.record.department_id,
        display_name=result.record.display_name,
        provenance_source_filename=result.record.provenance_source_filename,
        version=result.record.version,
    )


@docs_router.get(
    "/{doc_id}/file",
    response_class=FileResponse,
    responses={
        200: {
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
            "description": "Tenant-owned original document source",
        }
    },
)
async def open_document_file(
    doc_id: ResourceId,
    request: Request,
    user: UserContext = Depends(require_permission(Permission.DOC_READ)),
    read_provider: DocumentReadProvider = Depends(get_document_read),
):
    """返回一份属于当前租户的已受理源文件，供浏览器预览或下载。

    Args:
        doc_id: 文档 ID。
        request: 当前 FastAPI 请求。
        user: 当前认证用户上下文。
        read_provider: 注入的文档读取提供者。
    """

    tenant_id = _tenant_from_request(request, user) or ""
    read_provider = _resolved_document_read_provider(request, read_provider)
    record = read_provider.get_document(doc_id, tenant_id=tenant_id)
    if record is None:
        raise _ingest_error(404, "document_not_found", "文档不存在或不可访问")

    try:
        source_path = read_provider.resolve_source_path(record, tenant_id=tenant_id)
    except DocumentReadError as error:
        raise _ingest_error(error.status_code, error.code, error.message) from error

    filename = Path(
        record.uploaded_filename
        or record.provenance_source_filename
        or record.display_name
        or doc_id
    ).name
    return FileResponse(
        source_path,
        media_type=record.mime_type or "application/octet-stream",
        filename=filename,
        content_disposition_type="inline",
    )


@docs_router.get("/{doc_id}/chunks", response_model=list[ChunkItem])
async def get_document_chunks(
    doc_id: ResourceId,
    request: Request,
    user: UserContext = Depends(require_permission(Permission.DOC_READ)),
    read_provider: DocumentReadProvider = Depends(get_document_read),
):
    """查看指定文档的分块内容。

    Args:
        doc_id: 文档 ID。
        request: 当前 FastAPI 请求。
        user: 当前认证用户上下文。
        read_provider: 注入的文档读取提供者。
    """
    tenant_id = _tenant_from_request(request, user)
    read_provider = _resolved_document_read_provider(request, read_provider)
    chunks = await read_provider.get_chunks(doc_id, tenant_id=tenant_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"文档不存在或无分块: {doc_id}")
    return [ChunkItem(**chunk) for chunk in chunks]


@docs_router.delete("/{doc_id}", response_model=DocDeleteResponse)
async def delete_document(
    doc_id: ResourceId,
    request: Request,
    user: UserContext = Depends(require_permission(Permission.DOC_DELETE)),
    lifecycle: DocumentLifecycleCoordinator = Depends(get_document_lifecycle),
):
    """通过图谱优先的生命周期协调器删除一份文档。

    Args:
        doc_id: 文档 ID。
        request: 当前 FastAPI 请求。
        user: 当前认证用户上下文。
        lifecycle: 文档生命周期协调器。
    """

    tenant_id = _tenant_from_request(request, user) or ""
    try:
        result = await lifecycle.delete_document(
            DocumentDeleteRequest(
                doc_id=doc_id,
                tenant_id=tenant_id,
                actor=user,
                ip=getattr(getattr(request, "client", None), "host", ""),
                user_agent=getattr(request, "headers", {}).get("user-agent", ""),
            )
        )
    except DocumentLifecycleError as error:
        # 语义化常见失败：未找到 404、部门越权 403，其余按协调器返回的状态码
        if error.code == "document_not_found":
            raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}") from error
        if error.code == "document_department_forbidden":
            raise HTTPException(status_code=403, detail="部门负责人只能管理本部门文档") from error
        raise _ingest_error(error.status_code, error.code, error.message) from error

    return DocDeleteResponse(
        doc_id=result.doc_id,
        vectors_deleted=result.vectors_deleted,
        entities_deleted=result.entities_deleted,
        file_deleted=result.file_deleted,
    )
