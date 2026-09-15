"""文档入库与文档查阅 HTTP 路由。"""

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

logger = get_logger(__name__)
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


ingest_router = APIRouter(prefix="/ingest", tags=["文档入库"])  # 文档入库路由

_BATCH_UPLOAD_CONCURRENCY = 4  # 单请求内并发上传的文件数上限


@dataclass(frozen=True)
class PreparedFolderMember:
    """一个已通过校验并持久落盘、等待 worker 发布的文件夹成员。"""

    record: DocumentRecord  # 已落盘的文档目录记录
    document_input: IngestDocumentInput  # 待发布到解析流水线的文档输入
    response: IngestResponse  # 面向客户端的受理结果


def _tenant_from_request(request: Request, user: UserContext | None = None) -> str | None:
    """返回请求上认证的租户 ID，缺失时回退到用户上下文。

    Args:
        request: 当前 FastAPI 请求。
        user: 可选的用户上下文。
    """
    tenant_id = getattr(request.state, "tenant_id", None) or None
    if tenant_id == "":
        tenant_id = None
    if tenant_id is None and user is not None:
        tenant_id = user.org_id or None
    return tenant_id


def _ingest_error(status_code: int, code: str, message: str) -> HTTPException:
    """构造带错误码、消息与请求 ID 的入库异常。

    Args:
        status_code: HTTP 状态码。
        code: 业务错误码。
        message: 面向用户的错误消息。
    """
    # 生成一次性请求标识，便于通过 X-Request-ID 响应头关联日志
    request_id = secrets.token_hex(16)
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


def _resolved_document_read_provider(
    request: Request,
    candidate: DocumentReadProvider | object,
) -> DocumentReadProvider:
    """优先使用 FastAPI 注入的提供者，并保留受限的直连测试回退。

    Args:
        request: 当前 FastAPI 请求。
        candidate: 注入的文档读取提供者或任意回退对象。
    """

    if isinstance(candidate, DocumentReadProvider):
        return candidate
    return get_document_read(request)


def _normalize_upload_batch_id(value: str | None) -> str:
    """校验用于进度轮询的不透明客户端批次标识。

    Args:
        value: 原始批次标识字符串，可为空。
    """

    candidate = value.strip() if isinstance(value, str) else ""
    if len(candidate) > 128:
        raise _ingest_error(400, "upload_progress_id_invalid", "上传批次标识无效")
    try:
        return str(UUID(candidate)) if candidate else ""
    except ValueError as error:
        raise _ingest_error(400, "upload_progress_id_invalid", "上传批次标识无效") from error


def _upload_progress_metadata(
    metadata: dict | None,
    *,
    upload_id: str = "",
    client_file_id: str = "",
) -> dict:
    """追加受限的进度关联元数据，不暴露存储细节。

    Args:
        metadata: 原始元数据字典，可为空。
        upload_id: 上传批次 ID，为空则不写入。
        client_file_id: 客户端文件标识，为空则不写入。
    """

    next_metadata = dict(metadata or {})
    if upload_id:
        next_metadata["_upload_batch_id"] = upload_id
    if client_file_id:
        next_metadata["_upload_client_file_id"] = client_file_id
    return next_metadata


def _processing_progress_metadata(
    metadata: dict | None,
    *,
    upload_id: str = "",
    client_file_id: str = "",
    processing_step: str = DocumentIngestProcessingStep.PARSE.value,
) -> dict:
    """在进度元数据上附加当前长耗时的入库子步骤。

    Args:
        metadata: 原始元数据字典，可为空。
        upload_id: 上传批次 ID。
        client_file_id: 客户端文件标识。
        processing_step: 当前处理子步骤，默认为解析。
    """

    next_metadata = _upload_progress_metadata(
        metadata,
        upload_id=upload_id,
        client_file_id=client_file_id,
    )
    next_metadata["_processing_step"] = processing_step
    return next_metadata


def _allocate_document_id(external_source_id: str) -> str:
    """使用校验通过的外部 UUID，否则分配服务端持有的稳定标识。

    Args:
        external_source_id: 外部来源 ID，可为空。
    """

    candidate = (external_source_id or "").strip()
    if candidate:
        try:
            return str(UUID(candidate))
        except ValueError:
            pass
    return str(uuid4())


def _manifest_metadata_text(metadata: dict, key: str) -> str:
    """返回受限长度的字符串元数据值，不信任任意对象。

    Args:
        metadata: 元数据字典。
        key: 待读取的键名。
    """

    value = metadata.get(key, "")
    return str(value).strip()[:255] if value is not None else ""


def _folder_multipart_filename(
    raw_filename: str,
    *,
    root_folder_name: str,
    normalized_path: str,
) -> str:
    """从浏览器文件夹上传的 multipart 语义中提取安全的文件名。

    Args:
        raw_filename: multipart 原始文件名。
        root_folder_name: 清单中的根目录名。
        normalized_path: 清单归一化后的相对路径。
    """

    # ① 不含路径分隔符时直接按普通文件名归一化
    if "/" not in raw_filename and "\\" not in raw_filename:
        return normalize_uploaded_filename(raw_filename)

    # ② 带相对路径时先归一化，并接受“含/不含根目录前缀”两种等价形式
    multipart_path = normalize_relative_path(raw_filename)
    accepted_paths = {multipart_path}
    root_prefix = f"{root_folder_name}/"
    if multipart_path.startswith(root_prefix):
        accepted_paths.add(multipart_path[len(root_prefix):])
    # ③ 与清单路径对不上则拒绝
    if normalized_path not in accepted_paths:
        raise ValueError("multipart filename path does not match manifest")
    return Path(multipart_path).name


def _validate_folder_manifest(
    manifest: FolderUploadManifestRequest,
    files: list[UploadFile],
    *,
    tenant_id: str,
) -> list[tuple[UploadFile, FolderManifestFileRequest, str]]:
    """在任何文件落盘之前先校验清单整体结构。

    Args:
        manifest: 已解析的文件夹上传清单。
        files: multipart 上传的文件列表。
        tenant_id: 当前租户 ID。
    """

    # ① 安全关卡：清单公司必须属于当前租户
    if manifest.company_id != tenant_id:
        raise _ingest_error(403, "upload_company_forbidden", "公司不属于当前租户")
    # ② 文件数量必须与清单一一对应且不超过单批上限
    if not files or len(files) != len(manifest.files):
        raise _ingest_error(400, "upload_manifest_mismatch", "manifest 与上传文件数量不一致")
    if len(files) > settings.upload_max_batch_files:
        raise _ingest_error(
            400,
            "upload_batch_too_many_files",
            f"单次最多上传 {settings.upload_max_batch_files} 份文件。",
        )
    # ③ 根目录名必须合法
    try:
        normalize_uploaded_filename(manifest.root_folder_name)
    except ValueError as error:
        raise _ingest_error(400, "upload_root_folder_invalid", "根目录名称无效") from error

    # ④ 逐文件校验相对路径、multipart 文件名、重复项与部门映射
    normalized_paths: set[str] = set()
    validated: list[tuple[UploadFile, FolderManifestFileRequest, str]] = []
    client_ids: set[str] = set()
    for item_index, (file, entry) in enumerate(zip(files, manifest.files, strict=True)):
        try:
            normalized_path = normalize_relative_path(entry.relative_path)
        except ValueError as error:
            logger.warning(
                "folder_upload_manifest_member_invalid",
                field="relative_path",
                item_index=item_index,
                value_length=len(entry.relative_path),
            )
            raise _ingest_error(400, "upload_relative_path_invalid", "上传相对路径无效") from error
        raw_filename = file.filename or ""
        try:
            actual_filename = _folder_multipart_filename(
                raw_filename,
                root_folder_name=manifest.root_folder_name,
                normalized_path=normalized_path,
            )
        except ValueError as error:
            logger.warning(
                "folder_upload_manifest_member_invalid",
                field="filename",
                item_index=item_index,
                value_length=len(raw_filename),
                contains_forward_slash="/" in raw_filename,
                contains_backslash="\\" in raw_filename,
            )
            raise _ingest_error(400, "upload_filename_invalid", "上传文件名无效") from error
        if normalized_path in normalized_paths:
            raise _ingest_error(400, "upload_duplicate_path", "上传目录包含重复路径")
        if entry.client_file_id in client_ids:
            raise _ingest_error(400, "upload_duplicate_client_file_id", "client_file_id 重复")
        if Path(normalized_path).name != actual_filename:
            raise _ingest_error(400, "upload_filename_mismatch", "manifest 文件名与上传文件不一致")
        top_level = normalized_path.split("/", 1)[0]
        if manifest.department_mappings.get(top_level) != entry.department_id:
            raise _ingest_error(400, "upload_department_mapping_invalid", "目录与部门映射不一致")
        normalized_paths.add(normalized_path)
        client_ids.add(entry.client_file_id)
        validated.append((file, entry, normalized_path))
    return validated


def _ensure_manifest_departments(
    read_provider: DocumentReadProvider | Request,
    manifest: FolderUploadManifestRequest,
    *,
    tenant_id: str,
) -> None:
    """在整份清单校验通过后，解析属于当前租户的部门。

    Args:
        read_provider: 文档读取提供者或当前请求。
        manifest: 文件夹上传清单。
        tenant_id: 当前租户 ID。
    """

    if not isinstance(read_provider, DocumentReadProvider):
        read_provider = get_document_read(read_provider)
    for folder_key, department_id in manifest.department_mappings.items():
        existing = read_provider.get_department(department_id, tenant_id=tenant_id)
        # 部门已存在但归属其他公司则拒绝
        if existing is not None and existing.company_id != manifest.company_id:
            raise _ingest_error(403, "upload_department_forbidden", "部门不属于所选公司")
        # 部门不存在则按清单创建
        if existing is None:
            read_provider.upsert_department(
                DepartmentRecord(
                    department_id=department_id,
                    tenant_id=tenant_id,
                    company_id=manifest.company_id,
                    name=department_display_name(department_id, folder_key),
                    normalized_key=folder_key,
                )
            )


def _audit_upload_denial(
    submission: DocumentSubmissionCoordinator,
    request: Request,
    user: UserContext,
    *,
    code: str,
    size_bytes: int = 0,
    file_count: int | None = None,
) -> None:
    """委托记录上传拒绝审计，路由层不自行组装审计适配器。

    Args:
        submission: 文档提交协调器。
        request: 当前 FastAPI 请求。
        user: 当前认证用户上下文。
        code: 拒绝原因码。
        size_bytes: 涉及的文件大小（字节），默认 0。
        file_count: 涉及的文件数，可空。
    """
    metadata: dict[str, int | str] = {"code": code, "size_bytes": size_bytes}
    if file_count is not None:
        metadata["file_count"] = file_count
    submission.record_denial(
        DocumentSubmissionDenialRequest(
            tenant_id=_tenant_from_request(request, user) or "",
            user_id=user.user_id,
            username=user.username,
            org_id=user.org_id,
        ),
        code=code,
        size_bytes=size_bytes,
        metadata=metadata,
    )


async def _upload_one(
    request: Request,
    file: UploadFile,
    user: UserContext,
    submission: DocumentSubmissionCoordinator,
    *,
    upload_id: str = "",
    client_file_id: str = "",
) -> IngestResponse:
    """把单个 multipart 文件委派给与 HTTP 无关的提交协调器。

    Args:
        request: 当前 FastAPI 请求。
        file: 上传文件对象。
        user: 当前认证用户上下文。
        submission: 文档提交协调器。
        upload_id: 上传批次 ID，默认为空。
        client_file_id: 客户端文件标识，默认为空。
    """

    tenant_id = _tenant_from_request(request, user) or ""
    try:
        result = await submission.submit(
            DocumentSubmissionRequest(
                tenant_id=tenant_id,
                user_id=user.user_id,
                username=user.username,
                org_id=user.org_id,
                source=file.file,
                original_filename=file.filename or "unknown",
                content_type=file.content_type,
                upload_id=upload_id,
                client_file_id=client_file_id,
            )
        )
    except DocumentSubmissionError as error:
        raise _ingest_error(error.status_code, error.code, error.message) from error
    return IngestResponse(
        file_name=result.file_name,
        chunks_count=result.chunks_count,
        entities_count=result.entities_count,
        relations_count=result.relations_count,
        status="success",
        doc_id=result.doc_id,
        client_file_id=result.client_file_id,
    )


async def _prepare_folder_member(
    request: Request,
    file: UploadFile,
    entry: FolderManifestFileRequest,
    normalized_path: str,
    manifest: FolderUploadManifestRequest,
    user: UserContext,
    submission: DocumentSubmissionCoordinator,
) -> PreparedFolderMember | IngestResponse:
    """把单个已预校验的文件夹成员委派给提交协调器。

    Args:
        request: 当前 FastAPI 请求。
        file: 上传文件对象。
        entry: 清单中对应的文件条目。
        normalized_path: 归一化后的相对路径。
        manifest: 文件夹上传清单。
        user: 当前认证用户上下文。
        submission: 文档提交协调器。
    """

    tenant_id = _tenant_from_request(request, user) or ""
    original_name = _folder_multipart_filename(
        file.filename or "",
        root_folder_name=manifest.root_folder_name,
        normalized_path=normalized_path,
    )
    try:
        prepared = await submission.prepare_folder_submission(
            FolderSubmissionRequest(
                tenant_id=tenant_id,
                user_id=user.user_id,
                username=user.username,
                org_id=user.org_id,
                source=file.file,
                original_filename=original_name,
                content_type=file.content_type,
                company_id=manifest.company_id,
                department_id=entry.department_id,
                relative_path=entry.relative_path,
                normalized_relative_path=normalized_path,
                metadata=dict(entry.metadata),
                display_name=entry.display_name,
                provenance_source_filename=entry.provenance_source_filename,
                external_source_id=entry.external_source_id,
                upload_id=manifest.upload_id,
                client_file_id=entry.client_file_id,
            )
        )
    except DocumentSubmissionError as error:
        raise _ingest_error(error.status_code, error.code, error.message) from error

    # 未被受理的成员直接转为失败响应，不进入后续发布
    if not prepared.accepted:
        return IngestResponse(
            file_name=prepared.file_name,
            chunks_count=0,
            entities_count=0,
            relations_count=0,
            status="failed",
            doc_id="",
            client_file_id=prepared.client_file_id,
            relative_path=prepared.relative_path,
            department_id=prepared.department_id,
            display_name=prepared.display_name,
            provenance_source_filename=prepared.provenance_source_filename,
            error_code=prepared.error_code,
            message=prepared.message,
        )

    assert prepared.record is not None and prepared.document_input is not None
    return PreparedFolderMember(
        record=prepared.record,
        document_input=prepared.document_input,
        response=IngestResponse(
            file_name=prepared.file_name,
            chunks_count=0,
            entities_count=0,
            relations_count=0,
            status="accepted",
            doc_id=prepared.record.doc_id,
            client_file_id=prepared.client_file_id,
            relative_path=prepared.relative_path,
            department_id=prepared.department_id,
            display_name=prepared.record.display_name,
            provenance_source_filename=prepared.record.provenance_source_filename,
            version=prepared.record.version,
        ),
    )


@ingest_router.get("/progress/{upload_id}", response_model=IngestProgressResponse)
async def upload_progress(
    upload_id: ResourceId,
    request: Request,
    total_count: int = Query(default=1, ge=1, le=100),
    user: UserContext = Depends(require_permission(Permission.DOC_READ)),
    read_provider: DocumentReadProvider = Depends(get_document_read),
):
    """返回单个上传批次在租户范围内的生命周期里程碑。

    Args:
        upload_id: 上传批次 ID。
        request: 当前 FastAPI 请求。
        total_count: 客户端声明的批次文件数（1~100）。
        user: 当前认证用户上下文。
        read_provider: 注入的文档读取提供者。
    """

    # 安全关卡：进度查询限定在当前租户
    tenant_id = _tenant_from_request(request, user) or ""
    if not tenant_id or tenant_id != user.org_id:
        raise _ingest_error(403, "document_tenant_forbidden", "文档进度不属于当前租户")
    normalized_upload_id = _normalize_upload_batch_id(upload_id)
    if not normalized_upload_id:
        raise _ingest_error(400, "upload_progress_id_invalid", "上传批次标识无效")
    expected_count = max(1, min(total_count, settings.upload_max_batch_files))
    read_provider = _resolved_document_read_provider(request, read_provider)
    # ① 取回本批次全部目录记录（按批次标识元数据过滤）
    records = [
        record
        for record in read_provider.list_catalog_documents(tenant_id=tenant_id)
        if record.tenant_id == tenant_id
        and record.metadata.get("_upload_batch_id") == normalized_upload_id
    ]
    items: list[IngestProgressItem] = []
    for record in records:
        stage, stage_index, stage_total, stage_label = document_ingest_stage(
            record.ingest_status,
            record.metadata,
        )
        processing_step, processing_step_index, processing_step_total, processing_step_label = document_ingest_processing_step(
            record.metadata,
        )
        items.append(
            IngestProgressItem(
                doc_id=record.doc_id,
                client_file_id=str(record.metadata.get("_upload_client_file_id", "")),
                file_name=record.uploaded_filename or record.provenance_source_filename or record.display_name,
                status="success" if record.ingest_status == DocumentIngestStatus.INGESTED.value else record.ingest_status,
                ingest_stage=stage,
                ingest_stage_index=stage_index,
                ingest_stage_total=stage_total,
                ingest_stage_label=stage_label,
                processing_step=processing_step,
                processing_step_index=processing_step_index,
                processing_step_total=processing_step_total,
                processing_step_label=processing_step_label,
                error_code=record.error_code,
                message=record.error_code,
            )
        )
    items.sort(key=lambda item: (item.client_file_id, item.doc_id, item.file_name))
    # ② 终态判定：文件数已到齐且全部成功/失败
    terminal_states = {DocumentIngestStatus.INGESTED.value, DocumentIngestStatus.FAILED.value}
    terminal = len(records) >= expected_count and all(record.ingest_status in terminal_states for record in records)
    completed_count = sum(record.ingest_status == DocumentIngestStatus.INGESTED.value for record in records)
    failed_count = sum(record.ingest_status == DocumentIngestStatus.FAILED.value for record in records)
    # ③ 批次整体阶段取最慢文件的阶段；文件未到齐时补 0（表示仍在等待上传）
    observed_indexes = [document_ingest_stage(record.ingest_status, record.metadata)[1] for record in records]
    if len(records) < expected_count:
        observed_indexes.append(0)
    stage_index = min(observed_indexes) if observed_indexes else 0
    if stage_index <= 0:
        stage_label = "上传文件"
    elif failed_count and terminal:
        stage_label = "处理失败"
    else:
        stage_label = {
            1: "上传校验与落盘",
            2: "文件已接收",
            3: "解析与知识抽取",
            4: "入库完成",
        }[stage_index]
    # ④ 处理子步骤取最靠前的活跃文件；无活跃文件时取完成序号最大的文件
    active_candidates = [
        item
        for item in items
        if item.status not in {"success", DocumentIngestStatus.FAILED.value}
    ]
    active_step = (
        min(
            active_candidates,
            key=lambda item: (item.processing_step_index, item.client_file_id, item.doc_id, item.file_name),
        )
        if active_candidates
        else max(
            items,
            key=lambda item: (item.processing_step_index, item.client_file_id, item.doc_id, item.file_name),
        )
        if items
        else None
    )
    processing_step = active_step.processing_step if active_step else "parse"
    processing_step_index = active_step.processing_step_index if active_step else 1
    processing_step_total = active_step.processing_step_total if active_step else 6
    processing_step_label = active_step.processing_step_label if active_step else "解析文件"
    status = "failed" if terminal and failed_count else "success" if terminal else "processing"
    return IngestProgressResponse(
        upload_id=normalized_upload_id,
        total_count=expected_count,
        completed_count=completed_count,
        failed_count=failed_count,
        stage_index=stage_index,
        stage_label=stage_label,
        processing_step=processing_step,
        processing_step_index=processing_step_index,
        processing_step_total=processing_step_total,
        processing_step_label=processing_step_label,
        terminal=terminal,
        status=status,
        items=items,
    )


@ingest_router.post("/upload", response_model=IngestResponse)
@limiter.limit(RATE_LIMITS["doc_upload"], key_func=authenticated_composite_key)
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    user: UserContext = Depends(require_permission(Permission.DOC_WRITE)),
    submission: DocumentSubmissionCoordinator = Depends(get_document_submission),
    upload_id: str = Form(""),
    client_file_id: str = Form(""),
):
    """上传单个文档并触发入库。

    Args:
        request: 当前 FastAPI 请求（限流使用）。
        response: 当前响应对象。
        file: 上传文件对象。
        user: 当前认证用户上下文。
        submission: 文档提交协调器。
        upload_id: 上传批次标识表单字段，可为空。
        client_file_id: 客户端文件标识表单字段，可为空。
    """
    bind_context(user_id=user.user_id, action="doc.upload")
    return await _upload_one(
        request,
        file,
        user,
        submission,
        upload_id=_normalize_upload_batch_id(upload_id),
        client_file_id=client_file_id[:128] if isinstance(client_file_id, str) else "",
    )


@ingest_router.post("/folder", response_model=list[IngestResponse], status_code=202)
@limiter.limit(RATE_LIMITS["doc_batch_upload"], key_func=authenticated_composite_key)
async def upload_folder(
    request: Request,
    response: Response,
    files: list[UploadFile] = File(..., alias="files[]"),
    manifest: str = Form(...),
    user: UserContext = Depends(require_permission(Permission.DOC_WRITE)),
    submission: DocumentSubmissionCoordinator = Depends(get_document_submission),
    lifecycle: DocumentLifecycleCoordinator = Depends(get_document_lifecycle),
    read_provider: DocumentReadProvider = Depends(get_document_read),
):
    """接收一个逻辑文件夹，并把抽取发布为单个持久化后台任务。

    Args:
        request: 当前 FastAPI 请求（限流使用）。
        response: 当前响应对象。
        files: multipart 上传的文件列表。
        manifest: JSON 清单表单字段。
        user: 当前认证用户上下文。
        submission: 文档提交协调器。
        lifecycle: 文档生命周期协调器。
        read_provider: 注入的文档读取提供者。
    """

    bind_context(user_id=user.user_id, action="doc.folder_upload")
    # ① 安全关卡：必须存在与用户一致的有效租户
    tenant_id = _tenant_from_request(request, user) or ""
    if not tenant_id or tenant_id != user.org_id:
        raise _ingest_error(400, "upload_tenant_required", "上传需要有效租户")
    # ② 清单解析失败即拒绝并记录审计
    try:
        parsed_manifest = FolderUploadManifestRequest.model_validate_json(manifest)
    except ValidationError as error:
        _audit_upload_denial(
            submission,
            request,
            user,
            code="upload_manifest_invalid",
            file_count=len(files),
        )
        raise _ingest_error(400, "upload_manifest_invalid", "文件夹 manifest 无效") from error

    parsed_manifest = parsed_manifest.model_copy(
        update={"upload_id": _normalize_upload_batch_id(parsed_manifest.upload_id)}
    )
    # ③ 校验清单整体结构并确保引用的部门就绪
    validated = _validate_folder_manifest(parsed_manifest, files, tenant_id=tenant_id)
    read_provider = _resolved_document_read_provider(request, read_provider)
    _ensure_manifest_departments(
        read_provider,
        parsed_manifest,
        tenant_id=tenant_id,
    )
    concurrency = max(1, min(_BATCH_UPLOAD_CONCURRENCY, len(validated)))
    semaphore = asyncio.Semaphore(concurrency)

    async def prepare_with_limit(
        file: UploadFile,
        entry: FolderManifestFileRequest,
        normalized_path: str,
    ) -> PreparedFolderMember | IngestResponse:
        """在并发上限内准备单个文件夹成员。

        Args:
            file: 上传文件对象。
            entry: 清单中对应的文件条目。
            normalized_path: 归一化后的相对路径。
        """
        async with semaphore:
            return await _prepare_folder_member(
                request, file, entry, normalized_path, parsed_manifest, user, submission
            )

    # ④ 并发受限地准备全部成员
    outcomes = await asyncio.gather(
        *(
            prepare_with_limit(file, entry, normalized_path)
            for file, entry, normalized_path in validated
        )
    )
    prepared = [item for item in outcomes if isinstance(item, PreparedFolderMember)]
    # ⑤ 全部成员都被拒绝时退回 200 同步返回逐文件结果
    if not prepared:
        response.status_code = 200
        return outcomes

    # ⑥ 发布持久化文件夹任务；发布前已失活的文档标记为失败
    try:
        publication = await lifecycle.publish_folder(
            FolderPublicationRequest(
                tenant_id=tenant_id,
                actor=user,
                documents=[
                    AcceptedDocument(
                        record=item.record,
                        document_input=item.document_input,
                        client_file_id=item.response.client_file_id,
                    )
                    for item in prepared
                ],
            )
        )
    except DocumentLifecycleError as error:
        raise _ingest_error(error.status_code, error.code, error.message) from error

    active_doc_ids = set(publication.active_document_ids)
    task_id = publication.task_id
    response.status_code = 202
    return [
        item.response.model_copy(
            update={"status": "processing", "task_id": task_id}
        )
        if isinstance(item, PreparedFolderMember) and item.record.doc_id in active_doc_ids
        else item.response.model_copy(
            update={
                "status": "failed",
                "error_code": "document_no_longer_active",
                "message": "文档在任务发布前已不再处于可处理状态",
            }
        )
        if isinstance(item, PreparedFolderMember)
        else item
        for item in outcomes
    ]


@ingest_router.post("/batch", response_model=list[IngestResponse])
@limiter.limit(RATE_LIMITS["doc_batch_upload"], key_func=authenticated_composite_key)
async def upload_batch(
    request: Request,
    response: Response,
    files: list[UploadFile] = File(...),
    user: UserContext = Depends(require_permission(Permission.DOC_WRITE)),
    submission: DocumentSubmissionCoordinator = Depends(get_document_submission),
    upload_id: str = Form(""),
    client_file_ids: str = Form(""),
):
    """批量上传文档。

    Args:
        request: 当前 FastAPI 请求（限流使用）。
        response: 当前响应对象。
        files: multipart 上传的文件列表。
        user: 当前认证用户上下文。
        submission: 文档提交协调器。
        upload_id: 上传批次标识表单字段，可为空。
        client_file_ids: JSON 数组形式的客户端文件标识表单字段，可为空。
    """
    bind_context(user_id=user.user_id, action="doc.batch_upload")
    file_count = len(files)
    # ① 安全关卡：超出单批上限先审计再拒绝；审计服务不可用时按 503 拒绝
    if file_count > settings.upload_max_batch_files:
        try:
            _audit_upload_denial(
                submission,
                request,
                user,
                code="upload_batch_too_many_files",
                file_count=file_count,
            )
        except Exception as audit_error:
            logger.error(
                "doc_batch_upload_denial_audit_failed",
                error_type=type(audit_error).__name__,
            )
            raise _ingest_error(
                503,
                "audit_unavailable",
                "安全审计服务暂不可用，请稍后重试。",
            ) from audit_error
        raise _ingest_error(
            400,
            "upload_batch_too_many_files",
            f"单次最多上传 {settings.upload_max_batch_files} 份文件。",
        )

    normalized_upload_id = _normalize_upload_batch_id(upload_id)
    # ② 客户端文件标识必须为空或与文件数一致，且不得重复
    try:
        parsed_client_file_ids = json.loads(client_file_ids) if isinstance(client_file_ids, str) and client_file_ids else []
    except json.JSONDecodeError as error:
        raise _ingest_error(400, "upload_client_file_ids_invalid", "上传文件标识无效") from error
    if not isinstance(parsed_client_file_ids, list) or len(parsed_client_file_ids) not in {0, file_count}:
        raise _ingest_error(400, "upload_client_file_ids_invalid", "上传文件标识数量不匹配")
    normalized_client_file_ids = [str(item).strip()[:128] for item in parsed_client_file_ids]
    if len(set(normalized_client_file_ids)) != len(normalized_client_file_ids):
        raise _ingest_error(400, "upload_client_file_ids_invalid", "上传文件标识重复")

    concurrency = max(1, min(_BATCH_UPLOAD_CONCURRENCY, file_count))
    semaphore = asyncio.Semaphore(concurrency)

    async def upload_with_limit(index: int, file: UploadFile) -> IngestResponse:
        """在请求级并发上限内上传单个批次成员。

        Args:
            index: 文件在批次中的序号。
            file: 上传文件对象。
        """
        async with semaphore:
            # 仅在调用方提供对应标识时才传递批次与文件标识
            kwargs = {}
            if normalized_upload_id:
                kwargs["upload_id"] = normalized_upload_id
            if normalized_client_file_ids:
                kwargs["client_file_id"] = normalized_client_file_ids[index]
            return await _upload_one(request, file, user, submission, **kwargs)

    # ③ 并发受限地上传全部文件
    return await asyncio.gather(*(upload_with_limit(index, file) for index, file in enumerate(files)))


