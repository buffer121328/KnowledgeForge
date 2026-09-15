"""异步任务管理 API

提供:
  POST   /api/tasks/submit          # 提交入库任务
  POST   /api/tasks/submit-batch    # 提交批量入库
  POST   /api/tasks/update          # 提交知识更新
  GET    /api/tasks/{task_id}        # 查询任务状态
  GET    /api/tasks                  # 列出任务
  DELETE /api/tasks/{task_id}        # 取消任务
  POST   /api/tasks/cleanup          # 触发清理任务
"""

from __future__ import annotations

import re
import uuid

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.contracts import BatchLimit, FilePath, ResourceId, StrictRequestModel, TaskMutationResponse
from api.dependencies import get_current_user, require_permission
from domain.identity import Permission, UserContext
from infrastructure.celery_app import celery_app
from infrastructure.documents.local_uploads import LocalUploadStorage, UploadPolicyError
from infrastructure.celery_tasks import batch_ingest_task, cleanup_task, ingest_document_task, knowledge_update_task
from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service
from infrastructure.tasks.task_registry import (
    TaskRecord,
    TaskRegistryUnavailableError,
    get_task_registry,
)
from shared.utils.logging import (
    get_logger,
    safe_file_reference,
    sanitize_for_observability,
)
from shared.utils.metrics import task_registry_operations_total
from shared.utils.task_ids import new_task_id, task_belongs_to_org
from shared.config import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["任务管理"])


def _legacy_generated_task_id(task_id: str, org_id: str) -> bool:
    """Recognize only historical UUID-shaped status IDs for compatibility."""
    return bool(
        task_belongs_to_org(task_id, org_id)
        and re.fullmatch(r"tsk_[0-9a-f]{16}_[0-9a-f]{32}", task_id)
    )


def _require_owned_task(
    task_id: ResourceId,
    user: UserContext,
    operation: str,
) -> TaskRecord:
    """Return the require owned task."""
    try:
        record = get_task_registry().get_owned(task_id, user.org_id)
    except TaskRegistryUnavailableError as error:
        raise _task_error(
            503,
            "task_registry_unavailable",
            "任务注册服务暂不可用，请稍后重试。",
        ) from error
    if record is None:
        get_audit_service().log(
            user_id=user.user_id,
            username=user.username,
            org_id=user.org_id,
            action=AuditAction.TENANT_ACCESS_DENIED,
            resource=f"tenant/task/{task_id}",
            result=AuditResult.DENIED,
            metadata={
                "target_type": "task",
                "target_id": task_id,
                "operation": operation,
            },
        )
        raise HTTPException(status_code=404, detail="任务不存在")
    return record


def _task_error(status_code: int, code: str, message: str) -> HTTPException:
    """Return the task error."""
    request_id = uuid.uuid4().hex
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


def _reserve_task(
    task_id: ResourceId,
    user: UserContext,
    kind: str,
    file_reference: str = "",
) -> None:
    """Return the reserve task."""
    try:
        get_task_registry().reserve(
            task_id,
            org_id=user.org_id,
            actor_id=user.user_id,
            kind=kind,
            file_reference=file_reference,
        )
    except TaskRegistryUnavailableError as error:
        get_audit_service().log(
            user_id=user.user_id,
            username=user.username,
            org_id=user.org_id,
            action=AuditAction.SECURITY_STATE_UNAVAILABLE,
            result=AuditResult.FAILURE,
            resource="task-registry",
            metadata={"operation": "reserve"},
        )
        raise _task_error(
            503,
            "task_registry_unavailable",
            "任务注册服务暂不可用，请稍后重试。",
        ) from error


def _resolve_task_file(reference: str, org_id: str) -> str:
    """Resolve only tenant-owned accepted-upload references for worker dispatch."""
    try:
        return LocalUploadStorage(settings.upload_dir).resolve_accepted(reference, org_id)
    except UploadPolicyError as error:
        raise _task_error(
            404,
            "task_document_not_found",
            "任务文档不存在或不可访问。",
        ) from error


def _publish_failed(task_id: str, error: Exception) -> HTTPException:
    """Publish the failed."""
    try:
        get_task_registry().release(task_id)
    except TaskRegistryUnavailableError:
        pass
    logger.warning(
        "task_dispatch_failed",
        task_id=task_id,
        error_type=type(error).__name__,
    )
    return _task_error(
        503,
        "task_dispatch_unavailable",
        "任务暂无法提交，请稍后重试。",
    )


# ── 请求 / 响应 ────────────────────────────────────────────────

class TaskSubmitRequest(StrictRequestModel):
    """Represent a task submit request."""
    file_path: FilePath


class BatchTaskSubmitRequest(StrictRequestModel):
    """Represent a batch task submit request."""
    file_paths: list[FilePath] = Field(min_length=1, max_length=50)


class UpdateTaskRequest(StrictRequestModel):
    """Represent a update task request."""
    file_path: FilePath
    change_type: Literal["created", "modified", "deleted"] = "modified"


class TaskResponse(BaseModel):
    """Represent a task response."""
    task_id: ResourceId
    status: str
    file_path: str | None = None
    error: str | None = None
    description: str = ""


class TaskStatusResponse(BaseModel):
    """Represent a task status response."""
    task_id: ResourceId
    status: str
    ready: bool
    successful: bool | None = None
    result: dict | None = None
    error: str | None = None
    description: str = ""


# ── 接口 ───────────────────────────────────────────────────────

@router.post("/submit", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_ingest_task(
    req: TaskSubmitRequest,
    user: UserContext = Depends(require_permission(Permission.DOC_WRITE)),
):
    """提交文档入库异步任务"""
    resolved_path = _resolve_task_file(req.file_path, user.org_id)
    task_id = new_task_id(user.org_id)
    _reserve_task(
        task_id,
        user,
        "ingest",
        safe_file_reference(resolved_path),
    )
    try:
        ingest_document_task.apply_async(
            args=[resolved_path, user.org_id, user.user_id],
            task_id=task_id,
            queue="parser",
        )
        get_task_registry().update_state(task_id, "queued")
    except Exception as error:
        raise _publish_failed(task_id, error) from error
    logger.info(
        "task_submitted",
        task_id=task_id,
        file_reference=safe_file_reference(resolved_path),
        user_id=user.user_id,
    )
    task_registry_operations_total.labels("submit", "success").inc()
    get_audit_service().log(
        user_id=user.user_id,
        username=user.username,
        org_id=user.org_id,
        action=AuditAction.TASK_SUBMIT,
        resource=f"task/{task_id}",
        metadata={"kind": "ingest"},
    )
    return TaskResponse(
        task_id=task_id,
        status="PENDING",
        file_path=req.file_path,
        description=f"文档入库：{safe_file_reference(resolved_path)}",
    )


@router.post("/submit-batch", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_batch_task(
    req: BatchTaskSubmitRequest,
    user: UserContext = Depends(require_permission(Permission.DOC_WRITE)),
):
    """提交批量入库任务"""
    if not req.file_paths:
        raise HTTPException(status_code=400, detail="file_paths 不能为空")
    resolved_paths = [
        _resolve_task_file(reference, user.org_id) for reference in req.file_paths
    ]
    task_id = new_task_id(user.org_id)
    _reserve_task(task_id, user, "batch_ingest")
    try:
        batch_ingest_task.apply_async(
            args=[resolved_paths, user.org_id, user.user_id],
            task_id=task_id,
            queue="parser",
        )
        get_task_registry().update_state(task_id, "queued")
    except Exception as error:
        raise _publish_failed(task_id, error) from error
    logger.info(
        "batch_task_submitted",
        task_id=task_id,
        file_count=len(req.file_paths),
        user_id=user.user_id,
    )
    task_registry_operations_total.labels("submit", "success").inc()
    return TaskResponse(
        task_id=task_id,
        status="PENDING",
        description=f"批量文档入库：{len(req.file_paths)} 个文件",
    )


@router.post("/update", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_update_task(
    req: UpdateTaskRequest,
    user: UserContext = Depends(require_permission(Permission.DOC_WRITE)),
):
    """提交知识更新任务"""
    if req.change_type not in {"modified", "created", "deleted"}:
        raise HTTPException(status_code=400, detail="change_type 非法")
    resolved_path = _resolve_task_file(req.file_path, user.org_id)
    task_id = new_task_id(user.org_id)
    _reserve_task(
        task_id,
        user,
        "knowledge_update",
        safe_file_reference(resolved_path),
    )
    try:
        knowledge_update_task.apply_async(
            args=[resolved_path, req.change_type, user.org_id, user.user_id],
            task_id=task_id,
            queue="update",
        )
        get_task_registry().update_state(task_id, "queued")
    except Exception as error:
        raise _publish_failed(task_id, error) from error
    task_registry_operations_total.labels("submit", "success").inc()
    return TaskResponse(
        task_id=task_id,
        status="PENDING",
        file_path=req.file_path,
        description=f"知识更新：{safe_file_reference(resolved_path)}（{req.change_type}）",
    )


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: ResourceId,
    user: UserContext = Depends(get_current_user),
):
    """查询任务状态"""
    try:
        _require_owned_task(task_id, user, "status")
    except HTTPException as error:
        if not (error.status_code == 404 and _legacy_generated_task_id(task_id, user.org_id)):
            raise
    result = celery_app.AsyncResult(task_id)
    response = TaskStatusResponse(
        task_id=task_id,
        status=result.status,
        ready=result.ready(),
    )
    if result.ready():
        response.successful = result.successful()
        if result.successful():
            safe_result = sanitize_for_observability(result.result)
            response.result = (
                safe_result
                if isinstance(safe_result, dict)
                else {"value": safe_result}
            )
        else:
            response.error = "task_execution_failed"
    return response


@router.get("", response_model=list[TaskStatusResponse])
async def list_tasks(
    limit: BatchLimit = 50,
    user: UserContext = Depends(get_current_user),
):
    """List recent tasks from the durable organization registry."""
    try:
        records = get_task_registry().list_owned(user.org_id, limit)
    except TaskRegistryUnavailableError as error:
        raise _task_error(
            503,
            "task_registry_unavailable",
            "任务注册服务暂不可用，请稍后重试。",
        ) from error
    terminal = {"succeeded", "failed", "revoked", "publish_failed"}
    return [
        TaskStatusResponse(
            task_id=record.task_id,
            status=record.state.upper(),
            ready=record.state in terminal,
            successful=(
                record.state == "succeeded"
                if record.state in terminal
                else None
            ),
            error=(
                "task_execution_failed"
                if record.state in {"failed", "publish_failed"}
                else None
            ),
            description=record.description,
        )
        for record in records
    ]


@router.delete("/{task_id}", response_model=TaskMutationResponse)
async def cancel_task(
    task_id: ResourceId,
    user: UserContext = Depends(require_permission(Permission.DOC_WRITE)),
):
    """取消（撤销）任务"""
    _require_owned_task(task_id, user, "cancel")
    celery_app.control.revoke(task_id, terminate=True)
    try:
        get_task_registry().update_state(task_id, "revoked")
    except TaskRegistryUnavailableError as error:
        raise _task_error(
            503,
            "task_registry_unavailable",
            "任务注册服务暂不可用，请稍后重试。",
        ) from error
    logger.info("task_cancelled", task_id=task_id, user_id=user.user_id)
    task_registry_operations_total.labels("cancel", "success").inc()
    get_audit_service().log(
        user_id=user.user_id,
        username=user.username,
        org_id=user.org_id,
        action=AuditAction.TASK_CANCEL,
        resource=f"task/{task_id}",
    )
    return {"task_id": task_id, "status": "REVOKED"}


@router.post("/cleanup", response_model=TaskMutationResponse)
async def trigger_cleanup(
    user: UserContext = Depends(require_permission(Permission.ADMIN_MANAGE)),
):
    """触发清理任务"""
    task_id = new_task_id(user.org_id)
    _reserve_task(task_id, user, "cleanup")
    try:
        cleanup_task.apply_async(task_id=task_id, queue="maintenance")
        get_task_registry().update_state(task_id, "queued")
    except Exception as error:
        raise _publish_failed(task_id, error) from error
    return {"task_id": task_id, "status": "PENDING"}
