"""Webhook 管理 API 路由

提供:
  POST   /api/webhooks            - 注册 Webhook
  GET    /api/webhooks            - 列出所有 Webhook
  GET    /api/webhooks/events     - 可订阅事件列表
  DELETE /api/webhooks/{id}       - 删除 Webhook
  POST   /api/webhooks/{id}/test  - 测试 Webhook 连通性
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from api.contracts import MessageResponse, ResourceId, StrictRequestModel
from api.dependencies import require_permission
from domain.identity import Permission, UserContext
from infrastructure.webhooks.delivery import WebhookTargetValidationError
from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service
from infrastructure.webhooks.service import WebhookEvent, get_webhook_service
from shared.utils.ratelimit import RATE_LIMITS, authenticated_composite_key, limiter

router = APIRouter(prefix="/webhooks", tags=["Webhook 管理"])


def _audit_webhook_rejection(
    *,
    user: UserContext,
    target_id: str = "",
    target_type: str = "",
) -> None:
    """Record a bounded target rejection without exposing URL or validator details."""
    metadata: dict[str, str] = {"status_class": "validation"}
    if target_id:
        metadata["target_id"] = target_id
    if target_type:
        metadata["target_type"] = target_type
    try:
        get_audit_service().log_security_event(
            user_id=user.user_id,
            org_id=user.org_id,
            action=AuditAction.WEBHOOK_REJECTED,
            reason_code="target_validation_failed",
            metadata=metadata,
            result=AuditResult.DENIED,
            required=True,
        )
    except Exception:
        # The management operation is already rejected; do not disclose audit internals.
        return


def _audit_cross_org_webhook(
    service: Any,
    webhook_id: ResourceId,
    user: UserContext,
    operation: str,
) -> None:
    """Record an audit event for the cross org webhook."""
    del service
    get_audit_service().log(
        user_id=user.user_id,
        username=user.username,
        org_id=user.org_id,
        action=AuditAction.TENANT_ACCESS_DENIED,
        resource=f"tenant/webhook/{webhook_id}",
        result=AuditResult.DENIED,
        metadata={
            "target_type": "webhook",
            "target_id": webhook_id,
            "operation": operation,
        },
    )


class WebhookCreateRequest(StrictRequestModel):
    """Represent a webhook create request."""
    url: str = Field(min_length=8, max_length=2048)
    events: list[WebhookEvent] = Field(min_length=1, max_length=len(WebhookEvent))
    secret: str | None = Field(default=None, min_length=16, max_length=256)
    is_active: bool = True


class WebhookResponse(BaseModel):
    """Represent a webhook response."""
    id: str
    url: str
    events: list[str]
    secret: str | None = None
    is_active: bool
    created_at: str
    failure_count: int
    last_triggered_at: str | None = None
    last_response_code: int | None = None


class WebhookEventItem(BaseModel):
    """Represent webhook event item."""
    value: str
    label: str


class WebhookMutationResponse(MessageResponse):
    """Return the affected Webhook identifier."""

    webhook_id: ResourceId


class WebhookTestResponse(BaseModel):
    """Return a bounded Webhook connectivity result."""

    success: bool
    message: str | None = None
    error: str | None = None
    status_code: int | None = Field(default=None, ge=100, le=599)


# 事件中文说明（与前端 labels 对齐）
_EVENT_LABELS: dict[str, str] = {
    WebhookEvent.DOC_INGESTED.value: "文档入库",
    WebhookEvent.DOC_UPDATED.value: "文档更新",
    WebhookEvent.DOC_DELETED.value: "文档删除",
    WebhookEvent.QA_COMPLETED.value: "问答完成",
    WebhookEvent.QA_FEEDBACK.value: "问答反馈",
    WebhookEvent.SYSTEM_ERROR.value: "系统错误",
}


def _to_response(wh: Any, reveal_secret: bool = False) -> WebhookResponse:
    """转换为响应对象。reveal_secret=True 时返回明文 secret（仅创建时）"""
    if reveal_secret:
        secret_value: str | None = wh.secret
    else:
        secret_value = f"{wh.secret[:6]}...{wh.secret[-4:]}" if wh.secret else None
    return WebhookResponse(
        id=wh.id,
        url=wh.url,
        events=[e.value for e in wh.events],
        secret=secret_value,
        is_active=wh.is_active,
        created_at=wh.created_at,
        failure_count=wh.failure_count,
        last_triggered_at=wh.last_triggered_at,
        last_response_code=wh.last_response_code,
    )


@router.get("/events", response_model=list[WebhookEventItem])
async def list_webhook_events(
    user: UserContext = Depends(require_permission(Permission.ADMIN_MANAGE)),
):
    """返回后端实际支持的订阅事件"""
    return [
        WebhookEventItem(value=e.value, label=_EVENT_LABELS.get(e.value, e.value))
        for e in WebhookEvent
    ]


@router.post("", response_model=WebhookResponse)
async def register_webhook(
    req: WebhookCreateRequest,
    user: UserContext = Depends(require_permission(Permission.ADMIN_MANAGE)),
):
    """注册 Webhook"""
    try:
        events = list(req.events)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="无效的 Webhook 事件类型",
        ) from error
    if not events:
        raise HTTPException(status_code=400, detail="请至少选择一个有效事件")

    service = get_webhook_service()
    try:
        await service.validate_target(req.url)
        wh = service.register(
            url=req.url,
            events=events,
            secret=req.secret,
            is_active=req.is_active,
            org_id=user.org_id,
        )
    except WebhookTargetValidationError as error:
        _audit_webhook_rejection(user=user, target_type="registration")
        raise HTTPException(status_code=400, detail="Webhook URL 不安全或不可解析") from error
    return _to_response(wh, reveal_secret=True)


@router.get("", response_model=list[WebhookResponse])
async def list_webhooks(
    user: UserContext = Depends(require_permission(Permission.ADMIN_MANAGE)),
):
    """列出所有 Webhook"""
    service = get_webhook_service()
    return [_to_response(wh) for wh in service.list_webhooks(user.org_id)]


@router.delete("/{webhook_id}", response_model=WebhookMutationResponse)
async def delete_webhook(
    webhook_id: ResourceId,
    user: UserContext = Depends(require_permission(Permission.ADMIN_MANAGE)),
):
    """删除 Webhook"""
    service = get_webhook_service()
    if not service.delete(webhook_id, user.org_id):
        _audit_cross_org_webhook(service, webhook_id, user, "delete")
        raise HTTPException(status_code=404, detail="Webhook 不存在")
    return {"message": "已删除", "webhook_id": webhook_id}


@router.post("/{webhook_id}/test", response_model=WebhookTestResponse)
@limiter.limit(RATE_LIMITS["webhook_test"], key_func=authenticated_composite_key)
async def test_webhook(
    webhook_id: ResourceId,
    request: Request,
    response: Response,
    user: UserContext = Depends(require_permission(Permission.ADMIN_MANAGE)),
):
    """测试 Webhook 连通性（真实 HTTP POST）"""
    service = get_webhook_service()
    try:
        result = await service.test(webhook_id, user.org_id)
    except WebhookTargetValidationError as error:
        _audit_webhook_rejection(user=user, target_id=webhook_id, target_type="test")
        raise HTTPException(status_code=400, detail="Webhook URL 不安全或不可解析") from error
    if result.get("error") == "Webhook 不存在":
        _audit_cross_org_webhook(service, webhook_id, user, "test")
        raise HTTPException(status_code=404, detail="Webhook 不存在")
    return result
