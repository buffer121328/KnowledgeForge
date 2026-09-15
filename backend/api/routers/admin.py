"""Administration, knowledge-update, permission-audit, and health routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel

from api.contracts import ResourceId
from api.dependencies import require_permission
from api.middleware.permission_audit import get_audit_logs
from api.routers.audit import audit_http_error, normalize_audit_time_range
from api.schemas import AuditLogResponse, RequestTrendResponse, StatsResponse, UpdateRequest, UpdateResponse
from domain.documents import DocumentIngestStatus
from domain.identity import Permission, UserContext
from domain.tasks import ChangeType, DocumentChange
from infrastructure.audit.log import (
    AuditAction,
    AuditIntegrityError,
    AuditResult,
    get_audit_service,
)
from infrastructure.audit.trends import get_trend, summarize_trend
from infrastructure.webhooks.service import WebhookEvent, get_webhook_service


class HealthResponse(BaseModel):
    """Expose the stable shallow health payload."""

    status: Literal["ok", "alive"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Expose aggregate and per-component readiness without connection details."""

    status: Literal["ready", "not_ready"]
    components: dict[str, Literal["ready", "unavailable"]]


admin_router = APIRouter(prefix="/admin", tags=["系统管理"])
health_router = APIRouter(prefix="/api", tags=["系统管理"])


_GRAPH_VISIBLE_DOCUMENT_STATUSES = {
    DocumentIngestStatus.INGESTED.value,
    DocumentIngestStatus.LEGACY.value,
}
_EMPTY_GRAPH_STATS = {
    "total_entities": 0,
    "total_relations": 0,
    "nodes": 0,
    "edges": 0,
    "status": "empty",
}


def _summarize_request_details(trend: list[dict]) -> dict[str, dict]:
    """Aggregate bounded 24-hour request details for the dashboard."""
    summary: dict[str, dict] = {"ai": {}, "system_api": {}}
    for point in trend:
        for request_class, detail in (point.get("details") or {}).items():
            if request_class not in summary or not isinstance(detail, dict):
                continue
            target = summary[request_class]
            target["count"] = int(target.get("count", 0)) + int(detail.get("count", 0))
            target["error_count"] = int(target.get("error_count", 0)) + int(detail.get("error_count", 0))
            target["total_latency_ms"] = round(float(target.get("total_latency_ms", 0)) + float(detail.get("total_latency_ms", 0)), 2)
            methods = target.setdefault("methods", {})
            for method, count in (detail.get("methods") or {}).items():
                methods[method] = int(methods.get(method, 0)) + int(count)
            routes = target.setdefault("routes", {})
            for route, values in (detail.get("routes") or {}).items():
                route_target = routes.setdefault(route, {"count": 0, "error_count": 0, "last_status": 0})
                route_target["count"] += int(values.get("count", 0))
                route_target["error_count"] += int(values.get("error_count", 0))
                route_target["last_status"] = int(values.get("last_status", 0))
    return summary


def _has_graph_documents(request: Request, tenant_id: str) -> bool:
    """Return whether the catalog contains current graph-eligible tenant data."""

    catalog = getattr(request.app.state, "document_catalog", None)
    if catalog is None:
        return True
    documents = catalog.list_documents(tenant_id=tenant_id, company_id=tenant_id)
    return any(
        document.department_id
        and document.ingest_status in _GRAPH_VISIBLE_DOCUMENT_STATUSES
        for document in documents
    )


@admin_router.get("/stats", response_model=StatsResponse)
async def get_stats(
    request: Request,
    response: Response,
    user: UserContext = Depends(require_permission(Permission.ADMIN_MANAGE)),
):
    """获取当前租户的系统统计信息"""
    response.headers["Cache-Control"] = "no-store"
    vs_stats = await request.app.state.vector_store.get_stats()
    if user.org_id and _has_graph_documents(request, user.org_id):
        kg_stats = await request.app.state.knowledge_graph.get_stats(tenant_id=user.org_id)
    else:
        kg_stats = dict(_EMPTY_GRAPH_STATS)
    trend = get_trend(user.org_id, "24h")
    return StatsResponse(
        vector_store=vs_stats,
        knowledge_graph=kg_stats,
        request_trend=trend,
        request_detail=_summarize_request_details(trend),
    )


@admin_router.get("/request-trends", response_model=RequestTrendResponse)
async def get_request_trends(
    response: Response,
    window: Literal["60m", "24h"] = Query(default="60m"),
    user: UserContext = Depends(require_permission(Permission.ADMIN_MANAGE)),
) -> RequestTrendResponse:
    """Return lightweight, tenant-isolated request activity for live polling."""
    response.headers["Cache-Control"] = "no-store"
    points = get_trend(user.org_id, window)
    return RequestTrendResponse(
        window=window,
        timezone="UTC",
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        points=points,
        summary=summarize_trend(points),
    )


@admin_router.post("/update", response_model=UpdateResponse)
async def trigger_update(
    request: Request,
    req: UpdateRequest,
    user: UserContext = Depends(require_permission(Permission.ADMIN_MANAGE)),
):
    """手动触发知识更新"""
    tenant_id = getattr(request.state, "tenant_id", "")
    update_wf = request.app.state.workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="Update workflow not initialized")

    change = DocumentChange(
        file_path=req.file_path,
        change_type=ChangeType(req.change_type),
    )
    result = await update_wf.ainvoke({"changes": [change], "tenant_id": tenant_id})
    results = result.get("results", [])
    if not results:
        # 审计日志（失败）
        get_audit_service().log(
            user_id=user.user_id,
            action=AuditAction.DOC_UPDATE,
            resource=f"doc/{req.file_path}",
            result=AuditResult.FAILURE,
            username=user.username,
            ip=request.client.host if request.client else "",
            user_agent=request.headers.get("user-agent", ""),
            org_id=user.org_id,
        )
        raise HTTPException(status_code=500, detail="Update failed")

    updated = results[0]
    # 审计日志（成功）
    get_audit_service().log(
        user_id=user.user_id,
        action=AuditAction.DOC_UPDATE,
        resource=f"doc/{req.file_path}",
        result=AuditResult.SUCCESS if updated.success else AuditResult.FAILURE,
        username=user.username,
        ip=request.client.host if request.client else "",
        user_agent=request.headers.get("user-agent", ""),
        org_id=user.org_id,
        metadata={
            "vectors_added": updated.vectors_added,
            "vectors_deleted": updated.vectors_deleted,
            "entities_added": updated.entities_added,
        },
    )

    # Webhook 触发
    webhook_service = get_webhook_service()
    await webhook_service.trigger(
        WebhookEvent.DOC_UPDATED,
        {
            "file_path": req.file_path,
            "change_type": req.change_type,
            "vectors_added": updated.vectors_added,
            "vectors_deleted": updated.vectors_deleted,
            "entities_added": updated.entities_added,
            "user_id": user.user_id,
            "tenant_id": tenant_id,
        },
        org_id=user.org_id,
    )

    return UpdateResponse(
        file_path=updated.change.file_path,
        vectors_added=updated.vectors_added,
        vectors_deleted=updated.vectors_deleted,
        entities_added=updated.entities_added,
        relations_added=updated.relations_added,
        success=updated.success,
        processing_time_ms=updated.processing_time_ms,
    )


def _liveness_payload() -> HealthResponse:
    """Return the shallow process-liveness payload."""
    return HealthResponse(status="alive", service="AgentKnowledgeHub", version="2.0.0")


@health_router.get("/health", response_model=HealthResponse)
async def health():
    """Preserve the legacy shallow health response for existing callers."""
    return HealthResponse(status="ok", service="AgentKnowledgeHub", version="2.0.0")


@health_router.get("/health/live", response_model=HealthResponse)
async def liveness():
    """Report process liveness without probing external dependencies."""
    return _liveness_payload()


@health_router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def readiness(request: Request, response: Response):
    """Report bounded availability for dependencies required by full API traffic."""
    report = await request.app.state.readiness.check()
    if report["status"] != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report


@admin_router.get("/audit-logs", response_model=list[AuditLogResponse])
async def list_audit_logs(
    user_id: ResourceId | None = None,
    start: str | None = Query(default=None, max_length=64),
    end: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0, le=1_000_000),
    user: UserContext = Depends(require_permission(Permission.ADMIN_AUDIT)),
):
    """查询权限校验审计日志"""
    try:
        start, end = normalize_audit_time_range(start, end)
    except ValueError:
        raise audit_http_error(
            400,
            "invalid_audit_time_range",
            "审计时间范围无效",
        ) from None
    try:
        logs = get_audit_logs(
            user_id=user_id,
            limit=limit,
            org_id=user.org_id,
            start=start,
            end=end,
            offset=offset,
        )
    except AuditIntegrityError:
        raise audit_http_error(
            503,
            "audit_integrity_error",
            "审计记录完整性校验失败",
        ) from None
    return [
        AuditLogResponse(
            timestamp=log.timestamp,
            user_id=log.user_id,
            username=log.username,
            method=log.method,
            path=log.path,
            required_permissions=log.required_permissions,
            user_permissions=log.user_permissions,
            granted=log.granted,
            reason=log.reason,
        )
        for log in logs
    ]
