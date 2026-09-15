"""审计日志查询 API

提供:
  GET /api/audit/logs             - 查询审计日志
  GET /api/audit/logs/export      - 导出审计日志（CSV）
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel

from api.contracts import ResourceId
from api.dependencies import require_permission
from domain.identity import Permission, UserContext
from infrastructure.audit.log import (
    AuditAction,
    AuditIntegrityError,
    AuditResult,
    AuditWriteError,
    get_audit_service,
)

router = APIRouter(prefix="/audit", tags=["审计日志"])


class AuditLogItem(BaseModel):
    """Represent audit log item."""
    audit_id: str
    timestamp: str
    user_id: str
    username: str
    action: str
    resource: str
    result: str
    ip: str
    user_agent: str
    org_id: str
    metadata: dict
    prev_hash: str
    hash: str


class AuditActionsResponse(BaseModel):
    """List stable audit action and result codes."""

    actions: list[str]
    results: list[str]


def normalize_audit_time_range(
    start: str | None,
    end: str | None,
) -> tuple[str | None, str | None]:
    """Validate aware timestamps and normalize them to UTC ISO-8601."""

    def normalize(value: str | None) -> str | None:
        """Normalize the module."""
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("invalid timestamp") from error
        if parsed.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return parsed.astimezone(timezone.utc).isoformat()

    normalized_start = normalize(start)
    normalized_end = normalize(end)
    if (
        normalized_start is not None
        and normalized_end is not None
        and normalized_start > normalized_end
    ):
        raise ValueError("start must not be after end")
    return normalized_start, normalized_end


def audit_http_error(status_code: int, code: str, message: str) -> HTTPException:
    """Record an audit event for the HTTP error."""
    request_id = uuid4().hex
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


def _csv_safe_cell(value: object) -> object:
    """Prevent spreadsheet formula interpretation of untrusted text fields."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"
    return value


@router.get("/logs", response_model=list[AuditLogItem])
async def query_audit_logs(
    user_id: ResourceId | None = None,
    action: str | None = Query(default=None, max_length=128),
    start: str | None = Query(default=None, max_length=64),
    end: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0, le=1_000_000),
    user: UserContext = Depends(require_permission(Permission.ADMIN_AUDIT)),
):
    """查询审计日志（支持按用户、动作、时间范围过滤）"""
    try:
        start, end = normalize_audit_time_range(start, end)
    except ValueError:
        raise audit_http_error(
            400,
            "invalid_audit_time_range",
            "审计时间范围无效",
        ) from None
    service = get_audit_service()
    try:
        logs = service.query(
            user_id=user_id,
            action=action,
            start=start,
            end=end,
            limit=limit,
            org_id=user.org_id,
            offset=offset,
        )
    except AuditIntegrityError:
        raise audit_http_error(
            503,
            "audit_integrity_error",
            "审计记录完整性校验失败",
        ) from None
    return [AuditLogItem(
        audit_id=log.audit_id,
        timestamp=log.timestamp,
        user_id=log.user_id,
        username=log.username,
        action=log.action,
        resource=log.resource,
        result=log.result,
        ip=log.ip,
        user_agent=log.user_agent,
        org_id=log.org_id,
        metadata=log.metadata,
        prev_hash=log.prev_hash,
        hash=log.hash,
    ) for log in logs]


@router.get("/actions", response_model=AuditActionsResponse)
async def list_audit_actions(
    user: UserContext = Depends(require_permission(Permission.ADMIN_AUDIT)),
):
    """列出所有审计动作类型"""
    return {"actions": [a.value for a in AuditAction], "results": [r.value for r in AuditResult]}


@router.get(
    "/logs/export",
    response_class=Response,
    responses={
        200: {
            "content": {
                "text/csv": {"schema": {"type": "string", "format": "binary"}}
            },
            "description": "CSV audit export",
        }
    },
)
async def export_audit_logs(
    user_id: ResourceId | None = None,
    action: str | None = Query(default=None, max_length=128),
    start: str | None = Query(default=None, max_length=64),
    end: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0, le=1_000_000),
    user: UserContext = Depends(require_permission(Permission.ADMIN_AUDIT)),
):
    """导出审计日志为 CSV"""
    import csv
    import io
    try:
        start, end = normalize_audit_time_range(start, end)
    except ValueError:
        raise audit_http_error(
            400,
            "invalid_audit_time_range",
            "审计时间范围无效",
        ) from None

    service = get_audit_service()
    try:
        logs = service.query(
            user_id=user_id,
            action=action,
            start=start,
            end=end,
            limit=limit,
            org_id=user.org_id,
            offset=offset,
        )
    except AuditIntegrityError:
        raise audit_http_error(
            503,
            "audit_integrity_error",
            "审计记录完整性校验失败",
        ) from None

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "audit_id", "timestamp", "user_id", "username", "action",
        "resource", "result", "ip", "org_id",
    ])
    for log in logs:
        writer.writerow(
            [
                _csv_safe_cell(value)
                for value in (
                    log.audit_id,
                    log.timestamp,
                    log.user_id,
                    log.username,
                    log.action,
                    log.resource,
                    log.result,
                    log.ip,
                    log.org_id,
                )
            ]
        )

    request_id = uuid4().hex
    try:
        service.log(
            user_id=user.user_id,
            username=user.username,
            org_id=user.org_id,
            action=AuditAction.AUDIT_EXPORT,
            resource="audit/export",
            result=AuditResult.SUCCESS,
            metadata={
                "request_id": request_id,
                "filters": {
                    "user_id": user_id,
                    "action": action,
                    "start": start,
                    "end": end,
                    "limit": limit,
                    "offset": offset,
                },
                "record_count": len(logs),
            },
            required=True,
        )
    except AuditWriteError:
        raise audit_http_error(
            503,
            "audit_persistence_unavailable",
            "安全审计暂时不可用",
        ) from None

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=audit_logs.csv",
            "X-Request-ID": request_id,
        },
    )
