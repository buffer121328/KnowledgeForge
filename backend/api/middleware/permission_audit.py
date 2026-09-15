"""Persistent permission-decision auditing with compatibility read models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from infrastructure.audit.log import (
    AuditAction,
    AuditResult,
    AuditService,
    get_audit_service,
)

if TYPE_CHECKING:
    from infrastructure.audit.log import AuditLog


logger = logging.getLogger("api.middleware.permission")

# ── 审计日志记录 ──────────────────────────────────────────────

@dataclass
class PermissionAuditLog:
    """权限校验审计日志"""
    timestamp: float
    user_id: str
    username: str
    method: str
    path: str
    required_permissions: list[str]
    user_permissions: list[str]
    granted: bool
    reason: str = ""
    org_id: str = ""


# 内存审计日志（生产环境替换为数据库/ELK）
_audit_logs: list[PermissionAuditLog] = []
_MAX_AUDIT_LOGS = 10000


def _record_audit(
    user_id: str,
    username: str,
    method: str,
    path: str,
    required: list[str],
    user_perms: list[str],
    granted: bool,
    reason: str = "",
    org_id: str = "",
    ip: str = "",
    user_agent: str = "",
    audit_service: AuditService | None = None,
):
    """Persist a permission decision before updating compatibility state."""
    log = PermissionAuditLog(
        timestamp=time.time(),
        user_id=user_id,
        username=username,
        method=method,
        path=path,
        required_permissions=required,
        user_permissions=user_perms,
        granted=granted,
        reason=reason,
        org_id=org_id,
    )
    service = audit_service or get_audit_service()
    service.log(
        user_id=user_id,
        username=username,
        org_id=org_id,
        ip=ip,
        user_agent=user_agent,
        action=AuditAction.PERMISSION_CHECK,
        resource=f"{method.upper()} {path}",
        result=AuditResult.SUCCESS if granted else AuditResult.DENIED,
        metadata={
            "required_permissions": list(required),
            "user_permissions": list(user_perms),
            "reason_code": "permission_granted" if granted else "permission_denied",
        },
        required=True,
    )

    # Retained only for import compatibility and local diagnostics. Persistent
    # AuditService records are authoritative for API reads.
    _audit_logs.append(log)
    # 滚动淘汰
    if len(_audit_logs) > _MAX_AUDIT_LOGS:
        _audit_logs.pop(0)

    level = logging.INFO if granted else logging.WARNING
    logger.log(
        level,
        "权限校验 %s | user=%s(%s) %s %s | 需要=%s 拥有=%s | %s",
        "通过" if granted else "拒绝",
        username,
        user_id,
        method,
        path,
        required,
        user_perms[:5],  # 只打印前5个
        reason or "",
    )


def get_audit_logs(
    user_id: str | None = None,
    limit: int = 100,
    *,
    org_id: str | None = None,
    action: str | None = None,
    start: str | None = None,
    end: str | None = None,
    offset: int = 0,
    audit_service: AuditService | None = None,
) -> list[PermissionAuditLog]:
    """Query persistent permission records when an organization is supplied."""
    if org_id is not None:
        service = audit_service or get_audit_service()
        records = service.query(
            user_id=user_id,
            action=action or AuditAction.PERMISSION_CHECK.value,
            start=start,
            end=end,
            limit=limit,
            org_id=org_id,
            offset=offset,
        )
        return [_to_permission_log(record) for record in records]

    # Compatibility behavior for callers that still inspect process-local state.
    logs = _audit_logs
    if user_id:
        logs = [l for l in logs if l.user_id == user_id]
    return logs[-limit:]


def _to_permission_log(record: "AuditLog") -> PermissionAuditLog:
    """Convert the module to permission log."""
    metadata = record.metadata
    return PermissionAuditLog(
        timestamp=datetime.fromisoformat(record.timestamp).timestamp(),
        user_id=record.user_id,
        username=record.username,
        method=record.resource.partition(" ")[0],
        path=record.resource.partition(" ")[2],
        required_permissions=list(metadata.get("required_permissions", [])),
        user_permissions=list(metadata.get("user_permissions", [])),
        granted=record.result == AuditResult.SUCCESS.value,
        reason=(
            ""
            if metadata.get("reason_code") == "permission_granted"
            else "权限不足"
        ),
        org_id=record.org_id,
    )
