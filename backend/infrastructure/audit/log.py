"""Public audit facade with models, service, and global accessor."""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum

from domain.audit import (
    AuditIntegrityError,
    AuditLog,
    AuditStore,
    AuditStoreError,
    AuditWriteError,
)
from shared.utils.logging import get_logger, sanitize_for_observability

from infrastructure.audit.stores import (
    FileAuditStore,
    MemoryAuditStore,
)

logger = get_logger(__name__)


class AuditAction(str, Enum):
    """Represent audit action."""
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token_refresh"
    APIKEY_CREATE = "auth.apikey_create"
    APIKEY_ROTATE = "auth.apikey_rotate"
    APIKEY_SCOPE_UPDATE = "auth.apikey_scope_update"
    APIKEY_TOGGLE = "auth.apikey_toggle"
    APIKEY_REVOKE = "auth.apikey_revoke"
    DOC_UPLOAD = "doc.upload"
    DOC_DELETE = "doc.delete"
    DOC_UPDATE = "doc.update"
    DOC_EXPORT = "doc.export"
    QA_QUERY = "qa.query"
    QA_FEEDBACK = "qa.feedback"
    ADMIN_CONFIG = "admin.config"
    USER_ROLE_CHANGE = "user.role_change"
    PERMISSION_CHECK = "permission.check"
    TENANT_ACCESS_DENIED = "tenant.access_denied"
    RATE_LIMITED = "security.rate_limited"
    TOOL_POLICY_DECISION = "security.tool_policy_decision"
    SECURITY_STATE_UNAVAILABLE = "security.security_state_unavailable"
    TASK_SUBMIT = "task.submit"
    TASK_CANCEL = "task.cancel"
    TASK_STATE_UPDATE = "task.state_update"
    AUDIT_EXPORT = "audit.export"
    BREAKER_OPEN = "security.breaker_open"
    WEBHOOK_REJECTED = "security.webhook_rejected"
    QA_SAFETY_REFUSAL = "security.qa_safety_refusal"
    EVIDENCE_DATASET_CREATE = "evaluation.dataset_create"
    EVIDENCE_DATASET_EDIT = "evaluation.dataset_edit"
    EVIDENCE_DATASET_DELETE = "evaluation.dataset_delete"
    EVIDENCE_DATASET_SUBMIT = "evaluation.dataset_submit"
    EVIDENCE_DATASET_REVIEW = "evaluation.dataset_review"
    EVIDENCE_DATASET_FREEZE = "evaluation.dataset_freeze"
    EVIDENCE_DATASET_DRAFT = "evaluation.dataset_draft"
    EVIDENCE_RELEASE_START = "evaluation.release_start"
    EVALUATION_SPOT_CHECK = "evaluation.spot_check"
    EVIDENCE_RELEASE_REVIEW = "evaluation.release_review"
    EVIDENCE_GATE_PROMOTE = "evaluation.gate_promote"
    EVIDENCE_GATE_ROLLBACK = "evaluation.gate_rollback"
    EVIDENCE_FIXTURE_UPDATE = "evaluation.fixture_update"
    # —— 以下为本次新增的评测审计动作 ——
    EVIDENCE_FIXTURE_VALIDATION_START = "evaluation.fixture_validation_start"  # 固定版本 Fixture 校验任务启动
    EVIDENCE_FIXTURE_VALIDATION_RETRY = "evaluation.fixture_validation_retry"  # 固定版本 Fixture 校验任务重试
    EVIDENCE_CURRENT_CORPUS_DRAFT_CREATE = "evaluation.current_corpus_draft_create"  # 现役语料草稿创建
    EVIDENCE_CURRENT_CORPUS_DRAFT_SUBMIT = "evaluation.current_corpus_draft_submit"  # 现役语料草稿提交
    EVIDENCE_CURRENT_CORPUS_DRAFT_REVIEW = "evaluation.current_corpus_draft_review"  # 现役语料草稿评审
    EVIDENCE_CURRENT_CORPUS_DATASET_CREATE = "evaluation.current_corpus_dataset_create"  # 现役语料数据集创建
    EVIDENCE_CURRENT_CORPUS_TEMPLATE_IMPORT = "evaluation.current_corpus_template_import"  # 现役语料模板导入
    EVIDENCE_REFERENCE_ANSWER_CANDIDATE = "evaluation.reference_answer_candidate"  # 冻结文档候选答案生成
    EVIDENCE_DATASET_REBIND = "evaluation.dataset_rebind"  # 受支持评测集重绑定本地真实 Chunk
    EVIDENCE_ROUTINE_REBIND = "evaluation.routine_rebind"  # 50 条日常基准重绑定本地 Chunk
    EVIDENCE_CURRENT_CORPUS_BULK_SUBMIT = "evaluation.current_corpus_bulk_submit"  # 现役语料批量提交
    EVIDENCE_CURRENT_CORPUS_BULK_REVIEW = "evaluation.current_corpus_bulk_review"  # 现役语料批量评审


class AuditResult(str, Enum):
    """Represent the result of audit processing."""
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"


_SECURITY_EVENT_ACTIONS = frozenset(
    {
        AuditAction.BREAKER_OPEN,
        AuditAction.WEBHOOK_REJECTED,
        AuditAction.QA_SAFETY_REFUSAL,
    }
)
_SAFE_SECURITY_CODE = re.compile(r"^[a-z][a-z0-9_.:-]{0,95}$")
_SAFE_SECURITY_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
_SAFE_SECURITY_METADATA_KEYS = frozenset(
    {
        "dependency",
        "failure_class",
        "question_fingerprint",
        "request_id",
        "security_action",
        "stage",
        "status_class",
        "target_id",
        "target_type",
    }
)


class AuditService:
    """Provide audit operations."""
    def __init__(self, store: AuditStore | None = None) -> None:
        """Initialize the audit service."""
        self.store = store or MemoryAuditStore()

    def log(
        self,
        user_id: str,
        action: AuditAction,
        resource: str = "",
        result: AuditResult = AuditResult.SUCCESS,
        username: str = "",
        ip: str = "",
        user_agent: str = "",
        org_id: str = "",
        metadata: dict | None = None,
        required: bool = False,
    ) -> AuditLog:
        """Record the audit service."""
        safe_metadata = sanitize_for_observability(metadata or {})
        log = AuditLog(
            audit_id=f"aud_{secrets.token_hex(8)}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            username=username,
            action=action.value,
            resource=resource,
            result=result.value,
            ip=ip,
            user_agent=user_agent,
            org_id=org_id,
            metadata=safe_metadata if isinstance(safe_metadata, dict) else {},
        )
        try:
            self.store.append(log)
        except Exception as error:
            logger.error(
                "audit_log_failed",
                error_type=type(error).__name__,
                action=action.value,
            )
            if required:
                if isinstance(error, AuditWriteError):
                    raise
                raise AuditWriteError("required audit record could not be persisted") from error
        return log

    def log_security_event(
        self,
        *,
        user_id: str,
        org_id: str,
        action: AuditAction,
        reason_code: str,
        request_id: str = "",
        metadata: Mapping[str, object] | None = None,
        result: AuditResult = AuditResult.DENIED,
        required: bool = False,
    ) -> AuditLog:
        """Persist a security event through a closed, bounded metadata vocabulary."""

        if action not in _SECURITY_EVENT_ACTIONS:
            raise ValueError("action is not a security-event audit action")
        if not _SAFE_SECURITY_CODE.fullmatch(reason_code):
            raise ValueError("reason_code must be a bounded stable code")

        bounded: dict[str, object] = {"reason_code": reason_code}
        if request_id:
            if not _SAFE_SECURITY_IDENTIFIER.fullmatch(request_id):
                raise ValueError("request_id must be a bounded stable identifier")
            bounded["request_id"] = request_id
        for key, value in (metadata or {}).items():
            if key not in _SAFE_SECURITY_METADATA_KEYS:
                continue
            if isinstance(value, str):
                normalized = value.strip()
                if normalized and _SAFE_SECURITY_IDENTIFIER.fullmatch(normalized):
                    bounded[key] = normalized
            elif isinstance(value, (int, float, bool)):
                bounded[key] = value

        return self.log(
            user_id=user_id,
            org_id=org_id,
            action=action,
            resource=f"security/{action.value.removeprefix('security.')}",
            result=result,
            metadata=bounded,
            required=required,
        )

    def query(
        self,
        user_id: str | None = None,
        action: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        org_id: str | None = None,
        offset: int = 0,
    ) -> list[AuditLog]:
        """Query records through the audit service."""
        return self.store.query(
            user_id=user_id,
            action=action,
            start=start,
            end=end,
            limit=limit,
            org_id=org_id,
            offset=offset,
        )


_audit_service: AuditService | None = None


def get_audit_service() -> AuditService:
    """Return the audit service."""
    global _audit_service
    if _audit_service is None:
        from shared.config import settings

        _audit_service = AuditService(
            FileAuditStore(
                settings.audit_log_file,
                max_bytes=settings.audit_log_max_bytes,
                backup_count=settings.audit_log_backup_count,
            )
        )
    return _audit_service


def set_audit_service(service: AuditService) -> None:
    """Store the audit service."""
    global _audit_service
    _audit_service = service


__all__ = [
    "AuditAction",
    "AuditIntegrityError",
    "AuditLog",
    "AuditResult",
    "AuditService",
    "AuditStore",
    "AuditStoreError",
    "AuditWriteError",
    "FileAuditStore",
    "MemoryAuditStore",
    "get_audit_service",
    "set_audit_service",
]
