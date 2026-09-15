"""评测路由共享的常量、单例、错误映射与审计/变更管道。"""

from __future__ import annotations

import math
from typing import Any, Callable

from domain.identity import UserContext, UserRole
from evaluation.evidence_gate.review_workspace import (
    EvidenceReviewConflict,
    EvidenceReviewNotFound,
    EvidenceReviewPermissionError,
    EvidenceReviewValidationError,
)
from evaluation.fixture_validation import FixtureValidationConflict
from api.schemas import (
    EvidenceBulkActionRequest,
    EvidenceDiagnosticRootCauseResponse,
    EvidenceReleaseAttemptResponse,
    EvidenceDiagnosticSummaryResponse,
    EvidenceReleaseWorkflowResponse,
)
from auth.user_service import UserService
from fastapi import APIRouter, HTTPException, status

from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service
from api.contracts import ResourceId
from shared.config import settings

from evaluation.fixture_validation import (
    FixtureValidationService,
)
from evaluation.current_corpus_drafts import (
    CurrentCorpusDraftNotFound,
    CurrentCorpusDraftConflict,
        CurrentCorpusDraftPermissionError,
    CurrentCorpusDraftService,
)
from evaluation.evidence_gate.review_workspace import EvidenceReviewWorkspace
from shared.paths import (
    BACKEND_ROOT,
    COMPANY_DEMO_CORPUS_MANIFEST,
    EVIDENCE_GATES_DATA_ROOT,
    EVIDENCE_REVIEW_ROOT,
)

def _reviewer_display_names(org_id: str) -> dict[str, str]:
    """把活跃审核人 ID 解析为不含敏感信息的展示名。

    Args:
        org_id: 组织 ID。
    """

    return {
        str(record["user_id"]): str(
            record.get("display_name") or record.get("username") or record["user_id"]
        )
        for record in USER_SERVICE.list_users(
            page=1, page_size=1_000, org_id=org_id, is_active=True
        )
    }


def _with_reviewer_display_names(user: UserContext, result: dict[str, Any]) -> dict[str, Any]:
    """为仅用于 API 输出的案例结果补充审核人展示名，不在工作区持久化用户目录数据。

    Args:
        user: 当前认证用户上下文。
        result: 工作区返回的案例结果字典。
    """

    names = _reviewer_display_names(user.org_id)
    decorated = dict(result)
    # 单案例结果：复制后补充 reviewer_display_name
    if isinstance(decorated.get("case"), dict):
        case = dict(decorated["case"])
        reviewer_id = str(case.get("reviewer_id") or "")
        case["reviewer_display_name"] = names.get(reviewer_id) or None
        editor_id = str(case.get("last_editor_id") or "")
        case["last_editor_display_name"] = names.get(editor_id) or None
        decorated["case"] = case
    # 案例列表结果：逐项补充 reviewer_display_name
    if isinstance(decorated.get("cases"), list):
        decorated["cases"] = [
            {
                **case,
                "reviewer_display_name": names.get(str(case.get("reviewer_id") or ""))
                or None,
                "last_editor_display_name": names.get(str(case.get("last_editor_id") or ""))
                or None,
            }
            for case in decorated["cases"]
        ]
    return decorated


def _with_freezer_display_names(user: UserContext, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add safe display names while retaining stable IDs for reproducibility."""
    names = _reviewer_display_names(user.org_id)
    return [
        {
            **value,
            "last_frozen_by_display_name": names.get(str(value.get("last_frozen_by") or "")) or None,
            "frozen_by_display_name": names.get(str(value.get("frozen_by") or "")) or None,
        }
        for value in values
    ]


def _release_initiator_display_name(user: UserContext, initiated_by: str) -> str:
    """在公司范围内解析可展示的发起人标签，不暴露内部用户 ID。

    Args:
        user: 当前认证用户上下文。
        initiated_by: 发起人用户 ID。
    """

    records = USER_SERVICE.list_users(page=1, page_size=1_000, org_id=user.org_id)
    record = next(
        (item for item in records if str(item.get("user_id") or "") == initiated_by),
        None,
    )
    # 用户已不存在或无可用名称时统一显示为历史用户
    if not record:
        return "历史用户"
    for field in ("display_name", "username"):
        value = str(record.get(field) or "").strip()
        if value:
            return value[:64]
    return "历史用户"


def _release_workflow_response(user: UserContext, result: dict[str, Any]) -> EvidenceReleaseWorkflowResponse:
    """在 API 边界为工作流响应补充发起人展示名；持久化仍基于用户 ID。

    Args:
        user: 当前认证用户上下文。
        result: 工作流结果字典。
    """

    payload = dict(result)
    payload["initiated_by_display_name"] = _release_initiator_display_name(
        user, str(payload.get("initiated_by") or "")
    )
    return EvidenceReleaseWorkflowResponse(**payload)


def _diagnostic_table(value: Any, *, limit: int) -> dict[str, dict[str, int | float | str | None]]:
    """Keep only bounded scalar diagnostic aggregates from one report table."""

    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, int | float | str | None]] = {}
    for name in sorted(value):
        if len(result) >= limit or not isinstance(name, str) or not 0 < len(name) <= 64:
            continue
        item = value[name]
        if not isinstance(item, dict):
            continue
        bounded: dict[str, int | float | str | None] = {}
        for field in _DIAGNOSTIC_VALUE_FIELDS:
            if field not in item:
                continue
            field_value = item.get(field)
            if field_value is None:
                bounded[field] = None
            elif isinstance(field_value, bool):
                continue
            elif isinstance(field_value, int) and 0 <= field_value <= 10_000:
                bounded[field] = field_value
            elif isinstance(field_value, float) and math.isfinite(field_value) and -1_000 <= field_value <= 1_000:
                bounded[field] = field_value
            elif isinstance(field_value, str) and 0 < len(field_value) <= 64:
                bounded[field] = field_value
        result[name] = bounded
    return result


def _diagnostic_summary_response(metrics: dict[str, Any]) -> EvidenceDiagnosticSummaryResponse:
    """Map a persisted diagnostic report through a strict additive allowlist."""

    source = metrics.get("diagnostic_summary")
    if not isinstance(source, dict):
        return EvidenceDiagnosticSummaryResponse(availability="not_available")
    schema_version = source.get("schema_version")
    if schema_version != _DIAGNOSTIC_SUMMARY_SCHEMA:
        return EvidenceDiagnosticSummaryResponse(
            availability="unsupported_schema",
            schema_version=(
                schema_version
                if isinstance(schema_version, str) and 0 < len(schema_version) <= 64
                else None
            ),
        )
    identities = source.get("identities")
    safe_identities = {
        field: value
        for field, value in (identities.items() if isinstance(identities, dict) else ())
        if field in _DIAGNOSTIC_IDENTITY_FIELDS
        and isinstance(value, str)
        and 0 < len(value) <= 200
    }
    route_transitions = source.get("route_transitions")
    safe_transitions = {
        name: count
        for name, count in list(
            sorted(route_transitions.items()) if isinstance(route_transitions, dict) else ()
        )[:32]
        if isinstance(name, str)
        and 0 < len(name) <= 96
        and isinstance(count, int)
        and not isinstance(count, bool)
        and 0 <= count <= 10_000
    }
    hard_gates = source.get("hard_gates")
    safe_gates = list(
        dict.fromkeys(
            value
            for value in (hard_gates if isinstance(hard_gates, list) else [])
            if isinstance(value, str) and 0 < len(value) <= 96
        )
    )[:20]
    case_ids = source.get("case_ids")
    safe_case_ids = list(
        dict.fromkeys(
            value
            for value in (case_ids if isinstance(case_ids, list) else [])
            if isinstance(value, str) and 0 < len(value) <= 128
        )
    )[:20]
    root = source.get("root_cause")
    safe_root = None
    if isinstance(root, dict):
        decision = root.get("decision")
        reason_code = root.get("reason_code")
        if (
            isinstance(decision, str)
            and 0 < len(decision) <= 64
            and isinstance(reason_code, str)
            and 0 < len(reason_code) <= 96
        ):
            compared = root.get("compared_variants")
            hashes = root.get("supporting_hashes")
            safe_root = EvidenceDiagnosticRootCauseResponse(
                decision=decision,
                reason_code=reason_code,
                paired_count=(
                    root.get("paired_count")
                    if isinstance(root.get("paired_count"), int)
                    and not isinstance(root.get("paired_count"), bool)
                    and 0 <= root["paired_count"] <= 10_000
                    else None
                ),
                effect_direction=(
                    root.get("effect_direction")
                    if isinstance(root.get("effect_direction"), str)
                    and 0 < len(root["effect_direction"]) <= 32
                    else None
                ),
                risk=(
                    root.get("risk")
                    if isinstance(root.get("risk"), str) and 0 < len(root["risk"]) <= 96
                    else None
                ),
                recommend_embedding_change=root.get("recommend_embedding_change") is True,
                compared_variants=[
                    value for value in (compared if isinstance(compared, list) else [])
                    if isinstance(value, str) and 0 < len(value) <= 64
                ][:10],
                supporting_hashes=[
                    value for value in (hashes if isinstance(hashes, list) else [])
                    if isinstance(value, str) and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value)
                ][:10],
            )
    return EvidenceDiagnosticSummaryResponse(
        availability="available",
        schema_version=_DIAGNOSTIC_SUMMARY_SCHEMA,
        policy_identities={
            field: value for field, value in safe_identities.items() if "policy" in field
        },
        variant_identities={
            field: value for field, value in safe_identities.items() if "policy" not in field
        },
        categories=_diagnostic_table(source.get("categories"), limit=11),
        stages=_diagnostic_table(source.get("stages"), limit=10),
        metric_coverage=_diagnostic_table(source.get("metric_coverage"), limit=32),
        route_transitions=safe_transitions,
        hard_gates=safe_gates,
        root_cause=safe_root,
        case_ids=safe_case_ids,
    )


def _release_attempt_response(item: dict[str, Any]) -> EvidenceReleaseAttemptResponse:
    """Expose legacy metrics plus a separately sanitized diagnostic envelope."""

    payload = dict(item)
    metrics = dict(payload.get("metrics") or {})
    payload["diagnostic_summary"] = _diagnostic_summary_response(metrics)
    metrics.pop("diagnostic_summary", None)
    payload["metrics"] = metrics
    return EvidenceReleaseAttemptResponse(**payload)


def _raise_release_error(error: Exception) -> None:
    """把发布工作流错误码映射为对应状态码的 HTTPException，并保留原始异常链。

    Args:
        error: 发布流程抛出的异常对象。
    """
    code = str(error).partition(":")[0] or "release_workflow_error"
    # ① 工作流不存在 → 404
    if code in {"workflow_not_found"}:
        status_code = status.HTTP_404_NOT_FOUND
    # ② 业务冲突类（版本冲突/未就绪/仍在运行/校验缺失/快照不一致等）→ 409
    elif code in {
        "workflow_revision_conflict",
        "workflow_not_ready",
        "workflow_still_running",
        "fixture_validation_required",
        "current_corpus_required",
        "current_corpus_draft_required",
        "current_corpus_draft_mismatch",
        "current_corpus_snapshot_mismatch",
        "fixture_runtime_corpus_unaligned",
        "fixture_runtime_corpus_unavailable",
        "release_self_approval_forbidden",
        "release_checker_unavailable",
        "release_already_reviewed",
        "release_not_approved",
        "gate_transition_invalid",
        "gate_no_prior_configuration",
        "gate_history_missing",
        "gate_revision_conflict",
    }:
        status_code = status.HTTP_409_CONFLICT
    # ③ 其余错误 → 422
    else:
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    raise HTTPException(status_code=status_code, detail={"code": code}) from error


def _dispatch_release_workflow(
    workflow: dict[str, Any],
    *,
    user: UserContext,
    dataset_id: ResourceId,
) -> bool:
    """预约并发布一个持久化发布任务，不向外暴露消息中间件故障。

    排队中的工作流是唯一事实来源；发布失败时会刻意保持其排队状态，
    并把任务预约标记为发布失败，之后重试可安全复用同一稳定的工作流任务 ID。

    Args:
        workflow: 工作流结果字典。
        user: 当前认证用户上下文。
        dataset_id: 数据集 ID。
    """
    registry: Any | None = None
    workflow_id = str(workflow["workflow_id"])
    try:
        from infrastructure.celery_tasks import evidence_release_workflow_task
        from infrastructure.tasks.task_registry import get_task_registry
        from shared.utils.metrics import evidence_release_workflows_total

        registry = get_task_registry()
        record = registry.get_owned(workflow_id, user.org_id)
        # 安全关卡：任务不属于发布类型说明被其他业务占用，直接放弃
        if record and record.kind != "evidence_release":
            evidence_release_workflows_total.labels("dispatch", "ownership_conflict").inc()
            return False
        if record and record.state in {"reserved", "queued", "started"}:
            # 既有的持久化预约即已发布或正在投递的 Celery 任务的幂等标记。
            return False
        if not record:
            registry.reserve(
                workflow_id,
                org_id=user.org_id,
                actor_id=user.user_id,
                kind="evidence_release",
                file_reference=f"dataset:{dataset_id}",
            )
        evidence_release_workflow_task.apply_async(
            args=(workflow_id, user.org_id),
            task_id=workflow_id,
        )
        registry.update_state(workflow_id, "queued")
        evidence_release_workflows_total.labels("dispatch", "queued").inc()
        return True
    except Exception:
        # 发布失败：释放预约并记 unavailable 指标，对外仍表现为可稍后重试
        if registry is not None:
            try:
                registry.release(workflow_id)
            except Exception:
                pass
        try:
            from shared.utils.metrics import evidence_release_workflows_total
            evidence_release_workflows_total.labels("dispatch", "unavailable").inc()
        except Exception:
            pass
        return False



def _active_organization_admin_count(org_id: str) -> int:
    """Count enabled company administrators from the trusted identity service."""
    return len(USER_SERVICE.list_users(
        page=1, page_size=1_000, role=UserRole.ORGANIZATION_ADMIN,
        org_id=org_id, is_active=True,
    ))




def _release_service():
    """惰性组装持久化发布编排服务，让路由层不直接依赖数据库细节。"""
    global _RELEASE_SERVICE
    if _RELEASE_SERVICE is None:
        from evaluation.release.workflow import PostgreSQLReleaseWorkflowRepository
        from evaluation.release.runtime import ReleaseWorkflowService
        from infrastructure.postgres.database import get_database_service
        root = BACKEND_ROOT
        _RELEASE_SERVICE = ReleaseWorkflowService(
            PostgreSQLReleaseWorkflowRepository(get_database_service()), REVIEW_WORKSPACE,
            source_bundle=SOURCE_BUNDLE,
            output_root=root / "evaluation" / "results" / "release-workflows",
            # API 与 Worker 使用 Compose 注入的配置，容器内不需要 .env 文件。
            env_file=None,
            fixture_validation_service=_fixture_validation_service(),
        )
    return _RELEASE_SERVICE



_DIAGNOSTIC_SUMMARY_SCHEMA = "category-aware-diagnostic-summary-v1"

_DIAGNOSTIC_IDENTITY_FIELDS = frozenset(
    {
        "category_policy_version", "category_policy_sha256", "variant_plan_sha256",
        "pairing_identity_sha256", "response_snapshot_sha256", "variant_id",
    }
)

_DIAGNOSTIC_VALUE_FIELDS = frozenset(
    {
        "count", "total", "scored", "not_applicable", "failed", "unsupported",
        "mean", "rate", "coverage", "macro", "status",
        "exact_recall_at_5", "source_recall_at_5", "mrr",
    }
)

evaluation_router = APIRouter(prefix="/evaluation", tags=["评测治理"])

RESULTS_ROOT = BACKEND_ROOT / "evaluation" / "results"  # 评测运行结果根目录
REVIEW_ROOT = EVIDENCE_REVIEW_ROOT  # 证据评审工作区根目录（shared.paths 单一来源）
SOURCE_BUNDLE = EVIDENCE_GATES_DATA_ROOT  # 证据门源数据包（shared.paths 单一来源）
COMPANY_DEMO_MANIFEST = COMPANY_DEMO_CORPUS_MANIFEST
REVIEW_WORKSPACE = EvidenceReviewWorkspace(REVIEW_ROOT, SOURCE_BUNDLE)  # 证据评审工作区实例
CURRENT_CORPUS_DRAFT_SERVICE = CurrentCorpusDraftService(REVIEW_ROOT / "current-corpus-drafts")  # 现行语料草稿服务
USER_SERVICE = UserService()  # 用于审核人资格筛选的用户服务
_RELEASE_SERVICE: Any | None = None  # 惰性初始化的发布工作流服务
_FIXTURE_VALIDATION_SERVICE: FixtureValidationService | None = None  # 惰性初始化的 Fixture 校验服务


def _fixture_validation_service() -> FixtureValidationService:
    """惰性构建并复用全局 Fixture 校验服务单例。"""
    global _FIXTURE_VALIDATION_SERVICE
    if _FIXTURE_VALIDATION_SERVICE is None:
        _FIXTURE_VALIDATION_SERVICE = FixtureValidationService(
            REVIEW_WORKSPACE,
            finance_department_id=settings.evaluation_fixture_finance_department_id,
        )
    return _FIXTURE_VALIDATION_SERVICE



def _error_code(error: Exception) -> str:
    """提取异常文本中冒号前的首个错误码，缺省时回退到通用工作区错误码。

    Args:
        error: 工作区抛出的异常对象。
    """
    return str(error).partition(":")[0] or "evidence_workspace_error"


def _raise_workspace_error(error: Exception) -> None:
    """把证据工作区异常映射为对应状态码的 HTTPException，并保留原始异常链。

    Args:
        error: 证据工作区抛出的异常对象。
    """
    code = _error_code(error)
    # 按异常类型选择状态码：权限不足 403 / 未找到 404 / 冲突 409 / 其余 422
    if isinstance(error, EvidenceReviewPermissionError):
        status_code = status.HTTP_403_FORBIDDEN
    elif isinstance(error, EvidenceReviewNotFound):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(error, EvidenceReviewConflict):
        status_code = status.HTTP_409_CONFLICT
    else:
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    raise HTTPException(status_code=status_code, detail={"code": code}) from error


def _raise_fixture_validation_error(error: Exception) -> None:
    """安全地映射校验冲突，不暴露任务或运行时细节。

    Args:
        error: Fixture 校验抛出的异常对象。
    """
    if isinstance(error, FixtureValidationConflict):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": _error_code(error)},
        ) from error
    _raise_workspace_error(error)


def _raise_current_corpus_draft_error(error: Exception) -> None:
    """映射现行语料草稿失败，不泄露目录或身份细节。

    Args:
        error: 现行语料草稿服务抛出的异常对象。
    """
    # 按异常类型选择状态码：未找到 404 / 冲突 409 / 权限不足 403 / 其余 422
    if isinstance(error, CurrentCorpusDraftNotFound):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(error, CurrentCorpusDraftConflict):
        status_code = status.HTTP_409_CONFLICT
    elif isinstance(error, CurrentCorpusDraftPermissionError):
        status_code = status.HTTP_403_FORBIDDEN
    else:
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    raise HTTPException(status_code=status_code, detail={"code": _error_code(error)}) from error


def _role_value(value: Any) -> str:
    """把 UserRole 枚举或原始值统一转换为角色字符串。

    Args:
        value: UserRole 枚举成员或任意可字符串化的角色值。
    """
    return value.value if isinstance(value, UserRole) else str(value)


def _eligible_reviewer_records(
    user: UserContext,
    dataset_id: str,
    case_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读取指定案例并筛选出有资格审核它的组织管理员列表。

    Args:
        user: 当前认证用户上下文。
        dataset_id: 数据集 ID。
        case_id: 案例 ID。
    """
    case = REVIEW_WORKSPACE.get_case(user.org_id, dataset_id, case_id)
    department_id = str(case.get("department_id") or "")
    if not department_id:
        raise EvidenceReviewValidationError("case_department_required")
    # 安全关卡：非组织管理员只能操作本部门案例
    if department_id != user.department_id and user.role != UserRole.ORGANIZATION_ADMIN:
        raise EvidenceReviewPermissionError("department_scope_mismatch")
    maker_id = str(case.get("last_editor_id") or "")
    # Maker-Checker 分离：只有与最近编辑人不同的活跃组织管理员才有资格审核
    reviewers = [
        record
        for record in USER_SERVICE.list_users(
            page=1,
            page_size=100,
            org_id=user.org_id,
            is_active=True,
        )
        if (
            _role_value(record.get("role")) == UserRole.ORGANIZATION_ADMIN.value
            and record.get("user_id") != maker_id
        )
    ]
    return case, reviewers


def _submit_with_reviewer(
    user: UserContext,
    dataset_id: str,
    case_id: str,
    *,
    expected_revision: int,
    reviewer_id: str,
) -> dict[str, Any]:
    """校验审核人资格后把案例提交给指定审核人。

    Args:
        user: 当前认证用户上下文。
        dataset_id: 数据集 ID。
        case_id: 案例 ID。
        expected_revision: 期望的工作区版本号（乐观锁）。
        reviewer_id: 指定审核人用户 ID。
    """
    _case, reviewers = _eligible_reviewer_records(user, dataset_id, case_id)
    # 安全关卡：审核人必须在合格名单内，防止越权指派
    if reviewer_id not in {str(record.get("user_id")) for record in reviewers}:
        raise EvidenceReviewValidationError("reviewer_not_eligible")
    return REVIEW_WORKSPACE.submit_case(
        user.org_id,
        dataset_id,
        case_id,
        expected_revision=expected_revision,
        actor_id=user.user_id,
        actor_department_id=user.department_id or "",
        reviewer_id=reviewer_id,
        actor_is_organization_admin=user.role == UserRole.ORGANIZATION_ADMIN,
    )


def _eligible_current_corpus_reviewer_records(
    user: UserContext,
    dataset_id: str,
) -> list[dict[str, Any]]:
    """为一次现行语料全量草稿提交列出同级公司管理员。

    Args:
        user: 当前认证用户上下文。
        dataset_id: 数据集 ID。
    """
    dataset = REVIEW_WORKSPACE.get_dataset(user.org_id, dataset_id)
    # 仅 current_corpus 来源的数据集允许全量草稿提交
    if dataset.get("source_type") != "current_corpus":
        raise EvidenceReviewValidationError("current_corpus_dataset_required")
    return [
        record
        for record in USER_SERVICE.list_users(
            page=1,
            page_size=1_000,
            org_id=user.org_id,
            is_active=True,
        )
        if (
            _role_value(record.get("role")) == UserRole.ORGANIZATION_ADMIN.value
            and str(record.get("user_id") or "") != user.user_id
        )
    ]


def _submit_current_corpus_cases_to_reviewer(
    user: UserContext,
    dataset_id: str,
    *,
    expected_revision: int,
    reviewer_id: str,
) -> dict[str, Any]:
    """校验审核人资格后把全部现行语料草稿提交给指定审核人。

    Args:
        user: 当前认证用户上下文。
        dataset_id: 数据集 ID。
        expected_revision: 期望的工作区版本号（乐观锁）。
        reviewer_id: 指定审核人用户 ID。
    """
    reviewers = _eligible_current_corpus_reviewer_records(user, dataset_id)
    # 安全关卡：审核人必须在同级管理员名单内，防止越权指派
    if reviewer_id not in {str(record.get("user_id")) for record in reviewers}:
        raise EvidenceReviewValidationError("reviewer_not_eligible")
    return REVIEW_WORKSPACE.bulk_submit_current_corpus_cases(
        user.org_id,
        dataset_id,
        expected_revision=expected_revision,
        reviewer_id=reviewer_id,
        actor_id=user.user_id,
        actor_department_id=user.department_id or "",
        actor_is_department_manager=user.is_department_manager,
        actor_is_organization_admin=user.role == UserRole.ORGANIZATION_ADMIN,
    )


def _audit(
    user: UserContext,
    action: AuditAction,
    *,
    dataset_id: ResourceId,
    result: AuditResult,
    case_id: str = "",
    revision: int | None = None,
    reason_code: str = "",
    version: str = "",
    selection_mode: str = "",
    matched_count: int | None = None,
    processed_count: int | None = None,
    skipped_count: int | None = None,
) -> None:
    """以受限元数据记录一次证据数据集审计事件。

    Args:
        user: 当前认证用户上下文。
        action: 审计动作类型。
        dataset_id: 数据集 ID。
        result: 审计结果（成功/拒绝/失败）。
        case_id: 关联案例 ID，默认为空。
        revision: 关联工作区版本号，可空。
        reason_code: 原因码，默认为空。
        version: 关联冻结版本号，默认为空。
        selection_mode: 批量选择模式，默认为空。
        matched_count: 批量匹配数，可空。
        processed_count: 批量处理数，可空。
        skipped_count: 批量跳过数，可空。
    """
    # 基础元数据；字符串字段统一截断，避免审计日志膨胀
    metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "reason_code": reason_code[:96],
        "department_id": user.department_id or "",
        "is_department_manager": user.is_department_manager,
    }
    if case_id:
        metadata["case_id"] = case_id[:128]
    if revision is not None:
        metadata["revision"] = revision
    if version:
        metadata["version"] = version[:128]
    if selection_mode:
        metadata["selection_mode"] = selection_mode[:16]
    # 计数字段统一收敛到 [0, 10000]，防止异常值写入审计
    for field, value in (
        ("matched_count", matched_count),
        ("processed_count", processed_count),
        ("skipped_count", skipped_count),
    ):
        if value is not None:
            metadata[field] = max(0, min(int(value), 10_000))
    get_audit_service().log(
        user_id=user.user_id,
        username=user.username,
        org_id=user.org_id,
        action=action,
        resource=f"evidence-dataset:{dataset_id}",
        result=result,
        metadata=metadata,
    )


def _mutate(
    user: UserContext,
    action: AuditAction,
    dataset_id: ResourceId,
    operation: Callable[[], dict[str, Any]],
    *,
    case_id: str = "",
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """执行单个数据集变更操作并在成功或失败时记录审计。

    Args:
        user: 当前认证用户上下文。
        action: 审计动作类型。
        dataset_id: 数据集 ID。
        operation: 实际执行工作区变更的可调用对象。
        case_id: 关联案例 ID，默认为空。
        expected_revision: 期望的工作区版本号（乐观锁），可空。
    """
    try:
        result = operation()
    except (
        EvidenceReviewNotFound,
        EvidenceReviewConflict,
        EvidenceReviewPermissionError,
        EvidenceReviewValidationError,
    ) as error:
        # 失败路径：冲突/权限类记为 DENIED，其余记为 FAILURE，再转换为 HTTP 错误
        _audit(
            user,
            action,
            dataset_id=dataset_id,
            case_id=case_id,
            revision=expected_revision,
            reason_code=_error_code(error),
            result=(
                AuditResult.DENIED
                if isinstance(error, (EvidenceReviewConflict, EvidenceReviewPermissionError))
                else AuditResult.FAILURE
            ),
        )
        _raise_workspace_error(error)
    # 成功路径：以工作区返回的权威版本号记录 success 审计
    _audit(
        user,
        action,
        dataset_id=dataset_id,
        case_id=case_id,
        revision=result.get("revision"),
        version=str(result.get("version") or ""),
        reason_code="success",
        result=AuditResult.SUCCESS,
    )
    return result


def _mutate_batch(
    user: UserContext,
    action: AuditAction,
    dataset_id: ResourceId,
    payload: EvidenceBulkActionRequest,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """执行一次批量变更，并只审计有限的聚合元数据。

    Args:
        user: 当前认证用户上下文。
        action: 审计动作类型。
        dataset_id: 数据集 ID。
        payload: 批量操作请求体。
        operation: 实际执行工作区批量变更的可调用对象。
    """
    try:
        result = operation()
    except (
        EvidenceReviewNotFound,
        EvidenceReviewConflict,
        EvidenceReviewPermissionError,
        EvidenceReviewValidationError,
    ) as error:
        # 失败路径：冲突/权限类记为 DENIED，其余记为 FAILURE，再转换为 HTTP 错误
        _audit(
            user,
            action,
            dataset_id=dataset_id,
            revision=payload.expected_revision,
            selection_mode=payload.selection_mode,
            reason_code=_error_code(error),
            result=(
                AuditResult.DENIED
                if isinstance(error, (EvidenceReviewConflict, EvidenceReviewPermissionError))
                else AuditResult.FAILURE
            ),
        )
        _raise_workspace_error(error)
    processed_count = int(result.get("processed_count") or 0)
    skipped_count = int(result.get("skipped_count") or 0)
    # 成功路径：按处理/跳过计数归类为 全部成功 / 部分成功 / 无变更
    reason_code = (
        "success"
        if processed_count and not skipped_count
        else "partial_success"
        if processed_count
        else "no_changes"
    )
    _audit(
        user,
        action,
        dataset_id=dataset_id,
        revision=result.get("revision"),
        selection_mode=payload.selection_mode,
        matched_count=result.get("matched_count"),
        processed_count=processed_count,
        skipped_count=skipped_count,
        reason_code=reason_code,
        result=AuditResult.SUCCESS,
    )
    return result


def _mutate_current_corpus_batch(
    user: UserContext,
    action: AuditAction,
    dataset_id: ResourceId,
    *,
    expected_revision: int,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """执行一次现行语料全量草稿批量操作，并只审计有限的聚合数据。

    Args:
        user: 当前认证用户上下文。
        action: 审计动作类型。
        dataset_id: 数据集 ID。
        expected_revision: 期望的工作区版本号（乐观锁）。
        operation: 实际执行批量变更的可调用对象。
    """
    try:
        result = operation()
    except (
        EvidenceReviewNotFound,
        EvidenceReviewConflict,
        EvidenceReviewPermissionError,
        EvidenceReviewValidationError,
    ) as error:
        # 失败路径：冲突/权限类记为 DENIED，其余记为 FAILURE，再转换为 HTTP 错误
        _audit(
            user,
            action,
            dataset_id=dataset_id,
            revision=expected_revision,
            selection_mode="current_corpus",
            reason_code=_error_code(error),
            result=(
                AuditResult.DENIED
                if isinstance(error, (EvidenceReviewConflict, EvidenceReviewPermissionError))
                else AuditResult.FAILURE
            ),
        )
        _raise_workspace_error(error)
    processed_count = int(result.get("processed_count") or 0)
    skipped_count = int(result.get("skipped_count") or 0)
    # 成功路径：按处理/跳过计数归类为 全部成功 / 部分成功 / 无变更
    reason_code = (
        "success"
        if processed_count and not skipped_count
        else "partial_success"
        if processed_count
        else "no_changes"
    )
    _audit(
        user,
        action,
        dataset_id=dataset_id,
        revision=result.get("revision"),
        selection_mode="current_corpus",
        matched_count=result.get("matched_count"),
        processed_count=processed_count,
        skipped_count=skipped_count,
        reason_code=reason_code,
        result=AuditResult.SUCCESS,
    )
    return result

__all__ = [
    "evaluation_router",
    "RESULTS_ROOT",
    "REVIEW_ROOT",
    "SOURCE_BUNDLE",
    "REVIEW_WORKSPACE",
    "CURRENT_CORPUS_DRAFT_SERVICE",
    "USER_SERVICE",
    "_fixture_validation_service",
    "_release_service",
    "_error_code",
    "_raise_workspace_error",
    "_raise_fixture_validation_error",
    "_raise_current_corpus_draft_error",
    "_raise_release_error",
    "_role_value",
    "_eligible_reviewer_records",
    "_submit_with_reviewer",
    "_eligible_current_corpus_reviewer_records",
    "_submit_current_corpus_cases_to_reviewer",
    "_audit",
    "_mutate",
    "_mutate_batch",
    "_mutate_current_corpus_batch",
    "_reviewer_display_names",
    "_with_reviewer_display_names",
    "_with_freezer_display_names",
    "_release_initiator_display_name",
    "_release_workflow_response",
    "_diagnostic_table",
    "_diagnostic_summary_response",
    "_release_attempt_response",
    "_dispatch_release_workflow",
    "_active_organization_admin_count",
]


def get_review_workspace() -> EvidenceReviewWorkspace:
    """读取工作区单例（函数间接层让测试可对 _common 打补丁）。"""

    return globals()["REVIEW_WORKSPACE"]


def get_current_corpus_draft_service() -> CurrentCorpusDraftService:
    """读取现行语料草稿服务单例。"""

    return globals()["CURRENT_CORPUS_DRAFT_SERVICE"]


def get_user_service() -> UserService:
    """读取审核人目录服务单例。"""

    return globals()["USER_SERVICE"]

def get_company_demo_manifest():
    """读取公司语料身份清单路径（函数间接层让测试可对 _common 打补丁）。"""

    return globals()["COMPANY_DEMO_MANIFEST"]
