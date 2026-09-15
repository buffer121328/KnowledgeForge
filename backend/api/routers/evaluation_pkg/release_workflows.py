"""评测结果读取与证据数据集治理路由。

运行结果端点保持只读；数据集变更端点只写入组织范围内的编辑工作区，
不会启动基准评测，也不会改变生产证据门模式。
"""

from __future__ import annotations

# MISSING-NAME: dataset_id
# MISSING-NAME: document_read
# MISSING-NAME: error
# MISSING-NAME: item
# MISSING-NAME: limit
# MISSING-NAME: payload
# MISSING-NAME: user
# MISSING-NAME: version
# MISSING-NAME: workflow_id


from domain.identity import UserContext, UserRole
from evaluation.fixture_validation import (
    build_catalog_fixture_corpus_probe,
)
from fastapi import Depends, Path as ApiPath, Query
from infrastructure.audit.log import AuditAction, AuditResult

from api.contracts import ResourceId
from api.dependencies import DocumentReadProvider, get_document_read, require_role
from api.schemas import (
    EvidenceReleaseAttemptListResponse,
    EvidenceReleaseWorkflowActionRequest,
    EvidenceReleaseWorkflowReviewRequest,
    EvidenceGateConfigurationResponse,
    EvidenceGateRollbackRequest,
    EvidenceReleaseWorkflowListResponse,
    EvidenceReleaseWorkflowResponse,
    EvidenceReleaseWorkflowStartRequest,
)


_DIAGNOSTIC_SUMMARY_SCHEMA = "category-aware-diagnostic-summary-v1"
_DIAGNOSTIC_VALUE_FIELDS = frozenset(
    {
        "count", "total", "scored", "not_applicable", "failed", "unsupported",
        "mean", "rate", "coverage", "macro", "status",
        "exact_recall_at_5", "source_recall_at_5", "mrr",
    }
)
_DIAGNOSTIC_IDENTITY_FIELDS = frozenset(
    {
        "category_policy_version", "category_policy_sha256", "variant_plan_sha256",
        "pairing_identity_sha256", "response_snapshot_sha256", "variant_id",
    }
)

from api.routers.evaluation_pkg._common import (
    evaluation_router,  # noqa: F401
    get_review_workspace,
    _audit,
    _fixture_validation_service,
    _release_service,
    _raise_release_error,
    _release_workflow_response,
    _release_attempt_response,
    _dispatch_release_workflow,
    _active_organization_admin_count,
)

@evaluation_router.post("/datasets/{dataset_id}/versions/{version}/release-workflows", response_model=EvidenceReleaseWorkflowResponse)
async def start_evidence_release_workflow(
    dataset_id: ResourceId,
    version: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    payload: EvidenceReleaseWorkflowStartRequest = ...,
    document_read: DocumentReadProvider = Depends(get_document_read),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceReleaseWorkflowResponse:
    """从已审核冻结的语料版本启动冒烟、日常或正式评测。"""
    try:
        # ① 路径指定的版本必须属于当前组织、来源为 current_corpus 且已冻结。
        dataset = get_review_workspace().get_dataset(user.org_id, dataset_id)
        if dataset.get("source_type") != "current_corpus":
            raise RuntimeError("current_corpus_required")
        if dataset.get("status") != "frozen":
            raise RuntimeError("frozen_version_required")
        frozen_version = get_review_workspace().get_version(user.org_id, dataset_id, version)
        # ② 冻结版本自身负责提供不可变 manifest；不再读取草稿或快照血缘。
        manifest_sha256 = str(frozen_version["manifest_sha256"])
        _fixture_validation_service().assert_runtime_corpus_aligned(
            org_id=user.org_id,
            dataset_id=dataset_id,
            version=version,
            manifest_sha256=manifest_sha256,
            corpus_probe=build_catalog_fixture_corpus_probe(document_read.catalog),
        )
        # ③ 评测套件必须是已审核冻结的受支持版本；冒烟 12、日常 50、正式 100
        # 都通过同一份不可变 manifest 校验。
        formal_suite = get_review_workspace().get_frozen_evaluation_suite(
            user.org_id,
            payload.evaluation_dataset_id,
            payload.evaluation_version,
        )
        supported_suite_counts = {
            "evidence-gates-v1": 100,
            "evidence-gates-v1-routine": 50,
        }
        if formal_suite["dataset_id"] == dataset_id and int(formal_suite["case_count"]) == 12:
            supported_suite_counts[dataset_id] = 12
        if (
            formal_suite["dataset_id"] not in supported_suite_counts
            or int(formal_suite["case_count"]) != supported_suite_counts[formal_suite["dataset_id"]]
        ):
            raise RuntimeError("formal_evaluation_suite_required")
        result = _release_service().start(
            company_namespace=user.org_id,
            dataset_id=dataset_id,
            version=version,
            actor_id=user.user_id,
            evaluation_dataset_id=str(formal_suite["dataset_id"]),
            evaluation_version=str(formal_suite["version"]),
            evaluation_manifest_sha256=str(formal_suite["manifest_sha256"]),
            evaluation_case_count=int(formal_suite["case_count"]),
        )
        _audit(
            user,
            AuditAction.EVIDENCE_RELEASE_START,
            dataset_id=dataset_id,
            result=AuditResult.SUCCESS,
            revision=int(dataset.get("revision") or 0),
            version=version,
        )
        if result["status"] == "queued":
            _dispatch_release_workflow(result, user=user, dataset_id=dataset_id)
        return _release_workflow_response(user, result)
    except Exception as error:
        if type(error).__name__ in {
            "ReleaseWorkflowError",
            "DatabaseUnavailableError",
            "RuntimeError",
            "FixtureValidationError",
            "FixtureValidationConflict",
            "EvidenceReviewNotFound",
            "EvidenceReviewConflict",
            "EvidenceReviewValidationError",
        }:
            _raise_release_error(error)
        raise



@evaluation_router.get("/release-workflows", response_model=EvidenceReleaseWorkflowListResponse)
async def list_evidence_release_workflows(
    limit: int = Query(default=20, ge=1, le=100),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceReleaseWorkflowListResponse:
    """仅列出当前认证组织有限的持久化发布历史。

    Args:
        limit: 返回数量上限（1~100）。
        user: 当前认证用户上下文。
    """
    try:
        workflows = _release_service().repository.list(
            company_namespace=user.org_id,
            limit=limit,
        )
        return EvidenceReleaseWorkflowListResponse(
            workflows=[_release_workflow_response(user, item) for item in workflows]
        )
    except Exception as error:
        _raise_release_error(error)


@evaluation_router.get("/release-workflows/{workflow_id}", response_model=EvidenceReleaseWorkflowResponse)
async def get_evidence_release_workflow(
    workflow_id: ResourceId, user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceReleaseWorkflowResponse:
    """读取单个发布工作流的当前状态。

    Args:
        workflow_id: 工作流 ID。
        user: 当前认证用户上下文。
    """
    try: return _release_workflow_response(user, _release_service().repository.get(workflow_id, company_namespace=user.org_id))
    except Exception as error: _raise_release_error(error)


@evaluation_router.get("/release-workflows/{workflow_id}/attempts", response_model=EvidenceReleaseAttemptListResponse)
async def list_evidence_release_attempts(
    workflow_id: ResourceId, user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceReleaseAttemptListResponse:
    """列出单个发布工作流的全部尝试记录。

    Args:
        workflow_id: 工作流 ID。
        user: 当前认证用户上下文。
    """
    try:
        service = _release_service()
        # 安全关卡：先确认工作流属于当前组织，再返回尝试记录
        service.repository.get(workflow_id, company_namespace=user.org_id)
        return EvidenceReleaseAttemptListResponse(
            attempts=[
                _release_attempt_response(item)
                for item in service.repository.attempts(workflow_id)
            ]
        )
    except Exception as error: _raise_release_error(error)


@evaluation_router.post("/release-workflows/{workflow_id}/retry", response_model=EvidenceReleaseWorkflowResponse)
async def retry_evidence_release_workflow(
    workflow_id: ResourceId, payload: EvidenceReleaseWorkflowActionRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceReleaseWorkflowResponse:
    """按乐观锁重试一个发布工作流并重新派发后台任务。

    Args:
        workflow_id: 工作流 ID。
        payload: 工作流操作请求体（含期望版本号）。
        user: 当前认证用户上下文。
    """
    try:
        service = _release_service()
        work = service.repository.get(workflow_id, company_namespace=user.org_id)
        queued = service.retry(workflow_id, expected_revision=payload.expected_revision)
        _dispatch_release_workflow(queued, user=user, dataset_id=work["dataset_id"])
        return _release_workflow_response(user, queued)
    except Exception as error: _raise_release_error(error)


@evaluation_router.post("/release-workflows/{workflow_id}/approve", response_model=EvidenceReleaseWorkflowResponse)
async def approve_evidence_release_workflow(
    workflow_id: ResourceId, payload: EvidenceReleaseWorkflowReviewRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceReleaseWorkflowResponse:
    """Approve completed release evidence using organization-admin maker-checker."""
    try:
        service = _release_service()
        service.repository.get(workflow_id, company_namespace=user.org_id)
        result = service.review(
            workflow_id, expected_revision=payload.expected_revision, reviewer_id=user.user_id,
            decision="approved", reason=payload.reason, target_mode=payload.target_mode,
            active_organization_admin_count=_active_organization_admin_count(user.org_id),
        )
        _audit(user, AuditAction.EVIDENCE_RELEASE_REVIEW, dataset_id=result["dataset_id"],
               result=AuditResult.SUCCESS, revision=result["revision"], version=result["version"])
        return _release_workflow_response(user, result)
    except Exception as error:
        _raise_release_error(error)


@evaluation_router.post("/release-workflows/{workflow_id}/reject", response_model=EvidenceReleaseWorkflowResponse)
async def reject_evidence_release_workflow(
    workflow_id: ResourceId, payload: EvidenceReleaseWorkflowReviewRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceReleaseWorkflowResponse:
    """Reject completed release evidence using organization-admin maker-checker."""
    try:
        service = _release_service()
        service.repository.get(workflow_id, company_namespace=user.org_id)
        result = service.review(
            workflow_id, expected_revision=payload.expected_revision, reviewer_id=user.user_id,
            decision="rejected", reason=payload.reason, target_mode=None,
            active_organization_admin_count=_active_organization_admin_count(user.org_id),
        )
        _audit(user, AuditAction.EVIDENCE_RELEASE_REVIEW, dataset_id=result["dataset_id"],
               result=AuditResult.SUCCESS, revision=result["revision"], version=result["version"])
        return _release_workflow_response(user, result)
    except Exception as error:
        _raise_release_error(error)


@evaluation_router.post("/release-workflows/{workflow_id}/promote", response_model=EvidenceReleaseWorkflowResponse)
async def promote_evidence_release_workflow(
    workflow_id: ResourceId, payload: EvidenceReleaseWorkflowActionRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceReleaseWorkflowResponse:
    """Activate the approved next company Gate mode without deployment changes."""
    try:
        service = _release_service()
        service.repository.get(workflow_id, company_namespace=user.org_id)
        result = service.promote(workflow_id, expected_revision=payload.expected_revision, actor_id=user.user_id)
        _audit(user, AuditAction.EVIDENCE_GATE_PROMOTE, dataset_id=result["dataset_id"],
               result=AuditResult.SUCCESS, revision=result["revision"], version=result["version"])
        return _release_workflow_response(user, result)
    except Exception as error:
        _raise_release_error(error)


@evaluation_router.get("/gate", response_model=EvidenceGateConfigurationResponse)
async def get_evidence_gate_configuration(
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceGateConfigurationResponse:
    """读取公司 Gate 的受控当前配置，不返回原始评测内容。"""
    try:
        return EvidenceGateConfigurationResponse(**_release_service().repository.gate())
    except Exception as error:
        _raise_release_error(error)


@evaluation_router.post("/gate/rollback", response_model=EvidenceGateConfigurationResponse)
async def rollback_evidence_gate_configuration(
    payload: EvidenceGateRollbackRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceGateConfigurationResponse:
    """按配置历史回滚一个 Gate 级别，不触发部署或服务重启。"""
    try:
        result = _release_service().rollback(
            expected_revision=payload.expected_revision,
            actor_id=user.user_id,
            reason=payload.reason,
        )
        _audit(
            user, AuditAction.EVIDENCE_GATE_ROLLBACK, dataset_id="evidence-gate",
            result=AuditResult.SUCCESS, revision=result["revision"],
            version=str(result.get("calibration_version") or ""),
        )
        return EvidenceGateConfigurationResponse(**result)
    except Exception as error:
        _raise_release_error(error)


