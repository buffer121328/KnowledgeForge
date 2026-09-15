"""评测结果读取与证据数据集治理路由。

运行结果端点保持只读；数据集变更端点只写入组织范围内的编辑工作区，
不会启动基准评测，也不会改变生产证据门模式。
"""

from __future__ import annotations

# MISSING-NAME: dataset_id
# MISSING-NAME: document_read
# MISSING-NAME: error
# MISSING-NAME: item
# MISSING-NAME: payload
# MISSING-NAME: run_id
# MISSING-NAME: user
# MISSING-NAME: version


from domain.identity import UserContext, UserRole
from evaluation.fixture_validation import (
    FixtureValidationConflict,
    FixtureValidationError,
    build_catalog_fixture_corpus_probe,
)
from evaluation.evidence_gate.review_workspace import (
    EvidenceReviewConflict,
    EvidenceReviewNotFound,
    EvidenceReviewValidationError,
)
from fastapi import Depends, Path as ApiPath
from infrastructure.audit.log import AuditAction, AuditResult

from api.contracts import ResourceId
from api.dependencies import DocumentReadProvider, get_document_read, require_role
from api.schemas import (
    EvidenceFixtureProfileResponse,
    EvidenceFixtureValidationRun,
    EvidenceFixtureValidationRunListResponse,
    EvidenceFixtureValidationStartRequest,
    EvidenceFreezeResponse,
    EvidenceRevisionRequest,
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
    _raise_fixture_validation_error,
    _raise_workspace_error,
    _fixture_validation_service,
    _mutate,
)

@evaluation_router.get("/datasets/{dataset_id}/fixtures", response_model=EvidenceFixtureProfileResponse)
async def get_evidence_fixtures(
    dataset_id: ResourceId,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceFixtureProfileResponse:
    """读取单个数据集的只读 Fixture 档案。

    Args:
        dataset_id: 数据集 ID。
        user: 当前认证用户上下文。
    """
    try:
        return EvidenceFixtureProfileResponse(
            **get_review_workspace().get_fixtures(user.org_id, dataset_id)
        )
    except (EvidenceReviewNotFound, EvidenceReviewConflict, EvidenceReviewValidationError) as error:
        _raise_workspace_error(error)


@evaluation_router.post(
    "/datasets/{dataset_id}/versions/{version}/fixture-validations",
    response_model=EvidenceFixtureValidationRun,
)
async def start_fixture_validation(
    dataset_id: ResourceId,
    payload: EvidenceFixtureValidationStartRequest,
    version: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    document_read: DocumentReadProvider = Depends(get_document_read),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceFixtureValidationRun:
    """针对冻结版本启动一次 Fixture 校验并派发后台任务。

    Args:
        dataset_id: 数据集 ID。
        payload: 启动请求体（含期望工作区版本号）。
        version: 冻结版本号。
        document_read: 文档读取提供者（用于构建目录探针）。
        user: 当前认证用户上下文。
    """
    try:
        dataset = get_review_workspace().get_dataset(user.org_id, dataset_id)
        # 仅 current_corpus 来源的数据集可启动 Fixture 校验
        if dataset.get("source_type") != "current_corpus":
            raise FixtureValidationConflict("fixture_validation_current_corpus_required")
        result = _fixture_validation_service().start(
            org_id=user.org_id,
            dataset_id=dataset_id,
            version=version,
            expected_revision=payload.expected_revision,
            initiated_by=user.user_id,
            corpus_probe=build_catalog_fixture_corpus_probe(document_read.catalog),
        )
        _audit(
            user,
            AuditAction.EVIDENCE_FIXTURE_VALIDATION_START,
            dataset_id=dataset_id,
            result=AuditResult.SUCCESS,
            revision=payload.expected_revision,
            version=version,
        )
        try:
            from infrastructure.tasks.celery_evaluation_tasks import evidence_fixture_validation_task
            evidence_fixture_validation_task.apply_async(args=(result["run_id"], user.org_id, dataset_id, version), task_id=result["run_id"])
        except Exception:
            # 留下可审计的受限失败记录，而不是遗留一个孤立排队任务。
            _fixture_validation_service().fail_dispatch(
                org_id=user.org_id, dataset_id=dataset_id, version=version, run_id=result["run_id"]
            )
        return EvidenceFixtureValidationRun(**_fixture_validation_service().get(
            org_id=user.org_id, dataset_id=dataset_id, version=version, run_id=result["run_id"]
        ))
    except (FixtureValidationError, EvidenceReviewNotFound, EvidenceReviewValidationError) as error:
        _raise_fixture_validation_error(error)


@evaluation_router.get(
    "/datasets/{dataset_id}/versions/{version}/fixture-validations",
    response_model=EvidenceFixtureValidationRunListResponse,
)
async def list_fixture_validations(
    dataset_id: ResourceId,
    version: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceFixtureValidationRunListResponse:
    """列出单个冻结版本的 Fixture 校验运行。

    Args:
        dataset_id: 数据集 ID。
        version: 冻结版本号。
        user: 当前认证用户上下文。
    """
    try:
        return EvidenceFixtureValidationRunListResponse(
            runs=[EvidenceFixtureValidationRun(**item) for item in _fixture_validation_service().list(org_id=user.org_id, dataset_id=dataset_id, version=version)]
        )
    except (FixtureValidationError, EvidenceReviewNotFound, EvidenceReviewValidationError) as error:
        _raise_workspace_error(error)


@evaluation_router.get(
    "/datasets/{dataset_id}/versions/{version}/fixture-validations/{run_id}",
    response_model=EvidenceFixtureValidationRun,
)
async def get_fixture_validation(
    dataset_id: ResourceId,
    run_id: ResourceId,
    version: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceFixtureValidationRun:
    """读取单次 Fixture 校验运行。

    Args:
        dataset_id: 数据集 ID。
        run_id: 校验运行 ID。
        version: 冻结版本号。
        user: 当前认证用户上下文。
    """
    try:
        return EvidenceFixtureValidationRun(**_fixture_validation_service().get(org_id=user.org_id, dataset_id=dataset_id, version=version, run_id=run_id))
    except (FixtureValidationError, EvidenceReviewNotFound, EvidenceReviewValidationError) as error:
        _raise_workspace_error(error)


@evaluation_router.post(
    "/datasets/{dataset_id}/versions/{version}/fixture-validations/{run_id}/retry",
    response_model=EvidenceFixtureValidationRun,
)
async def retry_fixture_validation(
    dataset_id: ResourceId,
    run_id: ResourceId,
    payload: EvidenceFixtureValidationStartRequest,
    version: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    document_read: DocumentReadProvider = Depends(get_document_read),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceFixtureValidationRun:
    """重试一次失败的 Fixture 校验并重新派发后台任务。

    Args:
        dataset_id: 数据集 ID。
        run_id: 校验运行 ID。
        payload: 启动请求体（含期望工作区版本号）。
        version: 冻结版本号。
        document_read: 文档读取提供者（用于构建目录探针）。
        user: 当前认证用户上下文。
    """
    try:
        dataset = get_review_workspace().get_dataset(user.org_id, dataset_id)
        # 仅 current_corpus 来源的数据集可重试 Fixture 校验
        if dataset.get("source_type") != "current_corpus":
            raise FixtureValidationConflict("fixture_validation_current_corpus_required")
        result = _fixture_validation_service().retry(
            org_id=user.org_id,
            dataset_id=dataset_id,
            version=version,
            run_id=run_id,
            expected_revision=payload.expected_revision,
            initiated_by=user.user_id,
            corpus_probe=build_catalog_fixture_corpus_probe(document_read.catalog),
        )
        _audit(user, AuditAction.EVIDENCE_FIXTURE_VALIDATION_RETRY, dataset_id=dataset_id, result=AuditResult.SUCCESS, revision=payload.expected_revision, version=version)
        try:
            from infrastructure.tasks.celery_evaluation_tasks import evidence_fixture_validation_task
            evidence_fixture_validation_task.apply_async(
                args=(result["run_id"], user.org_id, dataset_id, version),
                task_id=result["run_id"],
            )
        except Exception:
            _fixture_validation_service().fail_dispatch(
                org_id=user.org_id,
                dataset_id=dataset_id,
                version=version,
                run_id=result["run_id"],
            )
        return EvidenceFixtureValidationRun(**_fixture_validation_service().get(org_id=user.org_id, dataset_id=dataset_id, version=version, run_id=result["run_id"]))
    except (FixtureValidationError, EvidenceReviewNotFound, EvidenceReviewValidationError) as error:
        _raise_fixture_validation_error(error)


@evaluation_router.post("/datasets/{dataset_id}/freeze", response_model=EvidenceFreezeResponse)
async def freeze_evidence_dataset(
    dataset_id: ResourceId,
    payload: EvidenceRevisionRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceFreezeResponse:
    """冻结数据集生成不可变版本。

    Args:
        dataset_id: 数据集 ID。
        payload: 请求体（含期望工作区版本号）。
        user: 当前认证用户上下文。
    """
    result = _mutate(
        user,
        AuditAction.EVIDENCE_DATASET_FREEZE,
        dataset_id,
        lambda: get_review_workspace().freeze(
            user.org_id,
            dataset_id,
            expected_revision=payload.expected_revision,
            actor_id=user.user_id,
        ),
        expected_revision=payload.expected_revision,
    )
    # 剥除内部产物路径，只返回可公开的冻结标识
    result.pop("version_path", None)
    return EvidenceFreezeResponse(**result)
