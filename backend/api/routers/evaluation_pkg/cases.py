"""评测结果读取与证据数据集治理路由。

运行结果端点保持只读；数据集变更端点只写入组织范围内的编辑工作区，
不会启动基准评测，也不会改变生产证据门模式。
"""

from __future__ import annotations

# MISSING-NAME: case_id
# MISSING-NAME: category
# MISSING-NAME: chunk
# MISSING-NAME: dataset_id
# MISSING-NAME: document
# MISSING-NAME: document_read
# MISSING-NAME: error
# MISSING-NAME: page
# MISSING-NAME: page_size
# MISSING-NAME: payload
# MISSING-NAME: query
# MISSING-NAME: record
# MISSING-NAME: review_status
# MISSING-NAME: runtime_chunks
# MISSING-NAME: user

from typing import Any

from domain.identity import UserContext, UserRole
from evaluation.evidence_gate.review_workspace import (
    EvidenceReviewConflict,
    EvidenceReviewNotFound,
    EvidenceReviewPermissionError,
    EvidenceReviewValidationError,
)
from fastapi import Depends, HTTPException, Query, status
from infrastructure.audit.log import AuditAction, AuditResult

from api.contracts import ResourceId
from api.dependencies import DocumentReadProvider, get_document_read, require_role
from api.schemas import (
    EvidenceBulkActionRequest,
    EvidenceBulkActionResponse,
    EvidenceCurrentCorpusBulkApproveRequest,
    EvidenceCurrentCorpusBulkSubmitRequest,
    EvidenceCaseListResponse,
    EvidenceCaseMutationRequest,
    EvidenceDatasetSummary,
    EvidenceDeleteResponse,
    EvidenceMutationResponse,
    EvidenceReviewerItem,
    EvidenceReviewRequest,
    EvidenceRevisionRequest,
    EvidenceReferenceAnswerCandidateResponse,
    EvidenceRepresentativeTemplateImportResponse,
    EvidenceSubmitRequest,
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
    _error_code,
    _mutate,
    _mutate_batch,
    _mutate_current_corpus_batch,
    _raise_workspace_error,
    _eligible_reviewer_records,
    _submit_with_reviewer,
    _eligible_current_corpus_reviewer_records,
    _submit_current_corpus_cases_to_reviewer,
    _with_reviewer_display_names,
)

@evaluation_router.get("/datasets/{dataset_id}/cases", response_model=EvidenceCaseListResponse)
async def list_evidence_cases(
    dataset_id: ResourceId,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: str | None = Query(None, max_length=128),
    review_status: str | None = Query(None, max_length=64),
    query: str | None = Query(None, max_length=200),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceCaseListResponse:
    """返回一页受限、可过滤的编辑案例。

    Args:
        dataset_id: 数据集 ID。
        page: 页码（从 1 开始）。
        page_size: 每页数量（1~100）。
        category: 过滤条件：分类，可空。
        review_status: 过滤条件：审核状态，可空。
        query: 过滤条件：检索词，可空。
        user: 当前认证用户上下文。
    """
    try:
        return EvidenceCaseListResponse(
            **_with_reviewer_display_names(
                user,
                get_review_workspace().list_cases(
                    user.org_id,
                    dataset_id,
                    page=page,
                    page_size=page_size,
                    category=category,
                    review_status=review_status,
                    query=query,
                ),
            )
        )
    except (EvidenceReviewNotFound, EvidenceReviewConflict, EvidenceReviewValidationError) as error:
        _raise_workspace_error(error)


@evaluation_router.post(
    "/datasets/{dataset_id}/current-corpus-representative-templates",
    response_model=EvidenceRepresentativeTemplateImportResponse,
)
async def import_current_corpus_representative_templates(
    dataset_id: ResourceId,
    payload: EvidenceRevisionRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceRepresentativeTemplateImportResponse:
    """仅从受限的历史代表性模板为现行语料工作区填充可编辑案例。

    Args:
        dataset_id: 数据集 ID。
        payload: 请求体（含期望工作区版本号）。
        user: 当前认证用户上下文。
    """

    try:
        result = get_review_workspace().import_current_corpus_representative_templates(
            user.org_id,
            dataset_id,
            expected_revision=payload.expected_revision,
            actor_id=user.user_id,
        )
    except (
        EvidenceReviewNotFound,
        EvidenceReviewConflict,
        EvidenceReviewPermissionError,
        EvidenceReviewValidationError,
    ) as error:
        _audit(
            user,
            AuditAction.EVIDENCE_CURRENT_CORPUS_TEMPLATE_IMPORT,
            dataset_id=dataset_id,
            revision=payload.expected_revision,
            reason_code=_error_code(error),
            result=(
                AuditResult.DENIED
                if isinstance(error, (EvidenceReviewConflict, EvidenceReviewPermissionError))
                else AuditResult.FAILURE
            ),
        )
        _raise_workspace_error(error)
    _audit(
        user,
        AuditAction.EVIDENCE_CURRENT_CORPUS_TEMPLATE_IMPORT,
        dataset_id=dataset_id,
        revision=int(result["revision"]),
        processed_count=len(result.get("imported_case_ids") or []),
        skipped_count=len(result.get("skipped") or []),
        reason_code="success" if result.get("imported_case_ids") else "no_changes",
        result=AuditResult.SUCCESS,
    )
    return EvidenceRepresentativeTemplateImportResponse(**result)


@evaluation_router.post(
    "/datasets/{dataset_id}/reference-answer-candidates",
    response_model=EvidenceReferenceAnswerCandidateResponse,
)
async def populate_evidence_reference_answer_candidates(
    dataset_id: ResourceId,
    payload: EvidenceRevisionRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceReferenceAnswerCandidateResponse:
    """填入已冻结文档生成的候选答案，仍须经工作台人工复核。"""

    try:
        result = get_review_workspace().populate_reference_answer_candidates(
            user.org_id,
            dataset_id,
            expected_revision=payload.expected_revision,
            actor_id=user.user_id,
        )
    except (
        EvidenceReviewNotFound,
        EvidenceReviewConflict,
        EvidenceReviewPermissionError,
        EvidenceReviewValidationError,
    ) as error:
        _audit(
            user,
            AuditAction.EVIDENCE_REFERENCE_ANSWER_CANDIDATE,
            dataset_id=dataset_id,
            revision=payload.expected_revision,
            reason_code=_error_code(error),
            result=(
                AuditResult.DENIED
                if isinstance(error, (EvidenceReviewConflict, EvidenceReviewPermissionError))
                else AuditResult.FAILURE
            ),
        )
        _raise_workspace_error(error)
    _audit(
        user,
        AuditAction.EVIDENCE_REFERENCE_ANSWER_CANDIDATE,
        dataset_id=dataset_id,
        revision=int(result["revision"]),
        processed_count=len(result["populated_case_ids"]),
        skipped_count=len(result["skipped"]),
        reason_code="success" if result["populated_case_ids"] else "no_changes",
        result=AuditResult.SUCCESS,
    )
    return EvidenceReferenceAnswerCandidateResponse(**result)


@evaluation_router.post(
    "/datasets/{dataset_id}/rebind-current-corpus",
    response_model=EvidenceDatasetSummary,
)
async def rebind_evaluation_suite_to_current_corpus(
    dataset_id: ResourceId,
    payload: EvidenceRevisionRequest,
    document_read: DocumentReadProvider = Depends(get_document_read),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceDatasetSummary:
    """Atomically replace supported suite Context IDs with local runtime chunks."""

    try:
        runtime_chunks: list[dict[str, Any]] = []
        for document in document_read.list_catalog_documents(tenant_id=user.org_id):
            for chunk in await document_read.get_chunks(document.doc_id, tenant_id=user.org_id):
                runtime_chunks.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "chunk_index": chunk.get("chunk_index"),
                        "content": chunk.get("content"),
                        "source_document_id": document.doc_id,
                        "title": document.display_name or document.provenance_source_filename or document.doc_id,
                        "department": document.department_id,
                    }
                )
        result = get_review_workspace().rebind_suite_to_current_chunks(
            user.org_id,
            dataset_id,
            expected_revision=payload.expected_revision,
            actor_id=user.user_id,
            current_chunks=runtime_chunks,
        )
    except (
        EvidenceReviewNotFound,
        EvidenceReviewConflict,
        EvidenceReviewPermissionError,
        EvidenceReviewValidationError,
    ) as error:
        _audit(
            user,
            AuditAction.EVIDENCE_DATASET_REBIND,
            dataset_id=dataset_id,
            revision=payload.expected_revision,
            reason_code=_error_code(error),
            result=AuditResult.DENIED if isinstance(error, (EvidenceReviewConflict, EvidenceReviewPermissionError)) else AuditResult.FAILURE,
        )
        _raise_workspace_error(error)
    _audit(
        user,
        AuditAction.EVIDENCE_DATASET_REBIND,
        dataset_id=dataset_id,
        revision=int(result["revision"]),
        processed_count=int(result["case_count"]),
        reason_code="success",
        result=AuditResult.SUCCESS,
    )
    return EvidenceDatasetSummary(**result)


@evaluation_router.post("/datasets/{dataset_id}/cases", response_model=EvidenceMutationResponse)
async def create_evidence_case(
    dataset_id: ResourceId,
    payload: EvidenceCaseMutationRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceMutationResponse:
    """创建单个证据案例。

    Args:
        dataset_id: 数据集 ID。
        payload: 案例变更请求体。
        user: 当前认证用户上下文。
    """
    # 案例部门缺省继承当前用户部门；仍为空则拒绝创建
    department_id = str(payload.case.get("department_id") or user.department_id or "")
    if not department_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "case_department_required"},
        )
    result = _mutate(
        user,
        AuditAction.EVIDENCE_DATASET_CREATE,
        dataset_id,
        lambda: get_review_workspace().create_case(
            user.org_id,
            dataset_id,
            payload.case,
            expected_revision=payload.expected_revision,
            actor_id=user.user_id,
            actor_department_id=department_id,
        ),
        case_id=str(payload.case.get("id") or ""),
        expected_revision=payload.expected_revision,
    )
    return EvidenceMutationResponse(**_with_reviewer_display_names(user, result))


@evaluation_router.put("/datasets/{dataset_id}/cases/{case_id}", response_model=EvidenceMutationResponse)
async def update_evidence_case(
    dataset_id: ResourceId,
    case_id: ResourceId,
    payload: EvidenceCaseMutationRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceMutationResponse:
    """整体替换单个证据案例。

    Args:
        dataset_id: 数据集 ID。
        case_id: 案例 ID。
        payload: 案例变更请求体。
        user: 当前认证用户上下文。
    """
    result = _mutate(
        user,
        AuditAction.EVIDENCE_DATASET_EDIT,
        dataset_id,
        lambda: get_review_workspace().update_case(
            user.org_id,
            dataset_id,
            case_id,
            payload.case,
            expected_revision=payload.expected_revision,
            actor_id=user.user_id,
            actor_department_id=user.department_id or "",
            actor_is_organization_admin=user.role == UserRole.ORGANIZATION_ADMIN,
        ),
        case_id=case_id,
        expected_revision=payload.expected_revision,
    )
    return EvidenceMutationResponse(**_with_reviewer_display_names(user, result))


@evaluation_router.delete(
    "/datasets/{dataset_id}/cases/{case_id}",
    response_model=EvidenceDeleteResponse,
)
async def delete_evidence_case(
    dataset_id: ResourceId,
    case_id: ResourceId,
    payload: EvidenceRevisionRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceDeleteResponse:
    """删除单个编辑案例；仅组织管理员可删除。

    Args:
        dataset_id: 数据集 ID。
        case_id: 案例 ID。
        payload: 请求体（含期望工作区版本号）。
        user: 当前认证用户上下文。
    """

    result = _mutate(
        user,
        AuditAction.EVIDENCE_DATASET_DELETE,
        dataset_id,
        lambda: get_review_workspace().delete_case(
            user.org_id,
            dataset_id,
            case_id,
            expected_revision=payload.expected_revision,
            actor_id=user.user_id,
            actor_is_organization_admin=user.role == UserRole.ORGANIZATION_ADMIN,
        ),
        case_id=case_id,
        expected_revision=payload.expected_revision,
    )
    return EvidenceDeleteResponse(**result)


@evaluation_router.post("/datasets/{dataset_id}/cases/{case_id}/submit", response_model=EvidenceMutationResponse)
async def submit_evidence_case(
    dataset_id: ResourceId,
    case_id: ResourceId,
    payload: EvidenceSubmitRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceMutationResponse:
    """把单个案例提交给指定审核人。

    Args:
        dataset_id: 数据集 ID。
        case_id: 案例 ID。
        payload: 提交请求体（含期望版本号与审核人 ID）。
        user: 当前认证用户上下文。
    """
    result = _mutate(
        user,
        AuditAction.EVIDENCE_DATASET_SUBMIT,
        dataset_id,
        lambda: _submit_with_reviewer(
            user,
            dataset_id,
            case_id,
            expected_revision=payload.expected_revision,
            reviewer_id=payload.reviewer_id,
        ),
        case_id=case_id,
        expected_revision=payload.expected_revision,
    )
    return EvidenceMutationResponse(**_with_reviewer_display_names(user, result))


@evaluation_router.post(
    "/datasets/{dataset_id}/cases/bulk-submit",
    response_model=EvidenceBulkActionResponse,
)
async def bulk_submit_evidence_cases(
    dataset_id: ResourceId,
    payload: EvidenceBulkActionRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceBulkActionResponse:
    """把选中或按条件过滤出的草稿分派给当前认证的合格管理员。

    Args:
        dataset_id: 数据集 ID。
        payload: 批量操作请求体。
        user: 当前认证用户上下文。
    """
    result = _mutate_batch(
        user,
        AuditAction.EVIDENCE_DATASET_SUBMIT,
        dataset_id,
        payload,
        lambda: get_review_workspace().bulk_submit_cases(
            user.org_id,
            dataset_id,
            expected_revision=payload.expected_revision,
            selection_mode=payload.selection_mode,
            case_ids=list(payload.case_ids),
            category=payload.category,
            review_status=payload.review_status,
            query=payload.query,
            actor_id=user.user_id,
            actor_department_id=user.department_id or "",
            actor_is_department_manager=user.is_department_manager,
            actor_is_organization_admin=user.role == UserRole.ORGANIZATION_ADMIN,
        ),
    )
    return EvidenceBulkActionResponse(**result)


@evaluation_router.post(
    "/datasets/{dataset_id}/cases/bulk-approve",
    response_model=EvidenceBulkActionResponse,
)
async def bulk_approve_evidence_cases(
    dataset_id: ResourceId,
    payload: EvidenceBulkActionRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceBulkActionResponse:
    """批量审核分派给当前认证管理员的选中或过滤案例。

    Args:
        dataset_id: 数据集 ID。
        payload: 批量操作请求体。
        user: 当前认证用户上下文。
    """
    result = _mutate_batch(
        user,
        AuditAction.EVIDENCE_DATASET_REVIEW,
        dataset_id,
        payload,
        lambda: get_review_workspace().bulk_approve_cases(
            user.org_id,
            dataset_id,
            expected_revision=payload.expected_revision,
            selection_mode=payload.selection_mode,
            case_ids=list(payload.case_ids),
            category=payload.category,
            review_status=payload.review_status,
            query=payload.query,
            actor_id=user.user_id,
            actor_department_id=user.department_id or "",
            actor_is_department_manager=user.is_department_manager,
            actor_is_organization_admin=user.role == UserRole.ORGANIZATION_ADMIN,
        ),
    )
    return EvidenceBulkActionResponse(**result)


@evaluation_router.get(
    "/datasets/{dataset_id}/current-corpus/reviewers",
    response_model=list[EvidenceReviewerItem],
)
async def list_current_corpus_bulk_reviewers(
    dataset_id: ResourceId,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> list[EvidenceReviewerItem]:
    """列出可供现行语料全量草稿提交选择的同级公司管理员。

    Args:
        dataset_id: 数据集 ID。
        user: 当前认证用户上下文。
    """
    try:
        records = _eligible_current_corpus_reviewer_records(user, dataset_id)
    except (
        EvidenceReviewNotFound,
        EvidenceReviewPermissionError,
        EvidenceReviewValidationError,
    ) as error:
        _raise_workspace_error(error)
    return [
        EvidenceReviewerItem(
            user_id=str(record["user_id"]),
            username=str(record.get("username") or ""),
            display_name=str(record.get("display_name") or record.get("username") or ""),
        )
        for record in records
    ]


@evaluation_router.post(
    "/datasets/{dataset_id}/current-corpus/bulk-submit",
    response_model=EvidenceBulkActionResponse,
)
async def bulk_submit_current_corpus_cases(
    dataset_id: ResourceId,
    payload: EvidenceCurrentCorpusBulkSubmitRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceBulkActionResponse:
    """把全部现行语料草稿分派给一名显式选择的同级公司管理员。

    Args:
        dataset_id: 数据集 ID。
        payload: 批量提交请求体（含期望版本号与审核人 ID）。
        user: 当前认证用户上下文。
    """
    result = _mutate_current_corpus_batch(
        user,
        AuditAction.EVIDENCE_CURRENT_CORPUS_BULK_SUBMIT,
        dataset_id,
        expected_revision=payload.expected_revision,
        operation=lambda: _submit_current_corpus_cases_to_reviewer(
            user,
            dataset_id,
            expected_revision=payload.expected_revision,
            reviewer_id=payload.reviewer_id,
        ),
    )
    return EvidenceBulkActionResponse(**result)


@evaluation_router.post(
    "/datasets/{dataset_id}/current-corpus/bulk-approve",
    response_model=EvidenceBulkActionResponse,
)
async def bulk_approve_current_corpus_cases(
    dataset_id: ResourceId,
    payload: EvidenceCurrentCorpusBulkApproveRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceBulkActionResponse:
    """仅审核分派给当前认证审核人的现行语料案例。

    Args:
        dataset_id: 数据集 ID。
        payload: 批量审核请求体（含期望版本号）。
        user: 当前认证用户上下文。
    """
    result = _mutate_current_corpus_batch(
        user,
        AuditAction.EVIDENCE_CURRENT_CORPUS_BULK_REVIEW,
        dataset_id,
        expected_revision=payload.expected_revision,
        operation=lambda: get_review_workspace().bulk_approve_current_corpus_cases(
            user.org_id,
            dataset_id,
            expected_revision=payload.expected_revision,
            reviewer_id=user.user_id,
            actor_id=user.user_id,
            actor_department_id=user.department_id or "",
            actor_is_department_manager=user.is_department_manager,
            actor_is_organization_admin=user.role == UserRole.ORGANIZATION_ADMIN,
        ),
    )
    return EvidenceBulkActionResponse(**result)


@evaluation_router.get(
    "/datasets/{dataset_id}/cases/{case_id}/reviewers",
    response_model=list[EvidenceReviewerItem],
)
async def list_evidence_case_reviewers(
    dataset_id: ResourceId,
    case_id: ResourceId,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> list[EvidenceReviewerItem]:
    """列出有资格审核该案例的活跃管理员。

    Args:
        dataset_id: 数据集 ID。
        case_id: 案例 ID。
        user: 当前认证用户上下文。
    """
    try:
        _case, records = _eligible_reviewer_records(user, dataset_id, case_id)
    except (
        EvidenceReviewNotFound,
        EvidenceReviewPermissionError,
        EvidenceReviewValidationError,
    ) as error:
        _raise_workspace_error(error)
    return [
        EvidenceReviewerItem(
            user_id=str(record["user_id"]),
            username=str(record.get("username") or ""),
            display_name=str(record.get("display_name") or record.get("username") or ""),
        )
        for record in records
    ]


@evaluation_router.post("/datasets/{dataset_id}/cases/{case_id}/review", response_model=EvidenceMutationResponse)
async def review_evidence_case(
    dataset_id: ResourceId,
    case_id: ResourceId,
    payload: EvidenceReviewRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceMutationResponse:
    """对单个案例作出审核决定。

    Args:
        dataset_id: 数据集 ID。
        case_id: 案例 ID。
        payload: 审核请求体（含决定与理由）。
        user: 当前认证用户上下文。
    """
    result = _mutate(
        user,
        AuditAction.EVIDENCE_DATASET_REVIEW,
        dataset_id,
        lambda: get_review_workspace().review_case(
            user.org_id,
            dataset_id,
            case_id,
            decision=payload.decision,
            reason=payload.reason,
            expected_revision=payload.expected_revision,
            actor_id=user.user_id,
            actor_department_id=user.department_id or "",
            actor_is_department_manager=True,
            actor_is_organization_admin=user.role == UserRole.ORGANIZATION_ADMIN,
        ),
        case_id=case_id,
        expected_revision=payload.expected_revision,
    )
    return EvidenceMutationResponse(**_with_reviewer_display_names(user, result))


