"""评测结果读取与证据数据集治理路由。

运行结果端点保持只读；数据集变更端点只写入组织范围内的编辑工作区，
不会启动基准评测，也不会改变生产证据门模式。
"""

from __future__ import annotations

# MISSING-NAME: COMPANY_DEMO_MANIFEST
# MISSING-NAME: chunk
# MISSING-NAME: current_chunks
# MISSING-NAME: dataset_id
# MISSING-NAME: document
# MISSING-NAME: document_read
# MISSING-NAME: error
# MISSING-NAME: limit
# MISSING-NAME: payload
# MISSING-NAME: user
# MISSING-NAME: version

import json
from typing import Any

from domain.identity import UserContext, UserRole
from evaluation.evidence_gate.review_workspace import (
    EvidenceReviewConflict,
    EvidenceReviewNotFound,
    EvidenceReviewPermissionError,
    EvidenceReviewValidationError,
)
from fastapi import Depends, HTTPException, Path as ApiPath, Query, status
from infrastructure.audit.log import AuditAction, AuditResult

from api.contracts import ResourceId
from api.dependencies import DocumentReadProvider, get_document_read, require_role
from api.schemas import (
    EvidenceDatasetListResponse,
    EvidenceDatasetSummary,
    EvidenceDraftFromVersionRequest,
    EvidenceFrozenVersion,
    EvidenceFrozenVersionListResponse,
    EvidenceSnapshotContextListResponse,
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
    _raise_workspace_error,
    _mutate,
    _with_freezer_display_names,
    get_company_demo_manifest,
)

@evaluation_router.get("/datasets", response_model=EvidenceDatasetListResponse)
async def list_evidence_datasets(
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceDatasetListResponse:
    """仅列出当前认证组织工作区中的数据集。

    Args:
        user: 当前认证用户上下文。
    """
    try:
        items = _with_freezer_display_names(user, get_review_workspace().list_datasets(user.org_id))
        return EvidenceDatasetListResponse(datasets=[EvidenceDatasetSummary(**item) for item in items])
    except (
        EvidenceReviewNotFound,
        EvidenceReviewConflict,
        EvidenceReviewPermissionError,
        EvidenceReviewValidationError,
    ) as error:
        _raise_workspace_error(error)


@evaluation_router.get("/datasets/{dataset_id}", response_model=EvidenceDatasetSummary)
async def get_evidence_dataset(
    dataset_id: ResourceId,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceDatasetSummary:
    """返回当前组织范围内单个数据集的摘要。

    Args:
        dataset_id: 数据集 ID。
        user: 当前认证用户上下文。
    """
    try:
        item = _with_freezer_display_names(user, [get_review_workspace().get_dataset(user.org_id, dataset_id)])[0]
        return EvidenceDatasetSummary(**item)
    except (EvidenceReviewNotFound, EvidenceReviewConflict, EvidenceReviewValidationError) as error:
        _raise_workspace_error(error)


@evaluation_router.get("/datasets/{dataset_id}/versions", response_model=EvidenceFrozenVersionListResponse)
async def list_evidence_dataset_versions(
    dataset_id: ResourceId,
    limit: int = Query(default=100, ge=1, le=100),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceFrozenVersionListResponse:
    """列出单个数据集的全部冻结版本。

    Args:
        dataset_id: 数据集 ID。
        user: 当前认证用户上下文。
    """
    try:
        items = _with_freezer_display_names(
            user,
            get_review_workspace().list_versions(user.org_id, dataset_id, limit=limit),
        )
        return EvidenceFrozenVersionListResponse(versions=[EvidenceFrozenVersion(**item) for item in items])
    except (EvidenceReviewNotFound, EvidenceReviewConflict, EvidenceReviewValidationError, EvidenceReviewPermissionError) as error:
        _raise_workspace_error(error)


@evaluation_router.get("/datasets/{dataset_id}/versions/{version}", response_model=EvidenceFrozenVersion)
async def get_evidence_dataset_version(
    dataset_id: ResourceId,
    version: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceFrozenVersion:
    """读取单个数据集的指定冻结版本。

    Args:
        dataset_id: 数据集 ID。
        version: 冻结版本号。
        user: 当前认证用户上下文。
    """
    try:
        item = _with_freezer_display_names(user, [get_review_workspace().get_version(user.org_id, dataset_id, version)])[0]
        return EvidenceFrozenVersion(**item)
    except (EvidenceReviewNotFound, EvidenceReviewConflict, EvidenceReviewValidationError, EvidenceReviewPermissionError) as error:
        _raise_workspace_error(error)


@evaluation_router.get(
    "/datasets/{dataset_id}/versions/{version}/contexts",
    response_model=EvidenceSnapshotContextListResponse,
)
async def list_frozen_snapshot_contexts(
    dataset_id: ResourceId,
    version: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceSnapshotContextListResponse:
    """返回冻结语料快照中的文档元数据，供管理员查阅。"""
    try:
        return EvidenceSnapshotContextListResponse(
            **get_review_workspace().list_version_contexts(user.org_id, dataset_id, version)
        )
    except (EvidenceReviewNotFound, EvidenceReviewConflict, EvidenceReviewValidationError, EvidenceReviewPermissionError) as error:
        _raise_workspace_error(error)


@evaluation_router.post("/datasets/{dataset_id}/refresh-current-corpus", response_model=EvidenceDatasetSummary)
async def refresh_current_corpus_dataset(
    dataset_id: ResourceId,
    document_read: DocumentReadProvider = Depends(get_document_read),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceDatasetSummary:
    """从当前组织真实入库目录刷新一份新的 current-corpus 工作区。"""
    try:
        manifest = json.loads(get_company_demo_manifest().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="41 份语料身份清单未随服务正确部署，请重新构建 API 镜像",
        ) from error
    expected_document_ids = {
        str(item.get("source_document_id") or "")
        for item in manifest.get("documents", [])
        if str(item.get("source_document_id") or "")
    }
    catalog_documents = [
        document
        for document in document_read.list_catalog_documents(tenant_id=user.org_id)
        if str(getattr(document.ingest_status, "value", document.ingest_status)) == "ingested"
        and document.doc_id in expected_document_ids
    ]
    source_documents = [
        {
            "document_id": document.doc_id,
            "source": document.display_name or document.provenance_source_filename or document.doc_id,
            "department": document.department_id,
            "version": document.version,
        }
        for document in catalog_documents
    ]
    if len(expected_document_ids) != 41 or len(source_documents) != 41:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"当前真实语料未完整入库：期望 41 份，实际可冻结 {len(source_documents)} 份",
        )
    current_chunks: list[dict[str, Any]] = []
    for document in catalog_documents:
        for chunk in await document_read.get_chunks(document.doc_id, tenant_id=user.org_id):
            current_chunks.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "content": chunk.get("content"),
                    "source_document_id": document.doc_id,
                }
            )
    try:
        result = get_review_workspace().refresh_current_corpus_dataset(
            user.org_id,
            dataset_id,
            source_documents=source_documents,
            current_chunks=current_chunks,
            actor_id=user.user_id,
        )
        # 历史快照没有独立审核的答案时，先从该不可变版本派生可编辑草稿；
        # 绝不直接改写旧版本，也不把候选答案自动视为已审核。
        if result.get("status") == "frozen":
            source_version = str(result.get("last_frozen_version") or "")
            if not source_version:
                raise EvidenceReviewConflict("source_frozen_version_required")
            result = get_review_workspace().create_draft_from_version(
                user.org_id,
                str(result["dataset_id"]),
                source_version,
                expected_revision=int(result["revision"]),
                actor_id=user.user_id,
            )
        get_review_workspace().populate_reference_answer_candidates(
            user.org_id,
            str(result["dataset_id"]),
            expected_revision=int(result["revision"]),
            actor_id=user.user_id,
        )
        result = get_review_workspace().get_dataset(user.org_id, str(result["dataset_id"]))
    except (EvidenceReviewNotFound, EvidenceReviewConflict, EvidenceReviewValidationError) as error:
        _raise_workspace_error(error)
    _audit(
        user,
        AuditAction.EVIDENCE_DATASET_DRAFT,
        dataset_id=str(result["dataset_id"]),
        revision=int(result["revision"]),
        reason_code="current_corpus_refreshed",
        result=AuditResult.SUCCESS,
    )
    return EvidenceDatasetSummary(**result)


@evaluation_router.post("/datasets/{dataset_id}/drafts", response_model=EvidenceDatasetSummary)
async def create_evidence_dataset_draft(
    dataset_id: ResourceId,
    payload: EvidenceDraftFromVersionRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvidenceDatasetSummary:
    """从指定冻结版本派生可编辑草稿。

    Args:
        dataset_id: 数据集 ID。
        payload: 请求体（含来源版本号与期望工作区版本号）。
        user: 当前认证用户上下文。
    """
    result = _mutate(
        user,
        AuditAction.EVIDENCE_DATASET_DRAFT,
        dataset_id,
        lambda: get_review_workspace().create_draft_from_version(
            user.org_id,
            dataset_id,
            payload.version,
            expected_revision=payload.expected_revision,
            actor_id=user.user_id,
        ),
        expected_revision=payload.expected_revision,
    )
    return EvidenceDatasetSummary(**result)


