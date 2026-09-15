"""评测结果读取与证据数据集治理路由。

运行结果端点保持只读；数据集变更端点只写入组织范围内的编辑工作区，
不会启动基准评测，也不会改变生产证据门模式。
"""

from __future__ import annotations

# MISSING-NAME: draft_id
# MISSING-NAME: error
# MISSING-NAME: item
# MISSING-NAME: user


from domain.identity import UserContext, UserRole
from evaluation.current_corpus_drafts import (
    CurrentCorpusDraftConflict,
    CurrentCorpusDraftNotFound,
    CurrentCorpusDraftPermissionError,
    CurrentCorpusDraftValidationError,
)
from fastapi import Depends, Path as ApiPath

from api.dependencies import require_role
from api.schemas import (
    CurrentCorpusDraft,
    CurrentCorpusDraftListResponse,
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
    get_current_corpus_draft_service,
    _raise_current_corpus_draft_error,
)

@evaluation_router.get("/current-corpus-drafts", response_model=CurrentCorpusDraftListResponse)
async def list_current_corpus_drafts(
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> CurrentCorpusDraftListResponse:
    """只读列出组织范围内的历史现行语料 Fixture 候选草稿。

    Args:
        user: 当前认证用户上下文。
    """
    try:
        return CurrentCorpusDraftListResponse(
            drafts=[CurrentCorpusDraft(**item) for item in get_current_corpus_draft_service().list(user.org_id)]
        )
    except (
        CurrentCorpusDraftNotFound,
        CurrentCorpusDraftConflict,
        CurrentCorpusDraftPermissionError,
        CurrentCorpusDraftValidationError,
    ) as error:
        _raise_current_corpus_draft_error(error)


@evaluation_router.get("/current-corpus-drafts/{draft_id}", response_model=CurrentCorpusDraft)
async def get_current_corpus_draft(
    draft_id: str = ApiPath(pattern=r"^ccd_[a-f0-9]{32}$"),
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> CurrentCorpusDraft:
    """只读读取单个历史现行语料草稿。

    Args:
        draft_id: 草稿 ID（ccd_ 前缀加 32 位十六进制）。
        user: 当前认证用户上下文。
    """
    try:
        return CurrentCorpusDraft(**get_current_corpus_draft_service().get(user.org_id, draft_id))
    except (
        CurrentCorpusDraftNotFound,
        CurrentCorpusDraftConflict,
        CurrentCorpusDraftPermissionError,
        CurrentCorpusDraftValidationError,
    ) as error:
        _raise_current_corpus_draft_error(error)


