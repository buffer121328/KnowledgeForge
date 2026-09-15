"""评测结果读取与证据数据集治理路由。

运行结果端点保持只读；数据集变更端点只写入组织范围内的编辑工作区，
不会启动基准评测，也不会改变生产证据门模式。
"""

from __future__ import annotations

# MISSING-NAME: error
# MISSING-NAME: page
# MISSING-NAME: page_size
# MISSING-NAME: payload
# MISSING-NAME: run
# MISSING-NAME: run_id
# MISSING-NAME: user


from domain.identity import UserContext, UserRole
from evaluation.benchmarks.result_reader import (
    EvaluationResultError,
    list_records,
    list_runs,
    load_run,
)
from evaluation.benchmarks.run_report import build_run_report
from evaluation.benchmarks.spot_check import list_anomalies, submit_spot_check
from fastapi import Depends, HTTPException, Query
from infrastructure.audit.log import AuditAction, AuditResult

from api.contracts import ResourceId
from api.dependencies import require_role
from api.schemas import (
    EvaluationRecordsResponse,
    EvaluationRunDetail,
    EvaluationRunListResponse,
    EvaluationRunSummary,
    EvaluationSpotCheckResponse,
    EvaluationSpotCheckSubmitRequest,
    EvaluationRunReportResponse,
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

# --- 共享单例与管道 ---
from api.routers.evaluation_pkg._common import (
    evaluation_router,
    _audit,
    RESULTS_ROOT,
)

def _results_root():
    """Read RESULTS_ROOT through _common at call time so tests can patch it."""
    from api.routers.evaluation_pkg import _common

    return _common.RESULTS_ROOT


@evaluation_router.get("/runs", response_model=EvaluationRunListResponse)
async def list_evaluation_runs(
    _user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvaluationRunListResponse:
    """按时间倒序返回评测运行列表，摘要字段受限。

    Args:
        _user: 已通过组织管理员角色校验的当前用户（未直接使用）。
    """
    return EvaluationRunListResponse(
        runs=[EvaluationRunSummary(**run) for run in list_runs(_results_root())]
    )


@evaluation_router.get("/runs/{run_id}", response_model=EvaluationRunDetail)
async def get_evaluation_run(
    run_id: ResourceId,
    _user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvaluationRunDetail:
    """返回单次运行的元数据、质量报告与 Ragas 评分摘要。

    Args:
        run_id: 评测运行 ID。
        _user: 已通过组织管理员角色校验的当前用户（未直接使用）。
    """
    try:
        return EvaluationRunDetail(**load_run(_results_root(), run_id))
    except EvaluationResultError as error:
        raise HTTPException(status_code=404, detail="评测运行不存在") from error


@evaluation_router.get("/runs/{run_id}/report", response_model=EvaluationRunReportResponse)
async def get_evaluation_run_report(
    run_id: ResourceId,
    _user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvaluationRunReportResponse:
    """返回单次运行的综合评测报告（RAGAS + 自定义指标 + 行为契约）。

    Args:
        run_id: 评测运行 ID。
        _user: 已通过组织管理员角色校验的当前用户（未直接使用）。
    """
    try:
        return EvaluationRunReportResponse(**build_run_report(_results_root(), run_id))
    except EvaluationResultError as error:
        raise HTTPException(status_code=404, detail="评测运行不存在") from error


@evaluation_router.get("/runs/{run_id}/records", response_model=EvaluationRecordsResponse)
async def get_evaluation_records(
    run_id: ResourceId,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    _user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvaluationRecordsResponse:
    """返回单页按问题维度的评测记录。

    Args:
        run_id: 评测运行 ID。
        page: 页码（从 1 开始）。
        page_size: 每页数量（1~100）。
        _user: 已通过组织管理员角色校验的当前用户（未直接使用）。
    """
    try:
        return EvaluationRecordsResponse(
            **list_records(_results_root(), run_id, page=page, page_size=page_size)
        )
    except EvaluationResultError as error:
        raise HTTPException(status_code=404, detail="评测运行不存在") from error


@evaluation_router.get("/runs/{run_id}/anomalies", response_model=EvaluationSpotCheckResponse)
async def get_evaluation_anomalies(
    run_id: ResourceId,
    _user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvaluationSpotCheckResponse:
    """返回单次运行中值得人工抽检的异常样本清单。

    Args:
        run_id: 评测运行 ID。
        _user: 已通过组织管理员角色校验的当前用户（未直接使用）。
    """
    try:
        return EvaluationSpotCheckResponse(**list_anomalies(_results_root(), run_id))
    except EvaluationResultError as error:
        raise HTTPException(status_code=404, detail="评测运行不存在") from error


@evaluation_router.post("/runs/{run_id}/spot-checks", response_model=EvaluationSpotCheckResponse)
async def submit_evaluation_spot_check(
    run_id: ResourceId,
    payload: EvaluationSpotCheckSubmitRequest,
    user: UserContext = Depends(require_role(UserRole.ORGANIZATION_ADMIN)),
) -> EvaluationSpotCheckResponse:
    """记录一条样本的人工抽检结论（owner-only 产物，可重复提交覆盖）。

    Args:
        run_id: 评测运行 ID。
        payload: 结论请求体（benchmark_id + verdict + note）。
        user: 当前管理员，其 user_id 作为复核人落盘。
    """
    try:
        result = submit_spot_check(
            RESULTS_ROOT,
            run_id,
            benchmark_id=payload.benchmark_id,
            verdict=payload.verdict,
            note=payload.note,
            reviewer_id=user.user_id,
        )
    except EvaluationResultError as error:
        raise HTTPException(status_code=404, detail="评测运行不存在") from error
    _audit(
        user,
        AuditAction.EVALUATION_SPOT_CHECK,
        dataset_id=run_id,
        result=AuditResult.SUCCESS,
        version=run_id,
    )
    return result


