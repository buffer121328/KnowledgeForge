"""框架无关的检索请求、结果与适配器协议定义。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

from domain.knowledge import RetrievedContext


class RetrievalStatus(str, Enum):
    """检索结果状态分类，避免向调用方泄露底层依赖异常。"""

    SUCCESS = "success"  # 成功返回上下文
    EMPTY = "empty"  # 正常执行但未命中任何上下文
    UNAVAILABLE = "unavailable"  # 依赖服务不可用（如熔断打开）
    INVALID = "invalid"  # 请求或返回数据不合法
    UNAUTHORIZED = "unauthorized"  # 缺少授权（如租户标识）被拒绝
    CONTRACT_ERROR = "contract_error"  # 依赖返回的数据违反契约
    EVIDENCE_FILTERED = "evidence_filtered"  # 上下文被证据资格过滤剔除


class RetrievalStrategy(str, Enum):
    """服务端选定的原生检索分支组合。"""

    DENSE = "dense"  # 仅稠密向量检索
    DENSE_GRAPH = "dense_graph"  # 稠密向量 + 图谱检索
    DENSE_BM25 = "dense_bm25"  # 稠密向量 + BM25 检索
    DENSE_BM25_GRAPH = "dense_bm25_graph"  # 稠密向量 + BM25 + 图谱检索

    @property
    def uses_bm25(self) -> bool:
        """该策略组合是否包含 BM25 稀疏检索分支。"""
        return self in {self.DENSE_BM25, self.DENSE_BM25_GRAPH}

    @property
    def uses_graph(self) -> bool:
        """该策略组合是否包含图谱检索分支。"""
        return self in {self.DENSE_GRAPH, self.DENSE_BM25_GRAPH}


def resolve_retrieval_strategy(
    retrieval_mode: str,
    configured_strategy: str | RetrievalStrategy,
) -> RetrievalStrategy:
    """把兼容的调用方检索模式映射为实际生效的服务端策略。

    Args:
        retrieval_mode: 调用方请求的检索模式字符串。
        configured_strategy: hybrid 模式下采用的服务端配置策略（枚举或字符串）。
    """
    # ① 统一小写并去除首尾空白
    normalized_mode = retrieval_mode.strip().lower()
    # ② vector 为兼容别名，恒定映射到纯稠密检索
    if normalized_mode == "vector":
        return RetrievalStrategy.DENSE
    # ③ hybrid 表示采用服务端配置的具体策略组合
    if normalized_mode == "hybrid":
        if isinstance(configured_strategy, RetrievalStrategy):
            return configured_strategy
        return RetrievalStrategy(configured_strategy.strip().lower())
    # ④ 其余取值必须直接对应某个策略名，否则抛出 ValueError
    try:
        return RetrievalStrategy(normalized_mode)
    except ValueError as error:
        raise ValueError(
            "retrieval_mode must be vector, hybrid, dense, dense_graph, "
            "dense_bm25, or dense_bm25_graph"
        ) from error


@dataclass(frozen=True)
class RetrievalScope:
    """由服务端鉴权推导出的检索范围。"""

    tenant_id: str  # 租户标识；空串会被适配器归一化为 None（不做租户过滤）
    visible_department_ids: tuple[str, ...] | None = None  # 可见部门 ID 元组；None 表示不做部门过滤


@dataclass(frozen=True)
class RetrievalRequest:
    """传给检索适配器的仓库自有请求对象。"""

    question: str  # 用户原始问题
    rewritten: dict[str, Any]  # 问题改写结果，含 queries/entities/keywords 等键
    scope: RetrievalScope  # 服务端鉴权推导的检索范围（租户/部门）
    user_id: str = ""  # 发起请求的用户标识，用于安全审计
    question_fingerprint: str = ""  # 问题指纹，用于审计日志中的脱敏关联


@dataclass(frozen=True)
class RetrievalOutcome:
    """检索结果：上下文列表 + 显式状态与不可用兼容标志。"""

    contexts: list[RetrievedContext]  # 检索到的上下文列表
    status: RetrievalStatus | str | None = None  # 显式状态；为 None 时在初始化后按结果推导
    unavailable: bool = False  # 兼容标志：依赖是否不可用，初始化后与 status 保持一致

    def __post_init__(self) -> None:
        """推导并归一化 status 与 unavailable 兼容标志。"""
        status = self.status
        # ① 未显式给出状态时按结果推导：不可用 > 有上下文（成功）> 空
        if status is None:
            if self.unavailable:
                status = RetrievalStatus.UNAVAILABLE
            elif self.contexts:
                status = RetrievalStatus.SUCCESS
            else:
                status = RetrievalStatus.EMPTY
        elif not isinstance(status, RetrievalStatus):
            # ② 传入的字符串状态统一收敛为枚举
            status = RetrievalStatus(status)
        # frozen dataclass 需绕过 __setattr__ 限制回写两个字段
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "unavailable", status is RetrievalStatus.UNAVAILABLE)


class RetrieverPort(Protocol):
    """由原生及可选检索器共同实现的适配器边界协议。"""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalOutcome:
        """对一次已授权的检索请求返回仓库自有的上下文集合。

        Args:
            request: 仓库自有的检索请求（含问题、改写与授权范围）。
        """
        ...


class CandidateStageRecorderPort(Protocol):
    """Optional request-local observer used only by offline evaluation execution."""

    candidate_budget: int

    def record(
        self,
        stage: str,
        contexts: Sequence[RetrievedContext],
        *,
        status: str,
        reason_code: str | None = None,
    ) -> None:
        """Capture a bounded stage without retaining raw context text."""
        ...
