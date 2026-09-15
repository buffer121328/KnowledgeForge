"""实现仓库自有检索端口（RetrieverPort）的原生检索适配器。"""

from __future__ import annotations

from typing import Any

from services.qa.retrieval import graph_retrieve, vector_retrieve
from domain.knowledge import RetrievedContext
from domain.retrieval import RetrievalOutcome, RetrievalRequest, RetrievalStatus


class NativeDenseRetriever:
    """将原生向量检索服务适配到框架无关的检索端口。"""

    def __init__(self, vector_store: Any, *, top_k: int = 8) -> None:
        """保存向量库实例与每条查询的候选块数上限。

        Args:
            vector_store: 提供语义搜索能力的向量库实例。
            top_k: 每条改写查询返回的候选块数上限。
        """
        self._vector_store = vector_store
        self._top_k = top_k

    async def retrieve(self, request: RetrievalRequest) -> RetrievalOutcome:
        """按授权范围执行原生稠密（向量）检索。

        Args:
            request: 仓库自有的检索请求（含问题、改写、租户/部门范围等）。
        """
        # scope.tenant_id 为空串时归一化为 None，交由底层决定租户过滤语义
        return await vector_retrieve(
            self._vector_store,
            request.rewritten,
            tenant_id=request.scope.tenant_id or None,
            user_id=request.user_id,
            question_fingerprint=request.question_fingerprint,
            visible_department_ids=request.scope.visible_department_ids,
            top_k=self._top_k,
        )


class NativeBM25Retriever:
    """将仓库自有的 BM25 稀疏索引适配到检索端口。"""

    def __init__(self, sparse_index: Any, *, top_k: int = 5) -> None:
        """保存稀疏索引实例与返回条数上限。

        Args:
            sparse_index: BM25 稀疏索引实例；为 None 表示该依赖不可用。
            top_k: 每次检索返回的最大上下文条数。
        """
        self._sparse_index = sparse_index
        self._top_k = top_k

    async def retrieve(self, request: RetrievalRequest) -> RetrievalOutcome:
        """按租户/部门过滤执行 BM25 检索，并以显式状态反映失败语义。

        Args:
            request: 仓库自有的检索请求（含问题、改写、租户/部门范围等）。
        """
        # 安全关卡：缺少租户标识直接判定未授权，不做任何检索
        if not request.scope.tenant_id:
            return RetrievalOutcome(contexts=[], status=RetrievalStatus.UNAUTHORIZED)
        # 索引未注入视为依赖不可用
        if self._sparse_index is None:
            return RetrievalOutcome(contexts=[], status=RetrievalStatus.UNAVAILABLE)

        # 延迟导入：避免模块加载期引入 BM25 基础设施依赖
        from infrastructure.retrieval.bm25_index import (
            BM25IndexContractError,
            BM25IndexUnavailable,
        )

        candidates: dict[str, RetrievedContext] = {}
        # 改写结果中的 queries 为空时回退到原始问题
        queries = request.rewritten.get("queries") or [request.question]
        try:
            # ① 最多取前 2 个改写查询，逐个调用稀疏索引检索
            for query in queries[:2]:
                results = await self._sparse_index.search(
                    str(query),
                    tenant_id=request.scope.tenant_id,
                    visible_department_ids=request.scope.visible_department_ids,
                    top_k=self._top_k,
                )
                for document, raw_score in results:
                    metadata = dict(document.get("metadata") or {})
                    source_document_id = (
                        document.get("source_document_id")
                        or metadata.get("source_document_id")
                        or metadata.get("doc_id")
                    )
                    if isinstance(source_document_id, str) and source_document_id.strip():
                        metadata["source_document_id"] = source_document_id
                    chunk_id = str(metadata.get("chunk_id") or "")
                    # 以 chunk_id（缺失时用 doc_id:chunk_index）作为去重身份
                    identity = chunk_id or f"{metadata.get('doc_id', '')}:{metadata.get('chunk_index', '')}"
                    context = RetrievedContext(
                        content=str(document.get("content") or ""),
                        source=str(document.get("source") or "bm25_index"),
                        score=float(raw_score),
                        retrieval_type="bm25",
                        metadata={
                            **metadata,
                            "raw_score": float(raw_score),
                        },
                    )
                    previous = candidates.get(identity)
                    # ② 同一身份只保留得分更高的结果
                    if previous is None or context.score > previous.score:
                        candidates[identity] = context
        # ③ 将底层依赖异常映射为显式检索状态，避免向上泄露实现细节
        except BM25IndexUnavailable:
            return RetrievalOutcome(contexts=[], status=RetrievalStatus.UNAVAILABLE)
        except BM25IndexContractError:
            return RetrievalOutcome(contexts=[], status=RetrievalStatus.CONTRACT_ERROR)
        except (TypeError, ValueError):
            return RetrievalOutcome(contexts=[], status=RetrievalStatus.INVALID)

        # ④ 确定性排序（分数降序，其次 chunk_id、来源）并截断到 top_k
        contexts = sorted(
            candidates.values(),
            key=lambda item: (
                -item.score,
                str(item.metadata.get("chunk_id") or ""),
                item.source,
            ),
        )[: self._top_k]
        # ⑤ 记录融合前的来源内排名，供后续 RRF 融合使用
        for source_rank, context in enumerate(contexts, start=1):
            context.metadata["source_rank"] = source_rank
        return RetrievalOutcome(contexts=contexts)


class ClaimAwareGraphRetriever:
    """将主张/证据图谱检索适配到框架无关的检索端口。"""

    def __init__(self, knowledge_graph: Any) -> None:
        """保存知识图谱实例。

        Args:
            knowledge_graph: 提供主张上下文搜索能力的知识图谱实例。
        """
        self._knowledge_graph = knowledge_graph

    async def retrieve(self, request: RetrievalRequest) -> RetrievalOutcome:
        """按授权范围执行有证据支撑的图谱检索。

        Args:
            request: 仓库自有的检索请求（含问题、改写、租户/部门范围等）。
        """
        # scope.tenant_id 为空串时归一化为 None，交由底层决定租户过滤语义
        return await graph_retrieve(
            self._knowledge_graph,
            request.question,
            request.rewritten,
            tenant_id=request.scope.tenant_id or None,
            user_id=request.user_id,
            question_fingerprint=request.question_fingerprint,
            visible_department_ids=request.scope.visible_department_ids,
        )


class HybridRetrievalOrchestrator:
    """融合各适配器产出的授权上下文，不向外暴露适配器实现细节。"""

    def __init__(
        self,
        *,
        rrf_k: int = 60,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        """保存 RRF 平滑常数与来源权重。

        Args:
            rrf_k: 倒数排名融合（RRF）的平滑参数。
            source_weights: 各检索来源的权重映射；为 None 时使用融合层默认权重。
        """
        self._rrf_k = rrf_k
        self._source_weights = source_weights

    def fuse(self, *outcomes: RetrievalOutcome) -> list[RetrievedContext]:
        """将各成功适配器结果中的上下文做确定性 RRF 融合并排序。

        Args:
            *outcomes: 待融合的适配器检索结果（通常只传入成功的结果）。
        """
        # 延迟导入：避免模块加载期引入排序依赖
        from services.qa.ranking import hybrid_rerank

        # 汇总所有结果中的上下文，统一交给 RRF 去重、融合与排序
        contexts = [context for outcome in outcomes for context in outcome.contexts]
        return hybrid_rerank(
            contexts,
            rrf_k=self._rrf_k,
            source_weights=self._source_weights,
        )
