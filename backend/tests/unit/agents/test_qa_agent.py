"""QAAgent 单元测试

覆盖:
  - 意图分类
  - 查询改写
  - 向量检索（带 tenant_id 过滤 + 熔断降级）
  - 图谱检索（带熔断降级）
  - 混合重排序
  - 置信度计算
  - 缓存命中/写入
  - 答案生成
"""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from infrastructure.evidence_gate_configuration import EffectiveEvidenceGateConfiguration

from agents.qa_agent import QAAgent
from domain.knowledge import QAResult, QueryIntent, RetrievedContext
from domain.retrieval import RetrievalOutcome, RetrievalStatus
from evaluation.benchmarks.candidate_stage_capture import CandidateStageRecorder
from shared.config import settings
from tests.qa_fakes import build_test_qa_agent_dependencies


@pytest.fixture
def agent():
    """使用显式测试依赖，避免真实模型初始化。"""
    return QAAgent(dependencies=build_test_qa_agent_dependencies())


def _make_ctx(content: str, score: float, rtype: str = "vector") -> RetrievedContext:
    return RetrievedContext(
        content=content,
        source="test.pdf",
        score=score,
        retrieval_type=rtype,
        metadata={"source": "test.pdf", "tenant_id": "org_001", "doc_id": "doc-1", "chunk_id": "doc-1#chunk-0"} if rtype == "graph" else {},
    )


class TestQueryIntent:
    def test_intent_values(self):
        assert QueryIntent.FACTOID.value == "factoid"
        assert QueryIntent.ANALYTICAL.value == "analytical"
        assert QueryIntent.COMPARATIVE.value == "comparative"
        assert QueryIntent.PROCEDURAL.value == "procedural"
        assert QueryIntent.EXPLORATORY.value == "exploratory"


class TestRetrievedContext:
    def test_default_metadata_is_empty_dict(self):
        ctx = RetrievedContext(content="x", source="s", score=0.5, retrieval_type="vector")
        assert ctx.metadata == {}

    def test_custom_metadata(self):
        ctx = RetrievedContext(
            content="x",
            source="s",
            score=0.5,
            retrieval_type="graph",
            metadata={"cypher": "MATCH (n) RETURN n"},
        )
        assert ctx.metadata["cypher"] == "MATCH (n) RETURN n"


class TestQAResult:
    def test_default_reasoning_steps_empty(self):
        r = QAResult(
            question="q",
            answer="a",
            contexts=[],
            intent=QueryIntent.FACTOID,
            confidence=0.0,
        )
        assert r.reasoning_steps == []


class TestHybridRerank:
    def test_graph_context_gets_higher_weight(self, agent):
        """图谱结果权重 1.2，向量结果权重 1.0"""
        v_ctx = _make_ctx("vector content", 0.8, "vector")
        g_ctx = _make_ctx("graph content", 0.8, "graph")
        reranked = agent._hybrid_rerank([v_ctx, g_ctx])
        assert reranked[0].retrieval_type == "graph"
        assert reranked[0].score > reranked[1].score

    def test_deduplication_by_content_prefix(self, agent):
        """相同前 100 字符的 context 去重"""
        ctx1 = _make_ctx("相同的内容开头" + "A" * 100, 0.9, "vector")
        ctx2 = _make_ctx("相同的内容开头" + "A" * 100, 0.8, "vector")
        reranked = agent._hybrid_rerank([ctx1, ctx2])
        assert len(reranked) == 1

    def test_sorted_by_score_desc(self, agent):
        ctx_low = _make_ctx("low", 0.3, "vector")
        ctx_mid = _make_ctx("mid", 0.6, "vector")
        ctx_high = _make_ctx("high", 0.9, "vector")
        reranked = agent._hybrid_rerank([ctx_low, ctx_mid, ctx_high])
        scores = [c.score for c in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self, agent):
        assert agent._hybrid_rerank([]) == []


class TestCalcConfidence:
    def test_empty_contexts_returns_zero(self, agent):
        assert agent._calc_confidence([]) == 0.0

    def test_avg_score_capped_at_1(self, agent):
        ctxs = [_make_ctx("a", 1.2, "vector"), _make_ctx("b", 1.1, "vector")]
        confidence = agent._calc_confidence(ctxs)
        assert confidence == 1.0

    def test_avg_score_normal_case(self, agent):
        ctxs = [_make_ctx("a", 0.6, "vector"), _make_ctx("b", 0.8, "vector")]
        confidence = agent._calc_confidence(ctxs)
        assert 0.6 < confidence < 0.8


class TestClassifyIntent:
    def test_classify_factoid(self, agent):
        intent = agent._classify_intent_local("张三是谁？")
        assert intent == QueryIntent.FACTOID

    def test_classify_analytical(self, agent):
        intent = agent._classify_intent_local("为什么会这样？")
        assert intent == QueryIntent.ANALYTICAL

    def test_classify_unknown_falls_back_to_factoid(self, agent):
        intent = agent._classify_intent_local("???")
        assert intent == QueryIntent.FACTOID


class TestRewriteQuery:
    def test_rewrite_preserves_original_query(self, agent):
        rewritten = agent._rewrite_query_local("原始问题")
        assert rewritten["queries"] == ["原始问题", "原始问题"]
        assert rewritten["entities"] == ["原始问题"]

    def test_rewrite_extracts_tokens_as_local_entities(self, agent):
        rewritten = agent._rewrite_query_local("如何 使用 知识 图谱")
        assert rewritten["queries"][0] == "如何 使用 知识 图谱"
        assert rewritten["entities"] == ["使用", "知识", "图谱"]
        assert rewritten["queries"][1] == "使用 知识 图谱"


class TestScopeContextExpansion:
    @pytest.mark.asyncio
    async def test_expands_toc_region_for_scope_question(self, agent):
        """目录型头块按序补拉同文档后续分块，且不重复已选块。"""
        head = RetrievedContext(
            content="公司人力资源管理制度 目 录 第一章 总则",
            source="vector_store",
            score=0.9,
            retrieval_type="vector",
            metadata={
                "source_document_id": "doc-a",
                "chunk_id": "doc-a#chunk-0",
                "chunk_index": 0,
            },
        )
        agent.vector_store = AsyncMock()
        agent.vector_store.fetch_chunks.return_value = [
            (
                {"content": "目录 第二章", "metadata": {"chunk_id": "doc-a#chunk-1", "chunk_index": 1}},
                1.0,
            ),
            (
                {"content": "目录 第三章", "metadata": {"chunk_id": "doc-a#chunk-2", "chunk_index": 2}},
                1.0,
            ),
        ]

        result = await agent._expand_scope_contexts([head])

        agent.vector_store.fetch_chunks.assert_awaited_once_with(
            ["doc-a#chunk-1", "doc-a#chunk-2", "doc-a#chunk-3", "doc-a#chunk-4", "doc-a#chunk-5"]
        )
        assert [context.metadata["chunk_id"] for context in result] == [
            "doc-a#chunk-0",
            "doc-a#chunk-1",
            "doc-a#chunk-2",
        ]
        assert all(context.metadata.get("scope_expansion") is True for context in result[1:])

    @pytest.mark.asyncio
    async def test_expands_neighbors_of_top_contexts(self, agent):
        """前三名分块补拉同文档相邻块，用于覆盖隔壁的规则正文。"""
        head = RetrievedContext(
            content="绩效考核管理制度 第二条 考核结果可作为晋升重要依据",
            source="vector_store",
            score=0.9,
            retrieval_type="vector",
            metadata={
                "source_document_id": "doc-b",
                "chunk_id": "doc-b#chunk-10",
                "chunk_index": 10,
            },
        )
        agent.vector_store = AsyncMock()
        agent.vector_store.fetch_chunks.return_value = [
            (
                {"content": "第十一条 晋升提案", "metadata": {"chunk_id": "doc-b#chunk-9", "chunk_index": 9}},
                1.0,
            ),
        ]

        result = await agent._expand_scope_contexts([head])

        requested = agent.vector_store.fetch_chunks.await_args[0][0]
        assert "doc-b#chunk-9" in requested and "doc-b#chunk-11" in requested
        assert [context.metadata["chunk_id"] for context in result] == [
            "doc-b#chunk-10",
            "doc-b#chunk-9",
        ]

    @pytest.mark.asyncio
    async def test_skips_expansion_without_chunk_identity(self, agent):
        """头块缺少文档/分块标识时无法定位邻居，不触发扩展。"""
        head = RetrievedContext(
            content="印章使用登记表填写说明",
            source="vector_store",
            score=0.9,
            retrieval_type="vector",
            metadata={"context_id": "ctx-1"},
        )
        agent.vector_store = AsyncMock()

        result = await agent._expand_scope_contexts([head])

        agent.vector_store.fetch_chunks.assert_not_awaited()
        assert result == [head]

    @pytest.mark.asyncio
    async def test_expansion_failure_returns_original_contexts(self, agent):
        """扩展取块失败时静默保留原上下文。"""
        head = RetrievedContext(
            content="目 录",
            source="vector_store",
            score=0.9,
            retrieval_type="vector",
            metadata={"source_document_id": "doc-a", "chunk_id": "doc-a#chunk-0", "chunk_index": 0},
        )
        agent.vector_store = AsyncMock()
        agent.vector_store.fetch_chunks.side_effect = RuntimeError("boom")

        result = await agent._expand_scope_contexts([head])

        assert result == [head]


class TestVectorRetrieve:
    @pytest.mark.asyncio
    async def test_vector_retrieve_no_store(self, agent):
        """无 vector_store 时返回空列表"""
        agent.vector_store = None
        rewritten = {"queries": ["q1"], "entities": [], "keywords": []}
        result = await agent._vector_retrieve(rewritten, tenant_id="org_001")
        assert result.contexts == []
        assert result.unavailable is False

    @pytest.mark.asyncio
    async def test_vector_retrieve_with_tenant_id(self, agent):
        """检索时应传递 tenant_id 参数"""
        mock_vs = AsyncMock()
        mock_vs.search.return_value = [({"content": "doc", "source": "s"}, 0.85)]
        agent.vector_store = mock_vs

        rewritten = {"queries": ["q1"], "entities": [], "keywords": []}
        result = await agent._vector_retrieve(rewritten, tenant_id="org_001")

        assert len(result.contexts) == 1
        assert result.contexts[0].retrieval_type == "vector"
        assert result.contexts[0].score == 0.85
        assert result.unavailable is False
        mock_vs.search.assert_called_once()
        call_kwargs = mock_vs.search.call_args
        assert call_kwargs[1].get("tenant_id") == "org_001" or call_kwargs[0][2] == "org_001"
        # 每条查询的候选块数必须来自 qa_vector_top_k 配置，而非写死的常量
        assert call_kwargs[0][1] == settings.qa_vector_top_k

    @pytest.mark.asyncio
    async def test_vector_retrieve_swallows_errors(self, agent):
        """单条 query 失败不应中断整个检索"""
        mock_vs = AsyncMock()
        mock_vs.search.side_effect = Exception("boom")
        agent.vector_store = mock_vs

        rewritten = {"queries": ["q1", "q2"], "entities": [], "keywords": []}
        result = await agent._vector_retrieve(rewritten, tenant_id=None)
        assert result.contexts == []
        assert result.unavailable is True


class TestGraphRetrieve:
    @pytest.mark.asyncio
    async def test_graph_retrieve_no_kg(self, agent):
        agent.knowledge_graph = None
        result = await agent._graph_retrieve("q", {"entities": []}, tenant_id="org_001")
        assert result.contexts == []
        assert result.unavailable is False

    @pytest.mark.asyncio
    async def test_graph_retrieve_with_records(self, agent):
        mock_kg = AsyncMock()
        mock_kg.search_claim_contexts.return_value = [
            {
                "claim_id": "claim-1",
                "relation_type": "WORKS_AT",
                "head_name": "张三",
                "head_type": "Person",
                "tail_name": "示例公司",
                "tail_type": "Organization",
                "claim_confidence": 0.9,
                "evidence_count": 1,
                "evidences": [
                    {
                        "evidence_id": "evidence-1",
                        "doc_id": "doc-1",
                        "chunk_id": "doc-1#chunk-0",
                        "department_id": "hr",
                        "source": "handbook.pdf",
                    }
                ],
            }
        ]

        agent.knowledge_graph = mock_kg
        result = await agent._graph_retrieve("张三", {"entities": ["张三"]}, tenant_id="org_001")

        assert len(result.contexts) == 1
        assert result.contexts[0].retrieval_type == "graph"
        assert result.contexts[0].source == "handbook.pdf"
        assert result.contexts[0].metadata["claim_id"] == "claim-1"
        assert result.contexts[0].metadata["chunk_ids"] == ["doc-1#chunk-0"]
        assert result.unavailable is False

    @pytest.mark.asyncio
    async def test_graph_retrieve_invalid_json(self, agent):
        agent.knowledge_graph = AsyncMock()
        agent.knowledge_graph.search_claim_contexts.return_value = []
        result = await agent._graph_retrieve("q", {"entities": []}, tenant_id=None)
        assert result.contexts == []
        assert result.unavailable is False

    @pytest.mark.asyncio
    async def test_bm25_cache_identity_includes_index_and_fusion_versions(self, agent):
        cache = AsyncMock()
        cache.get.return_value = None
        agent.qa_cache = cache
        agent._vector_retrieve = AsyncMock(return_value=[])
        agent._bm25_retrieve = AsyncMock(return_value=[])
        agent._graph_retrieve = AsyncMock(return_value=[])

        with patch("agents.qa_agent.settings.qa_retrieval_strategy", "dense_bm25"):
            await agent.answer("问题", tenant_id="org_001", user_id="user-1")

        cache_identity = cache.get.await_args.args[2]
        assert cache_identity.startswith("dense_bm25@native-bm25-v1")
        assert "unicode-cjk-bigram-v1" in cache_identity
        assert "rrf=60" in cache_identity


class TestRetrievalScopePropagation:
    @pytest.mark.asyncio
    async def test_answer_propagates_department_scope_to_every_retrieval_branch(self, agent):
        """A server-derived department scope must reach dense, BM25, and graph retrieval."""
        agent.qa_cache = None
        agent.semantic_cache = None
        agent._vector_retrieve = AsyncMock(return_value=RetrievalOutcome([]))
        agent._bm25_retrieve = AsyncMock(return_value=RetrievalOutcome([]))
        agent._graph_retrieve = AsyncMock(return_value=RetrievalOutcome([]))

        with patch("agents.qa_agent.settings.qa_retrieval_strategy", "dense_bm25_graph"):
            await agent.answer(
                "部门范围传播测试",
                tenant_id="org_001",
                user_id="user-1",
                retrieval_mode="dense_bm25_graph",
                visible_department_ids=("finance",),
            )

        for retriever in (
            agent._vector_retrieve,
            agent._bm25_retrieve,
            agent._graph_retrieve,
        ):
            assert retriever.await_args.kwargs["visible_department_ids"] == ("finance",)


class TestEvaluationCandidateStageCapture:
    @pytest.mark.asyncio
    async def test_ordinary_online_result_matches_evaluation_result_contract(self, agent):
        """Opt-in observation does not alter answer/context/cache identity outputs."""

        context = RetrievedContext(
            content="stable evidence",
            source="source.pdf",
            score=0.8,
            retrieval_type="vector",
            metadata={"source_document_id": "doc-a", "context_id": "ctx-a"},
        )
        agent.qa_cache = None
        agent.semantic_cache = None
        agent._vector_retrieve = AsyncMock(return_value=RetrievalOutcome([context]))
        agent._generate_answer = AsyncMock(return_value=("stable answer", ["reason"]))

        ordinary = await agent.answer(
            "同一问题", tenant_id="org_001", retrieval_mode="dense"
        )
        recorder = CandidateStageRecorder(candidate_budget=2)
        observed = await agent.answer(
            "同一问题",
            tenant_id="org_001",
            retrieval_mode="dense",
            evaluation_candidate_recorder=recorder,
        )

        assert ordinary.answer == observed.answer
        assert ordinary.contexts == observed.contexts
        assert ordinary.intent == observed.intent
        assert ordinary.confidence == observed.confidence
        assert ordinary.reasoning_steps == observed.reasoning_steps
        assert ordinary.degradation_code == observed.degradation_code
        assert ordinary.cache_hit_type == observed.cache_hit_type
        assert not hasattr(ordinary, "candidate_stages")

    @pytest.mark.asyncio
    async def test_records_bounded_pre_and_post_ranking_stages(self, agent):
        """Evaluation capture distinguishes hits, outages, and unconfigured branches."""

        dense_contexts = [
            RetrievedContext(
                content=f"dense evidence {index}",
                source="private.pdf",
                score=0.9 - index / 10,
                retrieval_type="vector",
                metadata={
                    "source_document_id": f"doc-{index}",
                    "context_id": f"ctx-{index}",
                    "content_sha256": f"{index + 1:064x}",
                },
            )
            for index in range(3)
        ]
        agent.qa_cache = None
        agent.semantic_cache = None
        agent._vector_retrieve = AsyncMock(
            return_value=RetrievalOutcome(dense_contexts)
        )
        agent._bm25_retrieve = AsyncMock(
            return_value=RetrievalOutcome(
                [], status=RetrievalStatus.UNAVAILABLE
            )
        )
        agent._graph_retrieve = AsyncMock(
            side_effect=AssertionError("graph must remain not configured")
        )
        agent._generate_answer = AsyncMock(return_value=("answer", ["reason"]))
        recorder = CandidateStageRecorder(candidate_budget=2)

        result = await agent.answer(
            "评测问题",
            tenant_id="org_001",
            retrieval_mode="dense_bm25",
            evaluation_candidate_recorder=recorder,
        )

        stages = {item["stage"]: item for item in recorder.snapshot()}
        assert list(stages) == ["dense", "bm25", "graph", "fused", "reranked"]
        assert stages["dense"]["status"] == "executed"
        assert stages["dense"]["candidate_budget"] == 2
        assert [item["rank"] for item in stages["dense"]["candidates"]] == [1, 2]
        assert stages["dense"]["candidates"][0] == {
            "rank": 1,
            "source_document_id": "doc-0",
            "context_id": "ctx-0",
            "content_sha256": hashlib.sha256(b"dense evidence 0").hexdigest(),
            "branch": "dense",
            "score": 0.9,
        }
        assert stages["bm25"]["status"] == "unavailable"
        assert stages["bm25"]["reason_code"] == "unavailable"
        assert stages["graph"]["status"] == "not_executed"
        assert stages["graph"]["reason_code"] == "branch_not_configured"
        assert stages["fused"]["status"] == "executed"
        assert stages["reranked"]["status"] == "not_executed"
        assert stages["reranked"]["reason_code"] == "reranker_disabled"
        assert not hasattr(result, "candidate_stages")

    @pytest.mark.asyncio
    async def test_records_executed_empty_separately_from_unavailable(self, agent):
        agent.qa_cache = None
        agent.semantic_cache = None
        agent._vector_retrieve = AsyncMock(return_value=RetrievalOutcome([]))
        agent._bm25_retrieve = AsyncMock(
            return_value=RetrievalOutcome([], status=RetrievalStatus.UNAVAILABLE)
        )
        recorder = CandidateStageRecorder(candidate_budget=2)

        await agent.answer(
            "无结果问题",
            tenant_id="org_001",
            retrieval_mode="dense_bm25",
            evaluation_candidate_recorder=recorder,
        )

        stages = {item["stage"]: item for item in recorder.snapshot()}
        assert stages["dense"]["status"] == "executed_empty"
        assert stages["bm25"]["status"] == "unavailable"
        assert stages["fused"]["status"] == "executed_empty"


class TestCacheIntegration:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_workflow(self, agent):
        """缓存命中时应直接返回，不执行后续流程"""
        cached_payload = {
            "question": "问题",
            "answer": "缓存答案",
            "contexts": [],
            "intent": "factoid",
            "confidence": 0.9,
            "reasoning_steps": ["cached"],
        }
        mock_cache = AsyncMock()
        mock_cache.get.return_value = cached_payload
        agent.qa_cache = mock_cache

        with patch.object(agent, "_classify_intent_local") as mock_classify:
            result = await agent.answer("问题", tenant_id="org_001", user_id="u1")
            mock_classify.assert_not_called()

        assert result.answer == "缓存答案"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_degraded_cache_miss_does_not_write_cache(self, agent):
        """没有授权来源且降级的回答不得进入缓存。"""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        agent.qa_cache = mock_cache

        agent.vector_store = None
        agent.knowledge_graph = None

        result = await agent.answer("问题", tenant_id="org_001", user_id="u1")

        assert result.answer == "未检索到可验证的证据，无法生成可靠答案。"
        mock_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_disabled_when_none(self, agent):
        """无 qa_cache 时不应尝试缓存"""
        agent.qa_cache = None
        agent.vector_store = None
        agent.knowledge_graph = None

        with patch.object(
            agent,
            "_call_llm_with_breaker",
            AsyncMock(),
        ):
            result = await agent.answer("问题", tenant_id=None, user_id="u1")
        assert result.answer == "未检索到可验证的证据，无法生成可靠答案。"


class TestLLMCallWithBreaker:
    @pytest.mark.asyncio
    async def test_llm_call_success_increments_metric(self, agent):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="ok")
        agent.llm = mock_llm

        result = await agent._call_llm_with_breaker([])
        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_llm_call_failure_reraises(self, agent):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RuntimeError("llm down")
        agent.llm = mock_llm

        with pytest.raises(RuntimeError):
            await agent._call_llm_with_breaker([])

class TestRetrievalMode:
    @pytest.mark.asyncio
    async def test_default_mode_keeps_parallel_hybrid_retrieval(self, agent):
        vector_context = _make_ctx("vector context", 0.8, "vector")
        graph_context = _make_ctx("graph context", 0.8, "graph")
        agent._vector_retrieve = AsyncMock(return_value=[vector_context])
        agent._graph_retrieve = AsyncMock(return_value=[graph_context])
        agent._generate_answer = AsyncMock(return_value=("hybrid answer", ["generated"]))

        result = await agent.answer("问题", tenant_id="org_001")

        agent._vector_retrieve.assert_awaited_once()
        agent._graph_retrieve.assert_awaited_once()
        assert {context.retrieval_type for context in result.contexts} == {"vector", "graph"}

    @pytest.mark.asyncio
    async def test_vector_mode_skips_graph_retrieval(self, agent):
        vector_context = _make_ctx("vector context", 0.8, "vector")
        agent._vector_retrieve = AsyncMock(return_value=[vector_context])
        agent._graph_retrieve = AsyncMock(return_value=[])
        agent._generate_answer = AsyncMock(return_value=("vector answer", ["generated"]))

        result = await agent.answer("问题", tenant_id="org_001", retrieval_mode="vector")

        agent._vector_retrieve.assert_awaited_once()
        agent._graph_retrieve.assert_not_awaited()
        assert [context.retrieval_type for context in result.contexts] == ["vector"]

    @pytest.mark.asyncio
    async def test_configured_dense_bm25_graph_invokes_all_native_branches(self, agent):
        vector_context = _make_ctx("vector context", 0.8, "vector")
        bm25_context = _make_ctx("bm25 context", 2.4, "bm25")
        graph_context = _make_ctx("graph context", 0.8, "graph")
        agent._vector_retrieve = AsyncMock(return_value=[vector_context])
        agent._bm25_retrieve = AsyncMock(return_value=[bm25_context])
        agent._graph_retrieve = AsyncMock(return_value=[graph_context])
        agent._generate_answer = AsyncMock(return_value=("fusion answer", ["generated"]))

        with patch("agents.qa_agent.settings.qa_retrieval_strategy", "dense_bm25_graph"):
            result = await agent.answer("问题", tenant_id="org_001")

        agent._vector_retrieve.assert_awaited_once()
        agent._bm25_retrieve.assert_awaited_once()
        agent._graph_retrieve.assert_awaited_once()
        assert {context.retrieval_type for context in result.contexts} == {
            "vector",
            "bm25",
            "graph",
        }

    @pytest.mark.asyncio
    async def test_vector_mode_rolls_back_without_invoking_bm25_or_graph(self, agent):
        vector_context = _make_ctx("vector context", 0.8, "vector")
        agent._vector_retrieve = AsyncMock(return_value=[vector_context])
        agent._bm25_retrieve = AsyncMock()
        agent._graph_retrieve = AsyncMock()
        agent._generate_answer = AsyncMock(return_value=("vector answer", ["generated"]))

        with patch("agents.qa_agent.settings.qa_retrieval_strategy", "dense_bm25_graph"):
            result = await agent.answer("问题", tenant_id="org_001", retrieval_mode="vector")

        agent._bm25_retrieve.assert_not_awaited()
        agent._graph_retrieve.assert_not_awaited()
        assert [context.retrieval_type for context in result.contexts] == ["vector"]

    @pytest.mark.asyncio
    async def test_rejects_unknown_retrieval_mode_before_work(self, agent):
        agent._vector_retrieve = AsyncMock()
        agent._graph_retrieve = AsyncMock()

        with pytest.raises(ValueError, match="retrieval_mode"):
            await agent.answer("问题", retrieval_mode="unsupported")  # type: ignore[arg-type]

        agent._vector_retrieve.assert_not_awaited()
        agent._graph_retrieve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_lookup_distinguishes_retrieval_mode(self, agent):
        cached_payload = {
            "question": "问题",
            "answer": "向量缓存答案",
            "contexts": [],
            "intent": "factoid",
            "confidence": 0.9,
            "reasoning_steps": ["cached"],
        }
        cache = AsyncMock()
        cache.get.side_effect = [cached_payload, None]
        agent.qa_cache = cache
        agent.vector_store = None
        agent.knowledge_graph = None

        vector_result = await agent.answer("问题", user_id="user-1", retrieval_mode="vector")
        hybrid_result = await agent.answer("问题", user_id="user-1", retrieval_mode="hybrid")

        assert vector_result.answer == "向量缓存答案"
        assert hybrid_result.answer == "未检索到可验证的证据，无法生成可靠答案。"
        vector_cache_mode = cache.get.await_args_list[0].args[2]
        hybrid_cache_mode = cache.get.await_args_list[1].args[2]
        assert cache.get.await_args_list[0].args[:2] == ("问题", "user-1")
        assert cache.get.await_args_list[1].args[:2] == ("问题", "user-1")
        assert vector_cache_mode.startswith("dense|gate=off")
        assert hybrid_cache_mode.startswith("dense_bm25_graph@")
        assert "|gate=off" in hybrid_cache_mode
        assert vector_cache_mode != hybrid_cache_mode
        assert cache.get.await_args_list[0].args[3] == ""
        assert cache.get.await_args_list[1].args[3] == ""
        cache.set.assert_not_awaited()


class TestOptionalCrossEncoderIntegration:
    @pytest.mark.asyncio
    async def test_disabled_reranker_preserves_existing_answer_path(self, agent):
        context = _make_ctx("员工年度培训预算是100万元。", 0.9)
        agent._vector_retrieve = AsyncMock(return_value=[context])
        agent._generate_answer = AsyncMock(return_value=("预算是100万元。", ["generated"]))

        result = await agent.answer("员工年度培训预算是多少", retrieval_mode="vector")

        assert result.answer == "预算是100万元。"
        assert result.contexts == [context]
        assert any(event.operation == "qa.rerank_completed" for event in result.trace.events)


class TestEvidenceGateRollout:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["off", "shadow"])
    async def test_off_and_shadow_preserve_legacy_generation(self, agent, monkeypatch, mode):
        monkeypatch.setattr("agents.qa_agent.settings.qa_evidence_gate_mode", mode)
        monkeypatch.setattr(
            "agents.qa_cache.get_effective_evidence_gate_configuration",
            lambda: EffectiveEvidenceGateConfiguration(mode=mode, revision=0, calibration_version="test", source="environment_default"),
        )
        background = _make_ctx("公司重视员工培训，并每年组织学习活动。", 0.9)
        background.metadata["context_id"] = "ctx_background"
        agent._vector_retrieve = AsyncMock(return_value=[background])
        agent._generate_answer = AsyncMock(return_value=("旧路径回答", ["generated"]))

        result = await agent.answer("员工年度培训预算是多少", retrieval_mode="vector")

        assert result.answer == "旧路径回答"
        agent._generate_answer.assert_awaited_once()
        assert result.evidence_assessment is not None
        assert result.evidence_assessment.response_status.value == "insufficient_evidence"

    @pytest.mark.asyncio
    async def test_enforce_stops_background_only_generation(self, agent, monkeypatch):
        monkeypatch.setattr("agents.qa_agent.settings.qa_evidence_gate_mode", "enforce")
        monkeypatch.setattr(
            "agents.qa_cache.get_effective_evidence_gate_configuration",
            lambda: EffectiveEvidenceGateConfiguration(mode="enforce", revision=0, calibration_version="test", source="environment_default"),
        )
        background = _make_ctx("公司重视员工培训，并每年组织学习活动。", 0.9)
        background.metadata["context_id"] = "ctx_background"
        agent._vector_retrieve = AsyncMock(return_value=[background])
        agent._generate_answer = AsyncMock(return_value=("不应返回", ["generated"]))

        result = await agent.answer("员工年度培训预算是多少", retrieval_mode="vector")

        agent._generate_answer.assert_not_awaited()
        assert result.response_status.value == "insufficient_evidence"
        assert result.degradation_code == "insufficient_verified_evidence"
        assert result.answer == "未检索到可验证的证据，无法生成可靠答案。"
        assert any(
            event.operation == "qa.evidence_qualified"
            for event in result.trace.events
        )
