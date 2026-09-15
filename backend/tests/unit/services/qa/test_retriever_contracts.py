"""Contract tests for framework-neutral retrieval outcomes and deterministic fusion."""

from __future__ import annotations

from services.qa.ranking import hybrid_rerank
from domain.knowledge import RetrievedContext
from domain.retrieval import RetrievalOutcome, RetrievalRequest, RetrievalScope, RetrievalStatus


def _context(
    content: str,
    score: float,
    retrieval_type: str,
    *,
    chunk_id: str = "",
    claim_id: str = "",
) -> RetrievedContext:
    metadata = {}
    if chunk_id:
        metadata["chunk_id"] = chunk_id
    if claim_id:
        metadata["claim_id"] = claim_id
    return RetrievedContext(
        content=content,
        source=f"{retrieval_type}.source",
        score=score,
        retrieval_type=retrieval_type,
        metadata=metadata,
    )


def test_retrieval_outcome_distinguishes_empty_and_unavailable() -> None:
    assert RetrievalOutcome(contexts=[]).status is RetrievalStatus.EMPTY
    assert RetrievalOutcome(contexts=[], unavailable=True).status is RetrievalStatus.UNAVAILABLE
    assert RetrievalOutcome(contexts=[], status=RetrievalStatus.UNAUTHORIZED).unavailable is False
    assert RetrievalOutcome(contexts=[_context("x", 0.2, "vector")]).status is RetrievalStatus.SUCCESS
    for status in (
        RetrievalStatus.INVALID,
        RetrievalStatus.UNAUTHORIZED,
        RetrievalStatus.CONTRACT_ERROR,
        RetrievalStatus.EVIDENCE_FILTERED,
    ):
        outcome = RetrievalOutcome(contexts=[], status=status)
        assert outcome.status is status
        assert outcome.unavailable is False


def test_retrieval_request_keeps_server_scope_separate_from_query_data() -> None:
    scope = RetrievalScope(tenant_id="tenant-a", visible_department_ids=("finance",))
    request = RetrievalRequest(
        question="问题",
        rewritten={"queries": ["问题"], "tenant_id": "tenant-b"},
        scope=scope,
        user_id="user-a",
        question_fingerprint="fingerprint",
    )

    assert request.scope.tenant_id == "tenant-a"
    assert request.scope.visible_department_ids == ("finance",)
    assert request.rewritten["tenant_id"] == "tenant-b"


def test_rrf_is_deterministic_and_does_not_compare_raw_scales_directly() -> None:
    candidates = [
        _context("vector-high", 0.99, "vector", chunk_id="chunk-v1"),
        _context("vector-low", 0.20, "vector", chunk_id="chunk-v2"),
        _context("graph-top", 0.51, "graph", claim_id="claim-g1"),
        _context("graph-second", 0.50, "graph", claim_id="claim-g2"),
    ]

    first = hybrid_rerank(candidates)
    second = hybrid_rerank(
        [
            _context("vector-high", 0.99, "vector", chunk_id="chunk-v1"),
            _context("vector-low", 0.20, "vector", chunk_id="chunk-v2"),
            _context("graph-top", 0.51, "graph", claim_id="claim-g1"),
            _context("graph-second", 0.50, "graph", claim_id="claim-g2"),
        ]
    )

    assert [item.content for item in first] == [item.content for item in second]
    assert first[0].content in {"graph-top", "vector-high"}
    assert first[0].metadata["fusion"]["source_rank"] == 1
    assert first[0].metadata["fusion"]["method"] == "rrf"
    assert [item.score for item in first] == sorted(
        [item.score for item in first], reverse=True
    )


def test_rrf_merges_duplicate_provenance_and_records_source_ranks() -> None:
    vector = _context("same evidence", 0.91, "vector", chunk_id="chunk-1")
    graph = _context("same evidence", 0.45, "graph", chunk_id="chunk-1")

    ranked = hybrid_rerank([vector, graph])

    assert len(ranked) == 1
    fusion = ranked[0].metadata["fusion"]
    assert fusion["source_ranks"] == {"graph": 1, "vector": 1}
    assert fusion["final_rank"] == 1
    assert fusion["score"] == 1.0


def test_rrf_prefers_canonical_chunk_content_over_graph_claim_text() -> None:
    vector = _context("canonical chunk", 0.40, "vector", chunk_id="chunk-1")
    vector.metadata["source_document_id"] = "doc-1"
    graph = _context("synthetic graph claim", 0.99, "graph", chunk_id="chunk-1")
    graph.metadata.update({"source_document_id": "doc-1", "claim_id": "claim-1"})

    ranked = hybrid_rerank([graph, vector])

    assert len(ranked) == 1
    assert ranked[0].content == "canonical chunk"
    assert ranked[0].metadata["fusion"]["source_ranks"] == {"graph": 1, "vector": 1}
    assert ranked[0].metadata["fusion"]["claim_ids"] == ["claim-1"]


def test_rrf_does_not_merge_identical_text_with_different_chunk_ids() -> None:
    ranked = hybrid_rerank([
        _context("same", 0.9, "vector", chunk_id="doc-1#chunk-0"),
        _context("same", 0.8, "bm25", chunk_id="doc-2#chunk-0"),
    ])

    assert len(ranked) == 2


def test_native_strategy_resolution_preserves_existing_modes() -> None:
    from domain.retrieval import RetrievalStrategy, resolve_retrieval_strategy

    assert resolve_retrieval_strategy("vector", "dense_bm25_graph") is RetrievalStrategy.DENSE
    assert resolve_retrieval_strategy("hybrid", "dense_graph") is RetrievalStrategy.DENSE_GRAPH
    assert resolve_retrieval_strategy("hybrid", "dense_bm25") is RetrievalStrategy.DENSE_BM25
    assert resolve_retrieval_strategy("dense", "dense_graph") is RetrievalStrategy.DENSE
    assert resolve_retrieval_strategy("dense_bm25_graph", "dense_graph") is RetrievalStrategy.DENSE_BM25_GRAPH
    assert RetrievalStrategy.DENSE_BM25_GRAPH.uses_bm25 is True
    assert RetrievalStrategy.DENSE_GRAPH.uses_bm25 is False
    assert RetrievalStrategy.DENSE_GRAPH.uses_graph is True


def test_settings_validate_native_retrieval_strategy_and_bm25_parameters() -> None:
    import pytest
    from pydantic import ValidationError

    from shared.config.settings import Settings

    configured = Settings(
        qa_retrieval_strategy="DENSE_BM25_GRAPH",
        qa_bm25_k1=1.2,
        qa_bm25_b=0.6,
        qa_bm25_top_k=7,
    )

    assert configured.qa_retrieval_strategy == "dense_bm25_graph"
    assert configured.qa_strategy_uses_bm25 is True
    with pytest.raises(ValidationError):
        Settings(qa_retrieval_strategy="framework_fusion")
    with pytest.raises(ValidationError):
        Settings(qa_bm25_b=1.5)

import pytest


@pytest.mark.asyncio
async def test_native_bm25_context_normalizes_metadata_document_id_for_provenance() -> None:
    from services.qa.retrievers import NativeBM25Retriever

    document_id = "4ccfe084-4b94-4f76-8604-8c3db4eaa71f"

    class SparseIndex:
        async def search(self, *_args, **_kwargs):
            return [(
                {
                    "content": "已授权内容",
                    "source": "policy.docx",
                    "metadata": {"doc_id": document_id, "chunk_id": "chunk-1"},
                },
                0.91,
            )]

    outcome = await NativeBM25Retriever(SparseIndex()).retrieve(
        RetrievalRequest(
            question="问题",
            rewritten={"queries": ["问题"]},
            scope=RetrievalScope(tenant_id="tenant-a"),
            user_id="user-a",
            question_fingerprint="fingerprint",
        )
    )

    assert outcome.contexts[0].metadata["source_document_id"] == document_id

@pytest.mark.asyncio
async def test_native_dense_context_normalizes_metadata_document_id_for_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.qa import retrieval as qa_retrieval

    document_id = "d778eb66-9a1b-4a93-aa1c-599a3fa8f509"

    async def execute_policy(_policy, operation, **_kwargs):
        return await operation()

    async def fallback(_breaker, _method, *_args, **_kwargs):
        return [(
            {
                "content": "已授权内容",
                "source": "policy.docx",
                "metadata": {"doc_id": document_id, "chunk_id": "chunk-1"},
            },
            0.92,
        )]

    monkeypatch.setattr(qa_retrieval, "execute_with_policy", execute_policy)
    monkeypatch.setattr(qa_retrieval, "call_with_fallback", fallback)
    class VectorStore:
        async def search(self, *_args, **_kwargs):
            raise AssertionError("fallback is expected to provide the mocked result")

    outcome = await qa_retrieval.vector_retrieve(
        VectorStore(), {"queries": ["问题"]}, tenant_id="tenant-a"
    )

    assert outcome.contexts[0].metadata["source_document_id"] == document_id
