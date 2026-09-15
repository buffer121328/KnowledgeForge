"""Contract tests for optional CrossEncoder reranking and RRF fallback."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

import pytest

from domain.knowledge import RetrievedContext
from infrastructure.retrieval.cross_encoder import (
    CrossEncoderScore,
    RerankStatus,
    SafeCrossEncoderReranker,
)


def contexts(count: int = 4) -> list[RetrievedContext]:
    return [
        RetrievedContext(
            content=f"context {index}",
            source=f"source-{index}.md",
            score=1 / (index + 1),
            retrieval_type="vector",
            metadata={"context_id": f"ctx_{index}"},
        )
        for index in range(count)
    ]


@dataclass
class StaticAdapter:
    scores: list[CrossEncoderScore]
    seen_count: int = 0

    async def score(self, question: str, candidates: list[RetrievedContext]) -> list[CrossEncoderScore]:
        del question
        self.seen_count = len(candidates)
        return self.scores


class FailingAdapter:
    async def score(self, question: str, candidates: list[RetrievedContext]) -> list[CrossEncoderScore]:
        del question, candidates
        raise RuntimeError("model load failed with private path")


class SlowAdapter:
    async def score(self, question: str, candidates: list[RetrievedContext]) -> list[CrossEncoderScore]:
        del question, candidates
        await asyncio.sleep(0.05)
        return []


@pytest.mark.asyncio
async def test_disabled_mode_preserves_rrf_order_without_adapter() -> None:
    original = contexts()
    result = await SafeCrossEncoderReranker(mode="disabled").rerank("question", original)

    assert result.status is RerankStatus.DISABLED
    assert result.contexts == original
    assert result.degradation_code is None


@pytest.mark.asyncio
async def test_enabled_mode_limits_candidates_and_keeps_configured_contexts() -> None:
    original = contexts(5)
    adapter = StaticAdapter(
        [
            CrossEncoderScore(context_id="ctx_2", score=0.9),
            CrossEncoderScore(context_id="ctx_1", score=0.8),
            CrossEncoderScore(context_id="ctx_0", score=0.1),
        ]
    )
    reranker = SafeCrossEncoderReranker(
        mode="local",
        adapter=adapter,
        candidate_limit=3,
        context_limit=2,
    )

    result = await reranker.rerank("question", original)

    assert adapter.seen_count == 3
    # ctx_0 是 RRF 头名，即使交叉编码器评分最低也必须保留在最终上下文中。
    assert [item.metadata["context_id"] for item in result.contexts] == ["ctx_0", "ctx_2"]
    assert result.status is RerankStatus.APPLIED
    assert {item.context_id: item.score for item in result.scores} == {
        "ctx_0": 0.1,
        "ctx_1": 0.8,
        "ctx_2": 0.9,
    }


def document_contexts() -> list[RetrievedContext]:
    specs = [
        ("doc-a", "chunk-1"),
        ("doc-a", "chunk-2"),
        ("doc-b", "chunk-1"),
        ("doc-a", "chunk-3"),
        ("doc-c", "chunk-1"),
    ]
    return [
        RetrievedContext(
            content=f"context {index}",
            source=f"source-{index}.md",
            score=1 / (index + 1),
            retrieval_type="vector",
            metadata={"chunk_id": f"{doc}#{chunk}"},
        )
        for index, (doc, chunk) in enumerate(specs)
    ]


@pytest.mark.asyncio
async def test_per_document_leaders_reserve_slots_for_other_documents() -> None:
    original = document_contexts()
    adapter = StaticAdapter(
        [
            CrossEncoderScore(context_id="doc-a#chunk-1", score=0.9),
            CrossEncoderScore(context_id="doc-a#chunk-2", score=0.85),
            CrossEncoderScore(context_id="doc-b#chunk-1", score=0.5),
            CrossEncoderScore(context_id="doc-a#chunk-3", score=0.4),
        ]
    )
    reranker = SafeCrossEncoderReranker(
        mode="remote",
        adapter=adapter,
        candidate_limit=4,
        context_limit=3,
        max_per_document=2,
    )

    result = await reranker.rerank("question", original)

    # 每份文档先保底一个最好分块（doc-b 不被 doc-a 挤出），
    # 剩余名额按相关性回填 doc-a 的次优分块。
    assert [item.metadata["chunk_id"] for item in result.contexts] == [
        "doc-a#chunk-1",
        "doc-b#chunk-1",
        "doc-a#chunk-2",
    ]
    assert result.status is RerankStatus.APPLIED


@pytest.mark.asyncio
async def test_document_leaders_fill_then_respect_per_document_cap() -> None:
    original = document_contexts()
    adapter = StaticAdapter(
        [
            CrossEncoderScore(context_id="doc-a#chunk-1", score=0.9),
            CrossEncoderScore(context_id="doc-a#chunk-2", score=0.85),
            CrossEncoderScore(context_id="doc-a#chunk-3", score=0.7),
            CrossEncoderScore(context_id="doc-b#chunk-1", score=0.6),
            CrossEncoderScore(context_id="doc-c#chunk-1", score=0.5),
        ]
    )
    reranker = SafeCrossEncoderReranker(
        mode="remote",
        adapter=adapter,
        candidate_limit=5,
        context_limit=5,
        max_per_document=3,
    )

    result = await reranker.rerank("question", original)

    # 领导位覆盖三份文档后，回填按相关性进行；doc-a 第三个分块回填后
    # 触达单文档上限，doc-a 的剩余分块被跳过。
    assert [item.metadata["chunk_id"] for item in result.contexts] == [
        "doc-a#chunk-1",
        "doc-b#chunk-1",
        "doc-c#chunk-1",
        "doc-a#chunk-2",
        "doc-a#chunk-3",
    ]
    assert result.status is RerankStatus.APPLIED


@pytest.mark.asyncio
async def test_rrf_leader_guard_restores_consensus_evidence() -> None:
    original = contexts(4)
    adapter = StaticAdapter(
        [
            CrossEncoderScore(context_id="ctx_1", score=0.9),
            CrossEncoderScore(context_id="ctx_2", score=0.8),
            CrossEncoderScore(context_id="ctx_3", score=0.7),
            CrossEncoderScore(context_id="ctx_0", score=0.1),
        ]
    )
    reranker = SafeCrossEncoderReranker(
        mode="remote",
        adapter=adapter,
        candidate_limit=4,
        context_limit=3,
    )

    result = await reranker.rerank("question", original)

    # RRF 头名 ctx_0 被重排挤出后必须回填，淘汰入选名单中评分最低的 ctx_3。
    assert [item.metadata["context_id"] for item in result.contexts] == [
        "ctx_0",
        "ctx_1",
        "ctx_2",
    ]
    assert result.status is RerankStatus.APPLIED


@pytest.mark.asyncio
async def test_rrf_leader_guard_noop_when_leader_already_retained() -> None:
    original = contexts(4)
    adapter = StaticAdapter(
        [
            CrossEncoderScore(context_id="ctx_0", score=0.9),
            CrossEncoderScore(context_id="ctx_1", score=0.8),
            CrossEncoderScore(context_id="ctx_2", score=0.7),
            CrossEncoderScore(context_id="ctx_3", score=0.6),
        ]
    )
    reranker = SafeCrossEncoderReranker(
        mode="remote",
        adapter=adapter,
        candidate_limit=4,
        context_limit=2,
    )

    result = await reranker.rerank("question", original)

    assert [item.metadata["context_id"] for item in result.contexts] == ["ctx_0", "ctx_1"]
    assert result.status is RerankStatus.APPLIED


@pytest.mark.asyncio
async def test_per_document_cap_defaults_off_without_document_identity() -> None:
    original = contexts(4)
    adapter = StaticAdapter(
        [
            CrossEncoderScore(context_id=f"ctx_{index}", score=0.9 - index / 10)
            for index in range(4)
        ]
    )
    reranker = SafeCrossEncoderReranker(
        mode="remote",
        adapter=adapter,
        candidate_limit=4,
        context_limit=4,
    )

    result = await reranker.rerank("question", original)

    assert [item.metadata["context_id"] for item in result.contexts] == [
        "ctx_0",
        "ctx_1",
        "ctx_2",
        "ctx_3",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scores", "code"),
    [
        ([CrossEncoderScore(context_id="unknown", score=0.5)], "cross_encoder_invalid_result"),
        ([CrossEncoderScore(context_id="ctx_0", score=math.nan)], "cross_encoder_invalid_result"),
        ([CrossEncoderScore(context_id="ctx_0", score=math.inf)], "cross_encoder_invalid_result"),
        ([CrossEncoderScore(context_id="ctx_0", score=0.5)], "cross_encoder_invalid_result"),
    ],
)
async def test_invalid_results_restore_exact_rrf_order(
    scores: list[CrossEncoderScore], code: str
) -> None:
    original = contexts(2)
    result = await SafeCrossEncoderReranker(
        mode="remote",
        adapter=StaticAdapter(scores),
        candidate_limit=2,
        context_limit=2,
    ).rerank("question", original)

    assert result.status is RerankStatus.FALLBACK
    assert result.contexts == original
    assert result.degradation_code == code


@pytest.mark.asyncio
async def test_local_load_failure_restores_exact_rrf_order() -> None:
    original = contexts(2)
    result = await SafeCrossEncoderReranker(
        mode="local", adapter=FailingAdapter(), candidate_limit=2, context_limit=2
    ).rerank("question", original)

    assert result.status is RerankStatus.FALLBACK
    assert result.contexts == original
    assert result.degradation_code == "cross_encoder_unavailable"
    assert "private path" not in (result.degradation_code or "")


@pytest.mark.asyncio
async def test_remote_timeout_restores_exact_rrf_order() -> None:
    original = contexts(2)
    result = await SafeCrossEncoderReranker(
        mode="remote",
        adapter=SlowAdapter(),
        candidate_limit=2,
        context_limit=2,
        timeout_seconds=0.001,
    ).rerank("question", original)

    assert result.status is RerankStatus.FALLBACK
    assert result.contexts == original
    assert result.degradation_code == "cross_encoder_timeout"


def test_invalid_mode_and_candidate_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        SafeCrossEncoderReranker(mode="auto")
    with pytest.raises(ValueError, match="context_limit"):
        SafeCrossEncoderReranker(
            mode="local",
            adapter=StaticAdapter([]),
            candidate_limit=1,
            context_limit=2,
        )
    with pytest.raises(ValueError, match="max_per_document"):
        SafeCrossEncoderReranker(
            mode="local",
            adapter=StaticAdapter([]),
            candidate_limit=2,
            context_limit=2,
            max_per_document=0,
        )
    with pytest.raises(ValueError, match="max_per_document"):
        SafeCrossEncoderReranker(
            mode="local",
            adapter=StaticAdapter([]),
            candidate_limit=2,
            context_limit=2,
            max_per_document=3,
        )
