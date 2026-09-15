"""Tests for versioned exact cache and scoped semantic confirmation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from infrastructure.cache.redis import KnowledgeRevisionStore, QACache
from infrastructure.cache.semantic import (
    BoundedSemanticIndexAdapter,
    SemanticCacheMode,
    SemanticCandidate,
    SemanticConfirmationService,
    SemanticQACache,
)


class MemoryCache:
    """Small async cache double with atomic take semantics."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    async def get(self, key: str):
        return self.values.get(key)

    async def set(self, key: str, value, ttl: int = 3600):
        self.values[key] = value

    async def delete(self, key: str):
        self.values.pop(key, None)

    async def take(self, key: str):
        return self.values.pop(key, None)

    async def incr(self, key: str, ttl: int | None = None) -> int:
        value = int(self.values.get(key, 0)) + 1
        self.values[key] = value
        return value


def _candidate() -> SemanticCandidate:
    return SemanticCandidate(
        question="如何部署？",
        similarity=0.96,
        cached_at="2026-08-01T00:00:00+00:00",
        cached_result={"answer": "受保护答案", "contexts": []},
        candidate_run_id="run-1",
    )


def _scope(**overrides):
    values = {
        "question": "怎样部署？",
        "tenant_id": "org-1",
        "user_scope": "user-1",
        "retrieval_mode": "hybrid",
        "knowledge_revision": 3,
    }
    values.update(overrides)
    return values


@pytest.mark.asyncio
async def test_exact_key_is_normalized_versioned_and_scope_bound():
    cache = QACache(AsyncMock())
    first = cache._make_key("  HOW   To Deploy? ", "u1", "hybrid", "o1", 2)
    normalized = cache._make_key("how to deploy?", "u1", "hybrid", "o1", 2)
    assert first == normalized
    assert first.startswith("qa:cache:v2:")
    assert first != cache._make_key("how to deploy?", "u2", "hybrid", "o1", 2)
    assert first != cache._make_key("how to deploy?", "u1", "hybrid", "o1", 3)


@pytest.mark.asyncio
async def test_revision_advances_and_run_invalidation_is_bounded():
    cache = MemoryCache()
    revision = KnowledgeRevisionStore(cache)  # type: ignore[arg-type]
    assert await revision.get("org-1") == 0
    assert await revision.advance("org-1") == 1
    assert await revision.advance("org-1") == 2

    qa_cache = QACache(cache)  # type: ignore[arg-type]
    await qa_cache.set("q", "u", {"answer": "a"}, tenant_id="o", qa_run_id="run")
    exact_key = qa_cache._make_key("q", "u", tenant_id="o")
    assert exact_key in cache.values
    await qa_cache.invalidate_run("run")
    assert exact_key not in cache.values


@pytest.mark.asyncio
async def test_persisted_run_can_be_associated_after_answer_is_cached():
    cache = MemoryCache()
    qa_cache = QACache(cache)  # type: ignore[arg-type]
    await qa_cache.set("q", "u", {"answer": "a"}, tenant_id="o", knowledge_revision=2)
    await qa_cache.associate_answer(
        "run", "q", "u", tenant_id="o", knowledge_revision=2
    )

    await qa_cache.invalidate_run("run")

    assert qa_cache._make_key("q", "u", tenant_id="o", knowledge_revision=2) not in cache.values


@pytest.mark.asyncio
async def test_missing_vector_capability_never_calls_search_or_scan():
    search = AsyncMock()
    upsert = AsyncMock()
    adapter = BoundedSemanticIndexAdapter(
        capability_probe=lambda: False,
        search=search,
        upsert=upsert,
    )
    semantic = SemanticQACache(adapter, mode="shadow")
    assert await semantic.lookup(**_scope()) is None
    search.assert_not_awaited()
    upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_shadow_returns_candidate_to_orchestrator_but_mode_remains_shadow():
    adapter = BoundedSemanticIndexAdapter(
        capability_probe=lambda: True,
        search=AsyncMock(return_value=[_candidate()]),
        upsert=AsyncMock(),
    )
    semantic = SemanticQACache(adapter, mode="shadow", similarity_threshold=0.9)
    result = await semantic.lookup(**_scope())
    assert result == _candidate()
    assert semantic.mode is SemanticCacheMode.SHADOW


@pytest.mark.asyncio
async def test_confirmation_is_scope_revision_and_one_time_bound():
    cache = MemoryCache()
    service = SemanticConfirmationService(cache, ttl_seconds=300)  # type: ignore[arg-type]
    token = await service.issue(_candidate(), **_scope())

    assert await service.consume_confirmation(token, **_scope(user_scope="other")) is None
    assert await service.consume_confirmation(token, **_scope()) is None

    token = await service.issue(_candidate(), **_scope())
    payload = await service.consume_confirmation(token, **_scope())
    assert payload is not None
    assert payload["cached_result"]["answer"] == "受保护答案"
    assert await service.consume_confirmation(token, **_scope()) is None


@pytest.mark.asyncio
async def test_reject_issues_one_time_scoped_bypass():
    cache = MemoryCache()
    service = SemanticConfirmationService(cache, ttl_seconds=300)  # type: ignore[arg-type]
    token = await service.issue(_candidate(), **_scope())
    bypass = await service.reject(token, **_scope())
    assert bypass
    assert await service.consume_bypass(bypass, **_scope()) is True
    assert await service.consume_bypass(bypass, **_scope()) is False
