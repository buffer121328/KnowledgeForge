"""Redis 缓存服务

提供:
  - CacheService:    通用 key-value 缓存（JSON 序列化）
  - QACache:         问答结果缓存（按问题哈希）
  - EmbeddingCache:  Embedding 向量缓存

多级缓存策略（L1 进程内 LRU + L2 Redis）由上层组合，
本模块只负责 L2 Redis 实现。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import redis.asyncio as redis

from shared.utils.logging import get_logger, safe_fingerprint

logger = get_logger(__name__)


class CacheService:
    """通用 Redis 缓存（异步）"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", **kwargs: Any) -> None:
        """Initialize the cache service."""
        self.redis = redis.from_url(redis_url, decode_responses=True, **kwargs)

    async def get(self, key: str) -> Any | None:
        """获取缓存值，返回反序列化后的对象；不存在返回 None"""
        try:
            data = await self.redis.get(key)
            if data is None:
                return None
            return json.loads(data)
        except json.JSONDecodeError:
            logger.warning(
                "cache_decode_failed",
                key_fingerprint=safe_fingerprint(key, namespace="cache-key"),
            )
            await self.redis.delete(key)
            return None
        except Exception as e:
            logger.warning(
                "cache_get_failed",
                key_fingerprint=safe_fingerprint(key, namespace="cache-key"),
                error_type=type(e).__name__,
            )
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """写入缓存，ttl 单位秒"""
        try:
            await self.redis.set(key, json.dumps(value, default=str), ex=ttl)
        except Exception as e:
            logger.warning(
                "cache_set_failed",
                key_fingerprint=safe_fingerprint(key, namespace="cache-key"),
                error_type=type(e).__name__,
            )

    async def delete(self, key: str) -> None:
        """Delete a record through the cache service."""
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.warning(
                "cache_delete_failed",
                key_fingerprint=safe_fingerprint(key, namespace="cache-key"),
                error_type=type(e).__name__,
            )

    async def take(self, key: str) -> Any | None:
        """Atomically return and remove one JSON value when Redis supports GETDEL."""
        try:
            data = await self.redis.getdel(key)
            if data is None:
                return None
            return json.loads(data)
        except json.JSONDecodeError:
            await self.redis.delete(key)
            return None
        except Exception as e:
            logger.warning(
                "cache_take_failed",
                key_fingerprint=safe_fingerprint(key, namespace="cache-key"),
                error_type=type(e).__name__,
            )
            return None

    async def exists(self, key: str) -> bool:
        """Report whether the exists."""
        try:
            return bool(await self.redis.exists(key))
        except Exception:
            return False

    async def incr(self, key: str, ttl: int | None = None) -> int:
        """自增计数器，可选 ttl"""
        try:
            count = await self.redis.incr(key)
            if ttl and count == 1:
                await self.redis.expire(key, ttl)
            return count
        except Exception as e:
            logger.warning(
                "cache_incr_failed",
                key_fingerprint=safe_fingerprint(key, namespace="cache-key"),
                error_type=type(e).__name__,
            )
            return 0

    async def close(self) -> None:
        """Release resources held by the cache service."""
        await self.redis.aclose()


# ── QA 结果缓存 ────────────────────────────────────────────────

class QACache:
    """Versioned exact QA cache scoped by tenant, user, mode and revision."""

    KEY_SCHEMA_VERSION = "v2"

    def __init__(self, cache: CacheService, ttl: int = 3600) -> None:
        """Initialize the question-answering cache."""
        self.cache = cache
        self.ttl = ttl

    def _make_key(
        self,
        question: str,
        user_id: str,
        retrieval_mode: str = "hybrid",
        tenant_id: str = "",
        knowledge_revision: int = 0,
    ) -> str:
        """Create a non-reversible key for an exact, normalized question."""
        normalized = re.sub(r"\s+", " ", question.strip()).casefold()
        question_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        scope_digest = hashlib.sha256(
            f"{tenant_id}:{user_id}:{retrieval_mode}:{knowledge_revision}:{question_digest}".encode(
                "utf-8"
            )
        ).hexdigest()
        return f"qa:cache:{self.KEY_SCHEMA_VERSION}:{scope_digest}"

    async def get(
        self,
        question: str,
        user_id: str,
        retrieval_mode: str = "hybrid",
        tenant_id: str = "",
        knowledge_revision: int = 0,
    ) -> dict | None:
        """Return the requested value from the question-answering cache."""
        return await self.cache.get(
            self._make_key(question, user_id, retrieval_mode, tenant_id, knowledge_revision)
        )

    async def set(
        self,
        question: str,
        user_id: str,
        result: dict,
        retrieval_mode: str = "hybrid",
        tenant_id: str = "",
        knowledge_revision: int = 0,
        qa_run_id: str = "",
    ) -> None:
        """Store the question-answering cache."""
        key = self._make_key(question, user_id, retrieval_mode, tenant_id, knowledge_revision)
        await self.cache.set(
            key,
            result,
            ttl=self.ttl,
        )
        if qa_run_id:
            await self.cache.set(
                self._run_key(qa_run_id),
                {"cache_keys": [key]},
                ttl=self.ttl,
            )

    async def invalidate(
        self,
        question: str,
        user_id: str,
        retrieval_mode: str = "hybrid",
        tenant_id: str = "",
        knowledge_revision: int = 0,
    ) -> None:
        """Invalidate the question-answering cache."""
        await self.cache.delete(
            self._make_key(question, user_id, retrieval_mode, tenant_id, knowledge_revision)
        )

    @staticmethod
    def _run_key(qa_run_id: str) -> str:
        digest = hashlib.sha256(qa_run_id.encode("utf-8")).hexdigest()
        return f"qa:run-cache:v1:{digest}"

    async def associate_run(self, qa_run_id: str, cache_key: str) -> None:
        """Associate a persisted run with a bounded exact-cache key list."""
        await self.cache.set(self._run_key(qa_run_id), {"cache_keys": [cache_key]}, ttl=self.ttl)

    async def associate_answer(
        self,
        qa_run_id: str,
        question: str,
        user_id: str,
        retrieval_mode: str = "hybrid",
        tenant_id: str = "",
        knowledge_revision: int = 0,
    ) -> None:
        """Associate a persisted run with its scoped exact-answer key."""
        await self.associate_run(
            qa_run_id,
            self._make_key(
                question,
                user_id,
                retrieval_mode,
                tenant_id,
                knowledge_revision,
            ),
        )

    async def invalidate_run(self, qa_run_id: str) -> None:
        """Invalidate the bounded cache keys associated with one QA run."""
        association_key = self._run_key(qa_run_id)
        association = await self.cache.take(association_key)
        for key in (association or {}).get("cache_keys", [])[:4]:
            if isinstance(key, str) and key.startswith("qa:cache:v2:"):
                await self.cache.delete(key)


class KnowledgeRevisionStore:
    """Maintain a monotonic per-tenant knowledge revision in Redis."""

    def __init__(self, cache: CacheService) -> None:
        self.cache = cache

    @staticmethod
    def _key(tenant_id: str) -> str:
        digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        return f"qa:knowledge-revision:v1:{digest}"

    async def get(self, tenant_id: str) -> int:
        """Return the current revision; unavailable Redis safely yields zero."""
        value = await self.cache.get(self._key(tenant_id))
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    async def advance(self, tenant_id: str) -> int:
        """Advance the revision after a committed tenant knowledge mutation."""
        return await self.cache.incr(self._key(tenant_id))


# ── Embedding 缓存 ─────────────────────────────────────────────

class EmbeddingCache:
    """Embedding 向量缓存 - 按文本哈希作为 key"""

    def __init__(self, cache: CacheService, ttl: int = 7 * 24 * 3600) -> None:
        """Initialize the embedding cache."""
        self.cache = cache
        self.ttl = ttl

    def _make_key(self, text: str, model: str = "default") -> str:
        """Create the key."""
        t_hash = hashlib.sha256(f"{model}:{text}".encode("utf-8")).hexdigest()
        return f"embedding:{t_hash}"

    async def get(self, text: str, model: str = "default") -> list[float] | None:
        """Return the requested value from the embedding cache."""
        return await self.cache.get(self._make_key(text, model))

    async def set(self, text: str, embedding: list[float], model: str = "default") -> None:
        """Store the embedding cache."""
        await self.cache.set(self._make_key(text, model), embedding, ttl=self.ttl)


# ── 单例 ───────────────────────────────────────────────────────

_cache_service: CacheService | None = None


def get_cache_service() -> CacheService:
    """获取全局 CacheService 单例"""
    global _cache_service
    if _cache_service is None:
        from shared.config import settings
        redis_url = getattr(settings, "redis_url", "redis://localhost:6379/0")
        timeout = getattr(settings, "readiness_probe_timeout_seconds", 2.0)
        _cache_service = CacheService(
            redis_url,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
            retry_on_timeout=False,
        )
    return _cache_service
