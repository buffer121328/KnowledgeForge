"""Bounded semantic QA cache and one-time confirmation boundaries.

The adapter deliberately exposes only bounded vector operations.  It has no
key-enumeration fallback, so a Redis deployment without vector search simply
disables semantic reuse while exact cache and full RAG continue to work.
"""

from __future__ import annotations

import hashlib
import inspect
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Awaitable, Callable, Protocol

from infrastructure.cache.redis import CacheService
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class SemanticCacheMode(StrEnum):
    """Supported semantic-cache rollout modes."""

    DISABLED = "disabled"
    SHADOW = "shadow"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class SemanticCandidate:
    """One already scope-filtered candidate returned by a bounded index."""

    question: str
    similarity: float
    cached_at: str
    cached_result: dict[str, Any]
    candidate_run_id: str = ""


@dataclass(frozen=True)
class SemanticConfirmationRequired:
    """Safe API-facing candidate metadata; cached answer is intentionally absent."""

    similar_question: str
    similarity: float
    cached_at: str
    confirmation_token: str


class SemanticIndex(Protocol):
    """Minimal bounded vector-index capability used by semantic cache."""

    async def available(self) -> bool: ...

    async def search(
        self,
        *,
        question: str,
        tenant_id: str,
        user_scope: str,
        retrieval_mode: str,
        knowledge_revision: int,
        limit: int,
    ) -> list[SemanticCandidate]: ...

    async def upsert(
        self,
        *,
        question: str,
        tenant_id: str,
        user_scope: str,
        retrieval_mode: str,
        knowledge_revision: int,
        result: dict[str, Any],
        qa_run_id: str,
    ) -> None: ...


class DisabledSemanticIndex:
    """No-op index used when vector-search capability is unavailable."""

    async def available(self) -> bool:
        return False

    async def search(self, **_kwargs: Any) -> list[SemanticCandidate]:
        return []

    async def upsert(self, **_kwargs: Any) -> None:
        return None


class BoundedSemanticIndexAdapter:
    """Adapter for a Redis Search/vector implementation supplied at assembly.

    The callbacks must implement a bounded, server-side vector query.  There is
    intentionally no Redis ``SCAN`` or in-process whole-cache fallback.
    """

    def __init__(
        self,
        *,
        capability_probe: Callable[[], bool | Awaitable[bool]],
        search: Callable[..., list[SemanticCandidate] | Awaitable[list[SemanticCandidate]]],
        upsert: Callable[..., None | Awaitable[None]],
    ) -> None:
        self._capability_probe = capability_probe
        self._search = search
        self._upsert = upsert

    @staticmethod
    async def _resolve(value: Any) -> Any:
        return await value if inspect.isawaitable(value) else value

    async def available(self) -> bool:
        try:
            return bool(await self._resolve(self._capability_probe()))
        except Exception as error:
            logger.warning("semantic_cache_capability_unavailable", error_type=type(error).__name__)
            return False

    async def search(self, **kwargs: Any) -> list[SemanticCandidate]:
        if not await self.available():
            return []
        try:
            values = await self._resolve(self._search(**kwargs))
            return list(values)[: max(1, min(int(kwargs.get("limit", 1)), 20))]
        except Exception as error:
            logger.warning("semantic_cache_lookup_failed", error_type=type(error).__name__)
            return []

    async def upsert(self, **kwargs: Any) -> None:
        if not await self.available():
            return
        try:
            await self._resolve(self._upsert(**kwargs))
        except Exception as error:
            logger.warning("semantic_cache_write_failed", error_type=type(error).__name__)


class SemanticQACache:
    """Coordinate disabled, shadow and explicit-confirm semantic reuse."""

    def __init__(
        self,
        index: SemanticIndex,
        *,
        mode: str = "shadow",
        similarity_threshold: float = 0.9,
        candidate_limit: int = 3,
    ) -> None:
        self.index = index
        self.mode = SemanticCacheMode(mode)
        self.similarity_threshold = similarity_threshold
        self.candidate_limit = max(1, min(candidate_limit, 20))

    async def lookup(
        self,
        *,
        question: str,
        tenant_id: str,
        user_scope: str,
        retrieval_mode: str,
        knowledge_revision: int,
    ) -> SemanticCandidate | None:
        """Return only the best qualifying bounded candidate."""
        if self.mode is SemanticCacheMode.DISABLED or not await self.index.available():
            return None
        candidates = await self.index.search(
            question=question,
            tenant_id=tenant_id,
            user_scope=user_scope,
            retrieval_mode=retrieval_mode,
            knowledge_revision=knowledge_revision,
            limit=self.candidate_limit,
        )
        eligible = [item for item in candidates if item.similarity >= self.similarity_threshold]
        return max(eligible, key=lambda item: item.similarity, default=None)

    async def record(
        self,
        *,
        question: str,
        tenant_id: str,
        user_scope: str,
        retrieval_mode: str,
        knowledge_revision: int,
        result: dict[str, Any],
        qa_run_id: str = "",
    ) -> None:
        """Write one eligible answer only when a bounded index is available."""
        if self.mode is SemanticCacheMode.DISABLED:
            return
        await self.index.upsert(
            question=question,
            tenant_id=tenant_id,
            user_scope=user_scope,
            retrieval_mode=retrieval_mode,
            knowledge_revision=knowledge_revision,
            result=result,
            qa_run_id=qa_run_id,
        )


class SemanticConfirmationService:
    """Issue and atomically consume scoped semantic confirmation/bypass tokens."""

    def __init__(self, cache: CacheService, *, ttl_seconds: int = 300) -> None:
        self.cache = cache
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(kind: str, token: str) -> str:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return f"qa:semantic-{kind}:v1:{digest}"

    async def issue(
        self,
        candidate: SemanticCandidate,
        *,
        question: str,
        tenant_id: str,
        user_scope: str,
        retrieval_mode: str,
        knowledge_revision: int,
    ) -> str:
        """Store a short-lived opaque token without exposing its cached answer."""
        token = secrets.token_urlsafe(32)
        await self.cache.set(
            self._key("confirmation", token),
            {
                "question": question,
                "tenant_id": tenant_id,
                "user_scope": user_scope,
                "retrieval_mode": retrieval_mode,
                "knowledge_revision": knowledge_revision,
                "candidate_question": candidate.question,
                "similarity": candidate.similarity,
                "cached_at": candidate.cached_at,
                "candidate_run_id": candidate.candidate_run_id,
                "cached_result": candidate.cached_result,
            },
            ttl=self.ttl_seconds,
        )
        return token

    @staticmethod
    def _matches(
        payload: dict[str, Any],
        *,
        question: str,
        tenant_id: str,
        user_scope: str,
        retrieval_mode: str,
        knowledge_revision: int,
    ) -> bool:
        return all(
            (
                payload.get("question") == question,
                payload.get("tenant_id") == tenant_id,
                payload.get("user_scope") == user_scope,
                payload.get("retrieval_mode") == retrieval_mode,
                int(payload.get("knowledge_revision", -1)) == knowledge_revision,
            )
        )

    async def consume_confirmation(self, token: str, **scope: Any) -> dict[str, Any] | None:
        """Consume a confirmation token once and return it only for the same scope."""
        payload = await self.cache.take(self._key("confirmation", token))
        if not isinstance(payload, dict) or not self._matches(payload, **scope):
            return None
        return payload

    async def reject(self, token: str, **scope: Any) -> str | None:
        """Consume a valid confirmation and issue one equally scoped bypass."""
        payload = await self.consume_confirmation(token, **scope)
        if payload is None:
            return None
        bypass = secrets.token_urlsafe(32)
        await self.cache.set(
            self._key("bypass", bypass),
            {
                **{name: scope[name] for name in (
                    "question", "tenant_id", "user_scope", "retrieval_mode", "knowledge_revision"
                )},
                "issued_at": datetime.now(UTC).isoformat(),
            },
            ttl=self.ttl_seconds,
        )
        return bypass

    async def consume_bypass(self, token: str, **scope: Any) -> bool:
        """Consume one bypass and verify every confirmation boundary."""
        payload = await self.cache.take(self._key("bypass", token))
        return isinstance(payload, dict) and self._matches(payload, **scope)
