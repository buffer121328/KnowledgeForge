"""Optional CrossEncoder boundary with bounded candidates and exact RRF fallback."""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence

from domain.knowledge import RetrievedContext


class RerankStatus(StrEnum):
    """Bounded outcomes for optional reranking."""

    DISABLED = "disabled"
    APPLIED = "applied"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class CrossEncoderScore:
    """One adapter score keyed by a current-run context identifier."""

    context_id: str
    score: float


@dataclass(frozen=True, slots=True)
class CrossEncoderResult:
    """Reranked contexts or the unchanged deterministic RRF fallback."""

    contexts: list[RetrievedContext]
    status: RerankStatus
    degradation_code: str | None = None
    scored_count: int = 0
    scores: tuple[CrossEncoderScore, ...] = ()


class CrossEncoderAdapter(Protocol):
    """Narrow interface implemented by approved local or remote adapters."""

    async def score(
        self,
        question: str,
        candidates: list[RetrievedContext],
    ) -> list[CrossEncoderScore]:
        """Return exactly one finite score for every supplied candidate."""


def context_id_for(context: RetrievedContext, index: int) -> str:
    """Return the bounded identifier used to join adapter scores to contexts."""
    metadata = context.metadata or {}
    for key in ("chunk_id", "context_id", "id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value[:128]
    digest = hashlib.sha256(
        f"{context.source}\0{context.content}".encode("utf-8")
    ).hexdigest()[:20]
    return f"ctx_{index}_{digest}"


class SafeCrossEncoderReranker:
    """Apply optional scoring without weakening RRF availability or ordering."""

    def __init__(
        self,
        *,
        mode: str,
        adapter: CrossEncoderAdapter | None = None,
        candidate_limit: int = 8,
        context_limit: int = 5,
        timeout_seconds: float = 2.0,
        max_per_document: int = 2,
    ) -> None:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"disabled", "remote", "local"}:
            raise ValueError("mode must be disabled, remote, or local")
        if candidate_limit <= 0 or candidate_limit > 50:
            raise ValueError("candidate_limit must be between 1 and 50")
        if context_limit <= 0 or context_limit > candidate_limit:
            raise ValueError("context_limit must be between 1 and candidate_limit")
        if timeout_seconds <= 0 or timeout_seconds > 30:
            raise ValueError("timeout_seconds must be between 0 and 30")
        if max_per_document <= 0 or max_per_document > candidate_limit:
            raise ValueError("max_per_document must be between 1 and candidate_limit")
        if normalized_mode != "disabled" and adapter is None:
            raise ValueError("an approved adapter is required outside disabled mode")
        self.mode = normalized_mode
        self.adapter = adapter
        self.candidate_limit = candidate_limit
        self.context_limit = context_limit
        self.timeout_seconds = timeout_seconds
        self.max_per_document = max_per_document

    async def rerank(
        self,
        question: str,
        contexts: Sequence[RetrievedContext],
    ) -> CrossEncoderResult:
        """Rerank bounded authorized candidates or preserve exact RRF order."""
        original = list(contexts)
        if self.mode == "disabled" or not original:
            return CrossEncoderResult(
                contexts=original,
                status=RerankStatus.DISABLED,
            )

        candidates = original[: self.candidate_limit]
        context_ids = [self._context_id(context, index) for index, context in enumerate(candidates)]
        try:
            assert self.adapter is not None
            scores = await asyncio.wait_for(
                self.adapter.score(question, candidates),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            return self._fallback(original, "cross_encoder_timeout")
        except Exception:
            return self._fallback(original, "cross_encoder_unavailable")

        if not self._valid_scores(context_ids, scores):
            return self._fallback(original, "cross_encoder_invalid_result")
        by_id = {score.context_id: score.score for score in scores}
        indexed = list(enumerate(zip(context_ids, candidates, strict=True)))
        indexed.sort(key=lambda item: (-by_id[item[1][0]], item[0]))
        reranked = self._select_with_document_guard(
            original, [context for _, (_, context) in indexed]
        )
        return CrossEncoderResult(
            contexts=reranked,
            status=RerankStatus.APPLIED,
            scored_count=len(scores),
            scores=tuple(scores),
        )

    def _select_with_document_guard(
        self,
        original: list[RetrievedContext],
        ordered_candidates: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        """Pick scored candidates per-document first, then fill by relevance.

        纯相关性重排会容忍同文档多个分块挤占全部名额，把矛盾/对比类问题
        需要的另一份制度整份挤出最终上下文。这里先给每份候选文档保留一个
        最好分块（文档按其最高分排序），剩余名额再按相关性回填；并始终
        保留 RRF 融合的头名（稠密+稀疏双重共识证据）。单文档回填由
        max_per_document 封顶，避免退化成单文档独占。
        """
        groups: dict[str, list[tuple[int, RetrievedContext]]] = {}
        for position, candidate in enumerate(ordered_candidates):
            identity = self._document_identity(candidate) or f"\x00single-{position}"
            groups.setdefault(identity, []).append((position, candidate))

        selected: list[RetrievedContext] = []
        per_document: dict[str, int] = {}
        # ① 每份文档的最好分块先入局（组按其最高分降序排列）
        for identity, group in groups.items():
            if len(selected) >= self.context_limit:
                break
            per_document[identity] = 1
            selected.append(group[0][1])
        # ② 剩余名额按相关性回填（position 升序即 CE 分数降序），
        #    单文档不超过 max_per_document
        remaining = sorted(
            ((position, candidate)
             for group in groups.values()
             for position, candidate in group[1:]),
            key=lambda item: item[0],
        )
        for _, candidate in remaining:
            if len(selected) >= self.context_limit:
                break
            identity = self._document_identity(candidate) or ""
            count = per_document.get(identity, 0)
            if count >= self.max_per_document:
                continue
            per_document[identity] = count + 1
            selected.append(candidate)

        leader = original[0]
        if all(candidate is not leader for candidate in selected):
            if len(selected) >= self.context_limit:
                selected.pop()
            selected.insert(0, leader)
        return selected

    @staticmethod
    def _document_identity(context: RetrievedContext) -> str:
        """Resolve the source document identity used by the per-document cap."""
        metadata = context.metadata or {}
        for key in ("source_document_id", "doc_id"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
        for key in ("chunk_id", "context_id", "id"):
            value = str(metadata.get(key) or "").strip()
            if "#" in value:
                return value.split("#", 1)[0]
        return ""

    @staticmethod
    def _fallback(
        original: list[RetrievedContext], degradation_code: str
    ) -> CrossEncoderResult:
        return CrossEncoderResult(
            contexts=original,
            status=RerankStatus.FALLBACK,
            degradation_code=degradation_code,
        )

    @staticmethod
    def _valid_scores(
        expected_context_ids: list[str],
        scores: list[CrossEncoderScore],
    ) -> bool:
        returned_ids = [score.context_id for score in scores]
        return bool(
            len(scores) == len(expected_context_ids)
            and len(returned_ids) == len(set(returned_ids))
            and set(returned_ids) == set(expected_context_ids)
            and all(math.isfinite(score.score) for score in scores)
        )

    @staticmethod
    def _context_id(context: RetrievedContext, index: int) -> str:
        return context_id_for(context, index)
