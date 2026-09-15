"""QA context fusion, de-duplication, ranking, and confidence helpers."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from domain.knowledge import RetrievedContext

_DEFAULT_SOURCE_WEIGHTS = {
    "vector": 1.0,
    "graph": 1.05,
    "bm25": 1.0,
    "hybrid": 1.0,
}


def _provenance_key(context: RetrievedContext) -> str:
    """Build a stable evidence identity without relying on raw score scales."""

    metadata = context.metadata
    chunk_id = str(metadata.get("chunk_id") or "").strip()
    if chunk_id:
        return f"chunk:{chunk_id}"
    claim_id = str(metadata.get("claim_id") or "").strip()
    if claim_id:
        return f"claim:{claim_id}"
    source_document_id = str(metadata.get("source_document_id") or "").strip()
    chunk_index = metadata.get("chunk_index")
    if source_document_id:
        return f"document:{source_document_id}:{chunk_index}"
    digest = hashlib.sha256(context.content.strip().encode("utf-8")).hexdigest()
    return f"content:{digest}"


def hybrid_rerank(
    contexts: list[RetrievedContext],
    *,
    rrf_k: int = 60,
    source_weights: dict[str, float] | None = None,
) -> list[RetrievedContext]:
    """Fuse per-source ranks with deterministic reciprocal-rank fusion."""

    if not contexts:
        return []
    bounded_k = max(1, min(int(rrf_k), 10_000))
    weights = {**_DEFAULT_SOURCE_WEIGHTS, **(source_weights or {})}
    by_source: dict[str, list[RetrievedContext]] = defaultdict(list)
    for context in contexts:
        by_source[context.retrieval_type or "unknown"].append(context)

    contributions: dict[str, float] = defaultdict(float)
    source_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    raw_scores: dict[str, dict[str, float]] = defaultdict(dict)
    representatives: dict[str, RetrievedContext] = {}
    claim_ids: dict[str, set[str]] = defaultdict(set)

    for source in sorted(by_source):
        ordered = sorted(
            by_source[source],
            key=lambda item: (-float(item.score), _provenance_key(item), item.content),
        )
        seen_in_source: set[str] = set()
        source_rank = 0
        for context in ordered:
            key = _provenance_key(context)
            if key in seen_in_source:
                continue
            seen_in_source.add(key)
            source_rank += 1
            source_ranks[key][source] = source_rank
            raw_scores[key][source] = float(context.score)
            contributions[key] += float(weights.get(source, 1.0)) / (bounded_k + source_rank)
            claim_id = str(context.metadata.get("claim_id") or "").strip()
            if claim_id:
                claim_ids[key].add(claim_id)
            current = representatives.get(key)
            prefer_canonical_content = (
                current is not None
                and current.retrieval_type == "graph"
                and context.retrieval_type != "graph"
            )
            same_content_kind = (
                current is None
                or current.retrieval_type == "graph"
                or context.retrieval_type != "graph"
            )
            if (
                current is None
                or prefer_canonical_content
                or (same_content_kind and float(context.score) > float(current.score))
            ):
                representatives[key] = context

    ordered_keys = sorted(
        representatives,
        key=lambda key: (-contributions[key], key),
    )
    max_contribution = max(contributions.values(), default=1.0) or 1.0
    ranked: list[RetrievedContext] = []
    for final_rank, key in enumerate(ordered_keys, start=1):
        context = representatives[key]
        normalized_score = contributions[key] / max_contribution
        ranks = dict(sorted(source_ranks[key].items()))
        scores = dict(sorted(raw_scores[key].items()))
        context.score = normalized_score
        context.metadata = {
            **context.metadata,
            "fusion": {
                "method": "rrf",
                "rrf_k": bounded_k,
                "source_rank": min(ranks.values()),
                "source_ranks": ranks,
                "source_raw_scores": scores,
                "source_weights": {
                    source: float(weights.get(source, 1.0)) for source in ranks
                },
                "final_rank": final_rank,
                "score": normalized_score,
                "provenance_key": key,
                "claim_ids": sorted(claim_ids[key]),
            },
        }
        ranked.append(context)
    return ranked


def calc_confidence(contexts: list[RetrievedContext]) -> float:
    """Calculate the confidence."""
    if not contexts:
        return 0.0
    return min(sum(context.score for context in contexts) / len(contexts), 1.0)
