"""Pure offline retrieval metrics with explicit source and exact namespaces."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

STAGE_RETRIEVAL_METRICS_VERSION = "stage-retrieval-metrics-v1"
_STAGES = frozenset({"dense", "bm25", "graph", "fused", "reranked"})
_STAGE_STATUSES = frozenset(
    {"executed", "executed_empty", "not_executed", "unavailable"}
)
_NAMESPACES = ("source_document", "exact_evidence")


class RetrievalMetricError(ValueError):
    """A stage artifact cannot be scored without mixing or inferring identities."""


@dataclass(frozen=True, slots=True)
class ExactEvidenceIdentity:
    """One reviewed chunk, addressable by stable context ID and/or content hash."""

    context_id: str | None = None
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        if not _text(self.context_id) and not _sha256(self.content_sha256):
            raise RetrievalMetricError("reviewed_exact_identity_invalid")
        if self.content_sha256 is not None and not _sha256(self.content_sha256):
            raise RetrievalMetricError("reviewed_content_sha256_invalid")


@dataclass(frozen=True, slots=True)
class ReviewedRelevance:
    """Frozen relevance judgments kept in their original identity namespaces."""

    source_document_ids: frozenset[str]
    exact_evidence: tuple[ExactEvidenceIdentity, ...]

    def __post_init__(self) -> None:
        if not self.source_document_ids or not all(
            _text(item) for item in self.source_document_ids
        ):
            raise RetrievalMetricError("reviewed_source_identity_invalid")
        if not self.exact_evidence:
            raise RetrievalMetricError("reviewed_exact_identity_missing")


def _text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128


def _sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _validated_k_values(k_values: Sequence[int], *, budget: int) -> tuple[int, ...]:
    values = tuple(k_values)
    if not values or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in values):
        raise RetrievalMetricError("retrieval_k_values_invalid")
    if len(values) != len(set(values)):
        raise RetrievalMetricError("retrieval_k_values_duplicate")
    if max(values) > budget:
        raise RetrievalMetricError("k_exceeds_candidate_budget")
    return tuple(sorted(values))


def _exact_match(candidate: Mapping[str, Any], relevant: ExactEvidenceIdentity) -> bool:
    context_id = candidate.get("context_id")
    content_sha256 = candidate.get("content_sha256")
    return bool(
        (_text(context_id) and context_id == relevant.context_id)
        or (_sha256(content_sha256) and content_sha256 == relevant.content_sha256)
    )


def _binary_rank_metrics(
    matched_relevance: Sequence[int | None],
    *,
    relevant_count: int,
    k_values: Sequence[int],
) -> dict[str, Any]:
    first_rank = next(
        (rank for rank, match in enumerate(matched_relevance, start=1) if match is not None),
        None,
    )
    recall_at_k: dict[str, float] = {}
    ndcg_at_k: dict[str, float] = {}
    for k in k_values:
        prefix = matched_relevance[:k]
        unique_hits = {match for match in prefix if match is not None}
        recall_at_k[str(k)] = len(unique_hits) / relevant_count
        seen: set[int] = set()
        dcg = 0.0
        for rank, match in enumerate(prefix, start=1):
            if match is None or match in seen:
                continue
            seen.add(match)
            dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(relevant_count, k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg_at_k[str(k)] = dcg / ideal_dcg if ideal_dcg else 0.0
    return {
        "status": "scored",
        "recall_at_k": recall_at_k,
        "first_relevant_rank": first_rank,
        "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
        "ndcg_at_k": ndcg_at_k,
    }


def score_stage_retrieval(
    stage_record: Mapping[str, Any],
    *,
    relevance: ReviewedRelevance,
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Score one bounded stage using ranks and identities, never raw branch scores."""

    stage = stage_record.get("stage")
    status = stage_record.get("status")
    budget = stage_record.get("candidate_budget")
    candidates = stage_record.get("candidates")
    if stage not in _STAGES:
        raise RetrievalMetricError("retrieval_stage_invalid")
    if status not in _STAGE_STATUSES:
        raise RetrievalMetricError("retrieval_stage_status_invalid")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise RetrievalMetricError("candidate_budget_invalid")
    ks = _validated_k_values(k_values, budget=budget)
    if not isinstance(candidates, list) or len(candidates) > budget:
        raise RetrievalMetricError("candidate_collection_invalid")
    if status in {"not_executed", "unavailable"}:
        if candidates:
            raise RetrievalMetricError("unexecuted_stage_has_candidates")
        reason = "stage_not_executed" if status == "not_executed" else "stage_unavailable"
        return {
            "schema_version": STAGE_RETRIEVAL_METRICS_VERSION,
            "stage": stage,
            "stage_status": status,
            "candidate_budget": budget,
            "candidate_count": 0,
            "k_values": list(ks),
            "namespaces": {
                namespace: {"status": status, "reason_code": reason}
                for namespace in _NAMESPACES
            },
        }
    if status == "executed_empty" and candidates:
        raise RetrievalMetricError("empty_stage_has_candidates")
    if status == "executed" and not candidates:
        raise RetrievalMetricError("executed_stage_candidates_missing")
    if any(not isinstance(candidate, dict) for candidate in candidates):
        raise RetrievalMetricError("candidate_record_invalid")
    if [candidate.get("rank") for candidate in candidates] != list(
        range(1, len(candidates) + 1)
    ):
        raise RetrievalMetricError("candidate_rank_sequence_invalid")

    if not candidates:
        zeroes = {str(k): 0.0 for k in ks}
        namespaces = {
            namespace: {
                "status": "scored",
                "recall_at_k": dict(zeroes),
                "first_relevant_rank": None,
                "mrr": 0.0,
                "ndcg_at_k": dict(zeroes),
            }
            for namespace in _NAMESPACES
        }
    else:
        source_valid = all(_text(candidate.get("source_document_id")) for candidate in candidates)
        exact_valid = all(
            _text(candidate.get("context_id")) or _sha256(candidate.get("content_sha256"))
            for candidate in candidates
        )
        namespaces: dict[str, Any] = {}
        if source_valid:
            source_relevant = sorted(relevance.source_document_ids)
            source_indexes = {identity: index for index, identity in enumerate(source_relevant)}
            namespaces["source_document"] = _binary_rank_metrics(
                [source_indexes.get(candidate["source_document_id"]) for candidate in candidates],
                relevant_count=len(source_relevant),
                k_values=ks,
            )
        else:
            namespaces["source_document"] = {
                "status": "invalid_provenance",
                "reason_code": "candidate_source_document_identity_missing",
            }
        if exact_valid:
            matches: list[int | None] = []
            for candidate in candidates:
                matches.append(
                    next(
                        (
                            index
                            for index, relevant in enumerate(relevance.exact_evidence)
                            if _exact_match(candidate, relevant)
                        ),
                        None,
                    )
                )
            namespaces["exact_evidence"] = _binary_rank_metrics(
                matches,
                relevant_count=len(relevance.exact_evidence),
                k_values=ks,
            )
        else:
            namespaces["exact_evidence"] = {
                "status": "invalid_provenance",
                "reason_code": "candidate_exact_identity_missing",
            }
    return {
        "schema_version": STAGE_RETRIEVAL_METRICS_VERSION,
        "stage": stage,
        "stage_status": status,
        "candidate_budget": budget,
        "candidate_count": len(candidates),
        "k_values": list(ks),
        "namespaces": namespaces,
    }


def _aggregate_namespace(
    rows: Sequence[Mapping[str, Any]], *, k_values: Sequence[int]
) -> dict[str, Any]:
    statuses = Counter(str(row.get("status")) for row in rows)
    scored = [row for row in rows if row.get("status") == "scored"]
    result: dict[str, Any] = {
        "status_counts": dict(sorted(statuses.items())),
        "scored": len(scored),
        "mean_recall_at_k": {
            str(k): (
                sum(float(row["recall_at_k"][str(k)]) for row in scored) / len(scored)
                if scored
                else None
            )
            for k in k_values
        },
        "mean_ndcg_at_k": {
            str(k): (
                sum(float(row["ndcg_at_k"][str(k)]) for row in scored) / len(scored)
                if scored
                else None
            )
            for k in k_values
        },
        "mean_mrr": (
            sum(float(row["mrr"]) for row in scored) / len(scored) if scored else None
        ),
    }
    return result


def aggregate_stage_retrieval(
    case_stage_results: Iterable[Mapping[str, Any]],
    *,
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Aggregate case metrics without combining stages or identity namespaces."""

    rows = list(case_stage_results)
    if not rows:
        raise RetrievalMetricError("stage_results_empty")
    ks = tuple(sorted(k_values))
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    category_grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        benchmark_id = row.get("benchmark_id")
        category = row.get("category")
        stage = row.get("stage")
        if not _text(benchmark_id) or not _text(category) or stage not in _STAGES:
            raise RetrievalMetricError("stage_result_identity_invalid")
        if row.get("schema_version") != STAGE_RETRIEVAL_METRICS_VERSION:
            raise RetrievalMetricError("stage_result_schema_invalid")
        if tuple(row.get("k_values") or ()) != ks:
            raise RetrievalMetricError("stage_result_k_values_mismatch")
        key = (benchmark_id, stage)
        if key in seen:
            raise RetrievalMetricError("stage_result_duplicate")
        seen.add(key)
        grouped[stage].append(row)
        category_grouped[(category, stage)].append(row)

    def summarize(stage_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            namespace: _aggregate_namespace(
                [row["namespaces"][namespace] for row in stage_rows], k_values=ks
            )
            for namespace in _NAMESPACES
        }

    return {
        "schema_version": STAGE_RETRIEVAL_METRICS_VERSION,
        "k_values": list(ks),
        "stages": {stage: summarize(grouped[stage]) for stage in sorted(grouped)},
        "categories": {
            category: {
                stage: summarize(category_grouped[(category, stage)])
                for candidate_category, stage in sorted(category_grouped)
                if candidate_category == category
            }
            for category in sorted({category for category, _stage in category_grouped})
        },
    }


def diagnose_stage_candidate_loss(
    stage_results: Sequence[Mapping[str, Any]], *, k: int
) -> dict[str, Any]:
    """Classify only directly observable candidate loss, never embedding quality.

    Dense recall attribution needs compatible alternatives plus Oracle repair and
    therefore remains outside this stage-only signal. Invalid candidate identity
    blocks attribution before any quality label is emitted.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise RetrievalMetricError("diagnostic_k_invalid")
    by_stage: dict[str, Mapping[str, Any]] = {}
    for result in stage_results:
        stage = result.get("stage")
        if stage not in _STAGES:
            raise RetrievalMetricError("retrieval_stage_invalid")
        if stage in by_stage:
            raise RetrievalMetricError("stage_result_duplicate")
        if result.get("schema_version") != STAGE_RETRIEVAL_METRICS_VERSION:
            raise RetrievalMetricError("stage_result_schema_invalid")
        by_stage[str(stage)] = result

    invalid_stages = sorted(
        stage
        for stage, result in by_stage.items()
        if any(
            namespace.get("status") == "invalid_provenance"
            for namespace in result.get("namespaces", {}).values()
            if isinstance(namespace, Mapping)
        )
    )
    if invalid_stages:
        return {
            "decision": "blocked",
            "reason_code": "invalid_candidate_provenance",
            "supporting_stages": invalid_stages,
        }

    def recall(stage: str, namespace: str) -> float | None:
        result = by_stage.get(stage)
        if result is None:
            return None
        values = result.get("namespaces")
        metric = values.get(namespace) if isinstance(values, Mapping) else None
        if not isinstance(metric, Mapping) or metric.get("status") != "scored":
            return None
        recalls = metric.get("recall_at_k")
        value = recalls.get(str(k)) if isinstance(recalls, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    dense_exact = recall("dense", "exact_evidence")
    fused_exact = recall("fused", "exact_evidence")
    reranked_exact = recall("reranked", "exact_evidence")
    if dense_exact is not None and dense_exact > 0 and fused_exact == 0:
        return {
            "decision": "fusion_or_candidate_selection",
            "reason_code": "dense_exact_hit_lost_after_fusion",
            "supporting_stages": ["dense", "fused"],
        }
    if (
        dense_exact is not None
        and dense_exact > 0
        and fused_exact is not None
        and fused_exact > 0
        and reranked_exact == 0
    ):
        return {
            "decision": "fusion_or_candidate_selection",
            "reason_code": "exact_hit_lost_after_reranking",
            "supporting_stages": ["dense", "fused", "reranked"],
        }

    dense_source = recall("dense", "source_document")
    if dense_source is not None and dense_source > 0 and dense_exact == 0:
        return {
            "decision": "inconclusive",
            "reason_code": "source_hit_exact_chunk_miss",
            "supporting_stages": ["dense"],
        }
    return {
        "decision": "inconclusive",
        "reason_code": "paired_oracle_evidence_required",
        "supporting_stages": sorted(by_stage),
    }


__all__ = [
    "STAGE_RETRIEVAL_METRICS_VERSION",
    "ExactEvidenceIdentity",
    "RetrievalMetricError",
    "ReviewedRelevance",
    "aggregate_stage_retrieval",
    "diagnose_stage_candidate_loss",
    "score_stage_retrieval",
]
