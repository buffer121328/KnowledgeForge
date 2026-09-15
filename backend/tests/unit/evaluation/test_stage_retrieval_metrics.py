"""Tests for deterministic, namespace-safe retrieval stage diagnostics."""

from __future__ import annotations

import math

import pytest

from evaluation.benchmarks.stage_retrieval_metrics import (
    ExactEvidenceIdentity,
    RetrievalMetricError,
    ReviewedRelevance,
    aggregate_stage_retrieval,
    diagnose_stage_candidate_loss,
    score_stage_retrieval,
)


RELEVANCE = ReviewedRelevance(
    source_document_ids=frozenset({"doc-a"}),
    exact_evidence=(ExactEvidenceIdentity("chunk-a", "a" * 64),),
)


def _stage(*candidates: dict[str, object], status: str = "executed") -> dict[str, object]:
    return {
        "stage": "dense",
        "status": status,
        "candidate_budget": 3,
        "candidates": list(candidates),
    }


def test_same_document_wrong_chunk_is_only_a_source_hit() -> None:
    result = score_stage_retrieval(
        _stage(
            {
                "rank": 1,
                "source_document_id": "doc-a",
                "context_id": "chunk-wrong",
                "content_sha256": "f" * 64,
                "score": 999.0,
            }
        ),
        relevance=RELEVANCE,
        k_values=(1, 3),
    )

    assert result["namespaces"]["source_document"]["recall_at_k"] == {
        "1": 1.0,
        "3": 1.0,
    }
    assert result["namespaces"]["exact_evidence"]["recall_at_k"] == {
        "1": 0.0,
        "3": 0.0,
    }
    assert result["namespaces"]["exact_evidence"]["first_relevant_rank"] is None


def test_multiple_relevant_chunks_use_rank_only_for_recall_mrr_and_binary_ndcg() -> None:
    relevance = ReviewedRelevance(
        source_document_ids=frozenset({"doc-a", "doc-b"}),
        exact_evidence=(
            ExactEvidenceIdentity("chunk-a", "a" * 64),
            ExactEvidenceIdentity("chunk-b", "b" * 64),
        ),
    )
    result = score_stage_retrieval(
        _stage(
            {"rank": 1, "source_document_id": "doc-a", "context_id": "chunk-a"},
            {"rank": 2, "source_document_id": "doc-x", "context_id": "chunk-x"},
            {"rank": 3, "source_document_id": "doc-b", "context_id": "chunk-b"},
        ),
        relevance=relevance,
        k_values=(1, 3),
    )
    exact = result["namespaces"]["exact_evidence"]

    assert exact["recall_at_k"] == {"1": 0.5, "3": 1.0}
    assert exact["first_relevant_rank"] == 1
    assert exact["mrr"] == 1.0
    expected_dcg = 1.0 + 1.0 / math.log2(4)
    ideal_dcg = 1.0 + 1.0 / math.log2(3)
    assert exact["ndcg_at_k"]["3"] == pytest.approx(expected_dcg / ideal_dcg)


def test_content_hash_can_prove_exact_identity_without_matching_context_id() -> None:
    result = score_stage_retrieval(
        _stage(
            {
                "rank": 1,
                "source_document_id": "doc-a",
                "context_id": "runtime-renumbered",
                "content_sha256": "a" * 64,
            }
        ),
        relevance=RELEVANCE,
        k_values=(1,),
    )

    assert result["namespaces"]["exact_evidence"]["recall_at_k"] == {"1": 1.0}


def test_executed_empty_and_not_executed_are_distinct() -> None:
    empty = score_stage_retrieval(
        _stage(status="executed_empty"), relevance=RELEVANCE, k_values=(1, 3)
    )
    not_executed = score_stage_retrieval(
        _stage(status="not_executed"), relevance=RELEVANCE, k_values=(1, 3)
    )

    assert empty["namespaces"]["exact_evidence"] == {
        "status": "scored",
        "recall_at_k": {"1": 0.0, "3": 0.0},
        "first_relevant_rank": None,
        "mrr": 0.0,
        "ndcg_at_k": {"1": 0.0, "3": 0.0},
    }
    assert not_executed["namespaces"]["exact_evidence"] == {
        "status": "not_executed",
        "reason_code": "stage_not_executed",
    }


def test_k_over_budget_and_invalid_provenance_fail_closed() -> None:
    with pytest.raises(RetrievalMetricError, match="k_exceeds_candidate_budget"):
        score_stage_retrieval(_stage(), relevance=RELEVANCE, k_values=(1, 5))

    result = score_stage_retrieval(
        _stage({"rank": 1}), relevance=RELEVANCE, k_values=(1,)
    )
    assert result["namespaces"]["source_document"] == {
        "status": "invalid_provenance",
        "reason_code": "candidate_source_document_identity_missing",
    }
    assert result["namespaces"]["exact_evidence"] == {
        "status": "invalid_provenance",
        "reason_code": "candidate_exact_identity_missing",
    }


def test_aggregation_keeps_stage_and_namespace_boundaries() -> None:
    dense_hit = score_stage_retrieval(
        _stage(
            {"rank": 1, "source_document_id": "doc-a", "context_id": "chunk-a"}
        ),
        relevance=RELEVANCE,
        k_values=(1,),
    )
    fused_miss = score_stage_retrieval(
        {
            **_stage(
                {"rank": 1, "source_document_id": "doc-a", "context_id": "wrong"}
            ),
            "stage": "fused",
        },
        relevance=RELEVANCE,
        k_values=(1,),
    )
    graph_absent = score_stage_retrieval(
        {**_stage(status="not_executed"), "stage": "graph"},
        relevance=RELEVANCE,
        k_values=(1,),
    )

    summary = aggregate_stage_retrieval(
        [
            {"benchmark_id": "case-1", "category": "fully_answerable", **dense_hit},
            {"benchmark_id": "case-1", "category": "fully_answerable", **fused_miss},
            {"benchmark_id": "case-1", "category": "fully_answerable", **graph_absent},
        ],
        k_values=(1,),
    )

    assert summary["stages"]["dense"]["exact_evidence"]["mean_recall_at_k"] == {
        "1": 1.0
    }
    assert summary["stages"]["fused"]["source_document"]["mean_recall_at_k"] == {
        "1": 1.0
    }
    assert summary["stages"]["fused"]["exact_evidence"]["mean_recall_at_k"] == {
        "1": 0.0
    }
    assert summary["stages"]["graph"]["exact_evidence"]["status_counts"] == {
        "not_executed": 1
    }


def test_dense_hit_lost_by_fusion_is_not_diagnosed_as_embedding_failure() -> None:
    dense_hit = score_stage_retrieval(
        _stage(
            {"rank": 1, "source_document_id": "doc-a", "context_id": "chunk-a"}
        ),
        relevance=RELEVANCE,
        k_values=(1,),
    )
    fused_miss = score_stage_retrieval(
        {
            **_stage(
                {"rank": 1, "source_document_id": "doc-a", "context_id": "wrong"}
            ),
            "stage": "fused",
        },
        relevance=RELEVANCE,
        k_values=(1,),
    )

    diagnosis = diagnose_stage_candidate_loss([dense_hit, fused_miss], k=1)

    assert diagnosis == {
        "decision": "fusion_or_candidate_selection",
        "reason_code": "dense_exact_hit_lost_after_fusion",
        "supporting_stages": ["dense", "fused"],
    }


def test_source_hit_without_exact_chunk_stays_inconclusive() -> None:
    wrong_chunk = score_stage_retrieval(
        _stage(
            {
                "rank": 1,
                "source_document_id": "doc-a",
                "context_id": "chunk-wrong",
            }
        ),
        relevance=RELEVANCE,
        k_values=(1,),
    )

    diagnosis = diagnose_stage_candidate_loss([wrong_chunk], k=1)

    assert diagnosis["decision"] == "inconclusive"
    assert diagnosis["reason_code"] == "source_hit_exact_chunk_miss"
    assert diagnosis["decision"] != "dense_recall"


def test_missing_candidate_provenance_blocks_root_cause_attribution() -> None:
    invalid = score_stage_retrieval(
        _stage({"rank": 1}), relevance=RELEVANCE, k_values=(1,)
    )

    diagnosis = diagnose_stage_candidate_loss([invalid], k=1)

    assert diagnosis == {
        "decision": "blocked",
        "reason_code": "invalid_candidate_provenance",
        "supporting_stages": ["dense"],
    }
