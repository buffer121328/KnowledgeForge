"""Ordered paired-evidence tests for bounded diagnostic attribution."""

from __future__ import annotations

from evaluation.diagnostic.root_cause import attribute_root_cause
from evaluation.diagnostic.variants import FrozenVariantPlan


def _plan() -> FrozenVariantPlan:
    return FrozenVariantPlan.create(
        {
            "corpus_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "authorization_scope_sha256": "c" * 64,
            "response_snapshot_sha256": "d" * 64,
            "query_rewrite_version": "query-v1",
            "embedding_model": "embedding-v1",
            "answer_model": "answer-v1",
            "candidate_budget_version": "budget-v1",
            "candidate_budget": 8,
            "category_policy_version": "policy-v1",
            "tokenizer_version": "tokenizer-v1",
            "index_version": "index-v1",
            "threshold_version": "threshold-v1",
            "prompt_version": "prompt-v1",
            "answer_schema_version": "answer-schema-v1",
            "grounding_policy_version": "grounding-v1",
            "cache_policy": "disabled",
        }
    )


def _scores(**overrides: list[float]) -> dict[str, list[float]]:
    values = {
        variant: [0.5] * 30
        for variant in (
            "observed",
            "oracle_context",
            "oracle_route",
            "dense_only",
            "bm25_only",
            "graph_only",
            "fused",
            "reranked",
        )
    }
    values.update(overrides)
    return values


def _recall(**overrides: list[float]) -> dict[str, list[float]]:
    values = {
        "dense_only": [1.0] * 30,
        "bm25_only": [1.0] * 30,
        "graph_only": [1.0] * 30,
    }
    values.update(overrides)
    return values


def test_hard_blocker_precedes_every_quality_attribution() -> None:
    result = attribute_root_cause(
        plan=_plan(),
        variant_scores=_scores(oracle_context=[1.0] * 30),
        retrieval_recall=_recall(dense_only=[0.0] * 30),
        stage_exact_hits={"dense": False, "bm25": True, "fused": True, "reranked": True},
        blockers=["authorization_leakage"],
    )

    assert result["decision"] == "blocked"
    assert result["reason_code"] == "hard_gate_blocked"
    assert result["recommend_embedding_change"] is False


def test_missing_required_pair_is_inconclusive() -> None:
    scores = _scores()
    scores.pop("oracle_route")

    result = attribute_root_cause(
        plan=_plan(),
        variant_scores=scores,
        retrieval_recall=_recall(),
        stage_exact_hits={},
    )

    assert result["decision"] == "inconclusive"
    assert result["reason_code"] == "required_paired_variant_missing"
    assert result["recommend_embedding_change"] is False


def test_branch_hit_lost_after_fusion_wins_before_embedding_rule() -> None:
    result = attribute_root_cause(
        plan=_plan(),
        variant_scores=_scores(oracle_context=[1.0] * 30),
        retrieval_recall=_recall(dense_only=[0.0] * 30),
        stage_exact_hits={"dense": True, "bm25": False, "graph": False, "fused": False, "reranked": False},
    )

    assert result["decision"] == "fusion_or_candidate_selection"
    assert result["reason_code"] == "branch_exact_hit_lost_after_fusion"
    assert result["recommend_embedding_change"] is False


def test_dense_recall_requires_alternative_recall_and_oracle_repair() -> None:
    result = attribute_root_cause(
        plan=_plan(),
        variant_scores=_scores(
            observed=[0.0] * 30,
            oracle_context=[1.0] * 30,
            oracle_route=[1.0] * 30,
        ),
        retrieval_recall=_recall(
            dense_only=[0.0] * 30,
            bm25_only=[1.0] * 30,
            graph_only=[0.0] * 30,
        ),
        stage_exact_hits={"dense": False, "bm25": True, "graph": False, "fused": True, "reranked": True},
    )

    assert result["decision"] == "dense_recall"
    assert result["reason_code"] == "alternative_recall_and_oracle_context_repair"
    assert result["recommend_embedding_change"] is True
    assert result["paired_count"] == 30
    assert result["pairing_identity_sha256"] == _plan().pairing_identity_sha256


def test_oracle_route_repair_is_qualification_or_routing() -> None:
    result = attribute_root_cause(
        plan=_plan(),
        variant_scores=_scores(
            observed=[0.0] * 30,
            oracle_context=[0.0] * 30,
            oracle_route=[1.0] * 30,
        ),
        retrieval_recall=_recall(),
        stage_exact_hits={"dense": True, "fused": True, "reranked": True},
    )

    assert result["decision"] == "qualification_or_routing"
    assert result["recommend_embedding_change"] is False


def test_low_answer_score_with_reviewed_context_and_route_is_generation() -> None:
    result = attribute_root_cause(
        plan=_plan(),
        variant_scores=_scores(
            observed=[0.2] * 30,
            oracle_context=[0.2] * 30,
            oracle_route=[0.2] * 30,
        ),
        retrieval_recall=_recall(),
        stage_exact_hits={"dense": True, "fused": True, "reranked": True},
    )

    assert result["decision"] == "generation"
    assert result["reason_code"] == "oracle_route_answer_below_threshold"
    assert result["recommend_embedding_change"] is False


def test_weak_paired_evidence_stays_inconclusive() -> None:
    result = attribute_root_cause(
        plan=_plan(),
        variant_scores=_scores(
            observed=[0.50] * 30,
            oracle_context=[0.55] * 30,
            oracle_route=[0.56] * 30,
        ),
        retrieval_recall=_recall(
            dense_only=[0.50] * 30,
            bm25_only=[0.55] * 30,
        ),
        stage_exact_hits={"dense": False, "bm25": False, "graph": False, "fused": False, "reranked": False},
    )

    assert result["decision"] == "inconclusive"
    assert result["reason_code"] == "paired_effect_inconclusive"
    assert result["recommend_embedding_change"] is False
