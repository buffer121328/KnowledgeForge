"""Ordered, evidence-bound root-cause attribution for paired diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.diagnostic.preregistration import (
    DIAGNOSTIC_PREREGISTRATION_VERSION,
    classify_paired_effect,
)
from evaluation.diagnostic.variants import (
    REQUIRED_DIAGNOSTIC_VARIANTS,
    FrozenVariantPlan,
    assert_pairable_variants,
)

GENERATION_ACCEPTANCE_THRESHOLD = 0.5


class RootCauseAttributionError(ValueError):
    """Paired diagnostic evidence is malformed or unsafe to compare."""


def _scores(values: Sequence[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise RootCauseAttributionError("root_cause_score_invalid")
        result.append(float(value))
    if not result:
        raise RootCauseAttributionError("root_cause_score_vector_empty")
    return result


def _base_result(plan: FrozenVariantPlan) -> dict[str, Any]:
    return {
        "schema_version": DIAGNOSTIC_PREREGISTRATION_VERSION,
        "variant_plan_sha256": plan.plan_sha256,
        "pairing_identity_sha256": plan.pairing_identity_sha256,
        "recommend_embedding_change": False,
    }


def _effect(
    scores: Mapping[str, list[float]], left: str, right: str
) -> dict[str, Any]:
    if len(scores[left]) != len(scores[right]):
        raise RootCauseAttributionError("root_cause_pair_count_mismatch")
    return classify_paired_effect(scores[left], scores[right])


def attribute_root_cause(
    *,
    plan: FrozenVariantPlan,
    variant_scores: Mapping[str, Sequence[float]],
    retrieval_recall: Mapping[str, Sequence[float]],
    stage_exact_hits: Mapping[str, bool],
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    """Apply pre-registered stage attribution rules in fail-closed order."""

    base = _base_result(plan)
    bounded_blockers = sorted(
        {
            blocker
            for blocker in blockers
            if isinstance(blocker, str) and 0 < len(blocker) <= 96
        }
    )
    if bounded_blockers:
        return {
            **base,
            "decision": "blocked",
            "reason_code": "hard_gate_blocked",
            "blockers": bounded_blockers[:20],
        }

    missing = [
        variant
        for variant in REQUIRED_DIAGNOSTIC_VARIANTS
        if variant not in variant_scores
    ]
    recall_variants = ("dense_only", "bm25_only", "graph_only")
    missing.extend(
        variant for variant in recall_variants if variant not in retrieval_recall
    )
    if missing:
        return {
            **base,
            "decision": "inconclusive",
            "reason_code": "required_paired_variant_missing",
            "missing_variants": sorted(set(missing)),
        }

    # Prove the compared entries are single-variable members of this exact plan.
    assert_pairable_variants(
        plan.variant("observed"),
        plan.variant("oracle_context"),
        declared_factor="context_source",
    )
    assert_pairable_variants(
        plan.variant("oracle_context"),
        plan.variant("oracle_route"),
        declared_factor="route_source",
    )
    assert_pairable_variants(
        plan.variant("dense_only"),
        plan.variant("bm25_only"),
        declared_factor="retrieval_pipeline",
    )
    assert_pairable_variants(
        plan.variant("dense_only"),
        plan.variant("graph_only"),
        declared_factor="retrieval_pipeline",
    )

    scores = {variant: _scores(variant_scores[variant]) for variant in REQUIRED_DIAGNOSTIC_VARIANTS}
    recalls = {variant: _scores(retrieval_recall[variant]) for variant in recall_variants}
    paired_count = len(scores["observed"])
    if any(len(values) != paired_count for values in scores.values()):
        raise RootCauseAttributionError("root_cause_pair_count_mismatch")
    if any(len(values) != paired_count for values in recalls.values()):
        raise RootCauseAttributionError("root_cause_pair_count_mismatch")

    hits = {
        stage: value
        for stage, value in stage_exact_hits.items()
        if stage in {"dense", "bm25", "graph", "fused", "reranked"}
        and isinstance(value, bool)
    }
    branch_hit = any(hits.get(stage) is True for stage in ("dense", "bm25", "graph"))
    if branch_hit and hits.get("fused") is False:
        return {
            **base,
            "decision": "fusion_or_candidate_selection",
            "reason_code": "branch_exact_hit_lost_after_fusion",
            "paired_count": paired_count,
            "compared_variants": ["dense_only", "bm25_only", "graph_only", "fused"],
        }
    if hits.get("fused") is True and hits.get("reranked") is False:
        return {
            **base,
            "decision": "fusion_or_candidate_selection",
            "reason_code": "exact_hit_lost_after_reranking",
            "paired_count": paired_count,
            "compared_variants": ["fused", "reranked"],
        }

    alternative = max(
        ("bm25_only", "graph_only"),
        key=lambda variant: sum(recalls[variant]) / len(recalls[variant]),
    )
    alternative_effect = _effect(
        {**scores, **recalls}, "dense_only", alternative
    )
    oracle_context_effect = _effect(scores, "observed", "oracle_context")
    if (
        alternative_effect["decision"] == "material_improvement"
        and oracle_context_effect["decision"] == "material_improvement"
    ):
        return {
            **base,
            "decision": "dense_recall",
            "reason_code": "alternative_recall_and_oracle_context_repair",
            "recommend_embedding_change": True,
            "paired_count": paired_count,
            "compared_variants": ["dense_only", alternative, "observed", "oracle_context"],
            "effects": {
                "alternative_recall": alternative_effect,
                "oracle_context": oracle_context_effect,
            },
        }

    oracle_route_effect = _effect(scores, "oracle_context", "oracle_route")
    if oracle_route_effect["decision"] == "material_improvement":
        return {
            **base,
            "decision": "qualification_or_routing",
            "reason_code": "oracle_route_material_repair",
            "paired_count": paired_count,
            "compared_variants": ["oracle_context", "oracle_route"],
            "effects": {"oracle_route": oracle_route_effect},
        }

    oracle_route_mean = sum(scores["oracle_route"]) / paired_count
    if oracle_route_mean < GENERATION_ACCEPTANCE_THRESHOLD:
        return {
            **base,
            "decision": "generation",
            "reason_code": "oracle_route_answer_below_threshold",
            "paired_count": paired_count,
            "compared_variants": ["oracle_context", "oracle_route"],
            "oracle_route_mean": round(oracle_route_mean, 6),
            "generation_acceptance_threshold": GENERATION_ACCEPTANCE_THRESHOLD,
        }

    return {
        **base,
        "decision": "inconclusive",
        "reason_code": "paired_effect_inconclusive",
        "paired_count": paired_count,
        "compared_variants": list(REQUIRED_DIAGNOSTIC_VARIANTS),
        "effects": {
            "alternative_recall": alternative_effect,
            "oracle_context": oracle_context_effect,
            "oracle_route": oracle_route_effect,
        },
    }


__all__ = [
    "GENERATION_ACCEPTANCE_THRESHOLD",
    "RootCauseAttributionError",
    "attribute_root_cause",
]
