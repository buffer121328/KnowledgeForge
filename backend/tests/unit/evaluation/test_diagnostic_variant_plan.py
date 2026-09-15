"""Acceptance tests for frozen, single-variable diagnostic variant plans."""

from __future__ import annotations

from dataclasses import replace

import pytest

from evaluation.diagnostic.variants import (
    DIAGNOSTIC_VARIANT_PLAN_VERSION,
    REQUIRED_DIAGNOSTIC_VARIANTS,
    DiagnosticVariantPlanError,
    FrozenVariantPlan,
    assert_pairable_variants,
)


def _identity(**overrides: object) -> dict[str, object]:
    identity: dict[str, object] = {
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
    identity.update(overrides)
    return identity


def test_frozen_plan_declares_oracle_chain_and_all_retrieval_ablations() -> None:
    plan = FrozenVariantPlan.create(_identity())

    assert plan.schema_version == DIAGNOSTIC_VARIANT_PLAN_VERSION
    assert tuple(item.variant_id for item in plan.variants) == REQUIRED_DIAGNOSTIC_VARIANTS
    assert len(plan.plan_sha256) == 64
    assert plan.variant("observed").resolved_identity["context_source"] == "observed"
    assert plan.variant("oracle_context").resolved_identity["context_source"] == "reviewed"
    assert plan.variant("oracle_route").resolved_identity["route_source"] == "reviewed"
    assert {
        plan.variant(name).resolved_identity["retrieval_pipeline"]
        for name in ("dense_only", "bm25_only", "graph_only", "fused", "reranked")
    } == {"dense", "bm25", "graph", "fused", "reranked"}


@pytest.mark.parametrize(
    ("left", "right", "factor"),
    [
        ("observed", "oracle_context", "context_source"),
        ("oracle_context", "oracle_route", "route_source"),
        ("dense_only", "bm25_only", "retrieval_pipeline"),
        ("bm25_only", "graph_only", "retrieval_pipeline"),
        ("graph_only", "fused", "retrieval_pipeline"),
        ("fused", "reranked", "retrieval_pipeline"),
    ],
)
def test_only_the_declared_variant_factor_may_change(
    left: str, right: str, factor: str
) -> None:
    plan = FrozenVariantPlan.create(_identity())

    comparison = assert_pairable_variants(
        plan.variant(left), plan.variant(right), declared_factor=factor
    )

    assert comparison["paired"] is True
    assert comparison["declared_factor"] == factor
    assert comparison["pairing_identity_sha256"] == plan.pairing_identity_sha256


@pytest.mark.parametrize(
    "field",
    [
        "corpus_sha256",
        "manifest_sha256",
        "authorization_scope_sha256",
        "response_snapshot_sha256",
        "query_rewrite_version",
        "embedding_model",
        "answer_model",
        "candidate_budget_version",
        "candidate_budget",
        "category_policy_version",
    ],
)
def test_pairing_rejects_every_frozen_identity_drift(field: str) -> None:
    first = FrozenVariantPlan.create(_identity())
    changed_value: object = 9 if field == "candidate_budget" else (
        "e" * 64 if field.endswith("sha256") else f"changed-{field}"
    )
    second = FrozenVariantPlan.create(_identity(**{field: changed_value}))

    with pytest.raises(
        DiagnosticVariantPlanError,
        match="variant_pair_identity_incompatible",
    ):
        assert_pairable_variants(
            first.variant("dense_only"),
            second.variant("bm25_only"),
            declared_factor="retrieval_pipeline",
        )


def test_pairing_rejects_an_undeclared_second_factor() -> None:
    plan = FrozenVariantPlan.create(_identity())
    tampered = replace(
        plan.variant("oracle_context"),
        factor_values={
            **plan.variant("oracle_context").factor_values,
            "retrieval_pipeline": "dense",
        },
    )

    with pytest.raises(
        DiagnosticVariantPlanError,
        match="variant_pair_non_target_identity_changed",
    ):
        assert_pairable_variants(
            plan.variant("observed"), tampered, declared_factor="context_source"
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"cache_policy": "enabled"},
        {"candidate_budget": 0},
        {"manifest_sha256": "short"},
    ],
)
def test_plan_rejects_unsafe_or_unbounded_identity(overrides: dict[str, object]) -> None:
    with pytest.raises(DiagnosticVariantPlanError):
        FrozenVariantPlan.create(_identity(**overrides))
