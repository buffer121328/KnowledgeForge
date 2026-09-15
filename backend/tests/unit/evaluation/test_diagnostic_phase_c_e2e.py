"""No-network acceptance of the complete Phase C isolation sequence."""

from __future__ import annotations

from typing import Any

import pytest

from evaluation.diagnostic.ablation import DiagnosticAblationOrchestrator
from evaluation.diagnostic.oracle import (
    DiagnosticOracleExecutor,
    ReviewedOracleContext,
    prepare_oracle_input,
)
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


@pytest.mark.asyncio
async def test_mocked_oracle_chain_and_five_ablations_are_isolated(tmp_path) -> None:
    plan = _plan()
    reviewed_context = ReviewedOracleContext(
        context_id="context-1",
        source_document_id="document-1",
        content_sha256="e" * 64,
        manifest_sha256="b" * 64,
        authorization_scope_sha256="c" * 64,
        content="reviewed private content",
        reviewed=True,
    )
    oracle_calls: list[str] = []

    async def oracle_callback(**kwargs: Any) -> dict[str, Any]:
        oracle_calls.append(kwargs["variant_id"])
        return {
            "status": "succeeded",
            "contract_score": 1.0 if kwargs["reviewed_route"] else 0.0,
        }

    executor = DiagnosticOracleExecutor(oracle_callback)
    oracle_context = prepare_oracle_input(
        plan=plan,
        variant_id="oracle_context",
        case_id="case-1",
        contexts=[reviewed_context],
        reviewed_route=None,
        runtime_manifest_sha256="b" * 64,
        runtime_authorization_scope_sha256="c" * 64,
    )
    oracle_route = prepare_oracle_input(
        plan=plan,
        variant_id="oracle_route",
        case_id="case-1",
        contexts=[reviewed_context],
        reviewed_route={
            "response_status": "answered",
            "evidence_states": ["direct_evidence"],
        },
        runtime_manifest_sha256="b" * 64,
        runtime_authorization_scope_sha256="c" * 64,
    )
    observed = {"status": "succeeded", "contract_score": 0.0}
    context_result = await executor.execute(oracle_context)
    route_result = await executor.execute(oracle_route)

    async def ablation_callback(**kwargs: Any) -> dict[str, Any]:
        index = ("dense_only", "bm25_only", "graph_only", "fused", "reranked").index(
            kwargs["variant_id"]
        )
        return {
            "status": "succeeded",
            "artifact_sha256": f"{index + 1:x}" * 64,
            "response_snapshot_sha256": "f" * 64,
        }

    ablations = await DiagnosticAblationOrchestrator(
        ablation_callback,
        checkpoint_path=tmp_path / "ablation.jsonl",
    ).run(plan, case_ids=["case-1"])

    assert [observed["contract_score"], context_result["contract_score"], route_result["contract_score"]] == [0.0, 0.0, 1.0]
    assert oracle_calls == ["oracle_context", "oracle_route"]
    assert {item["variant_id"] for item in ablations} == {
        "dense_only",
        "bm25_only",
        "graph_only",
        "fused",
        "reranked",
    }

    scores = {
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
    scores.update(
        {
            "observed": [0.0] * 30,
            "oracle_context": [0.0] * 30,
            "oracle_route": [1.0] * 30,
        }
    )
    diagnosis = attribute_root_cause(
        plan=plan,
        variant_scores=scores,
        retrieval_recall={
            "dense_only": [1.0] * 30,
            "bm25_only": [1.0] * 30,
            "graph_only": [1.0] * 30,
        },
        stage_exact_hits={"dense": True, "fused": True, "reranked": True},
    )
    assert diagnosis["decision"] == "qualification_or_routing"
    assert diagnosis["recommend_embedding_change"] is False
