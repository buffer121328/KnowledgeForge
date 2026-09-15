"""Tests for resumable, variant-isolated retrieval ablation execution."""

from __future__ import annotations

import json
from typing import Any

import pytest

from evaluation.diagnostic.ablation import DiagnosticAblationOrchestrator
from evaluation.diagnostic.variants import FrozenVariantPlan


def _plan(*, response: str = "d") -> FrozenVariantPlan:
    return FrozenVariantPlan.create(
        {
            "corpus_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "authorization_scope_sha256": "c" * 64,
            "response_snapshot_sha256": response * 64,
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
async def test_ablation_resumes_successes_and_retries_only_failed_pair(tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    failed_once = False

    async def execute(**kwargs: Any) -> dict[str, Any]:
        nonlocal failed_once
        pair = (kwargs["case_id"], kwargs["variant_id"])
        calls.append(pair)
        assert kwargs["evaluation_only"] is True
        assert kwargs["write_cache"] is False
        if pair == ("case-2", "graph_only") and not failed_once:
            failed_once = True
            raise RuntimeError("private provider payload")
        return {
            "status": "succeeded",
            "artifact_sha256": "e" * 64,
            "response_snapshot_sha256": "f" * 64,
        }

    checkpoint = tmp_path / "ablation.jsonl"
    runner = DiagnosticAblationOrchestrator(execute, checkpoint_path=checkpoint)
    first = await runner.run(_plan(), case_ids=["case-1", "case-2"])
    second = await runner.run(_plan(), case_ids=["case-1", "case-2"])

    assert len(first) == 10
    assert sum(item["status"] == "failed" for item in first) == 1
    assert all(item["status"] == "succeeded" for item in second)
    assert calls.count(("case-1", "dense_only")) == 1
    assert calls.count(("case-2", "graph_only")) == 2
    assert checkpoint.stat().st_mode & 0o077 == 0
    assert "private provider payload" not in checkpoint.read_text()


@pytest.mark.asyncio
async def test_changed_snapshot_cannot_reuse_prior_variant_or_judge_identity(
    tmp_path,
) -> None:
    calls: list[tuple[str, str]] = []

    async def execute(**kwargs: Any) -> dict[str, Any]:
        calls.append((kwargs["case_id"], kwargs["variant_id"]))
        return {
            "status": "succeeded",
            "artifact_sha256": "e" * 64,
            "response_snapshot_sha256": "f" * 64,
        }

    checkpoint = tmp_path / "ablation.jsonl"
    runner = DiagnosticAblationOrchestrator(execute, checkpoint_path=checkpoint)
    await runner.run(_plan(response="d"), case_ids=["case-1"])
    await runner.run(_plan(response="9"), case_ids=["case-1"])

    assert len(calls) == 10
    records = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert len(records) == 5
    assert {item["plan_sha256"] for item in records} == {
        _plan(response="9").plan_sha256
    }


@pytest.mark.asyncio
async def test_each_retrieval_variant_has_an_independent_checkpoint_key(tmp_path) -> None:
    async def execute(**kwargs: Any) -> dict[str, Any]:
        hashes = {
            "dense_only": "1",
            "bm25_only": "2",
            "graph_only": "3",
            "fused": "4",
            "reranked": "5",
        }
        return {
            "status": "succeeded",
            "artifact_sha256": hashes[kwargs["variant_id"]] * 64,
            "response_snapshot_sha256": "f" * 64,
        }

    outcomes = await DiagnosticAblationOrchestrator(
        execute, checkpoint_path=tmp_path / "ablation.jsonl"
    ).run(_plan(), case_ids=["case-1"])

    assert len(outcomes) == 5
    assert len({(item["case_id"], item["variant_id"]) for item in outcomes}) == 5
    assert {item["retrieval_pipeline"] for item in outcomes} == {
        "dense",
        "bm25",
        "graph",
        "fused",
        "reranked",
    }
