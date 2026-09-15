"""Offline end-to-end smoke test for the evaluation toolchain.

Runs the full offline chain -- benchmark loader -> BenchmarkRunner -> quality
report -- with a fake QA executor so it never contacts an LLM, vector store,
graph database, or external judge. This is the RAG end-to-end smoke gate from
the evaluation plan (M4): results must be reproducible and failures must not be
silent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.knowledge import QAResult, QueryIntent, RetrievedContext
from evaluation.benchmarks.benchmark import load_benchmark
from evaluation.release.quality_gates import aggregate_quality_report, evaluate_release_gate
from evaluation.runner import BenchmarkRunner

BENCHMARK = Path(__file__).parents[3] / "evaluation" / "data" / "company-demo" / "benchmark.jsonl"


async def _executor(*, question: str, tenant_id: str | None = None, **_kwargs):
    """Return a deterministic fake QA result that never touches external services."""
    return QAResult(
        question=question,
        answer="这是基于检索证据生成的回答。",
        contexts=[
            RetrievedContext(
                content="制度条款的完整检索文本。",
                source="company-demo",
                score=0.9,
                retrieval_type="vector",
                metadata={"source_document_id": "5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9"},
            )
        ],
        intent=QueryIntent.FACTOID,
        confidence=0.5,
    )


@pytest.mark.asyncio
async def test_end_to_end_smoke_chain_produces_gate_blocked_report(
    tmp_path: Path,
) -> None:
    """Runner artifacts feed the quality report which stays smoke-only."""
    benchmark = load_benchmark(BENCHMARK)
    assert len(benchmark) == 80

    run = await BenchmarkRunner(
        qa_executor=_executor,
        results_root=tmp_path,
    ).run(
        benchmark[:2],
        tenant_id="eval-smoke",
        retrieval_modes=("vector",),
        run_metadata={"purpose": "offline end-to-end smoke"},
    )

    responses = [json.loads(line) for line in run.responses_path.read_text(encoding="utf-8").splitlines()]
    assert len(responses) == 2
    assert {record["retrieval_mode"] for record in responses} == {"vector"}
    assert all(record["status"] == "succeeded" for record in responses)
    assert all(record["contexts"][0]["metadata"]["source_document_id"] for record in responses)

    metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    report = aggregate_quality_report(metadata, responses, smoke_threshold=30)
    assert report["run_classification"] == "smoke_only"
    assert report["counts"]["completed"] == 2

    decision = evaluate_release_gate(report)
    assert decision is not None and decision["decision"] == "blocked"


@pytest.mark.asyncio
async def test_end_to_end_smoke_records_failed_mode_without_masking(
    tmp_path: Path,
) -> None:
    """A failing retrieval mode is recorded explicitly, not silently replaced."""
    benchmark = load_benchmark(BENCHMARK)

    async def failing_executor(**_kwargs):
        raise RuntimeError("retrieval backend unavailable")

    run = await BenchmarkRunner(
        qa_executor=failing_executor,
        results_root=tmp_path,
    ).run(
        benchmark[:1],
        tenant_id="eval-smoke",
        retrieval_modes=("vector", "hybrid"),
    )

    responses = [json.loads(line) for line in run.responses_path.read_text(encoding="utf-8").splitlines()]
    assert len(responses) == 2
    assert all(record["status"] == "failed" for record in responses)
    assert all(record["exception"] == "RuntimeError" for record in responses)
