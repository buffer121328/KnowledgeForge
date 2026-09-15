"""Acceptance tests for the offline RAG benchmark runner.

These tests deliberately use fake QA and evaluator collaborators so they never
contact an LLM, vector database, graph database, or external benchmark service.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import types

import httpx
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, create_model
from domain.knowledge import QAResult, QueryIntent, RetrievedContext
from evaluation.ragas import adapter
from evaluation.benchmarks.benchmark import BenchmarkValidationError, load_benchmark
from evaluation.ragas.adapter import (
    build_openai_context_evaluators,
    build_openai_factual_correctness_evaluator,
    build_openai_faithfulness_evaluator,
    evaluate_context_quality,
    evaluate_evidence_gate_contract,
    evaluate_factual_correctness,
    evaluate_faithfulness,
)
from evaluation.runner import BenchmarkRunner

DOC_A = "7560dc54-81cb-4d28-bbb1-205f060ad678"
DOC_B = "08bb9178-7daa-40d3-b8df-601920a6df2c"


def _record(*, benchmark_id: str = "muda-case-001", **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": benchmark_id,
        "question": "请说明资本充足率。",
        "reference": "人工审核的参考答案。",
        "required_doc_ids": [DOC_A],
        "reference_context_ids": [DOC_A],
        "evidence": [
            {
                "source_document_id": DOC_A,
                "page": 10,
                "section": "资本管理",
                "claim": "资本充足率为人工审核值。",
            }
        ],
        "category": "single_document_fact",
        "expected_refusal": False,
        "notes": "fixture",
    }
    record.update(overrides)
    return record


def _write_jsonl(path: Path, *records: dict[str, Any]) -> Path:
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    return path


class TestBenchmarkLoader:
    def test_loads_pdf_grounded_record(self, tmp_path: Path):
        samples = load_benchmark(_write_jsonl(tmp_path / "benchmark.jsonl", _record()))

        assert len(samples) == 1
        assert samples[0].id == "muda-case-001"
        assert samples[0].reference_context_ids == (DOC_A,)
        assert samples[0].evidence[0].page == 10

    @pytest.mark.parametrize(
        "record, expected",
        [
            (_record(reference=" "), "reference"),
            (_record(category="unsupported"), "category"),
            (_record(reference_context_ids=[]), "reference_context_ids"),
            (_record(evidence=[]), "evidence"),
            (_record(evidence=[{**_record()["evidence"][0], "source_document_id": DOC_B}]), "required_doc_ids"),
        ],
    )
    def test_rejects_incomplete_or_unrelated_gold(self, tmp_path: Path, record: dict[str, Any], expected: str):
        with pytest.raises(BenchmarkValidationError, match=expected):
            load_benchmark(_write_jsonl(tmp_path / "benchmark.jsonl", record))

    def test_rejects_duplicate_ids_with_line_number(self, tmp_path: Path):
        path = _write_jsonl(tmp_path / "benchmark.jsonl", _record(), _record())

        with pytest.raises(BenchmarkValidationError, match=r"line 2.*duplicate"):
            load_benchmark(path)

    def test_rejects_malformed_json_with_line_number(self, tmp_path: Path):
        path = tmp_path / "benchmark.jsonl"
        path.write_text("{not-json}\n", encoding="utf-8")

        with pytest.raises(BenchmarkValidationError, match=r"line 1.*JSON"):
            load_benchmark(path)


class TestBenchmarkRunner:
    @pytest.mark.asyncio
    async def test_writes_untruncated_contexts_for_each_mode(self, tmp_path: Path):
        samples = load_benchmark(_write_jsonl(tmp_path / "benchmark.jsonl", _record()))
        long_context = "原始 PDF 完整上下文：" + "证据" * 300
        calls: list[dict[str, Any]] = []

        async def qa_executor(**kwargs: Any) -> QAResult:
            calls.append(kwargs)
            return QAResult(
                question=kwargs["question"],
                answer="回答受上下文支持。",
                contexts=[
                    RetrievedContext(
                        content=long_context,
                        source="muda.pdf",
                        score=0.91,
                        retrieval_type=kwargs["retrieval_mode"],
                        metadata={"source_document_id": DOC_A, "page": 10},
                    )
                ],
                intent=QueryIntent.FACTOID,
                confidence=0.91,
            )

        run = await BenchmarkRunner(qa_executor=qa_executor, results_root=tmp_path / "results").run(
            samples,
            tenant_id="eval-ragas",
            retrieval_modes=("vector", "hybrid"),
            run_metadata={"evaluator_model": "fake-judge"},
        )

        assert [call["retrieval_mode"] for call in calls] == ["vector", "hybrid"]
        assert all(call["user_id"] == "" for call in calls)
        assert [record["status"] for record in run.records] == ["succeeded", "succeeded"]
        assert run.records[0]["contexts"][0]["content"] == long_context
        assert len(run.records[0]["contexts"][0]["content"]) > 200
        assert run.records[0]["reference"] == "人工审核的参考答案。"
        assert run.records[0]["retrieved_context_ids"] == [DOC_A]
        assert run.records[0]["retrieved_source_document_ids"] == [DOC_A]
        assert run.records[0]["reference_source_document_ids"] == [DOC_A]
        assert run.records[0]["retrieved_chunk_ids"] == []
        assert run.records[0]["id_namespaces"]["retrieved_context_ids"] == "source_document"
        assert run.records[0]["category"] == "single_document_fact"
        assert run.records[0]["expected_refusal"] is False
        assert run.records[0]["refused"] is False
        assert run.metadata_path.exists()
        assert run.responses_path.exists()
        metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
        assert metadata["benchmark_sha256"]
        assert metadata["smoke_threshold"] == 30

    @pytest.mark.asyncio
    async def test_metadata_started_at_is_captured_before_delayed_executor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        samples = load_benchmark(_write_jsonl(tmp_path / "benchmark.jsonl", _record()))
        executor_started = False
        clock_observations: list[bool] = []

        def fake_utc_now() -> str:
            clock_observations.append(executor_started)
            return "before-executor" if not executor_started else "after-executor"

        monkeypatch.setattr("evaluation.runner._utc_now", fake_utc_now)

        async def qa_executor(**kwargs: Any) -> QAResult:
            nonlocal executor_started
            executor_started = True
            await asyncio.sleep(0)
            return QAResult(
                question=kwargs["question"],
                answer="回答受上下文支持。",
                contexts=[],
                intent=QueryIntent.FACTOID,
                confidence=0.0,
            )

        run = await BenchmarkRunner(qa_executor=qa_executor, results_root=tmp_path / "results").run(
            samples,
            tenant_id="eval-ragas",
            retrieval_modes=("vector",),
        )

        metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
        assert metadata["started_at"] == "before-executor"
        assert clock_observations == [False]

    @pytest.mark.asyncio
    async def test_records_failure_and_continues_other_modes(self, tmp_path: Path):
        samples = load_benchmark(_write_jsonl(tmp_path / "benchmark.jsonl", _record()))

        async def qa_executor(**kwargs: Any) -> QAResult:
            if kwargs["retrieval_mode"] == "vector":
                raise RuntimeError("vector unavailable")
            return QAResult(
                question=kwargs["question"],
                answer="hybrid answer",
                contexts=[],
                intent=QueryIntent.FACTOID,
                confidence=0.0,
            )

        run = await BenchmarkRunner(qa_executor=qa_executor, results_root=tmp_path / "results").run(
            samples,
            tenant_id="eval-ragas",
            retrieval_modes=("vector", "hybrid"),
        )

        assert [record["status"] for record in run.records] == ["failed", "succeeded"]
        assert run.records[0]["exception"] == "RuntimeError"
        assert "vector unavailable" not in run.responses_path.read_text(encoding="utf-8")
        assert run.records[1]["response"] == "hybrid answer"

    @pytest.mark.asyncio
    async def test_run_artifacts_are_owner_only_and_metadata_omits_absolute_source_path(self, tmp_path: Path):
        benchmark_path = tmp_path / "private" / "benchmark.jsonl"
        benchmark_path.parent.mkdir()
        samples = load_benchmark(_write_jsonl(benchmark_path, _record()))

        async def qa_executor(**kwargs: Any) -> QAResult:
            return QAResult(
                question=kwargs["question"],
                answer="answer",
                contexts=[],
                intent=QueryIntent.FACTOID,
                confidence=0.0,
            )

        run = await BenchmarkRunner(
            qa_executor=qa_executor,
            results_root=tmp_path / "results",
        ).run(samples, tenant_id="eval-ragas", retrieval_modes=("vector",))

        assert stat.S_IMODE(run.run_dir.stat().st_mode) == 0o700
        for path in (
            run.metadata_path,
            run.responses_path,
            run.invalid_provenance_path,
        ):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
        assert metadata["benchmark_source"] == "benchmark.jsonl"
        assert "benchmark_path" not in metadata
        assert str(benchmark_path) not in run.metadata_path.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_unmapped_context_is_retained_but_marks_invalid_provenance(self, tmp_path: Path):
        samples = load_benchmark(_write_jsonl(tmp_path / "benchmark.jsonl", _record()))

        async def qa_executor(**kwargs: Any) -> QAResult:
            return QAResult(
                question=kwargs["question"],
                answer="带有图谱关系的回答。",
                contexts=[
                    RetrievedContext(
                        content="实体关系文本，不能静默删除。",
                        source="knowledge_graph",
                        score=0.8,
                        retrieval_type="graph",
                        metadata={},
                    )
                ],
                intent=QueryIntent.FACTOID,
                confidence=0.8,
            )

        run = await BenchmarkRunner(qa_executor=qa_executor, results_root=tmp_path / "results").run(
            samples,
            tenant_id="eval-ragas",
            retrieval_modes=("hybrid",),
        )

        assert run.records[0]["status"] == "invalid_provenance"
        assert run.records[0]["contexts"][0]["content"] == "实体关系文本，不能静默删除。"
        assert run.records[0]["retrieved_context_ids"] == []
        invalid = [json.loads(line) for line in run.invalid_provenance_path.read_text(encoding="utf-8").splitlines()]
        assert invalid[0]["benchmark_id"] == "muda-case-001"
        assert invalid[0]["unmapped_contexts"][0]["rank"] == 1


class TestFaithfulnessAdapter:
    @pytest.mark.asyncio
    async def test_evaluates_only_successful_complete_contexts_and_continues_after_failure(self):
        long_context = "完整上下文" * 100
        records = [
            {
                "benchmark_id": "ok",
                "retrieval_mode": "hybrid",
                "status": "succeeded",
                "question": "问题",
                "response": "回答",
                "contexts": [{"content": long_context}],
            },
            {
                "benchmark_id": "failed-run",
                "retrieval_mode": "vector",
                "status": "failed",
                "question": "失败问题",
                "response": "",
                "contexts": [],
            },
            {
                "benchmark_id": "judge-fails",
                "retrieval_mode": "vector",
                "status": "succeeded",
                "question": "第二个问题",
                "response": "第二个回答",
                "contexts": [{"content": "第二段完整上下文"}],
            },
        ]
        received: list[dict[str, Any]] = []

        async def fake_evaluator(sample: dict[str, Any]) -> float:
            received.append(sample)
            if sample["user_input"] == "第二个问题":
                raise RuntimeError("judge unavailable")
            return 0.87

        outcomes = await evaluate_faithfulness(records, evaluator=fake_evaluator)

        assert received[0]["retrieved_contexts"] == [long_context]
        assert len(received[0]["retrieved_contexts"][0]) > 200
        assert [outcome["status"] for outcome in outcomes] == ["scored", "failed", "failed"]
        assert outcomes[0]["score"] == 0.87
        assert outcomes[2]["exception"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_uses_bounded_workers_and_retains_order_under_concurrent_failure(self):
        """Only the requested number of Judge calls may be in flight."""
        records = [
            {
                "benchmark_id": f"case-{number}",
                "retrieval_mode": "hybrid",
                "status": "succeeded",
                "question": f"问题-{number}",
                "response": f"回答-{number}",
                "contexts": [{"content": "完整上下文"}],
            }
            for number in range(1, 4)
        ]
        active = 0
        maximum_active = 0
        two_started = asyncio.Event()
        release = asyncio.Event()

        async def fake_evaluator(sample: dict[str, Any]) -> float:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                two_started.set()
            try:
                await release.wait()
                if sample["user_input"] == "问题-2":
                    raise RuntimeError("judge unavailable")
                return float(sample["user_input"].rsplit("-", 1)[1])
            finally:
                active -= 1

        scoring = asyncio.create_task(
            evaluate_faithfulness(
                records,
                evaluator=fake_evaluator,
                max_concurrency=2,
            )
        )
        await asyncio.wait_for(two_started.wait(), timeout=1)

        assert maximum_active == 2
        release.set()
        outcomes = await scoring

        assert [outcome["benchmark_id"] for outcome in outcomes] == [
            "case-1",
            "case-2",
            "case-3",
        ]
        assert [outcome["status"] for outcome in outcomes] == [
            "scored",
            "failed",
            "scored",
        ]
        assert outcomes[1]["exception"] == "RuntimeError"


class TestContextQualityAdapter:
    """Verify reference-grounded context metrics use complete saved snapshots."""

    @pytest.mark.asyncio
    async def test_context_metrics_receive_complete_inputs_and_isolate_failures(self):
        """Each selected metric scores independently and retains strategy identity."""
        long_context = "完整检索证据" * 100
        records = [
            {
                "benchmark_id": "ok",
                "chunking_strategy": "recursive",
                "status": "succeeded",
                "question": "问题",
                "reference": "审核参考答案",
                "contexts": [{"content": long_context}],
            },
            {
                "benchmark_id": "failed-run",
                "chunking_strategy": "semantic",
                "status": "failed",
                "question": "失败问题",
                "reference": "参考",
                "contexts": [],
            },
            {
                "benchmark_id": "judge-fails",
                "chunking_strategy": "fixed",
                "status": "succeeded",
                "question": "第二个问题",
                "reference": "第二个参考",
                "contexts": [{"content": "第二段完整上下文"}],
            },
        ]
        received: dict[str, list[dict[str, Any]]] = {
            "context_precision": [],
            "context_recall": [],
        }

        async def precision(sample: dict[str, Any]) -> float:
            received["context_precision"].append(sample)
            if sample["user_input"] == "第二个问题":
                raise RuntimeError("judge secret must not leak")
            return 0.8

        async def recall(sample: dict[str, Any]) -> float:
            received["context_recall"].append(sample)
            return 0.7

        outcomes = await evaluate_context_quality(
            records,
            evaluators={
                "context_precision": precision,
                "context_recall": recall,
            },
        )

        assert received["context_precision"][0] == {
            "user_input": "问题",
            "response": "",
            "reference": "审核参考答案",
            "retrieved_contexts": [long_context],
        }
        assert len(received["context_recall"][0]["retrieved_contexts"][0]) > 200
        assert [(item["metric"], item["status"]) for item in outcomes] == [
            ("context_precision", "scored"),
            ("context_recall", "scored"),
            ("context_precision", "failed"),
            ("context_recall", "failed"),
            ("context_precision", "failed"),
            ("context_recall", "scored"),
        ]
        assert outcomes[0]["chunking_strategy"] == "recursive"
        assert outcomes[4]["exception"] == "RuntimeError"
        assert "judge secret" not in json.dumps(outcomes)

    @pytest.mark.asyncio
    async def test_blank_reference_or_context_fails_without_calling_evaluator(self):
        """context_precision 是 without-reference 变体：空白参考仍可评分，
        但空检索文本必须失败；context_recall（reference 版）参考缺失必须失败。"""
        records = [
            {
                "benchmark_id": "blank-reference-precision-ok",
                "retrieval_mode": "vector",
                "status": "succeeded",
                "question": "问题",
                "response": "回答",
                "reference": " ",
                "contexts": [{"content": "文本"}],
            },
            {
                "benchmark_id": "blank-context",
                "retrieval_mode": "hybrid",
                "status": "invalid_provenance",
                "question": "问题",
                "response": "回答",
                "reference": "参考",
                "contexts": [{"content": " "}],
            },
        ]
        calls: dict[str, int] = {"context_precision": 0, "context_recall": 0}

        async def precision(sample: dict[str, Any]) -> float:
            calls["context_precision"] += 1
            return 1.0

        async def recall(sample: dict[str, Any]) -> float:
            calls["context_recall"] += 1
            return 1.0

        outcomes = await evaluate_context_quality(
            records,
            evaluators={"context_precision": precision, "context_recall": recall},
        )

        # precision（without-reference）对空白参考照常评分；recall 因缺参考失败
        assert calls["context_precision"] == 1
        assert calls["context_recall"] == 0
        statuses = [(o["metric"], o["status"]) for o in outcomes]
        assert ("context_precision", "scored") in statuses
        assert ("context_recall", "failed") in statuses
        assert ("context_precision", "failed") in statuses
        assert ("context_recall", "failed") in statuses

    @pytest.mark.asyncio
    async def test_builders_use_ragas_collections_ascore_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ragas 0.4 collection metrics expose ascore rather than the legacy API."""
        received: list[dict[str, Any]] = []

        class FakeClient:
            """Accept isolated client options without network I/O."""

            def __init__(self, **kwargs: Any) -> None:
                """Record no credential values in test output."""
                self.options = kwargs

        def metric_type(score: float):
            """Create a fake collections metric class with only ascore."""

            class FakeMetric:
                """Mimic the current collections metric surface."""

                def __init__(self, *, llm: Any) -> None:
                    """Retain the injected judge handle."""
                    self.llm = llm

                async def ascore(self, **kwargs: Any) -> Any:
                    """Record keyword inputs and return a value wrapper."""
                    received.append(kwargs)
                    return SimpleNamespace(value=score)

            return FakeMetric

        def llm_factory(model: str, *, client: Any, **_kwargs: Any) -> tuple[str, Any]:
            """Return a fake judge handle."""
            return model, client

        monkeypatch.setattr(
            adapter,
            "_ragas_components",
            lambda: (FakeClient, object, (llm_factory, metric_type(0.9))),
        )
        monkeypatch.setattr(
            adapter,
            "_context_ragas_components",
            lambda: (
                FakeClient,
                object,
                (llm_factory, metric_type(0.8), metric_type(0.7)),
            ),
        )
        faithfulness = build_openai_faithfulness_evaluator(
            api_key="test-key",
            model="test-model",
        )
        context = build_openai_context_evaluators(
            ("context_precision", "context_recall"),
            api_key="test-key",
            model="test-model",
        )

        assert await faithfulness(
            {
                "user_input": "问题",
                "response": "回答",
                "retrieved_contexts": ["上下文"],
            }
        ) == 0.9
        # context_precision 为 without-reference 变体：ascore 以 response 评分
        assert await context["context_precision"](
            {
                "user_input": "问题",
                "response": "回答",
                "reference": "参考",
                "retrieved_contexts": ["上下文"],
            }
        ) == 0.8
        assert await context["context_recall"](
            {
                "user_input": "问题",
                "response": "回答",
                "reference": "参考",
                "retrieved_contexts": ["上下文"],
            }
        ) == 0.7
        precision_call = next(
            kwargs for kwargs in received if "response" in kwargs and "reference" not in kwargs
        )
        assert precision_call["response"] == "回答"
        recall_call = next(
            kwargs for kwargs in received if "reference" in kwargs and "response" not in kwargs
        )
        assert recall_call["reference"] == "参考"


class TestFactualCorrectnessAdapter:
    """Verify the reference-grounded answer correctness metric."""

    @pytest.mark.asyncio
    async def test_scores_reviewed_references_and_isolates_failures(self) -> None:
        """FactualCorrectness uses response + reference and keeps per-record status."""
        records = [
            {
                "benchmark_id": "ok",
                "retrieval_mode": "hybrid",
                "status": "succeeded",
                "question": "问题",
                "response": "正确答案",
                "reference": "审核参考答案",
                "contexts": [{"content": "完整证据"}],
            },
            {
                "benchmark_id": "failed-run",
                "retrieval_mode": "vector",
                "status": "failed",
                "question": "失败问题",
                "response": "",
                "reference": "参考",
                "contexts": [],
            },
            {
                "benchmark_id": "judge-fails",
                "retrieval_mode": "vector",
                "status": "succeeded",
                "question": "第二个问题",
                "response": "第二个回答",
                "reference": "第二个参考",
                "contexts": [{"content": "证据"}],
            },
        ]
        received: list[dict[str, Any]] = []

        async def fake_evaluator(sample: dict[str, Any]) -> float:
            received.append(sample)
            if sample["response"] == "第二个回答":
                raise RuntimeError("judge unavailable")
            return 0.85

        outcomes = await evaluate_factual_correctness(
            records,
            evaluator=fake_evaluator,
        )

        assert received[0] == {
            "response": "正确答案",
            "reference": "审核参考答案",
            "retrieved_contexts": ["完整证据"],
        }
        assert [outcome["status"] for outcome in outcomes] == [
            "scored",
            "failed",
            "failed",
        ]
        assert outcomes[0]["score"] == 0.85
        assert outcomes[0]["metric"] == "factual_correctness"
        assert outcomes[2]["exception"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_blank_response_or_reference_fails_without_calling_evaluator(
        self,
    ) -> None:
        """FactualCorrectness requires both a response and a reviewed reference."""
        records = [
            {
                "benchmark_id": "blank-response",
                "retrieval_mode": "vector",
                "status": "succeeded",
                "question": "问题",
                "response": " ",
                "reference": "参考",
                "contexts": [{"content": "文本"}],
            },
            {
                "benchmark_id": "blank-reference",
                "retrieval_mode": "hybrid",
                "status": "invalid_provenance",
                "question": "问题",
                "response": "回答",
                "reference": "",
                "contexts": [{"content": "文本"}],
            },
        ]
        calls = 0

        async def evaluator(sample: dict[str, Any]) -> float:
            nonlocal calls
            calls += 1
            return 1.0

        outcomes = await evaluate_factual_correctness(
            records,
            evaluator=evaluator,
        )

        assert calls == 0
        assert [outcome["status"] for outcome in outcomes] == ["failed", "failed"]

    @pytest.mark.asyncio
    async def test_factual_builder_uses_ragas_collections_ascore_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The FactualCorrectness builder receives isolated judge settings."""
        received: list[dict[str, Any]] = []

        class FakeClient:
            """Accept isolated client options without network I/O."""

            def __init__(self, **kwargs: Any) -> None:
                """Record no credential values in test output."""
                self.options = kwargs

        class FakeMetric:
            """Mimic the current collections metric surface."""

            def __init__(self, *, llm: Any) -> None:
                """Retain the injected judge handle."""
                self.llm = llm

            async def ascore(self, **kwargs: Any) -> Any:
                """Record keyword inputs and return a value wrapper."""
                received.append(kwargs)
                return SimpleNamespace(value=0.95)

        def llm_factory(model: str, *, client: Any, **_kwargs: Any) -> tuple[str, Any]:
            """Return a fake judge handle."""
            return model, client

        monkeypatch.setattr(
            adapter,
            "_factual_ragas_components",
            lambda: (FakeClient, object, (llm_factory, FakeMetric)),
        )
        evaluator = build_openai_factual_correctness_evaluator(
            api_key="test-key",
            model="test-model",
        )

        assert await evaluator(
            {
                "response": "回答",
                "reference": "参考",
                "retrieved_contexts": ["上下文"],
            }
        ) == 0.95
        assert received[0] == {
            "response": "回答",
            "reference": "参考",
        }
        assert received[0].get("user_input") is None


@pytest.mark.asyncio
async def test_runner_executes_all_native_retrieval_strategies_with_frozen_metadata(
    tmp_path: Path,
) -> None:
    """Four-way comparison keeps the strategy variable explicit and metadata frozen."""
    samples = load_benchmark(_write_jsonl(tmp_path / "benchmark.jsonl", _record()))
    strategies = ("dense", "dense_graph", "dense_bm25", "dense_bm25_graph")
    calls: list[str] = []

    async def qa_executor(**kwargs: Any) -> QAResult:
        calls.append(kwargs["retrieval_mode"])
        return QAResult(
            question=kwargs["question"],
            answer="回答受上下文支持。",
            contexts=[
                RetrievedContext(
                    content="同一冻结语料的证据。",
                    source="company-demo",
                    score=0.9,
                    retrieval_type=kwargs["retrieval_mode"],
                    metadata={"source_document_id": DOC_A},
                )
            ],
            intent=QueryIntent.FACTOID,
            confidence=0.9,
        )

    frozen = {
        "corpus_revision": "company-demo-v1",
        "knowledge_revision": 7,
        "embedding_model": "text-embedding-v4",
        "answer_model": "deepseek-v4-flash",
        "prompt_id": "qa-generation-v1",
        "retrieval_config_id": "native-rrf-v1",
        "top_k": 8,
        "cache_policy": "disabled",
        "authorization_scope_fixture": "tenant-department-v1",
        "bm25": {
            "schema_version": "native-bm25-v1",
            "tokenizer_version": "unicode-cjk-bigram-v1",
            "k1": 1.5,
            "b": 0.75,
            "index_generation": "generation-1",
        },
    }
    run = await BenchmarkRunner(
        qa_executor=qa_executor,
        results_root=tmp_path / "results",
    ).run(
        samples,
        tenant_id="eval-ragas",
        retrieval_modes=strategies,
        run_metadata=frozen,
    )

    metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    assert calls == list(strategies)
    assert [record["retrieval_mode"] for record in run.records] == list(strategies)
    assert metadata["retrieval_modes"] == list(strategies)
    for field, value in frozen.items():
        assert metadata[field] == value


def test_ragas_configuration_falls_back_to_the_existing_ark_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty isolated key must not combine Ark credentials with OpenAI defaults."""
    from evaluation.ragas.adapter import validate_ragas_configuration

    monkeypatch.setenv("EVAL_OPENAI_API_KEY", "")
    monkeypatch.setenv("EVAL_OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("EVAL_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ark-test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    configuration = validate_ragas_configuration()

    assert configuration == {
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "model": "deepseek-v4-flash",
        "source": "deepseek_fallback",
    }


def test_ragas_configuration_accepts_mimo_fallback_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.ragas.adapter import validate_ragas_configuration

    monkeypatch.delenv("EVAL_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVAL_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("EVAL_OPENAI_MODEL", raising=False)
    monkeypatch.setenv("MIMO_API_KEY", "mimo-test-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setenv("MIMO_MODEL", "mimo-v2.5-pro")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    assert validate_ragas_configuration() == {
        "base_url": "https://api.xiaomimimo.com/v1",
        "model": "mimo-v2.5-pro",
        "source": "mimo_fallback",
    }


def test_context_record_uses_normalized_source_document_id_for_provenance() -> None:
    from evaluation.runner import _context_record, _provenance

    document_id = "4ccfe084-4b94-4f76-8604-8c3db4eaa71f"
    record = _context_record(
        RetrievedContext(
            content="受控内容",
            source="policy.docx",
            score=0.9,
            retrieval_type="bm25",
            metadata={"doc_id": document_id, "source_document_id": document_id},
        ),
        1,
    )

    retrieved, unmapped = _provenance([record])
    assert retrieved == [document_id]
    assert unmapped == []

@pytest.mark.asyncio
async def test_ragas_runner_reuses_durable_scored_outcomes_after_interruption_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.scripts import run_ragas_benchmark

    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        json.dumps({
            "benchmark_id": "case-1", "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded", "question": "问题", "response": "回答",
            "contexts": [{"content": "受控内容"}],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    calls = 0

    async def evaluator(_sample: dict[str, Any]) -> float:
        nonlocal calls
        calls += 1
        return 0.91

    monkeypatch.setattr(
        run_ragas_benchmark, "build_openai_faithfulness_evaluator", lambda: evaluator
    )
    args = SimpleNamespace(
        responses_jsonl=responses,
        metrics=["faithfulness"],
        output_jsonl=tmp_path / "ragas_scores.jsonl",
        concurrency=1,
    )

    await run_ragas_benchmark.run(args)
    await run_ragas_benchmark.run(args)

    assert calls == 1
    summary = json.loads(args.output_jsonl.with_suffix(".summary.json").read_text())
    assert summary["metrics"]["faithfulness"]["scored"] == 1
    assert args.output_jsonl.with_suffix(".resume.json").is_file()

def test_ragas_summary_exposes_only_bounded_failure_distribution() -> None:
    """Attempt summaries distinguish schema faults, truncation, and batch fallback."""
    from evaluation.scripts import run_ragas_benchmark

    summary = run_ragas_benchmark._score_summary([
        {
            "metric": "faithfulness", "status": "failed", "score": None,
            "reason_code": "judge_json_object_invalid",
            "exception": "RagasJudgeJsonObjectError",
            "judge_diagnostics": {"finish_reason": "length", "batch_fallback": True},
        },
        {
            "metric": "faithfulness", "status": "failed", "score": None,
            "reason_code": "judge_structured_output_retry_exhausted",
            "exception": "InstructorRetryException",
        },
        {
            "metric": "faithfulness", "status": "scored", "score": 0.75,
            "reason_code": None, "exception": None,
        },
    ])

    metric = summary["metrics"]["faithfulness"]
    assert metric["failure_reasons"] == {
        "judge_json_object_invalid": 1,
        "judge_structured_output_retry_exhausted": 1,
    }
    assert metric["failure_exceptions"] == {
        "RagasJudgeJsonObjectError": 1,
        "InstructorRetryException": 1,
    }
    assert metric["failure_finish_reasons"] == {"length": 1}
    assert metric["batch_fallbacks"] == 1
    assert metric["mean"] == 0.75


def test_contract_summary_exposes_bounded_mismatch_distribution() -> None:
    from evaluation.scripts import run_ragas_benchmark

    summary = run_ragas_benchmark._score_summary([
        {
            "metric": "evidence_gate_contract", "status": "scored", "score": 0.0,
            "category": "fully_answerable", "expected_response_status": "answered",
            "observed_response_status": "partially_answered",
        },
        {
            "metric": "evidence_gate_contract", "status": "scored", "score": 1.0,
            "category": "authorization_filtered", "expected_response_status": "insufficient_evidence",
            "observed_response_status": "insufficient_evidence",
        },
    ])

    metric = summary["metrics"]["evidence_gate_contract"]
    assert metric["mismatch_by_category"] == {"fully_answerable": 1}
    assert metric["route_mismatches"] == {"answered->partially_answered": 1}


@pytest.mark.asyncio
async def test_ragas_checkpoint_remains_available_when_soft_limit_interrupts_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.scripts import run_ragas_benchmark

    responses = tmp_path / "responses.jsonl"
    records = [
        {"benchmark_id": f"case-{index}", "retrieval_mode": "dense", "status": "succeeded", "question": "问题", "response": "回答", "contexts": [{"content": "内容"}]}
        for index in (1, 2)
    ]
    responses.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    class SoftTimeLimitExceeded(Exception):
        pass

    calls = 0
    async def evaluator(_sample: dict[str, Any]) -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SoftTimeLimitExceeded()
        return 0.88

    monkeypatch.setattr(run_ragas_benchmark, "build_openai_faithfulness_evaluator", lambda: evaluator)
    args = SimpleNamespace(responses_jsonl=responses, metrics=["faithfulness"], output_jsonl=tmp_path / "ragas_scores.jsonl", concurrency=1)

    with pytest.raises(SoftTimeLimitExceeded):
        await run_ragas_benchmark.run(args)

    persisted = [json.loads(line) for line in args.output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [(item["benchmark_id"], item["status"]) for item in persisted] == [("case-1", "scored")]

@pytest.mark.asyncio
async def test_ragas_builders_bound_judge_requests_with_one_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Judge uses one bounded transport and structured-output retry."""
    from evaluation.ragas import adapter
    from evaluation.ragas.adapter import (
        build_openai_context_evaluators,
        build_openai_factual_correctness_evaluator,
        build_openai_faithfulness_evaluator,
    )

    clients: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            clients.append(kwargs)

    class Metric:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def ascore(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(value=1.0)

    def llm_factory(_model: str, *, client: Any, **_kwargs: Any) -> Any:
        return client

    monkeypatch.setattr(adapter, "_ragas_components", lambda: (FakeClient, object, (llm_factory, Metric)))
    monkeypatch.setattr(adapter, "_context_ragas_components", lambda: (FakeClient, object, (llm_factory, Metric, Metric)))
    monkeypatch.setattr(adapter, "_factual_ragas_components", lambda: (FakeClient, object, (llm_factory, Metric)))

    build_openai_faithfulness_evaluator(api_key="k", model="m")
    build_openai_factual_correctness_evaluator(api_key="k", model="m")
    build_openai_context_evaluators(("context_precision",), api_key="k", model="m")

    assert len(clients) == 3
    assert all(client["timeout"] == 90.0 and client["max_retries"] == 1 for client in clients)


def test_json_object_judge_builders_request_json_object_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON-mode Judge calls explicitly select OpenAI-compatible JSON output."""
    factory_options: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    class Metric:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    def llm_factory(_model: str, *, client: Any, **kwargs: Any) -> Any:
        factory_options.append(kwargs)
        return client

    monkeypatch.setattr(
        adapter,
        "_ragas_components",
        lambda: (FakeClient, object, (llm_factory, Metric)),
    )
    monkeypatch.setattr(
        adapter,
        "_context_ragas_components",
        lambda: (FakeClient, object, (llm_factory, Metric, Metric)),
    )
    monkeypatch.setattr(
        adapter,
        "_factual_ragas_components",
        lambda: (FakeClient, object, (llm_factory, Metric)),
    )

    build_openai_faithfulness_evaluator(api_key="k", model="doubao-seed-2.0-lite")
    build_openai_factual_correctness_evaluator(api_key="k", model="doubao-seed-2.0-lite")
    build_openai_context_evaluators(
        ("context_precision",),
        api_key="k",
        model="doubao-seed-2.0-lite",
    )

    expected_options = {
        "max_retries": 1,
        "max_tokens": 16_384,
        "temperature": 0,
        "seed": adapter.RAGAS_JUDGE_SEED,
        "response_format": {"type": "json_object"},
        "system_prompt": "Return only a valid JSON object matching the requested response schema.",
    }
    assert factory_options == [expected_options] * 3

    factory_options.clear()
    build_openai_faithfulness_evaluator(api_key="k", model="doubao-seed-2.1-turbo")
    build_openai_factual_correctness_evaluator(api_key="k", model="doubao-seed-2.1-turbo")
    build_openai_context_evaluators(
        ("context_precision",),
        api_key="k",
        model="doubao-seed-2.1-turbo",
    )

    # The direct Ark path uses llm_factory only as Ragas' required
    # InstructorBaseRagasLLM shell; its generate/agenerate methods are then
    # replaced in memory by the direct JSON Object adapter.
    assert factory_options == [expected_options] * 3

    factory_options.clear()
    build_openai_faithfulness_evaluator(api_key="k", model="deepseek-v4-flash")
    build_openai_factual_correctness_evaluator(api_key="k", model="deepseek-v4-flash")
    build_openai_context_evaluators(
        ("context_precision",),
        api_key="k",
        model="deepseek-v4-flash",
    )

    assert factory_options == [expected_options] * 3
    assert adapter._ragas_model_options("unsupported-judge") == {
        "max_retries": 1,
        "max_tokens": 16_384,
        "temperature": 0,
        "seed": adapter.RAGAS_JUDGE_SEED,
    }


def test_mimo_judge_transport_uses_its_extended_timeout_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI client budget must not undercut the direct MiMo adapter budget."""
    clients: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            clients.append(kwargs)

    class Metric:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    def llm_factory(_model: str, *, client: Any, **_kwargs: Any) -> Any:
        return client

    monkeypatch.setattr(
        adapter,
        "_ragas_components",
        lambda: (FakeClient, object, (llm_factory, Metric)),
    )

    build_openai_faithfulness_evaluator(api_key="k", model="mimo-v2.5-pro")

    assert clients[0]["timeout"] == adapter.RAGAS_MIMO_JUDGE_TIMEOUT_SECONDS


def test_doubao_direct_json_adapter_keeps_ragas_collection_shell() -> None:
    """Collections accept the Doubao route before it makes any Judge request."""
    adapter._install_vertexai_compatibility_shim()
    from openai import AsyncOpenAI
    from ragas.llms.base import InstructorBaseRagasLLM, llm_factory
    from ragas.metrics.collections import (
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )

    llm = adapter._build_ragas_llm(
        AsyncOpenAI=AsyncOpenAI,
        llm_factory=llm_factory,
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="doubao-seed-2.1-turbo",
    )

    assert isinstance(llm, InstructorBaseRagasLLM)
    assert llm.ragas_batch_size == 2
    # These constructors carry the strict Ragas 0.4 collection-type check that
    # caused the August 21, 2026 startup failure. Construction must not contact Ark.
    assert isinstance(Faithfulness(llm=llm), Faithfulness)
    assert isinstance(FactualCorrectness(llm=llm), FactualCorrectness)
    assert isinstance(ContextPrecision(llm=llm), ContextPrecision)
    assert isinstance(ContextRecall(llm=llm), ContextRecall)


def test_deepseek_uses_langchain_adapter_and_stable_json_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepSeek construction uses ChatDeepSeek while preserving Ragas' shell."""
    import sys
    import types

    calls: list[dict[str, Any]] = []

    class FakeRunnable:
        def __init__(self, model: Any) -> None:
            self.model = model

    class FakeChatDeepSeek:
        def __init__(self, **kwargs: Any) -> None:
            calls.append({"init": kwargs})

        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
            calls.append({"structured": schema, **kwargs})
            return FakeRunnable(self)

    monkeypatch.setitem(
        sys.modules,
        "langchain_deepseek",
        types.SimpleNamespace(ChatDeepSeek=FakeChatDeepSeek),
    )
    llm = adapter._DeepSeekLangChainRagasLLM(
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://api.deepseek.com",
    )
    assert calls == [{"init": {
        "model": "deepseek-v4-flash",
        "api_key": "key",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.01,
        "max_tokens": 16_384,
        "timeout": 90.0,
        "max_retries": 1,
    }}]
    assert llm._chat is not None


def test_deepseek_omits_empty_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ChatDeepSeek must use its documented default endpoint when unset."""
    import sys
    import types

    calls: list[dict[str, Any]] = []

    class FakeChatDeepSeek:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_deepseek",
        types.SimpleNamespace(ChatDeepSeek=FakeChatDeepSeek),
    )

    adapter._DeepSeekLangChainRagasLLM(
        model="deepseek-v4-flash",
        api_key="key",
        base_url=None,
    )

    assert calls == [{
        "model": "deepseek-v4-flash",
        "api_key": "key",
        "temperature": 0.01,
        "max_tokens": 16_384,
        "timeout": 90.0,
        "max_retries": 1,
    }]


def test_deepseek_cache_usage_is_bounded_and_normalized() -> None:
    """Only allowlisted cache counters are retained from provider usage."""
    response = types.SimpleNamespace(
        usage_metadata={"input_token_details": {"cache_read": 12, "cache_creation": 34}},
        response_metadata={
            "token_usage": {
                "raw_provider_payload": "must-not-persist",
            }
        },
    )
    assert adapter._bounded_cache_usage(response) == {
        "prompt_cache_hit_tokens": 12,
        "prompt_cache_miss_tokens": 34,
    }


@pytest.mark.asyncio
async def test_deepseek_langchain_generation_reuses_prefix_and_exposes_bounded_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DS adapter sends a stable system prefix and retains cache counters only."""
    import sys

    requests: list[Any] = []
    structured_calls = 0

    class Runnable:
        async def ainvoke(self, messages: Any) -> Any:
            requests.append(messages)
            return {
                "raw": SimpleNamespace(
                    usage_metadata={"prompt_cache_hit_tokens": 11, "prompt_cache_miss_tokens": 5},
                ),
                "parsed": response_model(reason="ok", verdict=1),
                "parsing_error": None,
            }

    class FakeChatDeepSeek:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def with_structured_output(self, _schema: Any, **kwargs: Any) -> Any:
            nonlocal structured_calls
            assert kwargs == {"method": "json_mode", "include_raw": True}
            structured_calls += 1
            return Runnable()

    response_model = create_model("DeepSeekContextScore", reason=(str, ...), verdict=(int, ...))
    monkeypatch.setitem(
        sys.modules,
        "langchain_deepseek",
        types.SimpleNamespace(ChatDeepSeek=FakeChatDeepSeek),
    )
    llm = adapter._DeepSeekLangChainRagasLLM(
        model="deepseek-v4-flash", api_key="key", base_url="https://api.deepseek.com",
    )

    result = await llm.agenerate("RAGAS instruction and variable item 1", response_model)
    second_result = await llm.agenerate("RAGAS instruction and variable item 2", response_model)

    assert result.verdict == 1
    assert second_result.verdict == 1
    assert structured_calls == 1
    assert requests[0][0].content == adapter._DEEPSEEK_SYSTEM_PREFIX
    assert requests[0][1].content == "RAGAS instruction and variable item 1"
    assert requests[1][0].content == adapter._DEEPSEEK_SYSTEM_PREFIX
    assert requests[1][1].content == "RAGAS instruction and variable item 2"
    assert llm.diagnostics() == {
        "prompt_cache_hit_tokens": 22,
        "prompt_cache_miss_tokens": 10,
    }


@pytest.mark.asyncio
async def test_structured_output_retry_reasks_with_invalid_output_error_and_original_context() -> None:
    """Instructor's one retry retains context and asks the Judge to correct its JSON."""
    from openai import AsyncOpenAI
    from pydantic import BaseModel

    adapter._install_vertexai_compatibility_shim()
    try:
        from ragas.llms.base import llm_factory
    except ImportError:
        pytest.skip("Ragas judge dependencies are unavailable")

    class Score(BaseModel):
        score: float

    requests: list[dict[str, Any]] = []
    response_contents = ["not valid JSON", '{"score": 0.9}']

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = response_contents.pop(0)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-v4-flash",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://judge.invalid/v1",
        http_client=http_client,
        max_retries=0,
    )
    judge = llm_factory(
        "deepseek-v4-flash",
        client=client,
        **adapter._ragas_model_options("deepseek-v4-flash"),
    )
    try:
        result = await judge.agenerate("original RAGAS prompt with evaluation context", Score)
    finally:
        await client.close()

    assert result.score == 0.9
    assert len(requests) == 2
    assert requests[1]["messages"][:2] == requests[0]["messages"]
    assert requests[1]["messages"][-2] == {"role": "assistant", "content": "not valid JSON"}
    correction = requests[1]["messages"][-1]
    assert correction["role"] == "user"
    assert "Correct your JSON ONLY RESPONSE" in correction["content"]
    assert "not valid JSON" in correction["content"]
    assert requests[1]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_factual_correctness_builder_uses_only_supported_score_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Ragas factual metric rejects retrieved contexts as score arguments."""
    received: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    class Metric:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def ascore(self, *, response: str, reference: str) -> Any:
            received.append({"response": response, "reference": reference})
            return SimpleNamespace(value=0.75)

    def llm_factory(_model: str, *, client: Any, **_kwargs: Any) -> Any:
        return client

    monkeypatch.setattr(
        adapter,
        "_factual_ragas_components",
        lambda: (FakeClient, object, (llm_factory, Metric)),
    )

    evaluator = build_openai_factual_correctness_evaluator(api_key="k", model="m")

    assert await evaluator({
        "response": "回答",
        "reference": "参考",
        "retrieved_contexts": ["仅用于审计的上下文"],
    }) == 0.75
    # 默认 RAGAS_JUDGE_REPEATS=3：同一样本独立评三次取中位数，输入完全相同。
    assert received == [{"response": "回答", "reference": "参考"}] * 3


@pytest.mark.asyncio
async def test_faithfulness_incomplete_input_fails_with_a_reason() -> None:
    """Malformed saved records must not abort the remaining Faithfulness batch."""
    calls = 0

    async def evaluator(_sample: dict[str, Any]) -> float:
        nonlocal calls
        calls += 1
        return 1.0

    outcomes = await evaluate_faithfulness(
        [{
            "benchmark_id": "missing-context",
            "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded",
            "question": "问题",
            "response": "回答",
            "contexts": [],
        }],
        evaluator=evaluator,
    )

    assert calls == 0
    assert outcomes == [{
        "benchmark_id": "missing-context",
        "retrieval_mode": "dense_bm25_graph",
        "metric": "faithfulness",
        "score": None,
        "status": "failed",
        "exception": None,
        "reason_code": "missing_retrieved_contexts",
    }]


@pytest.mark.asyncio
async def test_audited_no_evidence_routes_are_not_applicable_not_skipped() -> None:
    """Formal refusal routes remain visible without fabricating RAGAS evidence."""
    record = {
        "benchmark_id": "audited-no-evidence",
        "retrieval_mode": "dense_bm25_graph",
        "status": "succeeded",
        "expected_refusal": True,
        "expected_response_status": "insufficient_evidence",
        "question": "问题",
        "response": "没有足够证据。",
        "reference": "",
        "contexts": [],
    }
    calls = 0

    async def evaluator(_sample: dict[str, Any]) -> float:
        nonlocal calls
        calls += 1
        return 1.0

    faithfulness = await evaluate_faithfulness([record], evaluator=evaluator)
    factual = await evaluate_factual_correctness([record], evaluator=evaluator)
    context = await evaluate_context_quality(
        [record], evaluators={"context_precision": evaluator}
    )

    assert calls == 0
    assert [outcome["status"] for outcome in (*faithfulness, *factual, *context)] == [
        "not_applicable",
        "not_applicable",
        "not_applicable",
    ]
    assert {outcome["reason_code"] for outcome in (*faithfulness, *factual, *context)} == {
        "audited_no_evidence_route"
    }


@pytest.mark.asyncio
async def test_correct_no_answer_route_does_not_receive_factual_zero() -> None:
    calls = 0

    async def evaluator(_sample: dict[str, Any]) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    record = {
        "benchmark_id": "authorized-refusal",
        "retrieval_mode": "dense_bm25_graph",
        "status": "succeeded",
        "expected_refusal": True,
        "expected_response_status": "insufficient_evidence",
        "response_status": "insufficient_evidence",
        "expected_evidence_states": ["insufficient_evidence"],
        "stages": {"qualification": {"evidence_states": ["insufficient_evidence"]}},
        "response": "没有可见证据。",
        "reference": "受权限保护的正向参考。",
        "contexts": [],
    }

    outcome = (await evaluate_factual_correctness([record], evaluator=evaluator))[0]

    assert calls == 0
    assert outcome["status"] == "not_applicable"
    assert outcome["reason_code"] == "audited_no_answer_route"


@pytest.mark.asyncio
async def test_false_refusal_on_answerable_route_remains_factual_scored() -> None:
    calls = 0

    async def evaluator(_sample: dict[str, Any]) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    record = {
        "benchmark_id": "false-refusal",
        "retrieval_mode": "dense_bm25_graph",
        "status": "succeeded",
        "expected_refusal": False,
        "expected_response_status": "answered",
        "response_status": "insufficient_evidence",
        "expected_evidence_states": ["direct_evidence"],
        "stages": {"qualification": {"evidence_states": ["insufficient_evidence"]}},
        "response": "没有足够证据。",
        "reference": "应当回答的参考。",
        "contexts": [],
    }

    outcome = (await evaluate_factual_correctness([record], evaluator=evaluator))[0]

    assert calls == 1
    assert outcome["status"] == "scored"
    assert outcome["score"] == 0.0


def test_evidence_gate_contract_scores_every_reviewed_response_route() -> None:
    """The all-case contract is deterministic and needs no Judge or raw text."""
    records = [
        {
            "benchmark_id": "expected-refusal",
            "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded",
            "expected_response_status": "insufficient_evidence",
            "response_status": "insufficient_evidence",
            "expected_evidence_states": ["insufficient_evidence"],
            "stages": {"qualification": {"evidence_states": ["insufficient_evidence"]}},
        },
        {
            "benchmark_id": "wrong-route",
            "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded",
            "expected_response_status": "source_unavailable",
            "response_status": "answered",
            "expected_evidence_states": ["insufficient_evidence"],
            "stages": {"qualification": {"evidence_states": ["direct_evidence"]}},
        },
    ]

    outcomes = evaluate_evidence_gate_contract(records)

    assert [(outcome["status"], outcome["score"], outcome["reason_code"]) for outcome in outcomes] == [
        ("scored", 1.0, None),
        ("scored", 0.0, None),
    ]


@pytest.mark.asyncio
async def test_non_finite_judge_score_is_failed_and_checkpointed() -> None:
    """NaN/Infinity must never become a durable scored outcome or metric mean."""
    checkpoints: list[dict[str, Any]] = []

    async def evaluator(_sample: dict[str, Any]) -> float:
        return float("nan")

    outcomes = await evaluate_faithfulness(
        [{
            "benchmark_id": "nan-score",
            "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded",
            "question": "问题",
            "response": "回答",
            "contexts": [{"content": "完整上下文"}],
        }],
        evaluator=evaluator,
        on_completed=lambda outcome: checkpoints.append(dict(outcome)),
    )

    assert outcomes[0]["status"] == "failed"
    assert outcomes[0]["reason_code"] == "judge_invalid_score"
    assert checkpoints == outcomes


@pytest.mark.asyncio
async def test_judge_connection_failure_is_retried_once_before_becoming_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-off Judge transport fault must not fail an otherwise complete run."""
    attempts = 0
    api_connection_error = type("APIConnectionError", (Exception,), {})

    async def evaluator(_sample: dict[str, Any]) -> float:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise api_connection_error()
        return 0.75

    monkeypatch.setattr(adapter, "RAGAS_SINGLE_SCORE_RETRY_DELAY_SECONDS", 0.0)
    outcomes = await evaluate_faithfulness(
        [{
            "benchmark_id": "transient-connection",
            "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded",
            "question": "问题",
            "response": "回答",
            "contexts": [{"content": "完整上下文"}],
        }],
        evaluator=evaluator,
    )

    assert attempts == 2
    assert outcomes[0]["status"] == "scored"
    assert outcomes[0]["score"] == 0.75


@pytest.mark.asyncio
async def test_doubao_json_adapter_repairs_known_schema_aliases() -> None:
    """Ark JSON Object responses get bounded local normalization before validation."""
    response_model = create_model("ClaimDecompositionOutput", claims=(list[str], ...))

    result = adapter._response_model_from_json(
        response_model,
        {"statements": ["第一条可验证陈述", "第二条可验证陈述"]},
    )

    assert result.claims == ["第一条可验证陈述", "第二条可验证陈述"]


@pytest.mark.asyncio
async def test_doubao_json_adapter_sends_json_object_without_instructor() -> None:
    """The Ark branch bypasses Instructor's strict parse/retry wrapper."""
    response_model = create_model("ContextPrecisionOutput", reason=(str, ...), verdict=(int, ...))
    requests: list[dict[str, Any]] = []

    class Completions:
        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content='{"explanation":"证据支持回答","attributed":"1"}',
                ))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    llm = adapter._DoubaoJsonObjectRagasLLM(
        client=client,
        model="doubao-seed-2.1-turbo",
    )

    result = await llm.agenerate("仅返回 JSON", response_model)

    assert result.reason == "证据支持回答"
    assert result.verdict == 1
    assert requests == [{
        "model": "doubao-seed-2.1-turbo",
        "messages": [
            {"role": "system", "content": "Return only a valid JSON object matching the requested response schema."},
            {"role": "user", "content": "仅返回 JSON"},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 16_384,
        "temperature": 0.01,
        "top_p": 0.1,
    }]


@pytest.mark.asyncio
async def test_mimo_json_adapter_uses_openai_compatible_json_object_mode() -> None:
    """Mimo uses the documented OpenAI-compatible JSON Object request contract."""
    response_model = create_model("ContextPrecisionOutput", reason=(str, ...), verdict=(int, ...))
    requests: list[dict[str, Any]] = []

    class Completions:
        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content='{"reason":"证据支持回答","verdict":1}',
                ))]
            )

    llm = adapter._MimoJsonObjectRagasLLM(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        model="mimo-v2.5-pro",
    )

    result = await llm.agenerate("仅返回 JSON", response_model)

    assert result.verdict == 1
    assert llm.ragas_batch_size == 2
    assert requests[0]["model"] == "mimo-v2.5-pro"
    assert requests[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_doubao_json_adapter_retains_only_allowlisted_completion_metadata() -> None:
    """Direct Ark JSON validation persists token/finish diagnostics but not model text."""
    response_model = create_model("ContextPrecisionOutput", reason=(str, ...), verdict=(int, ...))

    class Completions:
        async def create(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content="{not-json"),
                )],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=16_384, total_tokens=16_396),
            )

    llm = adapter._DoubaoJsonObjectRagasLLM(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        model="doubao-seed-2.1-turbo",
    )

    with pytest.raises(adapter.RagasJudgeJsonObjectError) as error:
        await llm.agenerate("仅返回 JSON", response_model)

    assert adapter._judge_failure_diagnostics(error.value) == {
        "configured_max_tokens": 16_384,
        "finish_reason": "length",
        "prompt_tokens": 12,
        "completion_tokens": 16_384,
        "total_tokens": 16_396,
    }
    assert "not-json" not in json.dumps(
        adapter._judge_failure_diagnostics(error.value),
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_doubao_json_adapter_retries_once_without_retaining_invalid_output() -> None:
    """A transient invalid object gets one direct correction attempt before failing."""
    response_model = create_model("ContextPrecisionOutput", reason=(str, ...), verdict=(int, ...))
    responses = ["{not-json", '{"reason":"可用","verdict":1}']
    requests: list[dict[str, Any]] = []

    class Completions:
        async def create(self, **kwargs: Any) -> Any:
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=responses.pop(0)))],
            )

    llm = adapter._DoubaoJsonObjectRagasLLM(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        model="doubao-seed-2.1-turbo",
    )

    result = await llm.agenerate("仅返回 JSON", response_model)

    assert result.verdict == 1
    assert len(requests) == 2
    assert "prior response did not pass" in requests[1]["messages"][0]["content"]
    assert "{not-json" not in json.dumps(requests[1], ensure_ascii=False)


@pytest.mark.asyncio
async def test_doubao_json_adapter_allows_concurrent_judge_requests() -> None:
    """One evaluator can use the caller's bounded RAGAS concurrency."""
    response_model = create_model("ContextPrecisionOutput", reason=(str, ...), verdict=(int, ...))
    active = 0
    maximum_active = 0

    class Completions:
        async def create(self, **_kwargs: Any) -> Any:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0)
            active -= 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content='{"reason":"可用","verdict":1}',
                ))]
            )

    llm = adapter._DoubaoJsonObjectRagasLLM(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        model="doubao-seed-2.1-turbo",
    )

    results = await asyncio.gather(
        llm.agenerate("第一题", response_model),
        llm.agenerate("第二题", response_model),
    )

    assert [result.verdict for result in results] == [1, 1]
    assert maximum_active == 2


@pytest.mark.asyncio
async def test_doubao_json_adapter_bounds_and_retries_stalled_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled direct request is cancelled and retried once before failing."""
    response_model = create_model("ContextPrecisionOutput", reason=(str, ...), verdict=(int, ...))
    attempts = 0

    class Completions:
        async def create(self, **_kwargs: Any) -> Any:
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(1)
            raise AssertionError("the outer timeout should cancel this call")

    monkeypatch.setattr(adapter, "RAGAS_DOUBAO_JUDGE_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(adapter, "RAGAS_DOUBAO_JUDGE_RETRY_DELAY_SECONDS", 0.0)
    llm = adapter._DoubaoJsonObjectRagasLLM(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        model="doubao-seed-2.1-turbo",
    )

    with pytest.raises(TimeoutError):
        await llm.agenerate("仅返回 JSON", response_model)

    assert attempts == 2


@pytest.mark.asyncio
async def test_doubao_json_adapter_retries_sdk_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI SDK timeout class receives the same bounded retry policy."""
    response_model = create_model("ContextPrecisionOutput", reason=(str, ...), verdict=(int, ...))
    attempts = 0
    api_timeout_error = type("APITimeoutError", (Exception,), {})

    class Completions:
        async def create(self, **_kwargs: Any) -> Any:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise api_timeout_error()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content='{"reason":"可用","verdict":1}',
                ))]
            )

    monkeypatch.setattr(adapter, "RAGAS_DOUBAO_JUDGE_RETRY_DELAY_SECONDS", 0.0)
    llm = adapter._DoubaoJsonObjectRagasLLM(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
        model="doubao-seed-2.1-turbo",
    )

    result = await llm.agenerate("仅返回 JSON", response_model)

    assert result.verdict == 1
    assert attempts == 2


@pytest.mark.asyncio
async def test_batchable_evaluator_repeats_median_and_disables_batching() -> None:
    """Repeat scoring takes the per-sample median and suppresses batch merging."""
    from evaluation.ragas.adapter import _BatchableEvaluator

    scores = [0.1, 0.9, 0.5, 0.7]
    calls = {"count": 0}

    async def flaky_evaluator(_sample: dict[str, Any]) -> float:
        value = scores[calls["count"] % len(scores)]
        calls["count"] += 1
        return value

    async def batch_evaluator(_samples: list[dict[str, Any]]) -> list[float]:
        raise AssertionError("batch path must be suppressed while repeats > 1")

    evaluator = _BatchableEvaluator(
        flaky_evaluator,
        batch_evaluator,
        batch_size=5,
        repeats=3,
    )

    assert evaluator.effective_batch_size == 1
    result = await evaluator({"question": "q"})
    assert calls["count"] == 3
    # 三次独立打分 [0.1, 0.9, 0.5] 的中位数抹平单次极端值。
    assert result == 0.5

    evaluator_single = _BatchableEvaluator(
        flaky_evaluator,
        batch_evaluator,
        batch_size=5,
        repeats=1,
    )
    assert evaluator_single.effective_batch_size == 5
    assert await evaluator_single({"question": "q"}) == 0.7


@pytest.mark.asyncio
async def test_doubao_adaptive_batches_graduate_to_two_samples_after_probe() -> None:
    """Successful probes enable two-item batches for the remaining work."""
    class AdaptiveEvaluator:
        def __init__(self) -> None:
            self.batch_size = 2
            self.adaptive_batches = True
            self._probe_complete = False
            self._batching_degraded = False
            self.single_calls = 0
            self.batch_sizes: list[int] = []

        @property
        def effective_batch_size(self) -> int:
            if self._batching_degraded or not self._probe_complete:
                return 1
            return self.batch_size

        def observe_single_result(self, succeeded: bool) -> None:
            if not self._probe_complete:
                self._probe_complete = True
                self._batching_degraded = not succeeded

        def observe_batch_failure(self) -> None:
            self._batching_degraded = True

        async def __call__(self, _sample: dict[str, Any]) -> float:
            self.single_calls += 1
            return 0.5

        async def score_batch(self, samples: list[dict[str, Any]]) -> list[float]:
            self.batch_sizes.append(len(samples))
            return [0.7] * len(samples)

    evaluator = AdaptiveEvaluator()
    records = [
        {
            "benchmark_id": f"adaptive-{index}",
            "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded",
            "question": f"问题-{index}",
            "response": f"回答-{index}",
            "contexts": [{"content": "完整上下文"}],
        }
        for index in range(6)
    ]

    outcomes = await evaluate_faithfulness(records, evaluator=evaluator)

    assert evaluator.single_calls == 4
    assert evaluator.batch_sizes == [2]
    assert [outcome["status"] for outcome in outcomes] == ["scored"] * 6


@pytest.mark.asyncio
async def test_ragas_batch_prompt_reorders_exact_item_coverage() -> None:
    """Batch parsing returns one validated structured result per stable item id."""
    class Input(BaseModel):
        question: str

    class Result(BaseModel):
        verdict: int

    class Prompt:
        instruction = "Return an independent verdict for each item."
        output_model = Result

        @staticmethod
        def _generate_examples() -> str:
            return "Example input and output."

    class FakeLLM:
        async def agenerate(self, _prompt: str, response_model: Any) -> Any:
            return response_model(
                items=[
                    {"item_id": 1, "result": {"verdict": 0}},
                    {"item_id": 0, "result": {"verdict": 1}},
                ]
            )

    results = await adapter._batch_prompt_results(
        FakeLLM(),
        Prompt(),
        [Input(question="第一题"), Input(question="第二题")],
    )

    assert [result.verdict for result in results] == [1, 0]


@pytest.mark.asyncio
async def test_ragas_batch_prompt_rejects_duplicate_item_ids() -> None:
    """Malformed aggregate Judge output must trigger the isolated fallback path."""
    class Input(BaseModel):
        question: str

    class Result(BaseModel):
        verdict: int

    class Prompt:
        instruction = "Return an independent verdict for each item."
        output_model = Result

        @staticmethod
        def _generate_examples() -> str:
            return "Example input and output."

    class FakeLLM:
        async def agenerate(self, _prompt: str, response_model: Any) -> Any:
            return response_model(
                items=[
                    {"item_id": 0, "result": {"verdict": 1}},
                    {"item_id": 0, "result": {"verdict": 0}},
                ]
            )

    with pytest.raises(ValueError, match="ragas_batch_output_invalid"):
        await adapter._batch_prompt_results(
            FakeLLM(),
            Prompt(),
            [Input(question="第一题"), Input(question="第二题")],
        )


@pytest.mark.asyncio
async def test_ragas_judge_scores_five_compatible_samples_in_one_batch() -> None:
    """A bounded batch retains one durable outcome and checkpoint per sample."""
    class BatchEvaluator:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []
            self.single_calls: list[str] = []

        async def __call__(self, sample: dict[str, Any]) -> float:
            self.single_calls.append(sample["user_input"])
            return 0.25

        async def score_batch(self, samples: list[dict[str, Any]]) -> list[float]:
            self.batch_sizes.append(len(samples))
            return [0.75 for _sample in samples]

    evaluator = BatchEvaluator()
    checkpoints: list[dict[str, Any]] = []
    records = [
        {
            "benchmark_id": f"batch-{index}",
            "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded",
            "question": f"问题-{index}",
            "response": f"回答-{index}",
            "contexts": [{"content": "完整上下文"}],
        }
        for index in range(6)
    ]

    outcomes = await evaluate_faithfulness(
        records,
        evaluator=evaluator,
        on_completed=lambda outcome: checkpoints.append(dict(outcome)),
    )

    assert evaluator.batch_sizes == [5]
    assert evaluator.single_calls == ["问题-5"]
    assert [outcome["score"] for outcome in outcomes] == [0.75] * 5 + [0.25]
    assert [outcome["status"] for outcome in outcomes] == ["scored"] * 6
    assert {outcome["benchmark_id"] for outcome in checkpoints} == {
        f"batch-{index}" for index in range(6)
    }


@pytest.mark.asyncio
async def test_ragas_judge_batch_fault_falls_back_to_isolated_single_scores() -> None:
    """A malformed or failed batch must not mark every member as failed."""
    class FallbackEvaluator:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.single_calls: list[str] = []

        async def __call__(self, sample: dict[str, Any]) -> float:
            self.single_calls.append(sample["user_input"])
            return 0.8

        async def score_batch(self, _samples: list[dict[str, Any]]) -> list[float]:
            self.batch_calls += 1
            raise ValueError("untrusted Judge response must not be retained")

    evaluator = FallbackEvaluator()
    records = [
        {
            "benchmark_id": f"fallback-{index}",
            "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded",
            "question": f"问题-{index}",
            "response": f"回答-{index}",
            "contexts": [{"content": "完整上下文"}],
        }
        for index in range(5)
    ]

    outcomes = await evaluate_faithfulness(records, evaluator=evaluator)

    assert evaluator.batch_calls == 1
    assert evaluator.single_calls == [f"问题-{index}" for index in range(5)]
    assert [outcome["status"] for outcome in outcomes] == ["scored"] * 5
    assert "untrusted Judge response" not in json.dumps(outcomes, ensure_ascii=False)


@pytest.mark.asyncio
async def test_ragas_judge_keeps_oversized_samples_out_of_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transient five-sample adapter never combines oversized input payloads."""
    class BatchEvaluator:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.single_calls = 0

        async def __call__(self, _sample: dict[str, Any]) -> float:
            self.single_calls += 1
            return 0.6

        async def score_batch(self, _samples: list[dict[str, Any]]) -> list[float]:
            self.batch_calls += 1
            return [0.9]

    monkeypatch.setattr(adapter, "RAGAS_JUDGE_BATCH_MAX_INPUT_CHARS", 100)
    evaluator = BatchEvaluator()
    records = [
        {
            "benchmark_id": f"large-{index}",
            "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded",
            "question": "问题-" + "长" * 100,
            "response": "回答",
            "contexts": [{"content": "完整上下文"}],
        }
        for index in range(2)
    ]

    outcomes = await evaluate_faithfulness(records, evaluator=evaluator)

    assert evaluator.batch_calls == 0
    assert evaluator.single_calls == 2
    assert [outcome["score"] for outcome in outcomes] == [0.6, 0.6]


@pytest.mark.asyncio
async def test_structured_output_retry_exhaustion_has_a_stable_reason() -> None:
    """Provider parsing retries must remain diagnosable without response text."""
    class InstructorRetryException(Exception):
        pass

    async def evaluator(_sample: dict[str, Any]) -> float:
        raise InstructorRetryException("provider output omitted")

    outcomes = await evaluate_faithfulness(
        [{
            "benchmark_id": "structured-output", "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded", "question": "问题", "response": "回答",
            "contexts": [{"content": "完整上下文"}],
        }],
        evaluator=evaluator,
    )

    assert outcomes[0]["status"] == "failed"
    assert outcomes[0]["exception"] == "InstructorRetryException"
    assert outcomes[0]["reason_code"] == "judge_structured_output_retry_exhausted"
    assert outcomes[0]["judge_diagnostics"] == {"configured_max_tokens": 16_384}


@pytest.mark.asyncio
async def test_incomplete_judge_output_persists_only_allowlisted_diagnostics() -> None:
    """Truncation diagnostics aid retry decisions without retaining Judge content."""
    class IncompleteOutputException(Exception):
        def __init__(self) -> None:
            self.last_completion = SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content='{"secret": "do not persist"}'),
                )],
                usage=SimpleNamespace(
                    prompt_tokens=17,
                    completion_tokens=16_384,
                    total_tokens=16_401,
                    cached_tokens=42,
                ),
            )

    async def evaluator(_sample: dict[str, Any]) -> float:
        raise IncompleteOutputException()

    outcomes = await evaluate_faithfulness(
        [{
            "benchmark_id": "truncated-output", "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded", "question": "问题", "response": "回答",
            "contexts": [{"content": "完整上下文"}],
        }],
        evaluator=evaluator,
    )

    outcome = outcomes[0]
    assert outcome["status"] == "failed"
    assert outcome["exception"] == "IncompleteOutputException"
    assert outcome["reason_code"] == "judge_structured_output_retry_exhausted"
    assert outcome["judge_diagnostics"] == {
        "configured_max_tokens": 16_384,
        "finish_reason": "length",
        "prompt_tokens": 17,
        "completion_tokens": 16_384,
        "total_tokens": 16_401,
    }
    assert "do not persist" not in json.dumps(outcome, ensure_ascii=False)


@pytest.mark.asyncio
async def test_ragas_runner_recomputes_duplicate_checkpoint_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambiguous checkpoint cannot choose an arbitrary previous score."""
    from evaluation.scripts import run_ragas_benchmark

    responses = tmp_path / "responses.jsonl"
    response = {
        "benchmark_id": "case-1", "retrieval_mode": "dense_bm25_graph",
        "status": "succeeded", "question": "问题", "response": "回答",
        "contexts": [{"content": "受控内容"}],
    }
    responses.write_text(json.dumps(response, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "ragas_scores.jsonl"
    duplicate = {
        "benchmark_id": "case-1", "retrieval_mode": "dense_bm25_graph",
        "metric": "faithfulness", "score": 0.1, "status": "scored",
        "exception": None,
    }
    output.write_text(
        "".join(json.dumps(duplicate, ensure_ascii=False) + "\n" for _ in range(2)),
        encoding="utf-8",
    )
    output.with_suffix(".resume.json").write_text(
        json.dumps({
            "responses_sha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
            "metrics": ["faithfulness"],
        }) + "\n",
        encoding="utf-8",
    )
    calls = 0

    async def evaluator(_sample: dict[str, Any]) -> float:
        nonlocal calls
        calls += 1
        return 0.91

    monkeypatch.setattr(
        run_ragas_benchmark, "build_openai_faithfulness_evaluator", lambda: evaluator
    )
    await run_ragas_benchmark.run(SimpleNamespace(
        responses_jsonl=responses, metrics=["faithfulness"], output_jsonl=output, concurrency=1,
    ))

    assert calls == 1
    persisted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert persisted == [{
        "benchmark_id": "case-1",
        "retrieval_mode": "dense_bm25_graph",
        "metric": "faithfulness",
        "score": 0.91,
        "status": "scored",
        "exception": None,
        "reason_code": None,
    }]
