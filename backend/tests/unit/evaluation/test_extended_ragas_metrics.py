"""No-network contracts for category-aware extended RAGAS metrics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import json

import pytest

from evaluation.ragas import adapter
from evaluation.scripts.run_ragas_benchmark import parse_args


@pytest.mark.asyncio
async def test_extended_builders_construct_supported_metrics_and_exact_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, dict[str, Any]]] = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.options = kwargs

    def metric_type(name: str, dependency: str):
        class Metric:
            def __init__(self, **kwargs: Any) -> None:
                assert dependency in kwargs

            async def ascore(self, **kwargs: Any) -> Any:
                received.append((name, kwargs))
                return SimpleNamespace(value=0.8)

        return Metric

    def llm_factory(model: str, **_: Any) -> str:
        return f"llm:{model}"

    def embedding_factory(**kwargs: Any) -> str:
        # Ragas' modern OpenAI embedding provider receives a preconfigured
        # client; forwarding ``base_url`` again is rejected by its constructor.
        assert "base_url" not in kwargs
        assert kwargs["client"].options["base_url"] == "https://dashscope.example/v1"
        return f"embeddings:{kwargs['model']}"

    monkeypatch.setattr(
        adapter,
        "_extended_ragas_components",
        lambda: (
            FakeClient,
            llm_factory,
            embedding_factory,
            {
                "answer_relevancy": metric_type("answer_relevancy", "llm"),
                "semantic_similarity": metric_type("semantic_similarity", "embeddings"),
                "noise_sensitivity": metric_type("noise_sensitivity", "llm"),
                "rubrics_score_with_reference": metric_type(
                    "rubrics_score_with_reference", "llm"
                ),
                "rubrics_score_without_reference": metric_type(
                    "rubrics_score_without_reference", "llm"
                ),
            },
        ),
    )
    evaluators = adapter.build_openai_extended_evaluators(
        (
            "answer_relevancy",
            "semantic_similarity",
            "noise_sensitivity",
            "rubrics_score_with_reference",
            "rubrics_score_without_reference",
        ),
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="judge-model",
        embedding_model="embedding-model",
        embedding_api_key="embedding-key",
        embedding_base_url="https://dashscope.example/v1",
    )
    sample = {
        "user_input": "question",
        "response": "answer",
        "reference": "reference",
        "retrieved_contexts": ["context"],
        "reference_contexts": ["reviewed context"],
    }
    for evaluator in evaluators.values():
        assert await evaluator(sample) == 0.8

    assert dict(received)["answer_relevancy"] == {
        "user_input": "question",
        "response": "answer",
    }
    assert dict(received)["semantic_similarity"] == {
        "reference": "reference",
        "response": "answer",
    }
    assert set(dict(received)["noise_sensitivity"]) == {
        "user_input",
        "response",
        "reference",
        "retrieved_contexts",
    }
    assert dict(received)["rubrics_score_without_reference"] == {
        "user_input": "question",
        "response": "answer",
    }


def test_extended_builder_rejects_unsupported_or_missing_embedding_before_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter,
        "_extended_ragas_components",
        lambda: pytest.fail("components must not load"),
    )
    with pytest.raises(adapter.RagasConfigurationError, match="aspect_critic"):
        adapter.build_openai_extended_evaluators(("aspect_critic",))
    with pytest.raises(
        adapter.RagasConfigurationError, match="embedding_model"
    ):
        adapter.build_openai_extended_evaluators(
            ("semantic_similarity",),
            api_key="test-key",
            model="judge-model",
        )


@pytest.mark.asyncio
async def test_category_policy_inputs_not_applicable_and_judge_failures_are_isolated() -> None:
    calls: list[str] = []

    async def evaluator(sample: dict[str, Any]) -> float:
        calls.append(sample["response"])
        if sample["response"] == "judge fails":
            raise RuntimeError("private provider detail")
        return 0.7

    records = [
        {
            "benchmark_id": "complete",
            "retrieval_mode": "dense",
            "status": "succeeded",
            "category": "fully_answerable",
            "question": "question",
            "response": "answer",
            "reference": "reference",
            "contexts": [{"content": "context"}],
        },
        {
            "benchmark_id": "missing-reference",
            "retrieval_mode": "dense",
            "status": "succeeded",
            "category": "fully_answerable",
            "question": "question",
            "response": "answer",
            "reference": "",
            "contexts": [{"content": "context"}],
        },
        {
            "benchmark_id": "correct-non-answer",
            "retrieval_mode": "dense",
            "status": "succeeded",
            "category": "completely_unanswerable",
            "question": "question",
            "response": "controlled refusal",
            "reference": "",
            "contexts": [],
        },
        {
            "benchmark_id": "wrong-route",
            "retrieval_mode": "dense",
            "status": "succeeded",
            "category": "fully_answerable",
            "question": "question",
            "response": "judge fails",
            "reference": "reference",
            "contexts": [{"content": "context"}],
        },
    ]

    outcomes = await adapter.evaluate_extended_ragas(
        records,
        evaluators={"semantic_similarity": evaluator},
    )

    assert [item["status"] for item in outcomes] == [
        "scored",
        "failed",
        "not_applicable",
        "failed",
    ]
    assert outcomes[1]["reason_code"] == "missing_reference"
    assert outcomes[2]["reason_code"].startswith("ordinary_answer_metric_invalid")
    assert outcomes[3]["reason_code"] == "judge_exception"
    assert outcomes[3]["exception"] == "RuntimeError"
    # 裸 RuntimeError 现在也会获得一次瞬态重试（judge_exception 类），
    # 因此该样本共被调用两次；两次都失败才落为 failed。
    assert calls == ["answer", "judge fails", "judge fails"]


@pytest.mark.asyncio
async def test_prompt_injection_safety_rubric_does_not_require_reference_or_contexts() -> None:
    received: list[dict[str, Any]] = []

    async def evaluator(sample: dict[str, Any]) -> float:
        received.append(sample)
        return 5.0

    outcomes = await adapter.evaluate_extended_ragas(
        [
            {
                "benchmark_id": "prompt-injection",
                "retrieval_mode": "dense_bm25_graph",
                "status": "succeeded",
                "category": "prompt_injection",
                "expected_refusal": True,
                "expected_response_status": "human_review_required",
                "question": "malicious input",
                "response": "request blocked and routed for review",
                "reference": "",
                "contexts": [],
            }
        ],
        evaluators={"rubrics_score_without_reference": evaluator},
    )

    assert outcomes[0]["status"] == "scored"
    assert outcomes[0]["score"] == 5.0
    assert received == [
        {
            "user_input": "malicious input",
            "response": "request blocked and routed for review",
        }
    ]


def test_cli_accepts_extended_metrics_and_explicit_embedding_model(tmp_path) -> None:
    args = parse_args(
        [
            "--responses-jsonl",
            str(tmp_path / "responses.jsonl"),
            "--metrics",
            "answer_relevancy",
            "semantic_similarity",
            "noise_sensitivity",
            "rubrics_score_with_reference",
            "rubrics_score_without_reference",
            "--embedding-model",
            "text-embedding-v4",
        ]
    )

    assert args.embedding_model == "text-embedding-v4"
    assert args.metrics == [
        "answer_relevancy",
        "semantic_similarity",
        "noise_sensitivity",
        "rubrics_score_with_reference",
        "rubrics_score_without_reference",
    ]


@pytest.mark.asyncio
async def test_legacy_ragas_metric_uses_category_policy_before_judge() -> None:
    calls = 0

    async def evaluator(_sample: dict[str, Any]) -> float:
        nonlocal calls
        calls += 1
        return 1.0

    outcomes = await adapter.evaluate_faithfulness(
        [
            {
                "benchmark_id": "safe-refusal",
                "retrieval_mode": "dense",
                "status": "succeeded",
                "category": "authorization_filtered",
                "question": "private",
                "response": "controlled refusal",
                "contexts": [],
            }
        ],
        evaluator=evaluator,
    )

    assert calls == 0
    assert outcomes[0]["status"] == "not_applicable"
    assert outcomes[0]["reason_code"] == (
        "ordinary_answer_metric_invalid_for_authorization_filtered"
    )
    assert outcomes[0]["category"] == "authorization_filtered"


@pytest.mark.asyncio
async def test_formal_cli_attaches_identity_and_preserves_deterministic_outcome(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.scripts import run_ragas_benchmark

    responses = tmp_path / "responses.jsonl"
    record = {
        "benchmark_id": "case-1",
        "retrieval_mode": "dense",
        "status": "succeeded",
        "category": "fully_answerable",
        "question": "question",
        "response": "answer",
        "reference": "reference",
        "contexts": [{"content": "context"}],
        "expected_response_status": "answered",
        "response_status": "answered",
        "expected_evidence_states": ["direct_evidence"],
        "stages": {"qualification": {"evidence_states": ["direct_evidence"]}},
    }
    responses.write_text(json.dumps(record) + "\n", encoding="utf-8")

    async def failed_judge(_sample: dict[str, Any]) -> float:
        raise TimeoutError("private timeout")

    monkeypatch.setattr(
        run_ragas_benchmark,
        "build_openai_faithfulness_evaluator",
        lambda: failed_judge,
    )
    monkeypatch.setattr(
        run_ragas_benchmark,
        "validate_ragas_configuration",
        lambda: {
            "source": "eval_openai",
            "model": "judge-v1",
            "base_url": "https://private.invalid/v1",
        },
    )
    output = tmp_path / "scores.jsonl"
    await run_ragas_benchmark.run(
        SimpleNamespace(
            responses_jsonl=responses,
            output_jsonl=output,
            metrics=["faithfulness", "evidence_gate_contract"],
            concurrency=1,
            variant_id="observed",
        )
    )

    outcomes = [json.loads(line) for line in output.read_text().splitlines()]
    by_metric = {item["metric"]: item for item in outcomes}
    assert by_metric["faithfulness"]["status"] == "failed"
    assert by_metric["faithfulness"]["metric_family"] == "ragas"
    assert by_metric["faithfulness"]["judge_identity"] == {
        "model": "judge-v1",
        "provider": "eval_openai",
    }
    assert by_metric["evidence_gate_contract"]["status"] == "scored"
    assert by_metric["evidence_gate_contract"]["metric_family"] == "deterministic"
    assert "judge_identity" not in by_metric["evidence_gate_contract"]
    assert all(item["variant_id"] == "observed" for item in outcomes)
    assert all(len(item["response_snapshot_sha256"]) == 64 for item in outcomes)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["families"] == {
        "deterministic": {"failed": 0, "not_applicable": 0, "scored": 1, "unsupported": 0},
        "ragas": {"failed": 1, "not_applicable": 0, "scored": 0, "unsupported": 0},
    }
    assert "overall_score" not in summary


@pytest.mark.asyncio
async def test_cli_rejects_unsupported_metric_before_loading_judge(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.scripts import run_ragas_benchmark

    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        json.dumps(
            {
                "benchmark_id": "case-1",
                "retrieval_mode": "dense",
                "status": "succeeded",
                "category": "fully_answerable",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_ragas_benchmark,
        "validate_ragas_configuration",
        lambda: pytest.fail("Judge configuration must not load"),
    )

    with pytest.raises(
        adapter.RagasConfigurationError,
        match="unsupported_metric:aspect_critic",
    ):
        await run_ragas_benchmark.run(
            SimpleNamespace(
                responses_jsonl=responses,
                output_jsonl=tmp_path / "scores.jsonl",
                metrics=["aspect_critic"],
                concurrency=1,
                variant_id="observed",
            )
        )
