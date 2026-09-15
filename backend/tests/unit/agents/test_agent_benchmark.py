from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.trace import TraceRecorder
from domain.knowledge import QAResult, QueryIntent
from evaluation.benchmarks.agent_benchmark import (
    AgentBenchmarkValidationError,
    evaluate_agent_trace,
    load_agent_benchmark,
)


FIXTURE = Path(__file__).parents[3] / "evaluation" / "data" / "agent-benchmark" / "agent_benchmark.jsonl"


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "id": "agent-test-001",
        "user_input": "请回答已入库年报中的资本充足率。",
        "reference": "应基于已入库证据回答。",
        "reference_topics": ["2021 年 A 股年报", "资本充足率"],
        "expected_refusal": False,
        "expected_operations": ["query.classify", "query.rewrite", "qa.completed"],
    }
    record.update(overrides)
    return record


def test_checked_in_agent_benchmark_is_valid() -> None:
    dataset = load_agent_benchmark(FIXTURE)

    assert dataset
    assert dataset[0].reference_topics
    assert "reasoning_steps" not in dataset[0].__dict__


def test_agent_benchmark_rejects_unknown_operation(tmp_path: Path) -> None:
    path = tmp_path / "agent.jsonl"
    path.write_text(json.dumps(_record(expected_operations=["made.up.operation"])) + "\n", encoding="utf-8")

    with pytest.raises(AgentBenchmarkValidationError, match="unknown operation"):
        load_agent_benchmark(path)


def test_agent_benchmark_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "agent.jsonl"
    path.write_text(
        "\n".join(json.dumps(_record()) for _ in range(2)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentBenchmarkValidationError, match="duplicate id"):
        load_agent_benchmark(path)


def _result(*, refused: bool, operations: list[str]) -> QAResult:
    recorder = TraceRecorder(trace_id="trace-test")
    for operation in operations:
        recorder.record(operation)
    return QAResult(
        question="question",
        answer="answer",
        contexts=[],
        intent=QueryIntent.FACTOID,
        confidence=0.0,
        degradation_code="insufficient_verified_evidence" if refused else None,
        trace=recorder.build(),
    )


def test_trace_evaluator_passes_matching_contract() -> None:
    sample = load_agent_benchmark(FIXTURE)[0]
    result = _result(refused=False, operations=list(sample.expected_operations))

    evaluation = evaluate_agent_trace(sample, result)

    assert evaluation["status"] == "passed"
    assert evaluation["missing_operations"] == []


def test_trace_evaluator_reports_missing_operation() -> None:
    sample = load_agent_benchmark(FIXTURE)[0]
    result = _result(refused=False, operations=["qa.completed"])

    evaluation = evaluate_agent_trace(sample, result)

    assert evaluation["status"] == "failed"
    assert evaluation["missing_operations"]
    assert "query.classify" in evaluation["missing_operations"]


def test_trace_evaluator_marks_missing_trace_invalid() -> None:
    sample = load_agent_benchmark(FIXTURE)[0]
    result = _result(refused=False, operations=[])
    result.trace = None

    evaluation = evaluate_agent_trace(sample, result)

    assert evaluation["status"] == "invalid_trace"
    assert evaluation["reason_code"] == "missing_trace"
