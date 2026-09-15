from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.trace import TRACE_OPERATIONS, TraceRecorder


def test_trace_serialization_is_bounded_and_json_safe() -> None:
    recorder = TraceRecorder(trace_id="trace-test")
    recorder.record(
        "retrieval.vector",
        metadata={
            "retrieval_mode": "vector",
            "contexts_count": 3,
            "question": "不要写入这段问题",
            "answer": "不要写入这段回答",
            "exception_message": "不要写入异常消息",
            "absolute_path": "/Users/cheng/secret.txt",
            "api_key": "secret-token",
        },
        duration_ms=1.25,
    )

    payload = recorder.build().to_dict()
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["trace_id"] == "trace-test"
    assert payload["schema_version"] == "v1"
    assert payload["events"][0]["operation"] == "retrieval.vector"
    assert payload["events"][0]["metadata"] == {
        "contexts_count": 3,
        "retrieval_mode": "vector",
    }
    assert "不要写入" not in serialized
    assert "/Users/cheng/secret.txt" not in serialized
    assert "secret-token" not in serialized


def test_trace_recorder_rejects_unknown_operations() -> None:
    recorder = TraceRecorder()

    with pytest.raises(ValueError, match="unknown trace operation"):
        recorder.record("tool.autonomous_call")


def test_trace_event_ids_are_unique_and_ordered() -> None:
    recorder = TraceRecorder()
    recorder.record("query.classify")
    recorder.record("query.rewrite")

    events = recorder.build().events

    assert [event.operation for event in events] == ["query.classify", "query.rewrite"]
    assert events[0].event_id != events[1].event_id
    assert all(event.duration_ms >= 0 for event in events)
    assert TRACE_OPERATIONS.issuperset({"query.classify", "query.rewrite"})


@pytest.mark.asyncio
async def test_capture_async_records_failure_type_without_message() -> None:
    recorder = TraceRecorder()

    async def fail() -> None:
        raise RuntimeError("raw failure detail")

    with pytest.raises(RuntimeError, match="raw failure detail"):
        await recorder.capture_async("retrieval.graph", fail)

    event = recorder.build().events[0]
    assert event.status == "failed"
    assert event.exception_type == "RuntimeError"
    assert event.metadata == {}
    assert event.error_message is None

@pytest.mark.asyncio
async def test_qa_agent_attaches_real_hybrid_trace_without_public_payloads() -> None:
    from unittest.mock import AsyncMock

    from agents.qa_agent import QAAgent
    from services.safety.pipeline import QASafetyPipeline

    from tests.qa_fakes import build_test_qa_agent_dependencies

    agent = QAAgent(
        dependencies=build_test_qa_agent_dependencies(
            safety_pipeline=QASafetyPipeline()
        )
    )
    agent._vector_retrieve = AsyncMock(return_value=[])
    agent._graph_retrieve = AsyncMock(return_value=[])

    result = await agent.answer("真实问题", tenant_id="tenant-1", retrieval_mode="hybrid")

    assert result.trace is not None
    operations = [event.operation for event in result.trace.events]
    assert operations[0] == "safety.validate_question"
    assert "retrieval.vector" in operations
    assert "retrieval.graph" in operations
    assert "qa.completed" in operations
    serialized = json.dumps(result.trace.to_dict(), ensure_ascii=False)
    assert "真实问题" not in serialized
    assert "tenant-1" not in serialized


@pytest.mark.asyncio
async def test_vector_trace_does_not_claim_graph_retrieval() -> None:
    from unittest.mock import AsyncMock

    from agents.qa_agent import QAAgent
    from services.safety.pipeline import QASafetyPipeline

    from tests.qa_fakes import build_test_qa_agent_dependencies

    agent = QAAgent(
        dependencies=build_test_qa_agent_dependencies(
            safety_pipeline=QASafetyPipeline()
        )
    )
    agent._vector_retrieve = AsyncMock(return_value=[])
    agent._graph_retrieve = AsyncMock(side_effect=AssertionError("graph must not run"))

    result = await agent.answer("vector question", retrieval_mode="vector")

    assert result.trace is not None
    operations = [event.operation for event in result.trace.events]
    assert "retrieval.vector" in operations
    assert "retrieval.graph" not in operations

@pytest.mark.asyncio
async def test_offline_runner_preserves_safe_trace_artifact(tmp_path) -> None:
    from evaluation.benchmarks.benchmark import load_benchmark
    from evaluation.runner import BenchmarkRunner
    from domain.knowledge import QAResult, QueryIntent

    benchmark = load_benchmark(Path(__file__).parents[3] / "evaluation" / "data" / "company-demo" / "benchmark.jsonl")
    recorder = TraceRecorder(trace_id="trace-runner")
    recorder.record("qa.completed")

    async def executor(**_kwargs):
        return QAResult(
            question="private question",
            answer="private answer",
            contexts=[],
            intent=QueryIntent.FACTOID,
            confidence=0.5,
            trace=recorder.build(),
        )

    run = await BenchmarkRunner(qa_executor=executor, results_root=tmp_path).run(
        benchmark[:1],
        tenant_id="tenant-1",
        retrieval_modes=("vector",),
    )

    assert run.records[0]["trace"]["trace_id"] == "trace-runner"
    assert "private question" not in run.responses_path.read_text(encoding="utf-8")


def test_public_question_response_does_not_expose_internal_trace() -> None:
    from api.schemas import QuestionResponse

    assert "trace" not in QuestionResponse.model_fields


def test_evidence_and_reranking_trace_metadata_is_bounded() -> None:
    recorder = TraceRecorder(trace_id="trace-evidence")
    recorder.record(
        "qa.rerank_completed",
        metadata={
            "mode": "disabled",
            "status": "disabled",
            "contexts_count": 3,
            "scored_count": 0,
            "endpoint": "https://private.example",
            "question": "private question",
        },
    )
    recorder.record(
        "qa.evidence_qualified",
        metadata={
            "evidence_state": "direct_evidence",
            "response_status": "answered",
            "evaluated_count": 3,
            "accepted_count": 2,
            "policy_version": "evidence-v1",
            "calibration_version": "calibration-v1",
            "source_text": "private source",
        },
    )

    payload = recorder.build().to_dict()

    assert payload["events"][0]["metadata"] == {
        "contexts_count": 3,
        "mode": "disabled",
        "scored_count": 0,
        "status": "disabled",
    }
    assert payload["events"][1]["metadata"] == {
        "accepted_count": 2,
        "calibration_version": "calibration-v1",
        "evaluated_count": 3,
        "evidence_state": "direct_evidence",
        "policy_version": "evidence-v1",
        "response_status": "answered",
    }


def test_structured_generation_and_grounding_trace_metadata_is_bounded() -> None:
    recorder = TraceRecorder(trace_id="trace-grounding")
    recorder.record(
        "qa.structured_generated",
        metadata={
            "status": "succeeded",
            "result_count": 2,
            "accepted_count": 1,
            "schema_version": "structured-answer-v1",
            "claim_text": "private generated claim",
            "provider_payload": "private provider payload",
        },
    )
    recorder.record(
        "qa.grounding_verified",
        metadata={
            "grounding_passed": False,
            "accepted_count": 1,
            "rejected_count": 1,
            "response_status": "partially_answered",
            "policy_version": "grounding-v1",
            "reason_code": "unsupported_critical_value",
            "source_text": "private source text",
        },
    )

    payload = recorder.build().to_dict()

    assert payload["events"][0]["metadata"] == {
        "accepted_count": 1,
        "result_count": 2,
        "schema_version": "structured-answer-v1",
        "status": "succeeded",
    }
    assert payload["events"][1]["metadata"] == {
        "accepted_count": 1,
        "grounding_passed": False,
        "policy_version": "grounding-v1",
        "reason_code": "unsupported_critical_value",
        "rejected_count": 1,
        "response_status": "partially_answered",
    }
