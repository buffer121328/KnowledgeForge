"""Acceptance tests for replay-safe evidence-gate runner artifacts."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from domain.evidence import (
    AnswerCitation,
    AnswerClaim,
    EvidenceAssessment,
    EvidenceReasonCode,
    EvidenceState,
    GroundingResult,
    QAResponseStatus,
    StructuredAnswer,
)
from domain.knowledge import QAResult, QueryIntent, RetrievedContext
from domain.trace import TraceRecorder
from evaluation.benchmarks.benchmark import BenchmarkSample
from evaluation.benchmarks.candidate_stage_capture import CandidateStageRecorder
from evaluation.evidence_gate.benchmark import EvidenceGateCase
from evaluation.runner import BenchmarkRunner


@pytest.mark.asyncio
async def test_runner_persists_bounded_stages_identity_and_resources(tmp_path: Path) -> None:
    trace = TraceRecorder(trace_id="trace-eval")
    trace.record("retrieval.vector", metadata={"available": True, "result_count": 1})
    trace.record("retrieval.bm25", status="failed", metadata={"available": False})
    trace.record("retrieval.graph", metadata={"available": True, "result_count": 0})
    assessment = EvidenceAssessment(
        states=(EvidenceState.DIRECT_EVIDENCE,),
        response_status=QAResponseStatus.ANSWERED,
        reason_codes=(EvidenceReasonCode.DIRECT_SUPPORT,),
        evaluated_context_ids=("ctx-1",),
        supporting_context_ids=("ctx-1",),
        policy_version="evidence-v1",
        calibration_version="threshold-v1",
    )
    structured = StructuredAnswer(
        status=QAResponseStatus.ANSWERED,
        answer="bounded answer",
        claims=(AnswerClaim("claim-1", "private claim text", ("cite-1",)),),
        citations=(AnswerCitation("cite-1", "ctx-1", "secret.pdf", "secret source text"),),
        schema_version="structured-answer-v1",
    )
    grounding = GroundingResult(
        passed=True,
        policy_version="grounding-v1",
        accepted_claim_ids=("claim-1",),
    )

    async def executor(**_: object) -> QAResult:
        return QAResult(
            question="private question",
            answer="bounded answer",
            contexts=[
                RetrievedContext(
                    content="secret source text",
                    source="secret.pdf",
                    score=0.8,
                    retrieval_type="dense",
                    metadata={
                        "source_document_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                        "context_id": "ctx-1",
                    },
                )
            ],
            intent=QueryIntent.FACTOID,
            confidence=0.9,
            trace=trace.build(),
            response_status=QAResponseStatus.ANSWERED,
            evidence_assessment=assessment,
            structured_answer=structured,
            grounding_result=grounding,
        )

    sample = BenchmarkSample(
        id="eg-1",
        question="private question",
        reference="private reference",
        required_doc_ids=("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",),
        reference_context_ids=("ctx-1",),
        evidence=(),
        category="single_document_fact",
        expected_refusal=False,
    )
    object.__setattr__(sample, "expected_response_status", "answered")
    object.__setattr__(sample, "expected_evidence_states", ("direct_evidence",))
    object.__setattr__(sample, "expected_citation_context_ids", ("ctx-1",))

    run = await BenchmarkRunner(qa_executor=executor, results_root=tmp_path).run(
        [sample],
        tenant_id="tenant-secret",
        retrieval_modes=("dense",),
        run_metadata={
            "dataset_version": "evidence-gates-v1",
            "dataset_manifest_sha256": "a" * 64,
            "answer_model": "answer-v1",
            "embedding_model": "embedding-v1",
            "retrieval_config_id": "retrieval-v1",
            "threshold_version": "threshold-v1",
            "candidate_budget_version": "candidate-v1",
            "reranker_mode": "disabled",
            "reranker_version": "disabled-v1",
            "structured_answer_schema_version": "structured-answer-v1",
            "grounding_policy_version": "grounding-v1",
        },
        smoke_threshold=1,
    )

    record = run.records[0]
    assert record["response_status"] == "answered"
    assert record["evidence_state"] == "direct_evidence"
    assert record["stages"]["retrieval"]["branch_availability"] == {
        "bm25": "unavailable",
        "dense": "available",
        "graph": "available",
    }
    assert record["stages"]["generation"] == {
        "schema_version": "structured-answer-v1",
        "claim_count": 1,
        "material_claim_count": 1,
        "citation_count": 1,
        "citation_validation_passed": True,
    }
    assert record["stages"]["grounding"]["passed"] is True
    assert record["resources"]["latency_ms"] >= 0
    assert record["resources"]["cpu_ms"] >= 0
    assert record["resources"]["rss_peak_mb"] >= 0
    assert record["run_identity"]["dataset_version"] == "evidence-gates-v1"

    replay_json = json.dumps(
        {
            "stages": record["stages"],
            "run_identity": record["run_identity"],
            "resources": record["resources"],
        }
    )
    assert "private question" not in replay_json
    assert "private claim text" not in replay_json
    assert "secret source text" not in replay_json
    assert "tenant-secret" not in replay_json


@pytest.mark.asyncio
async def test_runner_preserves_success_for_evidence_gate_case_without_inline_reference(
    tmp_path: Path,
) -> None:
    """Formal cases add their reviewed reference only after the QA snapshot is frozen."""

    async def executor(**kwargs: object) -> QAResult:
        return QAResult(
            question=str(kwargs["question"]),
            answer="已回答",
            contexts=[],
            intent=QueryIntent.FACTOID,
            confidence=0.9,
        )

    sample = EvidenceGateCase(
        id="eg-reference-later",
        question="正式评测问题",
        category="answerable",
        expected_response_status="answered",
        expected_evidence_states=("direct_evidence",),
        expected_reason_codes=("direct_support",),
        expected_citation_context_ids=(),
        expected_branch_availability={"dense": "available", "bm25": "available", "graph": "available"},
    )

    run = await BenchmarkRunner(qa_executor=executor, results_root=tmp_path).run(
        [sample],
        tenant_id="tenant-eval",
        retrieval_modes=("dense",),
        smoke_threshold=1,
    )

    assert run.records[0]["status"] == "succeeded"
    assert run.records[0]["reference"] == ""


@pytest.mark.asyncio
async def test_runner_writes_identity_only_candidate_stage_artifact(tmp_path: Path) -> None:
    """Candidate stages are separate, bounded, owner-only, and hash-bound."""

    async def executor(**kwargs: object) -> QAResult:
        recorder = kwargs["evaluation_candidate_recorder"]
        assert isinstance(recorder, CandidateStageRecorder)
        context = RetrievedContext(
            content="raw candidate text must stay out of candidate artifact",
            source="private-source.pdf",
            score=0.75,
            retrieval_type="vector",
            metadata={
                "source_document_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "context_id": "ctx-private",
                "content_sha256": "b" * 64,
            },
        )
        recorder.record("dense", [context], status="executed")
        recorder.record(
            "bm25", [], status="executed_empty", reason_code="empty"
        )
        recorder.record(
            "graph", [], status="not_executed", reason_code="branch_not_configured"
        )
        recorder.record("fused", [context], status="executed")
        recorder.record(
            "reranked", [], status="not_executed", reason_code="reranker_disabled"
        )
        return QAResult(
            question=str(kwargs["question"]),
            answer="answer",
            contexts=[context],
            intent=QueryIntent.FACTOID,
            confidence=0.75,
        )

    sample = BenchmarkSample(
        id="candidate-case",
        question="private evaluation question",
        reference="private reference",
        required_doc_ids=("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",),
        reference_context_ids=("ctx-private",),
        evidence=(),
        category="single_document_fact",
        expected_refusal=False,
    )
    run = await BenchmarkRunner(
        qa_executor=executor,
        results_root=tmp_path,
        candidate_budget=2,
        candidate_k_values=(1, 2),
    ).run(
        [sample],
        tenant_id="tenant-secret",
        retrieval_modes=("dense",),
        run_metadata={"diagnostic_variant": "observed"},
        smoke_threshold=1,
    )

    assert stat.S_IMODE(run.candidate_stages_path.stat().st_mode) == 0o600
    candidate_text = run.candidate_stages_path.read_text(encoding="utf-8")
    assert "private evaluation question" not in candidate_text
    assert "raw candidate text" not in candidate_text
    assert "private-source.pdf" not in candidate_text
    rows = [json.loads(line) for line in candidate_text.splitlines()]
    assert [row["stage"] for row in rows] == [
        "dense",
        "bm25",
        "graph",
        "fused",
        "reranked",
    ]
    assert rows[0]["candidates"] == [
        {
            "branch": "dense",
            "content_sha256": hashlib.sha256(
                b"raw candidate text must stay out of candidate artifact"
            ).hexdigest(),
            "context_id": "ctx-private",
            "rank": 1,
            "score": 0.75,
            "source_document_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        }
    ]
    metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    assert metadata["candidate_stages"] == {
        "artifact": "candidate-stages.jsonl",
        "candidate_budget": 2,
        "k_values": [1, 2],
        "records": 5,
        "schema_version": "candidate-stage-capture-v1",
        "sha256": run.candidate_stages_sha256,
    }
