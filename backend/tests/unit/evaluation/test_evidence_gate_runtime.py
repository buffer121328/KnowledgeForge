"""Acceptance tests for evidence-gate runtime preflight and blocked measurements."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.evidence_gate.preparation import prepare_evidence_gate_candidates
from evaluation.evidence_gate.runtime import assess_evidence_gate_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVIDENCE_GATE_ROOT = PROJECT_ROOT / "backend/evaluation/data/evidence-gates"
COMPANY_DEMO_ROOT = PROJECT_ROOT / "backend/evaluation/data/company-demo"


def test_candidate_stage_artifact_is_aggregated_for_positive_evidence_cases(tmp_path: Path) -> None:
    from evaluation.evidence_gate.runtime import _diagnostic_inputs

    stages = []
    for stage in ("dense", "bm25", "graph", "fused", "reranked"):
        stages.append({
            "benchmark_id": "case-1",
            "stage": stage,
            "status": "executed",
            "candidate_budget": 8,
            "candidates": [{
                "rank": 1,
                "source_document_id": "doc-1",
                "context_id": "doc-1#chunk-1",
                "content_sha256": "a" * 64,
                "branch": "dense",
            }],
        })
    path = tmp_path / "candidate-stages.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in stages) + "\n", encoding="utf-8")
    run = SimpleNamespace(
        candidate_stages_path=path,
        records=[{
            "benchmark_id": "case-1",
            "category": "fully_answerable",
            "status": "failed",
            "expected_source_document_ids": ["doc-1"],
            "expected_evidence_context_ids": ["doc-1#chunk-1"],
        }],
    )

    diagnostic = _diagnostic_inputs(run)

    assert diagnostic["retrieval_summary"]["stages"]["dense"]["exact_evidence"]["mean_recall_at_k"]["5"] == 1.0
    assert diagnostic["retrieval_summary"]["stages"]["fused"]["source_document"]["mean_mrr"] == 1.0


def test_diagnostic_inputs_does_not_fail_for_runs_without_positive_retrieval_cases(tmp_path: Path) -> None:
    from evaluation.evidence_gate.runtime import _diagnostic_inputs

    path = tmp_path / "candidate-stages.jsonl"
    path.write_text("", encoding="utf-8")
    run = SimpleNamespace(candidate_stages_path=path, records=[{
        "benchmark_id": "case-1",
        "category": "completely_unanswerable",
        "status": "succeeded",
    }])

    diagnostic = _diagnostic_inputs(run)

    assert diagnostic["retrieval_summary"]["status"] == "not_applicable"
    assert diagnostic["retrieval_summary"]["reason_code"] == "no_positive_retrieval_cases"


@pytest.mark.asyncio
async def test_bm25_unavailable_fixture_uses_the_native_unavailable_contract() -> None:
    """The intentional BM25 outage must become an outcome, not a RuntimeError."""
    from services.qa.retrievers import NativeBM25Retriever
    from domain.retrieval import RetrievalRequest, RetrievalScope, RetrievalStatus
    from evaluation.evidence_gate.runtime import _UnavailableDependency

    outcome = await NativeBM25Retriever(_UnavailableDependency("bm25")).retrieve(
        RetrievalRequest(
            question="bounded fixture question",
            rewritten={"queries": ["fixture"]},
            scope=RetrievalScope(tenant_id="org-a"),
        )
    )

    assert outcome.status is RetrievalStatus.UNAVAILABLE
    assert outcome.contexts == []


def test_prompt_injection_fixture_converts_direct_refusal_to_controlled_result() -> None:
    """A safety-policy success is a bounded human-review result, not a failed QA row."""
    from services.safety.qa_checks import QASafetyRefusalError
    from domain.evidence import EvidenceReasonCode, EvidenceState, QAResponseStatus
    from evaluation.evidence_gate.runtime import _controlled_safety_refusal_result

    result = _controlled_safety_refusal_result(
        "unsafe fixture question",
        QASafetyRefusalError("prompt_injection"),
    )

    assert result.response_status is QAResponseStatus.HUMAN_REVIEW_REQUIRED
    assert result.evidence_assessment is not None
    assert result.evidence_assessment.states == (EvidenceState.INVALID_PROVENANCE,)
    assert result.evidence_assessment.reason_codes == (EvidenceReasonCode.PROMPT_INJECTION_DETECTED,)
    assert result.security_actions == ["prompt_injection"]
    assert "unsafe fixture question" not in result.answer


def test_preflight_records_truthful_blockers_without_fake_metrics(tmp_path: Path) -> None:
    status_path = tmp_path / "runtime-preflight.json"

    result = assess_evidence_gate_runtime(
        manifest_path=EVIDENCE_GATE_ROOT / "manifest.json",
        baseline_path=EVIDENCE_GATE_ROOT / "baseline-template.json",
        output_path=status_path,
        gate_modes=("off", "shadow", "enforce"),
        tenant_id="",
        authoring=False,
        env_file=tmp_path / "missing.env",
        runtime_configuration={
            "deepseek_api_key": "",
            "dashscope_api_key": "",
            "neo4j_password": "password",
            "vector_store_type": "chroma",
        },
    )

    assert result["status"] == "blocked_missing_runtime_configuration"
    assert result["measured"] is False
    assert result["quality_metrics"] is None
    assert result["gate_modes"] == ["off", "shadow", "enforce"]
    assert "pending_human_review" in result["reason_codes"]
    assert "missing_runtime_env_file" in result["reason_codes"]
    assert "missing_generation_api_key" in result["reason_codes"]
    assert "missing_embedding_api_key" in result["reason_codes"]
    assert "missing_evaluation_tenant_id" in result["reason_codes"]
    assert "placeholder_graph_credentials" in result["reason_codes"]
    assert json.loads(status_path.read_text(encoding="utf-8")) == result


def test_authoring_preflight_allows_pending_bundle_but_not_missing_services(
    tmp_path: Path,
) -> None:
    prepared_root = tmp_path / "evidence-gates"
    prepare_evidence_gate_candidates(COMPANY_DEMO_ROOT, prepared_root)
    env_file = tmp_path / "configured.env"
    env_file.write_text("# test marker\n", encoding="utf-8")
    result = assess_evidence_gate_runtime(
        manifest_path=prepared_root / "manifest.json",
        baseline_path=prepared_root / "baseline-template.json",
        output_path=tmp_path / "runtime-preflight.json",
        gate_modes=("shadow",),
        tenant_id="tenant-eval",
        authoring=True,
        env_file=env_file,
        runtime_configuration={
            "deepseek_api_key": "configured",
            "dashscope_api_key": "configured",
            "neo4j_password": "configured",
            "vector_store_type": "chroma",
        },
    )

    assert "pending_human_review" not in result["reason_codes"]
    assert result["status"] == "ready_for_service_probe"
    assert result["dataset"]["case_count"] == 100
    assert result["dataset"]["review_status"] == "pending_human_review"


def test_compose_injected_runtime_does_not_require_an_env_file(tmp_path: Path) -> None:
    """Container workers validate injected settings, not a secret env-file path."""
    result = assess_evidence_gate_runtime(
        manifest_path=EVIDENCE_GATE_ROOT / "manifest.json",
        baseline_path=EVIDENCE_GATE_ROOT / "baseline-template.json",
        output_path=tmp_path / "runtime-preflight.json",
        gate_modes=("off",),
        tenant_id="company",
        authoring=False,
        env_file=None,
        runtime_configuration={
            "deepseek_api_key": "configured",
            "dashscope_api_key": "configured",
            "neo4j_password": "configured",
            "vector_store_type": "chroma",
        },
    )

    assert "missing_runtime_env_file" not in result["reason_codes"]


def test_server_fixture_probe_uses_one_controlled_finance_identity_for_both_denials(
    monkeypatch,
) -> None:
    """The Worker owns the finance scope and returns only bounded assertion facts."""
    import asyncio
    from types import SimpleNamespace

    from services.safety.qa_checks import QASafetyRefusalError
    from evaluation.evidence_gate.runtime import execute_frozen_fixture_probes
    from evaluation.fixture_validation import FinanceFixtureIdentityResolver

    class _Vector:
        async def init(self): pass
        async def health_check(self): return True

    class _Graph:
        async def init(self): pass
        async def health_check(self): return True
        async def close(self): pass

    questions = {
        "all": "all_retrieval_branches_unavailable",
        "partial": "bm25_unavailable_dense_graph_available",
        "conflict": "equal_authority_conflicting_documents",
        "admin": "finance_only_user_against_administration_document",
        "hr": "finance_only_user_against_hr_document",
        "absence": "frozen_corpus_absence_check",
        "injection": "prompt_injection_safety_fixture",
    }
    observed_calls: list[dict] = []
    agent_configurations: list[dict] = []

    class _Agent:
        def __init__(self, **kwargs):
            agent_configurations.append(kwargs)

        async def answer(self, question, **kwargs):
            observed_calls.append({"question": question, **kwargs})
            fixture = questions[question]
            if fixture == "all_retrieval_branches_unavailable":
                return SimpleNamespace(response_status="source_unavailable", contexts=[], security_actions=[])
            if fixture == "bm25_unavailable_dense_graph_available":
                return SimpleNamespace(
                    response_status="answered",
                    contexts=[],
                    security_actions=[],
                    trace=SimpleNamespace(
                        events=[
                            SimpleNamespace(operation="retrieval.vector", status="succeeded", metadata={"status": "success"}),
                            SimpleNamespace(operation="retrieval.bm25", status="succeeded", metadata={"status": "unavailable"}),
                            SimpleNamespace(operation="retrieval.graph", status="succeeded", metadata={"status": "empty"}),
                        ]
                    ),
                )
            if fixture == "equal_authority_conflicting_documents":
                return SimpleNamespace(
                    response_status="conflicting_evidence",
                    contexts=[
                        SimpleNamespace(source="catalog", metadata={"doc_id": "doc-a"}),
                        SimpleNamespace(source="graph", metadata={"doc_ids": ["doc-b"]}),
                    ],
                    security_actions=[],
                )
            if fixture == "prompt_injection_safety_fixture":
                raise QASafetyRefusalError("prompt_injection")
            return SimpleNamespace(response_status="insufficient_evidence", contexts=[], security_actions=[])

    monkeypatch.setattr("evaluation.evidence_gate.runtime.VectorStoreService", _Vector)
    monkeypatch.setattr("evaluation.evidence_gate.runtime.KnowledgeGraphService", _Graph)
    monkeypatch.setattr("evaluation.evidence_gate.runtime.NativeBM25Index", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("evaluation.evidence_gate.runtime.QAAgent", _Agent)
    monkeypatch.setattr("evaluation.evidence_gate.runtime.build_qa_agent_dependencies", lambda: object())

    cases = [
        SimpleNamespace(
            question=question,
            required_fixture=fixture,
            expected_source_document_ids=(
                ("doc-a", "doc-b")
                if fixture == "equal_authority_conflicting_documents"
                else (("restricted-doc",) if fixture.startswith("finance_only") else ())
            ),
            expected_reason_codes=(
                ("prompt_injection_detected",)
                if fixture == "prompt_injection_safety_fixture"
                else ()
            ),
        )
        for question, fixture in questions.items()
    ]
    observations = asyncio.run(
        execute_frozen_fixture_probes(
            cases=cases,
            tenant_id="org-a",
            identity_resolver=FinanceFixtureIdentityResolver("finance"),
        )
    )

    assert observations["equal_authority_conflicting_documents"][0]["expected_source_match_count"] == 2
    assert observations["bm25_unavailable_dense_graph_available"][0]["branch_availability"] == {
        "dense": "available",
        "bm25": "unavailable",
        "graph": "available",
    }
    assert observations["finance_only_user_against_hr_document"][0]["protected_context_disclosed"] is False
    assert observations["prompt_injection_safety_fixture"][0] == {
        "response_status": "refused",
        "branch_availability": {},
        "expected_source_match_count": 0,
        "expected_reason_match_count": 1,
        "protected_context_disclosed": False,
        "identity_scope_valid": True,
        "safety_action_applied": True,
    }
    assert sum(config["sparse_index"] is None for config in agent_configurations) == 2
    finance_calls = [call for call in observed_calls if call["question"] in {"admin", "hr"}]
    assert len(finance_calls) == 2
    assert all(call["user_id"] == "evaluation-fixture-finance" for call in finance_calls)
    assert all(call["visible_department_ids"] == ("finance",) for call in finance_calls)
    assert "restricted-doc" not in str(observations)
