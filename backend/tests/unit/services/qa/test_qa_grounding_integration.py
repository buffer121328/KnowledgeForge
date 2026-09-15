"""ATDD integration coverage for structured generation and grounding rollout modes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from infrastructure.evidence_gate_configuration import EffectiveEvidenceGateConfiguration
from infrastructure.retrieval.cross_encoder import (
    CrossEncoderResult,
    CrossEncoderScore,
    RerankStatus,
)

from agents.qa_agent import QAAgent
from services.qa.generation import StructuredGenerationError, StructuredGenerationResult
from domain.evidence import (
    AnswerCitation,
    AnswerClaim,
    EvidenceAssessment,
    EvidenceReasonCode,
    EvidenceState,
    QAResponseStatus,
    StructuredAnswer,
)
from domain.knowledge import RetrievedContext
from domain.retrieval import RetrievalOutcome


def retrieved_context() -> RetrievedContext:
    return RetrievedContext(
        content="培训预算为100万元，负责人是王强经理。",
        source="policy.md",
        score=0.95,
        retrieval_type="vector",
        metadata={
            "context_id": "ctx_1",
            "doc_id": "doc-1",
            "source": "policy.md",
            "tenant_id": "org-001",
        },
    )


def direct_assessment() -> EvidenceAssessment:
    return EvidenceAssessment(
        states=(EvidenceState.DIRECT_EVIDENCE,),
        response_status=QAResponseStatus.ANSWERED,
        reason_codes=(EvidenceReasonCode.DIRECT_SUPPORT,),
        evaluated_context_ids=("ctx_1",),
        supporting_context_ids=("ctx_1",),
        policy_version="evidence-v1",
        calibration_version="calibration-v1",
    )


def structured_generation(*claims: AnswerClaim) -> StructuredGenerationResult:
    evidence = retrieved_context()
    structured = StructuredAnswer(
        status=QAResponseStatus.ANSWERED,
        answer="".join(claim.text for claim in claims),
        claims=claims,
        citations=(
            AnswerCitation(
                citation_id="cite_current",
                context_id="ctx_1",
                source=evidence.source,
                content=evidence.content,
            ),
        ),
        schema_version="structured-answer-v1",
    )
    return StructuredGenerationResult(
        answer=structured,
        authorized_contexts={"ctx_1": evidence},
        reasoning_steps=("structured",),
        attempt_count=1,
    )


def make_agent(monkeypatch: pytest.MonkeyPatch, gate_mode: str) -> QAAgent:
    monkeypatch.setattr("agents.qa_agent.settings.qa_evidence_gate_mode", gate_mode)
    monkeypatch.setattr(
        "agents.qa_cache.get_effective_evidence_gate_configuration",
        lambda: EffectiveEvidenceGateConfiguration(mode=gate_mode, revision=0, calibration_version="test", source="environment_default"),
    )
    monkeypatch.setattr("agents.qa_agent.settings.qa_retrieval_strategy", "dense")
    from tests.qa_fakes import build_test_qa_agent_dependencies

    agent = QAAgent(
        dependencies=build_test_qa_agent_dependencies(
            evidence_qualifier=MagicMock()
        )
    )
    agent.evidence_qualifier.assess.return_value = direct_assessment()
    agent._vector_retrieve = AsyncMock(
        return_value=RetrievalOutcome(contexts=[retrieved_context()])
    )
    agent._generate_answer = AsyncMock(return_value=("legacy answer", ["legacy"]))
    return agent


@pytest.mark.asyncio
async def test_off_mode_does_not_run_structured_generation(monkeypatch) -> None:
    agent = make_agent(monkeypatch, "off")
    agent._generate_structured_answer = AsyncMock()

    result = await agent.answer("培训预算是多少？", tenant_id="org-001")

    assert result.answer == "legacy answer"
    assert result.structured_answer is None
    assert result.grounding_result is None
    agent._generate_structured_answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_applied_reranker_scores_reach_evidence_qualification(monkeypatch) -> None:
    agent = make_agent(monkeypatch, "off")
    context = retrieved_context()
    agent.cross_encoder_reranker.rerank = AsyncMock(
        return_value=CrossEncoderResult(
            contexts=[context],
            status=RerankStatus.APPLIED,
            scored_count=1,
            scores=(CrossEncoderScore(context_id="ctx_1", score=0.92),),
        )
    )

    await agent.answer("培训预算是多少？", tenant_id="org-001")

    assert agent.evidence_qualifier.assess.call_args.kwargs["reranker_scores"] == {
        "ctx_1": 0.92
    }


@pytest.mark.asyncio
async def test_shadow_mode_records_grounding_but_preserves_legacy_answer(monkeypatch) -> None:
    agent = make_agent(monkeypatch, "shadow")
    generated = structured_generation(
        AnswerClaim(
            claim_id="claim_budget",
            text="培训预算为100万元。",
            citation_ids=("cite_current",),
        )
    )
    agent._generate_structured_answer = AsyncMock(return_value=generated)

    result = await agent.answer("培训预算是多少？", tenant_id="org-001")

    assert result.answer == "legacy answer"
    assert result.structured_answer == generated.answer
    assert result.grounding_result is not None and result.grounding_result.passed is True
    operations = [event.operation for event in result.trace.events]
    assert "qa.structured_generated" in operations
    assert "qa.grounding_verified" in operations
    agent._generate_answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_enforce_mode_returns_fully_grounded_structured_answer(monkeypatch) -> None:
    agent = make_agent(monkeypatch, "enforce")
    generated = structured_generation(
        AnswerClaim(
            claim_id="claim_budget",
            text="培训预算为100万元。",
            citation_ids=("cite_current",),
        )
    )
    agent._generate_structured_answer = AsyncMock(return_value=generated)

    result = await agent.answer("培训预算是多少？", tenant_id="org-001")

    assert result.answer == "培训预算为100万元。"
    assert result.response_status is QAResponseStatus.ANSWERED
    assert result.structured_answer == generated.answer
    assert result.grounding_result is not None and result.grounding_result.passed is True
    agent._generate_answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_enforce_mode_filters_unsupported_claim_into_partial_answer(monkeypatch) -> None:
    agent = make_agent(monkeypatch, "enforce")
    agent._generate_structured_answer = AsyncMock(
        return_value=structured_generation(
            AnswerClaim(
                claim_id="claim_budget",
                text="培训预算为100万元。",
                citation_ids=("cite_current",),
            ),
            AnswerClaim(
                claim_id="claim_owner",
                text="负责人是李明经理。",
                citation_ids=("cite_current",),
            ),
        )
    )

    result = await agent.answer("培训预算和负责人是谁？", tenant_id="org-001")

    assert result.answer == "培训预算为100万元。"
    assert result.response_status is QAResponseStatus.PARTIALLY_ANSWERED
    assert [claim.claim_id for claim in result.structured_answer.claims] == ["claim_budget"]
    assert result.grounding_result is not None and result.grounding_result.passed is False
    assert result.grounding_result.rejected_claim_ids == ("claim_owner",)


@pytest.mark.asyncio
async def test_enforce_mode_discards_output_when_all_claims_fail_grounding(monkeypatch) -> None:
    agent = make_agent(monkeypatch, "enforce")
    agent._generate_structured_answer = AsyncMock(
        return_value=structured_generation(
            AnswerClaim(
                claim_id="claim_owner",
                text="负责人是李明经理。",
                citation_ids=("cite_current",),
            )
        )
    )

    result = await agent.answer("负责人是谁？", tenant_id="org-001")

    assert "可验证的证据" in result.answer
    assert result.response_status is QAResponseStatus.INSUFFICIENT_EVIDENCE
    assert result.degradation_code == "grounding_failed"
    assert result.structured_answer is None
    assert result.grounding_result is not None and result.grounding_result.passed is False
    agent._generate_answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_enforce_mode_does_not_trust_unstructured_text_after_parse_failure(
    monkeypatch,
) -> None:
    agent = make_agent(monkeypatch, "enforce")
    agent._generate_structured_answer = AsyncMock(side_effect=StructuredGenerationError())

    result = await agent.answer("培训预算是多少？", tenant_id="org-001")

    assert "可验证的证据" in result.answer
    assert result.response_status is QAResponseStatus.INSUFFICIENT_EVIDENCE
    assert result.degradation_code == "structured_output_invalid"
    assert result.structured_answer is None
    assert result.grounding_result is None
    agent._generate_answer.assert_not_awaited()
    structured_event = next(
        event for event in result.trace.events if event.operation == "qa.structured_generated"
    )
    assert structured_event.status == "failed"
    assert structured_event.metadata == {
        "reason_code": "structured_output_invalid",
        "schema_version": "structured-answer-v1",
        "status": "invalid",
    }


@pytest.mark.asyncio
async def test_enforce_mode_caches_only_fully_grounded_answer(monkeypatch) -> None:
    agent = make_agent(monkeypatch, "enforce")
    agent.qa_cache = AsyncMock()
    agent.qa_cache.get.return_value = None
    agent._generate_structured_answer = AsyncMock(
        return_value=structured_generation(
            AnswerClaim(
                claim_id="claim_budget",
                text="培训预算为100万元。",
                citation_ids=("cite_current",),
            )
        )
    )

    await agent.answer(
        "培训预算是多少？", tenant_id="org-001", user_id="user-1"
    )

    agent.qa_cache.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_grounding_failure_never_enters_reusable_cache(monkeypatch) -> None:
    agent = make_agent(monkeypatch, "enforce")
    agent.qa_cache = AsyncMock()
    agent.qa_cache.get.return_value = None
    agent._generate_structured_answer = AsyncMock(
        return_value=structured_generation(
            AnswerClaim(
                claim_id="claim_owner",
                text="负责人是李明经理。",
                citation_ids=("cite_current",),
            )
        )
    )

    result = await agent.answer(
        "负责人是谁？", tenant_id="org-001", user_id="user-1"
    )

    assert result.response_status is QAResponseStatus.INSUFFICIENT_EVIDENCE
    agent.qa_cache.set.assert_not_awaited()
