"""Compatibility tests for evidence-aware QA cache values."""

from __future__ import annotations

import pytest
from infrastructure.evidence_gate_configuration import EffectiveEvidenceGateConfiguration

from agents.qa_agent import QAAgent
from services.safety.pipeline import QASafetyPipeline
from domain.evidence import (
    EvidenceAssessment,
    EvidenceReasonCode,
    EvidenceState,
    QAResponseStatus,
)
from domain.knowledge import QAResult, QueryIntent, RetrievedContext
from tests.qa_fakes import build_test_qa_agent_dependencies


def cached_payload() -> dict:
    return {
        "question": "预算是多少",
        "answer": "预算是100万元。",
        "contexts": [
            {
                "content": "预算是100万元。",
                "source": "policy.md",
                "score": 0.9,
                "retrieval_type": "vector",
                "metadata": {"context_id": "ctx_1"},
            }
        ],
        "intent": "factoid",
        "confidence": 0.9,
        "reasoning_steps": ["cached"],
    }


def make_agent() -> QAAgent:
    return QAAgent(
        dependencies=build_test_qa_agent_dependencies(
            safety_pipeline=QASafetyPipeline()
        )
    )


@pytest.mark.asyncio
async def test_legacy_cache_is_readable_when_gate_is_off(monkeypatch) -> None:
    monkeypatch.setattr("agents.qa_agent.settings.qa_evidence_gate_mode", "off")
    monkeypatch.setattr(
        "agents.qa_cache.get_effective_evidence_gate_configuration",
        lambda: EffectiveEvidenceGateConfiguration(mode="off", revision=0, calibration_version="test", source="environment_default"),
    )

    result = await make_agent()._safe_cached_result(cached_payload(), tenant_id=None)

    assert result is not None
    assert result.answer == "预算是100万元。"
    assert result.evidence_assessment is None
    assert result.response_status is None


@pytest.mark.asyncio
async def test_legacy_cache_becomes_miss_when_gate_is_enforced(monkeypatch) -> None:
    monkeypatch.setattr("agents.qa_agent.settings.qa_evidence_gate_mode", "enforce")
    monkeypatch.setattr(
        "agents.qa_cache.get_effective_evidence_gate_configuration",
        lambda: EffectiveEvidenceGateConfiguration(mode="enforce", revision=0, calibration_version="test", source="environment_default"),
    )

    result = await make_agent()._safe_cached_result(cached_payload(), tenant_id=None)

    assert result is None


@pytest.mark.asyncio
async def test_evidence_aware_cache_round_trip(monkeypatch) -> None:
    monkeypatch.setattr("agents.qa_agent.settings.qa_evidence_gate_mode", "enforce")
    monkeypatch.setattr(
        "agents.qa_cache.get_effective_evidence_gate_configuration",
        lambda: EffectiveEvidenceGateConfiguration(mode="enforce", revision=0, calibration_version="test", source="environment_default"),
    )
    context = RetrievedContext(
        content="预算是100万元。",
        source="policy.md",
        score=0.9,
        retrieval_type="vector",
        metadata={"context_id": "ctx_1"},
    )
    assessment = EvidenceAssessment(
        states=(EvidenceState.DIRECT_EVIDENCE,),
        response_status=QAResponseStatus.ANSWERED,
        reason_codes=(EvidenceReasonCode.DIRECT_SUPPORT,),
        evaluated_context_ids=("ctx_1",),
        supporting_context_ids=("ctx_1",),
        policy_version="evidence-v1",
        calibration_version="calibration-v1",
    )
    original = QAResult(
        question="预算是多少",
        answer="预算是100万元。",
        contexts=[context],
        intent=QueryIntent.FACTOID,
        confidence=0.9,
        response_status=QAResponseStatus.ANSWERED,
        evidence_assessment=assessment,
    )

    serialized = QAAgent._serialize_cache_result(original)
    restored = await make_agent()._safe_cached_result(serialized, tenant_id=None)

    assert restored is not None
    assert restored.response_status is QAResponseStatus.ANSWERED
    assert restored.evidence_assessment == assessment
