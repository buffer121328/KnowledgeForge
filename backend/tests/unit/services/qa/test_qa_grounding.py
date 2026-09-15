"""Deterministic grounding tests for structured QA claims."""

from __future__ import annotations

import pytest

from services.qa.grounding import GroundingVerifier
from domain.evidence import (
    AnswerCitation,
    AnswerClaim,
    EvidenceReasonCode,
    MissingInformation,
    QAResponseStatus,
    StructuredAnswer,
)
from domain.knowledge import RetrievedContext


def context(content: str) -> RetrievedContext:
    return RetrievedContext(
        content=content,
        source="policy.md",
        score=0.9,
        retrieval_type="vector",
        metadata={"context_id": "ctx_1"},
    )


def answer(claim_text: str, *, context_id: str = "ctx_1") -> StructuredAnswer:
    return StructuredAnswer(
        status=QAResponseStatus.ANSWERED,
        answer=claim_text,
        claims=(
            AnswerClaim(
                claim_id="claim_1",
                text=claim_text,
                citation_ids=("cite_1",),
            ),
        ),
        citations=(
            AnswerCitation(citation_id="cite_1", context_id=context_id),
        ),
        schema_version="structured-answer-v1",
    )


def test_grounding_accepts_supported_number_date_version_and_clause() -> None:
    verifier = GroundingVerifier(policy_version="grounding-v1")
    structured = answer("v2版本预算为100万元，自2026年8月1日按第12条生效。")

    result = verifier.verify(
        structured,
        {"ctx_1": context("v2版本预算为100万元，自2026年8月1日按第12条生效。")},
    )

    assert result.passed is True
    assert result.accepted_claim_ids == ("claim_1",)


def test_grounding_rejects_unknown_current_run_context() -> None:
    result = GroundingVerifier().verify(
        answer("预算为100万元。", context_id="ctx_stale"),
        {"ctx_1": context("预算为100万元。")},
    )

    assert result.passed is False
    assert EvidenceReasonCode.UNKNOWN_CITATION in result.reason_codes


def test_grounding_rejects_unsupported_critical_value() -> None:
    structured = answer("预算为120万元。")

    result = GroundingVerifier().verify(
        structured,
        {"ctx_1": context("预算为100万元。")},
    )

    assert result.passed is False
    assert EvidenceReasonCode.UNSUPPORTED_CRITICAL_VALUE in result.reason_codes


def test_grounding_rejects_unsupported_named_person() -> None:
    structured = answer("负责人是李明经理。")

    result = GroundingVerifier().verify(
        structured,
        {"ctx_1": context("负责人是王强经理。")},
    )

    assert result.passed is False
    assert EvidenceReasonCode.UNSUPPORTED_CRITICAL_VALUE in result.reason_codes


def test_supported_subset_removes_rejected_claims_and_returns_partial() -> None:
    structured = StructuredAnswer(
        status=QAResponseStatus.ANSWERED,
        answer="预算为100万元，负责人是李明经理。",
        claims=(
            AnswerClaim(
                claim_id="claim_budget",
                text="预算为100万元。",
                citation_ids=("cite_1",),
            ),
            AnswerClaim(
                claim_id="claim_owner",
                text="负责人是李明经理。",
                citation_ids=("cite_1",),
            ),
        ),
        citations=(AnswerCitation(citation_id="cite_1", context_id="ctx_1"),),
        schema_version="structured-answer-v1",
    )
    verifier = GroundingVerifier()
    result = verifier.verify(
        structured,
        {"ctx_1": context("预算为100万元，负责人是王强经理。")},
    )

    filtered = verifier.supported_subset(structured, result)

    assert filtered is not None
    assert filtered.status is QAResponseStatus.PARTIALLY_ANSWERED
    assert [claim.claim_id for claim in filtered.claims] == ["claim_budget"]
    assert filtered.missing_information == (
        MissingInformation(
            field="unsupported_claims",
            description="部分结论缺少可验证的当前证据。",
        ),
    )


def test_supported_subset_returns_none_when_every_claim_is_rejected() -> None:
    structured = answer("预算为120万元。")
    verifier = GroundingVerifier()
    result = verifier.verify(structured, {"ctx_1": context("预算为100万元。")})

    assert verifier.supported_subset(structured, result) is None


@pytest.mark.parametrize(
    ("claim_text", "evidence_text"),
    [
        ("该政策自2026年9月1日生效。", "该政策自2026年8月1日生效。"),
        ("当前适用v3版本。", "当前适用v2版本。"),
        ("应遵守SEC-102条款。", "应遵守SEC-101条款。"),
    ],
    ids=["unsupported-date", "unsupported-version", "unsupported-policy-id"],
)
def test_grounding_rejects_unsupported_date_version_and_policy_identifier(
    claim_text: str,
    evidence_text: str,
) -> None:
    result = GroundingVerifier().verify(
        answer(claim_text),
        {"ctx_1": context(evidence_text)},
    )

    assert result.passed is False
    assert EvidenceReasonCode.UNSUPPORTED_CRITICAL_VALUE in result.reason_codes
