"""Domain contract tests for evidence qualification and grounded answers."""

from __future__ import annotations

import pytest

from domain.evidence import (
    AnswerCitation,
    AnswerClaim,
    EvidenceAssessment,
    EvidenceReasonCode,
    EvidenceState,
    GroundingResult,
    MissingInformation,
    QAResponseStatus,
    StructuredAnswer,
)


def test_direct_evidence_assessment_authorizes_generation() -> None:
    assessment = EvidenceAssessment(
        states=(EvidenceState.DIRECT_EVIDENCE,),
        response_status=QAResponseStatus.ANSWERED,
        reason_codes=(EvidenceReasonCode.DIRECT_SUPPORT,),
        evaluated_context_ids=("ctx_1",),
        supporting_context_ids=("ctx_1",),
        policy_version="evidence-v1",
        calibration_version="uncalibrated-v1",
    )

    assert assessment.generation_allowed is True
    assert assessment.primary_state is EvidenceState.DIRECT_EVIDENCE


def test_non_answer_route_cannot_claim_generation_is_allowed() -> None:
    assessment = EvidenceAssessment(
        states=(EvidenceState.RELEVANT_BACKGROUND,),
        response_status=QAResponseStatus.INSUFFICIENT_EVIDENCE,
        reason_codes=(EvidenceReasonCode.BACKGROUND_ONLY,),
        evaluated_context_ids=("ctx_1",),
        policy_version="evidence-v1",
        calibration_version="uncalibrated-v1",
    )

    assert assessment.generation_allowed is False


def test_answer_route_requires_supporting_context() -> None:
    with pytest.raises(ValueError, match="supporting context"):
        EvidenceAssessment(
            states=(EvidenceState.DIRECT_EVIDENCE,),
            response_status=QAResponseStatus.ANSWERED,
            reason_codes=(EvidenceReasonCode.DIRECT_SUPPORT,),
            policy_version="evidence-v1",
            calibration_version="uncalibrated-v1",
        )


def test_assessment_rejects_duplicate_context_ids() -> None:
    with pytest.raises(ValueError, match="evaluated_context_ids"):
        EvidenceAssessment(
            states=(EvidenceState.INSUFFICIENT_EVIDENCE,),
            response_status=QAResponseStatus.INSUFFICIENT_EVIDENCE,
            reason_codes=(EvidenceReasonCode.ZERO_RESULTS,),
            evaluated_context_ids=("ctx_1", "ctx_1"),
            policy_version="evidence-v1",
            calibration_version="uncalibrated-v1",
        )


def test_material_claim_requires_citations() -> None:
    with pytest.raises(ValueError, match="material claim"):
        AnswerClaim(claim_id="claim_1", text="年度预算为 100 万元。")


def test_structured_answer_requires_current_declared_citations() -> None:
    citation = AnswerCitation(
        citation_id="cite_1",
        context_id="ctx_1",
        source="policy.md",
    )
    with pytest.raises(ValueError, match="unknown citation"):
        StructuredAnswer(
            status=QAResponseStatus.ANSWERED,
            answer="预算为 100 万元。",
            claims=(
                AnswerClaim(
                    claim_id="claim_1",
                    text="预算为 100 万元。",
                    citation_ids=("cite_missing",),
                ),
            ),
            citations=(citation,),
            schema_version="structured-answer-v1",
        )


def test_structured_answer_rejects_duplicate_citation_ids() -> None:
    citation = AnswerCitation(citation_id="cite_1", context_id="ctx_1")
    with pytest.raises(ValueError, match="citation IDs"):
        StructuredAnswer(
            status=QAResponseStatus.ANSWERED,
            answer="有依据的回答。",
            claims=(
                AnswerClaim(
                    claim_id="claim_1",
                    text="有依据的回答。",
                    citation_ids=("cite_1",),
                ),
            ),
            citations=(citation, citation),
            schema_version="structured-answer-v1",
        )


def test_partial_answer_requires_missing_information() -> None:
    citation = AnswerCitation(citation_id="cite_1", context_id="ctx_1")
    with pytest.raises(ValueError, match="missing information"):
        StructuredAnswer(
            status=QAResponseStatus.PARTIALLY_ANSWERED,
            answer="仅能确认当前版本。",
            claims=(
                AnswerClaim(
                    claim_id="claim_1",
                    text="当前版本为 v2。",
                    citation_ids=("cite_1",),
                ),
            ),
            citations=(citation,),
            schema_version="structured-answer-v1",
        )


def test_valid_partial_answer_keeps_supported_claims_and_missing_fields() -> None:
    answer = StructuredAnswer(
        status=QAResponseStatus.PARTIALLY_ANSWERED,
        answer="仅能确认当前版本。",
        claims=(
            AnswerClaim(
                claim_id="claim_1",
                text="当前版本为 v2。",
                citation_ids=("cite_1",),
            ),
        ),
        citations=(AnswerCitation(citation_id="cite_1", context_id="ctx_1"),),
        missing_information=(
            MissingInformation(field="effective_date", description="缺少生效日期"),
        ),
        schema_version="structured-answer-v1",
    )

    assert answer.citation_map["cite_1"].context_id == "ctx_1"
    assert answer.claims[0].citation_ids == ("cite_1",)


def test_grounding_result_requires_reasons_when_failed() -> None:
    with pytest.raises(ValueError, match="reason code"):
        GroundingResult(
            passed=False,
            rejected_claim_ids=("claim_1",),
            policy_version="grounding-v1",
        )


def test_grounding_result_rejects_claim_in_both_sets() -> None:
    with pytest.raises(ValueError, match="both accepted and rejected"):
        GroundingResult(
            passed=False,
            accepted_claim_ids=("claim_1",),
            rejected_claim_ids=("claim_1",),
            reason_codes=(EvidenceReasonCode.UNSUPPORTED_CLAIM,),
            policy_version="grounding-v1",
        )


def test_evidence_contracts_round_trip_through_cache_shapes() -> None:
    from domain.evidence import (
        evidence_assessment_from_dict,
        evidence_assessment_to_dict,
        grounding_result_from_dict,
        grounding_result_to_dict,
        structured_answer_from_dict,
        structured_answer_to_dict,
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
    answer = StructuredAnswer(
        status=QAResponseStatus.ANSWERED,
        answer="预算为100万元。",
        claims=(
            AnswerClaim(
                claim_id="claim_1",
                text="预算为100万元。",
                citation_ids=("cite_1",),
            ),
        ),
        citations=(
            AnswerCitation(
                citation_id="cite_1",
                context_id="ctx_1",
                source="policy.md",
            ),
        ),
        schema_version="structured-answer-v1",
    )
    grounding = GroundingResult(
        passed=True,
        accepted_claim_ids=("claim_1",),
        policy_version="grounding-v1",
    )

    assert evidence_assessment_from_dict(evidence_assessment_to_dict(assessment)) == assessment
    assert structured_answer_from_dict(structured_answer_to_dict(answer)) == answer
    assert grounding_result_from_dict(grounding_result_to_dict(grounding)) == grounding
