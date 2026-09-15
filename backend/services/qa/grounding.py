"""Deterministic citation and critical-value grounding for structured QA answers."""

from __future__ import annotations

import re
from typing import Mapping

from domain.evidence import (
    AnswerClaim,
    EvidenceReasonCode,
    GroundingResult,
    MissingInformation,
    QAResponseStatus,
    StructuredAnswer,
)
from domain.knowledge import RetrievedContext


_NUMBER_DATE_VERSION_RE = re.compile(
    r"(?:v(?:ersion)?\s*)?\d+(?:\.\d+)*(?:%|万元|亿元|元|万|亿|年|月|日|版)?",
    re.IGNORECASE,
)
_CLAUSE_RE = re.compile(r"第[一二三四五六七八九十百千万\d]+条|[A-Z]{2,}-\d+", re.IGNORECASE)
_PERSON_RE = re.compile(
    r"[\u4e00-\u9fff]{2,4}(?:先生|女士|经理|总监|负责人)|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
)
_NON_WORD_RE = re.compile(r"[\s,，。；;：:（）()\[\]{}]+")


class GroundingVerifier:
    """Verify current-run citations and exact critical factual values."""

    def __init__(self, policy_version: str = "grounding-v1") -> None:
        if not policy_version.strip():
            raise ValueError("policy_version is required")
        self.policy_version = policy_version

    def verify(
        self,
        answer: StructuredAnswer,
        authorized_contexts: Mapping[str, RetrievedContext],
    ) -> GroundingResult:
        """Return per-claim grounding without trusting generated citation content."""
        citation_map = answer.citation_map
        accepted: list[str] = []
        rejected: list[str] = []
        reasons: list[EvidenceReasonCode] = []
        for claim in answer.claims:
            cited_contexts: list[RetrievedContext] = []
            citation_invalid = False
            for citation_id in claim.citation_ids:
                citation = citation_map.get(citation_id)
                context = (
                    authorized_contexts.get(citation.context_id)
                    if citation is not None
                    else None
                )
                if context is None:
                    citation_invalid = True
                    self._append_reason(reasons, EvidenceReasonCode.UNKNOWN_CITATION)
                    break
                cited_contexts.append(context)
            if citation_invalid or not cited_contexts:
                rejected.append(claim.claim_id)
                continue
            evidence_text = "\n".join(context.content for context in cited_contexts)
            if not self._critical_values_supported(claim, evidence_text):
                rejected.append(claim.claim_id)
                self._append_reason(
                    reasons,
                    EvidenceReasonCode.UNSUPPORTED_CRITICAL_VALUE,
                )
                continue
            accepted.append(claim.claim_id)
        passed = not rejected
        if rejected and not reasons:
            reasons.append(EvidenceReasonCode.UNSUPPORTED_CLAIM)
        return GroundingResult(
            passed=passed,
            accepted_claim_ids=tuple(accepted),
            rejected_claim_ids=tuple(rejected),
            reason_codes=tuple(reasons),
            policy_version=self.policy_version,
        )

    def supported_subset(
        self,
        answer: StructuredAnswer,
        result: GroundingResult,
    ) -> StructuredAnswer | None:
        """Drop rejected claims and return a truthful partial answer when possible."""
        accepted_ids = set(result.accepted_claim_ids)
        claims = tuple(claim for claim in answer.claims if claim.claim_id in accepted_ids)
        if not claims:
            return None
        referenced_citations = {
            citation_id for claim in claims for citation_id in claim.citation_ids
        }
        citations = tuple(
            citation
            for citation in answer.citations
            if citation.citation_id in referenced_citations
        )
        if result.rejected_claim_ids:
            missing = answer.missing_information or (
                MissingInformation(
                    field="unsupported_claims",
                    description="部分结论缺少可验证的当前证据。",
                ),
            )
            status = QAResponseStatus.PARTIALLY_ANSWERED
        else:
            missing = ()
            status = QAResponseStatus.ANSWERED
        return StructuredAnswer(
            status=status,
            answer="".join(claim.text for claim in claims),
            claims=claims,
            citations=citations,
            missing_information=missing,
            schema_version=answer.schema_version,
        )

    @classmethod
    def _critical_values_supported(cls, claim: AnswerClaim, evidence_text: str) -> bool:
        claim_values = cls._critical_values(claim.text)
        if not claim_values:
            return True
        evidence_values = cls._critical_values(evidence_text)
        return claim_values.issubset(evidence_values)

    @staticmethod
    def _critical_values(text: str) -> set[str]:
        values = {
            _NON_WORD_RE.sub("", match).lower()
            for regex in (_NUMBER_DATE_VERSION_RE, _CLAUSE_RE, _PERSON_RE)
            for match in regex.findall(text)
        }
        return {value for value in values if value}

    @staticmethod
    def _append_reason(
        reasons: list[EvidenceReasonCode], reason: EvidenceReasonCode
    ) -> None:
        if reason not in reasons:
            reasons.append(reason)
