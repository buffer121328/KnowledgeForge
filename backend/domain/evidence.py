"""Pure contracts for evidence qualification, structured answers, and grounding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


_MAX_IDENTIFIER_LENGTH = 128
_MAX_TEXT_LENGTH = 8_000
_MAX_REASON_CODES = 16
_MAX_CONTEXT_IDS = 50


class EvidenceState(StrEnum):
    """Bounded evidence states produced after authorization and safety filtering."""

    DIRECT_EVIDENCE = "direct_evidence"
    PARTIAL_EVIDENCE = "partial_evidence"
    RELEVANT_BACKGROUND = "relevant_background"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    IRRELEVANT = "irrelevant"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_PROVENANCE = "invalid_provenance"


class QAResponseStatus(StrEnum):
    """Stable externally visible QA response routes."""

    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NEEDS_CLARIFICATION = "needs_clarification"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    SOURCE_UNAVAILABLE = "source_unavailable"


class EvidenceReasonCode(StrEnum):
    """Low-cardinality evidence and grounding decision reasons."""

    DIRECT_SUPPORT = "direct_support"
    RERANKER_SUPPORT = "reranker_support"
    PARTIAL_SUPPORT = "partial_support"
    BACKGROUND_ONLY = "background_only"
    MATERIAL_CONFLICT = "material_conflict"
    IRRELEVANT_CONTEXT = "irrelevant_context"
    ZERO_RESULTS = "zero_results"
    AUTHORIZED_CONTEXTS_EMPTY = "authorized_contexts_empty"
    INVALID_CONTEXT_PROVENANCE = "invalid_context_provenance"
    PARTIAL_DEPENDENCY_UNAVAILABLE = "partial_dependency_unavailable"
    ALL_DEPENDENCIES_UNAVAILABLE = "all_dependencies_unavailable"
    MISSING_QUESTION_DETAIL = "missing_question_detail"
    HIGH_RISK_REVIEW = "high_risk_review"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    UNKNOWN_CITATION = "unknown_citation"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    UNSUPPORTED_CRITICAL_VALUE = "unsupported_critical_value"
    STRUCTURED_OUTPUT_INVALID = "structured_output_invalid"


def _validate_identifier(value: str, *, field_name: str) -> None:
    if not value or len(value) > _MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"{field_name} must be non-empty and at most {_MAX_IDENTIFIER_LENGTH} characters"
        )


def _validate_text(value: str, *, field_name: str, allow_empty: bool = False) -> None:
    if (not allow_empty and not value.strip()) or len(value) > _MAX_TEXT_LENGTH:
        raise ValueError(
            f"{field_name} must be non-empty and at most {_MAX_TEXT_LENGTH} characters"
        )


def _validate_unique(values: tuple[str, ...], *, field_name: str, limit: int) -> None:
    if len(values) > limit:
        raise ValueError(f"{field_name} exceeds the limit of {limit}")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")
    for value in values:
        _validate_identifier(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class MissingInformation:
    """A bounded, user-actionable description of information still required."""

    field: str
    description: str

    def __post_init__(self) -> None:
        _validate_identifier(self.field, field_name="missing information field")
        _validate_text(self.description, field_name="missing information description")


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Tenant-safe evidence decision produced before answer generation."""

    states: tuple[EvidenceState, ...]
    response_status: QAResponseStatus
    reason_codes: tuple[EvidenceReasonCode, ...]
    evaluated_context_ids: tuple[str, ...] = ()
    supporting_context_ids: tuple[str, ...] = ()
    missing_information: tuple[MissingInformation, ...] = ()
    policy_version: str = "evidence-composite-v2"
    calibration_version: str = "qualification-boundaries-v2"

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("evidence assessment requires at least one state")
        if len(self.states) != len(set(self.states)):
            raise ValueError("evidence states must be unique")
        if not self.reason_codes:
            raise ValueError("evidence assessment requires at least one reason code")
        if len(self.reason_codes) > _MAX_REASON_CODES:
            raise ValueError("evidence reason codes exceed the bounded limit")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("evidence reason codes must be unique")
        _validate_unique(
            self.evaluated_context_ids,
            field_name="evaluated_context_ids",
            limit=_MAX_CONTEXT_IDS,
        )
        _validate_unique(
            self.supporting_context_ids,
            field_name="supporting_context_ids",
            limit=_MAX_CONTEXT_IDS,
        )
        unknown_support = set(self.supporting_context_ids) - set(
            self.evaluated_context_ids
        )
        if unknown_support:
            raise ValueError("supporting context IDs must be evaluated in the current run")
        if self.generation_allowed and not self.supporting_context_ids:
            raise ValueError("answer routes require at least one supporting context")
        _validate_identifier(self.policy_version, field_name="policy_version")
        _validate_identifier(self.calibration_version, field_name="calibration_version")

    @property
    def primary_state(self) -> EvidenceState:
        """Return the highest-priority state selected by the qualifier."""
        return self.states[0]

    @property
    def generation_allowed(self) -> bool:
        """Return whether structured generation may run for this assessment."""
        return self.response_status in {
            QAResponseStatus.ANSWERED,
            QAResponseStatus.PARTIALLY_ANSWERED,
        }


@dataclass(frozen=True, slots=True)
class AnswerCitation:
    """A request-local citation mapped to an authorized retrieval context."""

    citation_id: str
    context_id: str
    source: str = ""
    content: str = field(default="", repr=False)
    document_id: str = ""
    chunk_id: str = ""
    chunk_index: int | None = None
    highlight: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.citation_id, field_name="citation_id")
        _validate_identifier(self.context_id, field_name="context_id")
        if len(self.source) > _MAX_IDENTIFIER_LENGTH * 4:
            raise ValueError("citation source is too long")
        if len(self.content) > _MAX_TEXT_LENGTH:
            raise ValueError("citation content is too long")
        if self.document_id:
            _validate_identifier(self.document_id, field_name="citation document ID")
        if self.chunk_id:
            _validate_identifier(self.chunk_id, field_name="citation chunk ID")
        if self.chunk_index is not None and self.chunk_index < 0:
            raise ValueError("citation chunk index must be non-negative")
        if len(self.highlight) > _MAX_TEXT_LENGTH:
            raise ValueError("citation highlight is too long")


@dataclass(frozen=True, slots=True)
class AnswerClaim:
    """One user-visible conclusion and its request-local citations."""

    claim_id: str
    text: str
    citation_ids: tuple[str, ...] = ()
    material: bool = True

    def __post_init__(self) -> None:
        _validate_identifier(self.claim_id, field_name="claim_id")
        _validate_text(self.text, field_name="claim text")
        _validate_unique(
            self.citation_ids,
            field_name="claim citation IDs",
            limit=_MAX_CONTEXT_IDS,
        )
        if self.material and not self.citation_ids:
            raise ValueError("material claim requires at least one citation")


@dataclass(frozen=True, slots=True)
class StructuredAnswer:
    """Structured generated answer eligible for deterministic grounding checks."""

    status: QAResponseStatus
    answer: str
    claims: tuple[AnswerClaim, ...]
    citations: tuple[AnswerCitation, ...]
    schema_version: str
    missing_information: tuple[MissingInformation, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {
            QAResponseStatus.ANSWERED,
            QAResponseStatus.PARTIALLY_ANSWERED,
        }:
            raise ValueError("structured answers support only answered or partially_answered")
        _validate_text(self.answer, field_name="structured answer")
        if not self.claims:
            raise ValueError("structured answer requires at least one claim")
        _validate_identifier(self.schema_version, field_name="schema_version")
        claim_ids = tuple(claim.claim_id for claim in self.claims)
        _validate_unique(claim_ids, field_name="claim IDs", limit=_MAX_CONTEXT_IDS)
        citation_ids = tuple(citation.citation_id for citation in self.citations)
        _validate_unique(
            citation_ids,
            field_name="citation IDs",
            limit=_MAX_CONTEXT_IDS,
        )
        declared = set(citation_ids)
        referenced = {
            citation_id
            for claim in self.claims
            for citation_id in claim.citation_ids
        }
        if not referenced.issubset(declared):
            raise ValueError("structured answer contains an unknown citation")
        if self.status is QAResponseStatus.PARTIALLY_ANSWERED and not self.missing_information:
            raise ValueError("partial answer requires missing information")
        if self.status is QAResponseStatus.ANSWERED and self.missing_information:
            raise ValueError("fully answered result cannot contain missing information")

    @property
    def citation_map(self) -> dict[str, AnswerCitation]:
        """Return a request-local citation lookup for grounding and presentation."""
        return {citation.citation_id: citation for citation in self.citations}


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Deterministic validation outcome for generated claims and citations."""

    passed: bool
    policy_version: str
    accepted_claim_ids: tuple[str, ...] = ()
    rejected_claim_ids: tuple[str, ...] = ()
    reason_codes: tuple[EvidenceReasonCode, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.policy_version, field_name="policy_version")
        _validate_unique(
            self.accepted_claim_ids,
            field_name="accepted_claim_ids",
            limit=_MAX_CONTEXT_IDS,
        )
        _validate_unique(
            self.rejected_claim_ids,
            field_name="rejected_claim_ids",
            limit=_MAX_CONTEXT_IDS,
        )
        if set(self.accepted_claim_ids) & set(self.rejected_claim_ids):
            raise ValueError("a claim cannot be both accepted and rejected")
        if self.passed and self.rejected_claim_ids:
            raise ValueError("passed grounding cannot contain rejected claims")
        if not self.passed and not self.reason_codes:
            raise ValueError("failed grounding requires at least one reason code")
        if len(self.reason_codes) > _MAX_REASON_CODES:
            raise ValueError("grounding reason codes exceed the bounded limit")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("grounding reason codes must be unique")



def evidence_assessment_to_dict(value: EvidenceAssessment | None) -> dict[str, Any] | None:
    """Serialize an assessment for cache/history compatibility boundaries."""
    if value is None:
        return None
    return {
        "states": [state.value for state in value.states],
        "response_status": value.response_status.value,
        "reason_codes": [reason.value for reason in value.reason_codes],
        "evaluated_context_ids": list(value.evaluated_context_ids),
        "supporting_context_ids": list(value.supporting_context_ids),
        "missing_information": [
            {"field": item.field, "description": item.description}
            for item in value.missing_information
        ],
        "policy_version": value.policy_version,
        "calibration_version": value.calibration_version,
    }


def evidence_assessment_from_dict(value: Mapping[str, Any]) -> EvidenceAssessment:
    """Reconstruct a validated assessment from a serialized boundary value."""
    return EvidenceAssessment(
        states=tuple(EvidenceState(item) for item in value.get("states") or ()),
        response_status=QAResponseStatus(value["response_status"]),
        reason_codes=tuple(
            EvidenceReasonCode(item) for item in value.get("reason_codes") or ()
        ),
        evaluated_context_ids=tuple(value.get("evaluated_context_ids") or ()),
        supporting_context_ids=tuple(value.get("supporting_context_ids") or ()),
        missing_information=tuple(
            MissingInformation(
                field=str(item["field"]),
                description=str(item["description"]),
            )
            for item in value.get("missing_information") or ()
        ),
        policy_version=str(value["policy_version"]),
        calibration_version=str(value["calibration_version"]),
    )


def structured_answer_to_dict(value: StructuredAnswer | None) -> dict[str, Any] | None:
    """Serialize a structured answer without trace or provider internals."""
    if value is None:
        return None
    return {
        "status": value.status.value,
        "answer": value.answer,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "citation_ids": list(claim.citation_ids),
                "material": claim.material,
            }
            for claim in value.claims
        ],
        "citations": [
            {
                "citation_id": citation.citation_id,
                "context_id": citation.context_id,
                "source": citation.source,
                "content": citation.content,
                "document_id": citation.document_id,
                "chunk_id": citation.chunk_id,
                "chunk_index": citation.chunk_index,
                "highlight": citation.highlight,
            }
            for citation in value.citations
        ],
        "missing_information": [
            {"field": item.field, "description": item.description}
            for item in value.missing_information
        ],
        "schema_version": value.schema_version,
    }


def structured_answer_from_dict(value: Mapping[str, Any]) -> StructuredAnswer:
    """Reconstruct and validate a structured answer boundary value."""
    return StructuredAnswer(
        status=QAResponseStatus(value["status"]),
        answer=str(value["answer"]),
        claims=tuple(
            AnswerClaim(
                claim_id=str(item["claim_id"]),
                text=str(item["text"]),
                citation_ids=tuple(item.get("citation_ids") or ()),
                material=bool(item.get("material", True)),
            )
            for item in value.get("claims") or ()
        ),
        citations=tuple(
            AnswerCitation(
                citation_id=str(item["citation_id"]),
                context_id=str(item["context_id"]),
                source=str(item.get("source") or ""),
                content=str(item.get("content") or ""),
                document_id=str(item.get("document_id") or ""),
                chunk_id=str(item.get("chunk_id") or ""),
                chunk_index=(
                    int(item["chunk_index"])
                    if item.get("chunk_index") is not None
                    else None
                ),
                highlight=str(item.get("highlight") or ""),
            )
            for item in value.get("citations") or ()
        ),
        missing_information=tuple(
            MissingInformation(
                field=str(item["field"]),
                description=str(item["description"]),
            )
            for item in value.get("missing_information") or ()
        ),
        schema_version=str(value["schema_version"]),
    )


def grounding_result_to_dict(value: GroundingResult | None) -> dict[str, Any] | None:
    """Serialize a grounding result for safe persistence."""
    if value is None:
        return None
    return {
        "passed": value.passed,
        "accepted_claim_ids": list(value.accepted_claim_ids),
        "rejected_claim_ids": list(value.rejected_claim_ids),
        "reason_codes": [reason.value for reason in value.reason_codes],
        "policy_version": value.policy_version,
    }


def grounding_result_from_dict(value: Mapping[str, Any]) -> GroundingResult:
    """Reconstruct and validate a grounding result boundary value."""
    return GroundingResult(
        passed=bool(value["passed"]),
        accepted_claim_ids=tuple(value.get("accepted_claim_ids") or ()),
        rejected_claim_ids=tuple(value.get("rejected_claim_ids") or ()),
        reason_codes=tuple(
            EvidenceReasonCode(item) for item in value.get("reason_codes") or ()
        ),
        policy_version=str(value["policy_version"]),
    )
