from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from domain.evidence import EvidenceAssessment, GroundingResult, QAResponseStatus, StructuredAnswer
from domain.trace import AgentTrace

__all__ = ["Entity", "ExtractionResult", "KnowledgeEvent", "QAResult", "QueryIntent", "Relation", "RetrievedContext"]


@dataclass
class Entity:
    """Represent entity."""
    name: str
    type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def node_label(self) -> str:
        """Handle node label for the entity."""
        return self.type.replace(" ", "_")


@dataclass
class Relation:
    """Represent relation."""
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEvent:
    """Represent a knowledge event."""
    trigger: str
    type: str
    participants: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Represent the result of extraction processing."""
    entities: list[Entity]
    relations: list[Relation]
    events: list[KnowledgeEvent]
    source_chunk_id: str = ""


class QueryIntent(str, Enum):
    """Represent query intent."""
    FACTOID = "factoid"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"


@dataclass
class RetrievedContext:
    """Represent retrieved context."""
    content: str
    source: str
    score: float
    retrieval_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    """Represent the result of question-answering processing."""
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)
    degradation_code: str | None = None
    security_actions: list[str] = field(default_factory=list)
    trace: AgentTrace | None = None
    cache_hit_type: str = "none"
    knowledge_revision: int = 0
    response_status: QAResponseStatus | None = None
    evidence_assessment: EvidenceAssessment | None = None
    structured_answer: StructuredAnswer | None = None
    grounding_result: GroundingResult | None = None
