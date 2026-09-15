"""Pinned RAGAS 0.4.3 metric capabilities used by diagnostic planning.

This module is deliberately declarative and does not import RAGAS.  The matching
no-network acceptance test constructs the public collection classes from the
pinned evaluation dependency so package drift fails visibly before a run plan is
accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

RAGAS_CAPABILITY_VERSION = "ragas-0.4.3-capabilities-v1"
RAGAS_PACKAGE_VERSION = "0.4.3"


@dataclass(frozen=True, slots=True)
class RagasMetricCapability:
    """One bounded public RAGAS collection metric contract."""

    metric_id: str
    collection_class: str
    accepted_inputs: tuple[str, ...]
    constructor_dependencies: tuple[Literal["llm", "embeddings"], ...]


@dataclass(frozen=True, slots=True)
class UnsupportedRagasMetric:
    """A requested metric family that lacks a supported public collection API."""

    requested_id: str
    reason_code: str
    detail: str


_CAPABILITIES = (
    RagasMetricCapability(
        metric_id="answer_relevancy",
        collection_class=(
            "ragas.metrics.collections.answer_relevancy.metric.AnswerRelevancy"
        ),
        accepted_inputs=("user_input", "response"),
        constructor_dependencies=("llm", "embeddings"),
    ),
    RagasMetricCapability(
        metric_id="semantic_similarity",
        collection_class="ragas.metrics.collections._semantic_similarity.SemanticSimilarity",
        accepted_inputs=("reference", "response"),
        constructor_dependencies=("embeddings",),
    ),
    RagasMetricCapability(
        metric_id="noise_sensitivity",
        collection_class=(
            "ragas.metrics.collections.noise_sensitivity.metric.NoiseSensitivity"
        ),
        accepted_inputs=(
            "user_input",
            "response",
            "reference",
            "retrieved_contexts",
        ),
        constructor_dependencies=("llm",),
    ),
    RagasMetricCapability(
        metric_id="rubrics_score_with_reference",
        collection_class=(
            "ragas.metrics.collections.domain_specific_rubrics.metric."
            "RubricsScoreWithReference"
        ),
        accepted_inputs=(
            "user_input",
            "response",
            "retrieved_contexts",
            "reference_contexts",
            "reference",
        ),
        constructor_dependencies=("llm",),
    ),
    RagasMetricCapability(
        metric_id="rubrics_score_without_reference",
        collection_class=(
            "ragas.metrics.collections.domain_specific_rubrics.metric."
            "RubricsScoreWithoutReference"
        ),
        # The reference-free safety rubric is used for prompt-injection routes.
        # Internal retrieval may still have executed before the request was
        # blocked, but neither reference evidence nor retrieved context is a
        # semantic prerequisite for judging the user-visible safety response.
        accepted_inputs=("user_input", "response"),
        constructor_dependencies=("llm",),
    ),
    RagasMetricCapability(
        metric_id="domain_specific_rubrics",
        collection_class=(
            "ragas.metrics.collections.domain_specific_rubrics.metric."
            "DomainSpecificRubrics"
        ),
        accepted_inputs=(
            "user_input",
            "response",
            "retrieved_contexts",
            "reference_contexts",
            "reference",
        ),
        constructor_dependencies=("llm",),
    ),
    RagasMetricCapability(
        metric_id="instance_specific_rubrics",
        collection_class=(
            "ragas.metrics.collections.instance_specific_rubrics.metric."
            "InstanceSpecificRubrics"
        ),
        accepted_inputs=(
            "rubrics",
            "user_input",
            "response",
            "retrieved_contexts",
            "reference_contexts",
            "reference",
        ),
        constructor_dependencies=("llm",),
    ),
)

RAGAS_EXTENDED_METRIC_CAPABILITIES: Mapping[str, RagasMetricCapability] = (
    MappingProxyType({capability.metric_id: capability for capability in _CAPABILITIES})
)

RAGAS_UNSUPPORTED_METRICS: Mapping[str, UnsupportedRagasMetric] = MappingProxyType(
    {
        "aspect_critic": UnsupportedRagasMetric(
            requested_id="aspect_critic",
            reason_code="ragas_public_collection_metric_unavailable",
            detail=(
                "RAGAS 0.4.3 exposes AspectCritic only through a deprecated legacy "
                "export while its public collections module has no AspectCritic; "
                "KnowledgeForge does not treat the deprecated API as a supported "
                "collection metric."
            ),
        )
    }
)


__all__ = [
    "RAGAS_CAPABILITY_VERSION",
    "RAGAS_EXTENDED_METRIC_CAPABILITIES",
    "RAGAS_PACKAGE_VERSION",
    "RAGAS_UNSUPPORTED_METRICS",
    "RagasMetricCapability",
    "UnsupportedRagasMetric",
]
