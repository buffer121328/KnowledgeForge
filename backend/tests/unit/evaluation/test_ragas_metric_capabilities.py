"""No-network acceptance coverage for the pinned RAGAS metric surface."""

from __future__ import annotations

import importlib.metadata
import inspect
from collections.abc import Callable
from typing import Any

import pytest

from evaluation.ragas import adapter
from evaluation.ragas.metric_capabilities import (
    RAGAS_CAPABILITY_VERSION,
    RAGAS_EXTENDED_METRIC_CAPABILITIES,
    RAGAS_PACKAGE_VERSION,
    RAGAS_UNSUPPORTED_METRICS,
)


def _public_parameters(callback: Callable[..., Any]) -> tuple[str, ...]:
    return tuple(inspect.signature(callback).parameters)


def test_pinned_ragas_extended_collection_metrics_construct_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declared 0.4.3 public collection surface must construct offline."""
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    adapter._install_vertexai_compatibility_shim()

    from openai import AsyncOpenAI
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        DomainSpecificRubrics,
        InstanceSpecificRubrics,
        NoiseSensitivity,
        RubricsScoreWithReference,
        RubricsScoreWithoutReference,
        SemanticSimilarity,
    )

    class OfflineEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            del text, kwargs
            return [1.0, 0.0]

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            del text, kwargs
            return [1.0, 0.0]

    llm = adapter._build_ragas_llm(
        AsyncOpenAI=AsyncOpenAI,
        llm_factory=llm_factory,
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="doubao-seed-2.1-turbo",
    )
    embeddings = OfflineEmbeddings()
    instances = {
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "semantic_similarity": SemanticSimilarity(embeddings=embeddings),
        "noise_sensitivity": NoiseSensitivity(llm=llm),
        "rubrics_score_with_reference": RubricsScoreWithReference(llm=llm),
        "rubrics_score_without_reference": RubricsScoreWithoutReference(llm=llm),
        "domain_specific_rubrics": DomainSpecificRubrics(llm=llm),
        "instance_specific_rubrics": InstanceSpecificRubrics(llm=llm),
    }

    assert importlib.metadata.version("ragas") == RAGAS_PACKAGE_VERSION
    assert RAGAS_CAPABILITY_VERSION == "ragas-0.4.3-capabilities-v1"
    assert set(instances) == set(RAGAS_EXTENDED_METRIC_CAPABILITIES)
    for metric_id, metric in instances.items():
        capability = RAGAS_EXTENDED_METRIC_CAPABILITIES[metric_id]
        assert metric.name == metric_id
        assert (
            f"{type(metric).__module__}.{type(metric).__name__}"
            == capability.collection_class
        )
        # capability 只声明适配器允许传入的输入；pinned ragas 类的签名可以
        # 更宽（例如免参考安全 rubric 刻意不声明 reference 类输入）。
        assert set(capability.accepted_inputs) <= set(_public_parameters(metric.ascore))


def test_aspect_critic_is_not_misrepresented_as_a_public_collection_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A private legacy class is not an approved RAGAS collection capability."""
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    adapter._install_vertexai_compatibility_shim()

    import ragas.metrics as metrics
    import ragas.metrics.collections as collections

    unsupported = RAGAS_UNSUPPORTED_METRICS["aspect_critic"]
    assert unsupported.reason_code == "ragas_public_collection_metric_unavailable"
    assert not hasattr(collections, "AspectCritic")
    with pytest.warns(DeprecationWarning):
        legacy_aspect_critic = metrics.AspectCritic
    assert legacy_aspect_critic.__module__ == "ragas.metrics._aspect_critic"
    assert "aspect_critic" not in RAGAS_EXTENDED_METRIC_CAPABILITIES
