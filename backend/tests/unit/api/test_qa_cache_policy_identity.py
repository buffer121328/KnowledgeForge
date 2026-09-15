"""Cache identity coverage for evidence-producing policy versions."""

from __future__ import annotations

from agents.qa_agent import QAAgent
from domain.retrieval import RetrievalStrategy


def test_cache_identity_contains_all_evidence_producing_versions() -> None:
    identity = QAAgent._cache_mode_id(RetrievalStrategy.DENSE_BM25_GRAPH)

    assert "gate=off" in identity
    assert "evidence=evidence-composite-v2" in identity
    assert "calibration=qualification-boundaries-v2" in identity
    assert "budget=candidate-budget-v1" in identity
    assert "reranker=disabled:disabled-v1" in identity
    assert "answer=structured-answer-v1" in identity
    assert "grounding=grounding-v1" in identity


def test_cache_identity_changes_when_evidence_policy_changes(monkeypatch) -> None:
    baseline = QAAgent._cache_mode_id(RetrievalStrategy.DENSE)

    monkeypatch.setattr("agents.qa_agent.settings.qa_evidence_policy_version", "evidence-v3")
    changed = QAAgent._cache_mode_id(RetrievalStrategy.DENSE)

    assert changed != baseline
    assert "evidence=evidence-v3" in changed
