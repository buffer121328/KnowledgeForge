"""Acceptance coverage for evidence-gate and optional reranker settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.config.settings import Settings


def make_settings(**overrides: object) -> Settings:
    """Create isolated settings without reading a developer .env file."""
    return Settings(_env_file=None, **overrides)


def test_evidence_and_cross_encoder_defaults_are_safe() -> None:
    configured = make_settings()

    assert configured.qa_evidence_gate_mode == "off"
    assert configured.qa_cross_encoder_mode == "disabled"
    assert configured.qa_cross_encoder_remote_endpoint == ""
    assert configured.qa_evidence_candidate_limit <= configured.qa_context_limit


@pytest.mark.parametrize("value", ["OFF", "shadow", " enforce "])
def test_evidence_gate_mode_is_normalized(value: str) -> None:
    assert make_settings(qa_evidence_gate_mode=value).qa_evidence_gate_mode == value.strip().lower()


def test_unknown_evidence_gate_mode_is_rejected() -> None:
    with pytest.raises(ValidationError, match="qa_evidence_gate_mode"):
        make_settings(qa_evidence_gate_mode="active")


def test_unknown_cross_encoder_mode_is_rejected() -> None:
    with pytest.raises(ValidationError, match="qa_cross_encoder_mode"):
        make_settings(qa_cross_encoder_mode="auto")


def test_remote_cross_encoder_requires_explicit_endpoint() -> None:
    with pytest.raises(ValidationError, match="qa_cross_encoder_remote_endpoint"):
        make_settings(qa_cross_encoder_mode="remote")


def test_remote_cross_encoder_requires_dashscope_credential() -> None:
    with pytest.raises(ValidationError, match="dashscope_api_key"):
        make_settings(
            qa_cross_encoder_mode="remote",
            qa_cross_encoder_remote_endpoint="https://dashscope.example/rerank",
        )


def test_remote_cross_encoder_accepts_qwen3_rerank_configuration() -> None:
    configured = make_settings(
        qa_cross_encoder_mode="remote",
        qa_cross_encoder_remote_endpoint="https://dashscope.example/rerank",
        dashscope_api_key="test-key",
    )

    assert configured.qa_cross_encoder_model == "qwen3-rerank"


def test_evidence_threshold_order_is_validated() -> None:
    with pytest.raises(ValidationError, match="evidence thresholds"):
        make_settings(
            qa_evidence_gray_zone_lower=0.8,
            qa_evidence_gray_zone_upper=0.6,
        )


def test_candidate_budgets_cannot_exceed_context_pool() -> None:
    with pytest.raises(ValidationError, match="qa_evidence_candidate_limit"):
        make_settings(qa_context_limit=4, qa_evidence_candidate_limit=5)

    with pytest.raises(ValidationError, match="qa_cross_encoder_context_limit"):
        make_settings(
            qa_cross_encoder_candidate_limit=4,
            qa_cross_encoder_context_limit=5,
        )


def test_policy_versions_remain_safe_low_cardinality_labels() -> None:
    with pytest.raises(ValidationError, match="policy versions"):
        make_settings(qa_evidence_policy_version="contains spaces")
