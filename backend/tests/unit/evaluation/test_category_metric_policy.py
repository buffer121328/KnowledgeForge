"""Acceptance tests for the bounded eleven-category diagnostic policy."""

from __future__ import annotations

import json

import pytest

from evaluation.benchmarks.category_metric_policy import (
    CATEGORY_METRIC_POLICY,
    CATEGORY_METRIC_POLICY_VERSION,
    CategoryMetricPolicyError,
    category_metric_policy_identity,
    get_category_metric_contract,
    metric_applicability,
    validate_category_metric_policy,
)
from evaluation.evidence_gate.benchmark import REQUIRED_EVIDENCE_GATE_CATEGORIES
from evaluation.ragas.metric_capabilities import RAGAS_CAPABILITY_VERSION


EXPECTED_RAGAS = {
    # 黄金锚点类目：保留 reference-required 指标作为回归基准
    "fully_answerable": {
        "faithfulness",
        "factual_correctness",
        "answer_relevancy",
        "semantic_similarity",
        "context_precision",
        "context_recall",
        "noise_sensitivity",
    },
    # 日常主轴类目：reference-free 优先
    "partially_answerable": {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    },
    "conflicting": {"context_precision"},
    "background_only": set(),
    "missing_version_or_date": set(),
    "missing_business_record": set(),
    "completely_unanswerable": set(),
    "authorization_filtered": set(),
    "single_branch_unavailable": {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    },
    "all_branches_unavailable": set(),
    "prompt_injection": {"rubrics_score_without_reference"},
}

EXPECTED_CUSTOM = {
    "fully_answerable": {
        "exact_evidence_retrieval",
        "response_route_correctness",
        "citation_correctness",
        "grounding_correctness",
    },
    "partially_answerable": {
        "partial_recognition",
        "supported_part_coverage",
        "unsupported_claim_rate",
        "missing_information_accuracy",
    },
    "conflicting": {
        "conflict_recognition",
        "opposing_evidence_coverage",
        "unsupported_resolution_rate",
    },
    "background_only": {
        "background_classification",
        "correct_non_answer_route",
        "unsupported_answer_rate",
    },
    "missing_version_or_date": {"clarification_route", "missing_slot_accuracy"},
    "missing_business_record": {
        "missing_record_route",
        "missing_field_accuracy",
        "unsupported_answer_rate",
    },
    "completely_unanswerable": {
        "zero_results_route",
        "refusal_correctness",
        "hallucination_rate",
    },
    "authorization_filtered": {
        "authorization_route",
        "unauthorized_citation_rate",
        "cross_scope_leakage_rate",
    },
    "single_branch_unavailable": {
        "partial_outage_recognition",
        "answer_continuity",
        "remaining_branch_retrieval_quality",
    },
    "all_branches_unavailable": {
        "source_unavailable_route",
        "branch_state_accuracy",
        "unsupported_answer_rate",
    },
    "prompt_injection": {
        "injection_detection",
        "human_review_route",
        "instruction_data_leakage_rate",
    },
}


def test_policy_has_exact_category_and_metric_contracts() -> None:
    assert set(CATEGORY_METRIC_POLICY) == REQUIRED_EVIDENCE_GATE_CATEGORIES
    assert set(EXPECTED_RAGAS) == REQUIRED_EVIDENCE_GATE_CATEGORIES
    assert set(EXPECTED_CUSTOM) == REQUIRED_EVIDENCE_GATE_CATEGORIES

    for category in REQUIRED_EVIDENCE_GATE_CATEGORIES:
        contract = get_category_metric_contract(category)
        assert {item.metric_id for item in contract.ragas if item.applicable} == (
            EXPECTED_RAGAS[category]
        )
        assert {item.metric_id for item in contract.custom} == EXPECTED_CUSTOM[category]
        assert all(item.applicable for item in contract.custom)
        assert {item.outcome_family for item in contract.metrics}.issubset(
            {"ragas", "deterministic", "retrieval", "safety"}
        )
        assert all(item.reason_code and len(item.reason_code) <= 96 for item in contract.metrics)


def test_inapplicable_ragas_metrics_are_explicit_and_auditable() -> None:
    for category, applicable in EXPECTED_RAGAS.items():
        contract = get_category_metric_contract(category)
        declared = {item.metric_id for item in contract.ragas}
        assert applicable.issubset(declared)
        assert all(
            item.applicable == (item.metric_id in applicable)
            for item in contract.ragas
        )

    outcome = metric_applicability("authorization_filtered", "faithfulness")
    assert outcome.family == "ragas"
    assert outcome.applicable is False
    assert outcome.reason_code == "ordinary_answer_metric_invalid_for_authorization_filtered"


def test_only_leakage_metrics_are_category_hard_gates() -> None:
    hard_gates = {
        (category, item.metric_id)
        for category, contract in CATEGORY_METRIC_POLICY.items()
        for item in contract.metrics
        if item.hard_gate
    }
    assert hard_gates == {
        ("authorization_filtered", "unauthorized_citation_rate"),
        ("authorization_filtered", "cross_scope_leakage_rate"),
        ("prompt_injection", "instruction_data_leakage_rate"),
    }


def test_unknown_category_and_metric_fail_closed() -> None:
    with pytest.raises(CategoryMetricPolicyError, match="unknown_category"):
        get_category_metric_contract("new_unreviewed_category")
    with pytest.raises(CategoryMetricPolicyError, match="unknown_metric"):
        metric_applicability("fully_answerable", "invented_judge_metric")


def test_policy_identity_is_stable_bounded_and_content_addressed() -> None:
    validate_category_metric_policy(CATEGORY_METRIC_POLICY)
    first = category_metric_policy_identity()
    second = category_metric_policy_identity()

    assert first == second
    assert first["category_policy_version"] == CATEGORY_METRIC_POLICY_VERSION
    assert first["ragas_capability_version"] == RAGAS_CAPABILITY_VERSION
    assert len(first["category_policy_sha256"]) == 64
    assert set(first) == {
        "category_policy_version",
        "category_policy_sha256",
        "ragas_capability_version",
    }
    serialized = json.dumps(first, sort_keys=True)
    assert "question" not in serialized
    assert "reference" not in serialized
    assert "context" not in serialized
    assert "credential" not in serialized
