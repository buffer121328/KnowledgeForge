"""Versioned, content-addressed metric policy for reviewed evidence-gate categories.

The declarations in this module contain identifiers, applicability decisions,
bounded reason codes, and hard-gate flags only.  Runtime questions, references,
contexts, prompts, provider payloads, and credentials do not belong in this
policy or its run-plan identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from evaluation.evidence_gate.benchmark import REQUIRED_EVIDENCE_GATE_CATEGORIES
from evaluation.ragas.metric_capabilities import (
    RAGAS_CAPABILITY_VERSION,
    RAGAS_EXTENDED_METRIC_CAPABILITIES,
)

CATEGORY_METRIC_POLICY_VERSION = "evidence-gate-category-metrics-v1"

MetricFamily = Literal["ragas", "custom"]


class CategoryMetricPolicyError(ValueError):
    """A bounded category policy cannot be used for a diagnostic run."""


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    """Applicability and release behavior for one metric in one category."""

    metric_id: str
    family: MetricFamily
    outcome_family: Literal["ragas", "deterministic", "retrieval", "safety"]
    applicable: bool
    reason_code: str
    hard_gate: bool = False


@dataclass(frozen=True, slots=True)
class CategoryMetricContract:
    """Complete RAGAS applicability plus required custom metrics for a category."""

    category: str
    ragas: tuple[MetricPolicy, ...]
    custom: tuple[MetricPolicy, ...]

    @property
    def metrics(self) -> tuple[MetricPolicy, ...]:
        """Return all independently reported metric outcomes in stable order."""

        return self.ragas + self.custom


_LEGACY_RAGAS_METRIC_IDS = frozenset(
    {"faithfulness", "factual_correctness", "context_precision", "context_recall"}
)
_SUPPORTED_RAGAS_METRIC_IDS = frozenset(
    _LEGACY_RAGAS_METRIC_IDS | set(RAGAS_EXTENDED_METRIC_CAPABILITIES)
)

_APPLICABLE_RAGAS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        # 黄金锚点类目：保留 reference-required 指标（factual_correctness /
        # context_recall / semantic_similarity）作为回归基准，参考答案须
        # 条款级对齐并随制度语料变更同步维护。
        "fully_answerable": frozenset(
            {
                "faithfulness",
                "factual_correctness",
                "answer_relevancy",
                "semantic_similarity",
                "context_precision",
                "context_recall",
                "noise_sensitivity",
            }
        ),
        # 日常主轴类目：reference-free 指标优先，避免参考覆盖面差异造成的
        # 评分失真（factual_correctness 会把"正确但超出参考"的内容计为错）。
        "partially_answerable": frozenset(
            {
                "faithfulness",
                "answer_relevancy",
                "context_precision",
            }
        ),
        "conflicting": frozenset({"context_precision"}),
        "background_only": frozenset(),
        "missing_version_or_date": frozenset(),
        "missing_business_record": frozenset(),
        "completely_unanswerable": frozenset(),
        "authorization_filtered": frozenset(),
        "single_branch_unavailable": frozenset(
            {
                "faithfulness",
                "answer_relevancy",
                "context_precision",
            }
        ),
        "all_branches_unavailable": frozenset(),
        "prompt_injection": frozenset({"rubrics_score_without_reference"}),
    }
)

_CUSTOM_METRICS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "fully_answerable": (
            "exact_evidence_retrieval",
            "response_route_correctness",
            "citation_correctness",
            "grounding_correctness",
        ),
        "partially_answerable": (
            "partial_recognition",
            "supported_part_coverage",
            "unsupported_claim_rate",
            "missing_information_accuracy",
        ),
        "conflicting": (
            "conflict_recognition",
            "opposing_evidence_coverage",
            "unsupported_resolution_rate",
        ),
        "background_only": (
            "background_classification",
            "correct_non_answer_route",
            "unsupported_answer_rate",
        ),
        "missing_version_or_date": (
            "clarification_route",
            "missing_slot_accuracy",
        ),
        "missing_business_record": (
            "missing_record_route",
            "missing_field_accuracy",
            "unsupported_answer_rate",
        ),
        "completely_unanswerable": (
            "zero_results_route",
            "refusal_correctness",
            "hallucination_rate",
        ),
        "authorization_filtered": (
            "authorization_route",
            "unauthorized_citation_rate",
            "cross_scope_leakage_rate",
        ),
        "single_branch_unavailable": (
            "partial_outage_recognition",
            "answer_continuity",
            "remaining_branch_retrieval_quality",
        ),
        "all_branches_unavailable": (
            "source_unavailable_route",
            "branch_state_accuracy",
            "unsupported_answer_rate",
        ),
        "prompt_injection": (
            "injection_detection",
            "human_review_route",
            "instruction_data_leakage_rate",
        ),
    }
)

_HARD_GATES = frozenset(
    {
        ("authorization_filtered", "unauthorized_citation_rate"),
        ("authorization_filtered", "cross_scope_leakage_rate"),
        ("prompt_injection", "instruction_data_leakage_rate"),
    }
)

_RETRIEVAL_CUSTOM_METRICS = frozenset(
    {
        "exact_evidence_retrieval",
        "partial_outage_recognition",
        "remaining_branch_retrieval_quality",
        "branch_state_accuracy",
    }
)
_SAFETY_CATEGORIES = frozenset({"authorization_filtered", "prompt_injection"})


def _contract(category: str) -> CategoryMetricContract:
    applicable_ragas = _APPLICABLE_RAGAS[category]
    category_reason = f"reviewed_{category}_contract"
    inapplicable_reason = f"ordinary_answer_metric_invalid_for_{category}"
    ragas = tuple(
        MetricPolicy(
            metric_id=metric_id,
            family="ragas",
            outcome_family="ragas",
            applicable=metric_id in applicable_ragas,
            reason_code=(
                category_reason if metric_id in applicable_ragas else inapplicable_reason
            ),
        )
        for metric_id in sorted(_SUPPORTED_RAGAS_METRIC_IDS)
    )
    custom = tuple(
        MetricPolicy(
            metric_id=metric_id,
            family="custom",
            outcome_family=(
                "safety"
                if category in _SAFETY_CATEGORIES
                else "retrieval"
                if metric_id in _RETRIEVAL_CUSTOM_METRICS
                else "deterministic"
            ),
            applicable=True,
            reason_code=category_reason,
            hard_gate=(category, metric_id) in _HARD_GATES,
        )
        for metric_id in _CUSTOM_METRICS[category]
    )
    return CategoryMetricContract(category=category, ragas=ragas, custom=custom)


CATEGORY_METRIC_POLICY: Mapping[str, CategoryMetricContract] = MappingProxyType(
    {category: _contract(category) for category in sorted(_APPLICABLE_RAGAS)}
)


def validate_category_metric_policy(
    policy: Mapping[str, CategoryMetricContract],
) -> None:
    """Reject incomplete categories, unknown metrics, duplicates, or unsafe fields."""

    categories = set(policy)
    if categories != REQUIRED_EVIDENCE_GATE_CATEGORIES:
        raise CategoryMetricPolicyError("category_policy_category_set_mismatch")
    supported_custom = {item for values in _CUSTOM_METRICS.values() for item in values}
    for category in sorted(categories):
        contract = policy[category]
        if contract.category != category:
            raise CategoryMetricPolicyError("category_policy_contract_key_mismatch")
        ragas_ids = [item.metric_id for item in contract.ragas]
        if set(ragas_ids) != _SUPPORTED_RAGAS_METRIC_IDS:
            raise CategoryMetricPolicyError("category_policy_unknown_or_missing_ragas_metric")
        metric_ids = [item.metric_id for item in contract.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise CategoryMetricPolicyError("category_policy_duplicate_metric")
        for item in contract.metrics:
            if item.family not in {"ragas", "custom"}:
                raise CategoryMetricPolicyError("category_policy_unknown_metric_family")
            if item.outcome_family not in {
                "ragas", "deterministic", "retrieval", "safety"
            }:
                raise CategoryMetricPolicyError("category_policy_unknown_outcome_family")
            if item.family == "custom" and item.metric_id not in supported_custom:
                raise CategoryMetricPolicyError("category_policy_unknown_custom_metric")
            if not item.metric_id or len(item.metric_id) > 96:
                raise CategoryMetricPolicyError("category_policy_metric_id_invalid")
            if not item.reason_code or len(item.reason_code) > 96:
                raise CategoryMetricPolicyError("category_policy_reason_code_invalid")
            if item.hard_gate and (item.family != "custom" or not item.applicable):
                raise CategoryMetricPolicyError("category_policy_hard_gate_invalid")


def get_category_metric_contract(category: str) -> CategoryMetricContract:
    """Resolve a reviewed category or fail before any external scoring starts."""

    contract = CATEGORY_METRIC_POLICY.get(category)
    if contract is None:
        raise CategoryMetricPolicyError("unknown_category")
    return contract


def metric_applicability(category: str, metric_id: str) -> MetricPolicy:
    """Resolve one declared category/metric outcome, including audited N/A."""

    contract = get_category_metric_contract(category)
    for item in contract.metrics:
        if item.metric_id == metric_id:
            return item
    raise CategoryMetricPolicyError("unknown_metric")


def _canonical_policy_bytes() -> bytes:
    records = [
        {
            "category": category,
            "metrics": [
                {
                    "metric_id": item.metric_id,
                    "family": item.family,
                    "outcome_family": item.outcome_family,
                    "applicable": item.applicable,
                    "reason_code": item.reason_code,
                    "hard_gate": item.hard_gate,
                }
                for item in CATEGORY_METRIC_POLICY[category].metrics
            ],
        }
        for category in sorted(CATEGORY_METRIC_POLICY)
    ]
    payload = {
        "schema_version": CATEGORY_METRIC_POLICY_VERSION,
        "ragas_capability_version": RAGAS_CAPABILITY_VERSION,
        "categories": records,
    }
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def category_metric_policy_identity() -> dict[str, str]:
    """Return the bounded immutable identity embedded in diagnostic run plans."""

    validate_category_metric_policy(CATEGORY_METRIC_POLICY)
    return {
        "category_policy_version": CATEGORY_METRIC_POLICY_VERSION,
        "category_policy_sha256": hashlib.sha256(_canonical_policy_bytes()).hexdigest(),
        "ragas_capability_version": RAGAS_CAPABILITY_VERSION,
    }


__all__ = [
    "CATEGORY_METRIC_POLICY",
    "CATEGORY_METRIC_POLICY_VERSION",
    "CategoryMetricContract",
    "CategoryMetricPolicyError",
    "MetricPolicy",
    "category_metric_policy_identity",
    "get_category_metric_contract",
    "metric_applicability",
    "validate_category_metric_policy",
]
