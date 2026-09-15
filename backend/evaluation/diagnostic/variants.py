"""Frozen single-variable plans for diagnostic Oracle and retrieval variants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

DIAGNOSTIC_VARIANT_PLAN_VERSION = "diagnostic-variant-plan-v1"
REQUIRED_DIAGNOSTIC_VARIANTS = (
    "observed",
    "oracle_context",
    "oracle_route",
    "dense_only",
    "bm25_only",
    "graph_only",
    "fused",
    "reranked",
)

_HASH_FIELDS = frozenset(
    {
        "corpus_sha256",
        "manifest_sha256",
        "authorization_scope_sha256",
        "response_snapshot_sha256",
    }
)
_STRING_FIELDS = frozenset(
    {
        "query_rewrite_version",
        "embedding_model",
        "answer_model",
        "candidate_budget_version",
        "category_policy_version",
        "tokenizer_version",
        "index_version",
        "threshold_version",
        "prompt_version",
        "answer_schema_version",
        "grounding_policy_version",
        "cache_policy",
    }
)
_REQUIRED_COMMON_FIELDS = _HASH_FIELDS | _STRING_FIELDS | {"candidate_budget"}
_FACTOR_FIELDS = frozenset(
    {"context_source", "route_source", "retrieval_pipeline"}
)
_FACTOR_VALUES = {
    "context_source": frozenset({"observed", "reviewed"}),
    "route_source": frozenset({"observed", "reviewed"}),
    "retrieval_pipeline": frozenset(
        {"observed", "dense", "bm25", "graph", "fused", "reranked"}
    ),
}


class DiagnosticVariantPlanError(ValueError):
    """A diagnostic plan is unsafe, incomplete, or not pair-compatible."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_common_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if set(identity) != _REQUIRED_COMMON_FIELDS:
        raise DiagnosticVariantPlanError("variant_plan_identity_fields_invalid")
    for field in _HASH_FIELDS:
        value = identity[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise DiagnosticVariantPlanError("variant_plan_hash_identity_invalid")
    for field in _STRING_FIELDS:
        value = identity[field]
        if not isinstance(value, str) or not 0 < len(value) <= 200:
            raise DiagnosticVariantPlanError("variant_plan_string_identity_invalid")
    budget = identity["candidate_budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 50:
        raise DiagnosticVariantPlanError("variant_plan_candidate_budget_invalid")
    if identity["cache_policy"] != "disabled":
        raise DiagnosticVariantPlanError("variant_plan_cache_policy_invalid")
    return dict(identity)


@dataclass(frozen=True, slots=True)
class DiagnosticVariant:
    """One immutable variant resolved against a common frozen identity."""

    variant_id: str
    common_identity: Mapping[str, Any]
    factor_values: Mapping[str, str]
    pairing_identity_sha256: str

    @property
    def resolved_identity(self) -> Mapping[str, Any]:
        return MappingProxyType({**self.common_identity, **self.factor_values})


@dataclass(frozen=True, slots=True)
class FrozenVariantPlan:
    """The complete Oracle/ablation plan bound to one frozen run identity."""

    schema_version: str
    common_identity: Mapping[str, Any]
    variants: tuple[DiagnosticVariant, ...]
    pairing_identity_sha256: str
    plan_sha256: str

    @classmethod
    def create(cls, common_identity: Mapping[str, Any]) -> "FrozenVariantPlan":
        validated = MappingProxyType(_validate_common_identity(common_identity))
        pairing_sha256 = _canonical_sha256(dict(validated))
        factors = {
            "observed": ("observed", "observed", "observed"),
            "oracle_context": ("reviewed", "observed", "observed"),
            "oracle_route": ("reviewed", "reviewed", "observed"),
            "dense_only": ("observed", "observed", "dense"),
            "bm25_only": ("observed", "observed", "bm25"),
            "graph_only": ("observed", "observed", "graph"),
            "fused": ("observed", "observed", "fused"),
            "reranked": ("observed", "observed", "reranked"),
        }
        variants = tuple(
            DiagnosticVariant(
                variant_id=variant_id,
                common_identity=validated,
                factor_values=MappingProxyType(
                    dict(
                        zip(
                            ("context_source", "route_source", "retrieval_pipeline"),
                            factors[variant_id],
                            strict=True,
                        )
                    )
                ),
                pairing_identity_sha256=pairing_sha256,
            )
            for variant_id in REQUIRED_DIAGNOSTIC_VARIANTS
        )
        plan_payload = {
            "schema_version": DIAGNOSTIC_VARIANT_PLAN_VERSION,
            "common_identity": dict(validated),
            "variants": [
                {"variant_id": item.variant_id, **dict(item.factor_values)}
                for item in variants
            ],
        }
        return cls(
            schema_version=DIAGNOSTIC_VARIANT_PLAN_VERSION,
            common_identity=validated,
            variants=variants,
            pairing_identity_sha256=pairing_sha256,
            plan_sha256=_canonical_sha256(plan_payload),
        )

    def variant(self, variant_id: str) -> DiagnosticVariant:
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        raise DiagnosticVariantPlanError("diagnostic_variant_unknown")


def assert_pairable_variants(
    left: DiagnosticVariant,
    right: DiagnosticVariant,
    *,
    declared_factor: str,
) -> dict[str, Any]:
    """Fail closed unless two variants differ only in the declared factor."""

    if declared_factor not in _FACTOR_FIELDS:
        raise DiagnosticVariantPlanError("variant_pair_factor_invalid")
    if (
        left.pairing_identity_sha256 != right.pairing_identity_sha256
        or dict(left.common_identity) != dict(right.common_identity)
    ):
        raise DiagnosticVariantPlanError("variant_pair_identity_incompatible")
    left_identity = dict(left.resolved_identity)
    right_identity = dict(right.resolved_identity)
    changed = {
        field
        for field in set(left_identity) | set(right_identity)
        if left_identity.get(field) != right_identity.get(field)
    }
    if changed != {declared_factor}:
        raise DiagnosticVariantPlanError("variant_pair_non_target_identity_changed")
    for variant in (left, right):
        values = dict(variant.factor_values)
        if set(values) != _FACTOR_FIELDS or any(
            values[field] not in _FACTOR_VALUES[field] for field in _FACTOR_FIELDS
        ):
            raise DiagnosticVariantPlanError("variant_pair_factor_value_invalid")
    return {
        "paired": True,
        "declared_factor": declared_factor,
        "left_variant_id": left.variant_id,
        "right_variant_id": right.variant_id,
        "pairing_identity_sha256": left.pairing_identity_sha256,
    }


__all__ = [
    "DIAGNOSTIC_VARIANT_PLAN_VERSION",
    "REQUIRED_DIAGNOSTIC_VARIANTS",
    "DiagnosticVariant",
    "DiagnosticVariantPlanError",
    "FrozenVariantPlan",
    "assert_pairable_variants",
]
