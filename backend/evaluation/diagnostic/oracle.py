"""Diagnostic-only Oracle preparation and execution with zero online writes."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from evaluation.diagnostic.variants import DiagnosticVariant, FrozenVariantPlan

OracleEvaluationCallback = Callable[..., Awaitable[Mapping[str, Any]]]


class DiagnosticOracleError(ValueError):
    """Oracle preparation or dispatch crossed a frozen safety boundary."""


def _valid_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bounded_text(value: Any, *, maximum: int = 200) -> bool:
    return isinstance(value, str) and 0 < len(value) <= maximum


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewedOracleContext:
    """One reviewed context whose raw content remains in the evaluation process."""

    context_id: str
    source_document_id: str
    content_sha256: str
    manifest_sha256: str
    authorization_scope_sha256: str
    content: str
    reviewed: bool


@dataclass(frozen=True, slots=True)
class PreparedOracleInput:
    """Authorization-validated input for an injected evaluation executor."""

    variant: DiagnosticVariant
    case_id: str
    contexts: tuple[ReviewedOracleContext, ...]
    reviewed_route: Mapping[str, Any] | None

    def safe_identity(self) -> dict[str, Any]:
        """Return IDs and hashes only; never serialize context or route content."""

        summary: dict[str, Any] = {
            "variant_id": self.variant.variant_id,
            "pairing_identity_sha256": self.variant.pairing_identity_sha256,
            "case_id": self.case_id,
            "reviewed_contexts": [
                {
                    "context_id": context.context_id,
                    "source_document_id": context.source_document_id,
                    "content_sha256": context.content_sha256,
                }
                for context in self.contexts
            ],
        }
        if self.reviewed_route is not None:
            summary["reviewed_route_sha256"] = _canonical_sha256(
                dict(self.reviewed_route)
            )
        return summary


def _validated_route(route: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if route is None:
        return None
    if set(route) != {"response_status", "evidence_states"}:
        raise DiagnosticOracleError("oracle_reviewed_route_invalid")
    response_status = route.get("response_status")
    evidence_states = route.get("evidence_states")
    if not _bounded_text(response_status, maximum=64) or not isinstance(
        evidence_states, Sequence
    ) or isinstance(evidence_states, (str, bytes)):
        raise DiagnosticOracleError("oracle_reviewed_route_invalid")
    if not evidence_states or not all(
        _bounded_text(value, maximum=64) for value in evidence_states
    ):
        raise DiagnosticOracleError("oracle_reviewed_route_invalid")
    return MappingProxyType(
        {
            "response_status": response_status,
            "evidence_states": tuple(evidence_states),
        }
    )


def prepare_oracle_input(
    *,
    plan: FrozenVariantPlan,
    variant_id: str,
    case_id: str,
    contexts: Sequence[ReviewedOracleContext],
    reviewed_route: Mapping[str, Any] | None,
    runtime_manifest_sha256: str,
    runtime_authorization_scope_sha256: str,
) -> PreparedOracleInput:
    """Validate frozen lineage and authorization before any Oracle dispatch."""

    if variant_id not in {"oracle_context", "oracle_route"}:
        raise DiagnosticOracleError("oracle_variant_required")
    variant = plan.variant(variant_id)
    expected_manifest = plan.common_identity["manifest_sha256"]
    expected_scope = plan.common_identity["authorization_scope_sha256"]
    if runtime_manifest_sha256 != expected_manifest:
        raise DiagnosticOracleError("oracle_manifest_mismatch")
    if runtime_authorization_scope_sha256 != expected_scope:
        raise DiagnosticOracleError("oracle_authorization_scope_mismatch")
    if not _bounded_text(case_id, maximum=128):
        raise DiagnosticOracleError("oracle_case_identity_invalid")
    if not contexts or len(contexts) > plan.common_identity["candidate_budget"]:
        raise DiagnosticOracleError("oracle_context_count_invalid")

    seen: set[str] = set()
    for context in contexts:
        if context.reviewed is not True:
            raise DiagnosticOracleError("oracle_context_not_reviewed")
        if context.manifest_sha256 != expected_manifest:
            raise DiagnosticOracleError("oracle_context_lineage_mismatch")
        if context.authorization_scope_sha256 != expected_scope:
            raise DiagnosticOracleError("oracle_context_authorization_mismatch")
        if not all(
            _bounded_text(value, maximum=128)
            for value in (context.context_id, context.source_document_id)
        ) or not _valid_hash(context.content_sha256):
            raise DiagnosticOracleError("oracle_context_identity_invalid")
        if not isinstance(context.content, str) or not context.content:
            raise DiagnosticOracleError("oracle_context_content_invalid")
        if context.context_id in seen:
            raise DiagnosticOracleError("oracle_context_identity_duplicate")
        seen.add(context.context_id)

    route = _validated_route(reviewed_route)
    if variant_id == "oracle_route" and route is None:
        raise DiagnosticOracleError("oracle_reviewed_route_required")
    if variant_id == "oracle_context" and route is not None:
        raise DiagnosticOracleError("oracle_reviewed_route_forbidden")
    return PreparedOracleInput(
        variant=variant,
        case_id=case_id,
        contexts=tuple(contexts),
        reviewed_route=route,
    )


class DiagnosticOracleExecutor:
    """Dispatch Oracle inputs only through an injected evaluation callback."""

    def __init__(self, callback: OracleEvaluationCallback) -> None:
        if not callable(callback):
            raise TypeError("Oracle evaluation callback must be callable")
        self._callback = callback

    async def execute(self, prepared: PreparedOracleInput) -> dict[str, Any]:
        """Execute without exposing an online mode or permitting persistence."""

        if prepared.variant.variant_id not in {"oracle_context", "oracle_route"}:
            raise DiagnosticOracleError("oracle_variant_required")
        result = self._callback(
            case_id=prepared.case_id,
            variant_id=prepared.variant.variant_id,
            contexts=prepared.contexts,
            reviewed_route=prepared.reviewed_route,
            evaluation_only=True,
            write_indexes=False,
            write_history=False,
            write_cache=False,
        )
        if not inspect.isawaitable(result):
            raise DiagnosticOracleError("oracle_executor_not_async")
        resolved = await result
        if not isinstance(resolved, Mapping):
            raise DiagnosticOracleError("oracle_executor_result_invalid")
        return dict(resolved)


__all__ = [
    "DiagnosticOracleError",
    "DiagnosticOracleExecutor",
    "PreparedOracleInput",
    "ReviewedOracleContext",
    "prepare_oracle_input",
]
