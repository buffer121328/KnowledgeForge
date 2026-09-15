"""Bounded identities and checkpoint keys for category-aware outcomes."""

from __future__ import annotations

from typing import Any, Literal, Mapping

MetricFamily = Literal["ragas", "deterministic", "retrieval", "safety"]
OutcomeKey = tuple[str, ...]

_FAMILIES = frozenset({"ragas", "deterministic", "retrieval", "safety"})
_IDENTITY_FIELDS = frozenset(
    {
        "category_policy_version",
        "category_policy_sha256",
        "variant_id",
        "response_snapshot_sha256",
    }
)
_JUDGE_FIELDS = frozenset({"provider", "model", "config_sha256"})


class EvaluationOutcomeError(ValueError):
    """A diagnostic outcome identity is malformed or unsafe to persist."""


def _bounded_string(value: Any, *, maximum: int = 200) -> bool:
    return isinstance(value, str) and 0 < len(value) <= maximum


def attach_diagnostic_identity(
    outcome: dict[str, Any],
    *,
    metric_family: MetricFamily | str,
    metric_version: str,
    identity: Mapping[str, Any],
    judge_identity: Mapping[str, Any] | None = None,
) -> None:
    """Attach only allowlisted run, policy, metric, and optional Judge identity."""

    if metric_family not in _FAMILIES or not _bounded_string(metric_version):
        raise EvaluationOutcomeError("metric_identity_invalid")
    if set(identity) != _IDENTITY_FIELDS:
        raise EvaluationOutcomeError("identity_fields_invalid")
    if not all(_bounded_string(identity[field]) for field in _IDENTITY_FIELDS):
        raise EvaluationOutcomeError("identity_fields_invalid")
    if len(identity["category_policy_sha256"]) != 64:
        raise EvaluationOutcomeError("identity_fields_invalid")
    if len(identity["response_snapshot_sha256"]) != 64:
        raise EvaluationOutcomeError("identity_fields_invalid")
    if judge_identity is not None:
        if not set(judge_identity).issubset(_JUDGE_FIELDS) or not judge_identity:
            raise EvaluationOutcomeError("judge_identity_invalid")
        if not all(_bounded_string(value) for value in judge_identity.values()):
            raise EvaluationOutcomeError("judge_identity_invalid")

    outcome.update(
        {
            "metric_family": metric_family,
            "metric_version": metric_version,
            **{field: identity[field] for field in sorted(_IDENTITY_FIELDS)},
        }
    )
    if judge_identity is not None:
        outcome["judge_identity"] = {
            field: judge_identity[field] for field in sorted(judge_identity)
        }


def outcome_checkpoint_key(outcome: Mapping[str, Any]) -> OutcomeKey | None:
    """Return a legacy-compatible key that isolates every diagnostic identity."""

    legacy = tuple(
        outcome.get(field) for field in ("benchmark_id", "retrieval_mode", "metric")
    )
    if not all(_bounded_string(value) for value in legacy):
        return None
    present = {field for field in _IDENTITY_FIELDS | {"metric_family"} if field in outcome}
    if not present:
        return legacy  # type: ignore[return-value]
    required = _IDENTITY_FIELDS | {"metric_family"}
    if present != required or outcome.get("metric_family") not in _FAMILIES:
        return None
    diagnostic = tuple(outcome.get(field) for field in sorted(required))
    if not all(_bounded_string(value) for value in diagnostic):
        return None
    return (*legacy, *diagnostic)  # type: ignore[return-value]


__all__ = [
    "EvaluationOutcomeError",
    "MetricFamily",
    "OutcomeKey",
    "attach_diagnostic_identity",
    "outcome_checkpoint_key",
]
