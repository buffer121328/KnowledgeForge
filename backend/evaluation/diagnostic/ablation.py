"""Resumable single-variable retrieval ablation orchestration."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from evaluation.diagnostic.variants import FrozenVariantPlan

AblationCallback = Callable[..., Awaitable[Mapping[str, Any]]]
RETRIEVAL_ABLATION_VARIANTS = (
    "dense_only",
    "bm25_only",
    "graph_only",
    "fused",
    "reranked",
)


def _bounded_text(value: Any, *, maximum: int = 128) -> bool:
    return isinstance(value, str) and 0 < len(value) <= maximum


def _valid_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_jsonl_secure(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record, ensure_ascii=True, separators=(",", ":"), sort_keys=True
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _outcome_key(outcome: Mapping[str, Any]) -> tuple[str, str] | None:
    case_id = outcome.get("case_id")
    variant_id = outcome.get("variant_id")
    if not _bounded_text(case_id) or variant_id not in RETRIEVAL_ABLATION_VARIANTS:
        return None
    return str(case_id), str(variant_id)


class DiagnosticAblationOrchestrator:
    """Run five frozen retrieval variants with safe per-pair checkpoints."""

    def __init__(self, callback: AblationCallback, *, checkpoint_path: Path) -> None:
        if not callable(callback):
            raise TypeError("Ablation callback must be callable")
        self._callback = callback
        self._checkpoint_path = Path(checkpoint_path)

    def _load(
        self, plan: FrozenVariantPlan, valid_keys: set[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        if not self._checkpoint_path.is_file():
            return []
        candidates: list[dict[str, Any]] = []
        try:
            for line in self._checkpoint_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        candidates.append(value)
        except (OSError, json.JSONDecodeError):
            return []
        eligible = [
            item
            for item in candidates
            if _outcome_key(item) in valid_keys
            and item.get("status") == "succeeded"
            and item.get("plan_sha256") == plan.plan_sha256
            and item.get("pairing_identity_sha256")
            == plan.pairing_identity_sha256
            and _valid_hash(item.get("artifact_sha256"))
            and _valid_hash(item.get("response_snapshot_sha256"))
        ]
        counts = Counter(_outcome_key(item) for item in eligible)
        return [item for item in eligible if counts[_outcome_key(item)] == 1]

    async def run(
        self, plan: FrozenVariantPlan, *, case_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Resume exact successes and retry every missing or failed pair."""

        cases = tuple(case_ids)
        if (
            not cases
            or len(cases) != len(set(cases))
            or len(cases) > 1000
            or not all(_bounded_text(case_id) for case_id in cases)
        ):
            raise ValueError("ablation_case_ids_invalid")
        valid_keys = {
            (case_id, variant_id)
            for case_id in cases
            for variant_id in RETRIEVAL_ABLATION_VARIANTS
        }
        outcomes = self._load(plan, valid_keys)
        completed = {
            key for item in outcomes if (key := _outcome_key(item)) is not None
        }

        def checkpoint(outcome: dict[str, Any]) -> None:
            key = _outcome_key(outcome)
            if key is None:
                return
            for index, current in enumerate(outcomes):
                if _outcome_key(current) == key:
                    outcomes[index] = outcome
                    break
            else:
                outcomes.append(outcome)
            outcomes.sort(key=lambda item: (cases.index(item["case_id"]), RETRIEVAL_ABLATION_VARIANTS.index(item["variant_id"])))
            _write_jsonl_secure(self._checkpoint_path, outcomes)

        for case_id in cases:
            for variant_id in RETRIEVAL_ABLATION_VARIANTS:
                if (case_id, variant_id) in completed:
                    continue
                variant = plan.variant(variant_id)
                pipeline = variant.factor_values["retrieval_pipeline"]
                outcome: dict[str, Any] = {
                    "case_id": case_id,
                    "variant_id": variant_id,
                    "retrieval_pipeline": pipeline,
                    "status": "failed",
                    "reason_code": "variant_executor_failed",
                    "artifact_sha256": None,
                    "response_snapshot_sha256": None,
                    "plan_sha256": plan.plan_sha256,
                    "pairing_identity_sha256": plan.pairing_identity_sha256,
                }
                try:
                    result = await self._callback(
                        case_id=case_id,
                        variant_id=variant_id,
                        retrieval_pipeline=pipeline,
                        candidate_budget=plan.common_identity["candidate_budget"],
                        evaluation_only=True,
                        write_cache=False,
                    )
                    artifact_sha256 = result.get("artifact_sha256")
                    response_sha256 = result.get("response_snapshot_sha256")
                    if (
                        result.get("status") != "succeeded"
                        or not _valid_hash(artifact_sha256)
                        or not _valid_hash(response_sha256)
                    ):
                        outcome["reason_code"] = "variant_executor_result_invalid"
                    else:
                        outcome.update(
                            {
                                "status": "succeeded",
                                "reason_code": None,
                                "artifact_sha256": artifact_sha256,
                                "response_snapshot_sha256": response_sha256,
                            }
                        )
                except Exception as error:
                    if type(error).__name__ == "SoftTimeLimitExceeded":
                        raise
                checkpoint(outcome)
        return outcomes


__all__ = [
    "RETRIEVAL_ABLATION_VARIANTS",
    "DiagnosticAblationOrchestrator",
]
