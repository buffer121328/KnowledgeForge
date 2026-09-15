"""Bounded identity-only candidate capture for evaluation-scoped QA runs."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

from domain.knowledge import RetrievedContext

CANDIDATE_STAGE_SCHEMA_VERSION = "candidate-stage-capture-v1"
CANDIDATE_STAGES = ("dense", "bm25", "graph", "fused", "reranked")
CANDIDATE_STAGE_STATUSES = frozenset(
    {"executed", "executed_empty", "not_executed", "unavailable"}
)


def _bounded_identity(value: Any) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 128 else None


def _content_sha256(context: RetrievedContext) -> str:
    """Hash the retrieved Chunk body, not the source document file digest.

    候选工件必须绑定运行时真正参与排序的正文；适配器自报的摘要
    可能过期或被伪造，一律忽略。
    """
    return hashlib.sha256(context.content.encode("utf-8")).hexdigest()


def _branch_name(context: RetrievedContext) -> str | None:
    value = _bounded_identity(context.retrieval_type)
    if value == "vector":
        return "dense"
    return value


class CandidateStageRecorder:
    """Collect one bounded, raw-text-free snapshot for each retrieval stage."""

    def __init__(self, *, candidate_budget: int) -> None:
        if (
            isinstance(candidate_budget, bool)
            or not isinstance(candidate_budget, int)
            or not 1 <= candidate_budget <= 50
        ):
            raise ValueError("candidate_budget must be between 1 and 50")
        self.candidate_budget = candidate_budget
        self._records: dict[str, dict[str, Any]] = {}

    def record(
        self,
        stage: str,
        contexts: Sequence[RetrievedContext],
        *,
        status: str,
        reason_code: str | None = None,
    ) -> None:
        """Record a stage once, retaining only bounded identities, rank, and score."""

        if stage not in CANDIDATE_STAGES:
            raise ValueError("candidate stage is invalid")
        if stage in self._records:
            raise ValueError(f"candidate stage already recorded: {stage}")
        if status not in CANDIDATE_STAGE_STATUSES:
            raise ValueError("candidate stage status is invalid")
        values = list(contexts)
        if status == "executed" and not values:
            raise ValueError("executed candidate stage requires candidates")
        if status != "executed" and values:
            raise ValueError("non-executed candidate stage cannot contain candidates")
        if reason_code is not None and _bounded_identity(reason_code) is None:
            raise ValueError("candidate stage reason_code is invalid")

        candidates: list[dict[str, Any]] = []
        for rank, context in enumerate(values[: self.candidate_budget], start=1):
            metadata = context.metadata if isinstance(context.metadata, dict) else {}
            context_id = next(
                (
                    identity
                    for key in ("chunk_id", "context_id")
                    if (identity := _bounded_identity(metadata.get(key))) is not None
                ),
                None,
            )
            candidate: dict[str, Any] = {
                "rank": rank,
                "source_document_id": _bounded_identity(
                    metadata.get("source_document_id")
                ),
                "context_id": context_id,
                "content_sha256": _content_sha256(context),
                "branch": _branch_name(context),
            }
            if (
                not isinstance(context.score, bool)
                and isinstance(context.score, (int, float))
                and math.isfinite(float(context.score))
            ):
                candidate["score"] = float(context.score)
            candidates.append(candidate)

        record: dict[str, Any] = {
            "stage": stage,
            "status": status,
            "candidate_budget": self.candidate_budget,
            "candidates": candidates,
        }
        if reason_code is not None:
            record["reason_code"] = reason_code
        self._records[stage] = record

    def snapshot(self) -> list[dict[str, Any]]:
        """Return all stages in stable order, filling genuinely unobserved stages."""

        return [
            dict(
                self._records.get(
                    stage,
                    {
                        "stage": stage,
                        "status": "not_executed",
                        "candidate_budget": self.candidate_budget,
                        "candidates": [],
                        "reason_code": "stage_not_observed",
                    },
                )
            )
            for stage in CANDIDATE_STAGES
        ]


__all__ = [
    "CANDIDATE_STAGE_SCHEMA_VERSION",
    "CANDIDATE_STAGE_STATUSES",
    "CANDIDATE_STAGES",
    "CandidateStageRecorder",
]
