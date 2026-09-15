"""Safety acceptance tests for the diagnostic-only Oracle boundary."""

from __future__ import annotations

from typing import Any

import pytest

from evaluation.diagnostic.oracle import (
    DiagnosticOracleError,
    DiagnosticOracleExecutor,
    ReviewedOracleContext,
    prepare_oracle_input,
)
from evaluation.diagnostic.variants import FrozenVariantPlan


def _plan() -> FrozenVariantPlan:
    return FrozenVariantPlan.create(
        {
            "corpus_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "authorization_scope_sha256": "c" * 64,
            "response_snapshot_sha256": "d" * 64,
            "query_rewrite_version": "query-v1",
            "embedding_model": "embedding-v1",
            "answer_model": "answer-v1",
            "candidate_budget_version": "budget-v1",
            "candidate_budget": 8,
            "category_policy_version": "policy-v1",
            "tokenizer_version": "tokenizer-v1",
            "index_version": "index-v1",
            "threshold_version": "threshold-v1",
            "prompt_version": "prompt-v1",
            "answer_schema_version": "answer-schema-v1",
            "grounding_policy_version": "grounding-v1",
            "cache_policy": "disabled",
        }
    )


def _context(**overrides: Any) -> ReviewedOracleContext:
    values: dict[str, Any] = {
        "context_id": "context-1",
        "source_document_id": "document-1",
        "content_sha256": "e" * 64,
        "manifest_sha256": "b" * 64,
        "authorization_scope_sha256": "c" * 64,
        "content": "private reviewed context text",
        "reviewed": True,
    }
    values.update(overrides)
    return ReviewedOracleContext(**values)


def test_oracle_preparation_binds_reviewed_lineage_and_safe_identity_only() -> None:
    prepared = prepare_oracle_input(
        plan=_plan(),
        variant_id="oracle_route",
        case_id="case-1",
        contexts=[_context()],
        reviewed_route={
            "response_status": "answered",
            "evidence_states": ["direct_evidence"],
        },
        runtime_manifest_sha256="b" * 64,
        runtime_authorization_scope_sha256="c" * 64,
    )

    summary = prepared.safe_identity()
    rendered = repr(summary)
    assert summary["variant_id"] == "oracle_route"
    assert summary["reviewed_contexts"] == [
        {
            "context_id": "context-1",
            "source_document_id": "document-1",
            "content_sha256": "e" * 64,
        }
    ]
    assert len(summary["reviewed_route_sha256"]) == 64
    assert "private reviewed context text" not in rendered
    assert "/Users/" not in rendered


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        (
            {"runtime_authorization_scope_sha256": "f" * 64},
            "oracle_authorization_scope_mismatch",
        ),
        (
            {"runtime_manifest_sha256": "f" * 64},
            "oracle_manifest_mismatch",
        ),
        (
            {"contexts": [_context(reviewed=False)]},
            "oracle_context_not_reviewed",
        ),
        (
            {"contexts": [_context(manifest_sha256="f" * 64)]},
            "oracle_context_lineage_mismatch",
        ),
        (
            {"contexts": [_context(authorization_scope_sha256="f" * 64)]},
            "oracle_context_authorization_mismatch",
        ),
    ],
)
def test_oracle_preparation_fails_before_dispatch(
    kwargs: dict[str, Any], reason: str
) -> None:
    parameters: dict[str, Any] = {
        "plan": _plan(),
        "variant_id": "oracle_context",
        "case_id": "case-1",
        "contexts": [_context()],
        "reviewed_route": None,
        "runtime_manifest_sha256": "b" * 64,
        "runtime_authorization_scope_sha256": "c" * 64,
    }
    parameters.update(kwargs)

    with pytest.raises(DiagnosticOracleError, match=reason):
        prepare_oracle_input(**parameters)


@pytest.mark.asyncio
async def test_executor_is_evaluation_only_and_disables_online_side_effects() -> None:
    calls: list[dict[str, Any]] = []

    async def execute(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "succeeded", "response_snapshot_sha256": "f" * 64}

    prepared = prepare_oracle_input(
        plan=_plan(),
        variant_id="oracle_context",
        case_id="case-1",
        contexts=[_context()],
        reviewed_route=None,
        runtime_manifest_sha256="b" * 64,
        runtime_authorization_scope_sha256="c" * 64,
    )

    result = await DiagnosticOracleExecutor(execute).execute(prepared)

    assert result["status"] == "succeeded"
    assert len(calls) == 1
    assert calls[0]["evaluation_only"] is True
    assert calls[0]["write_indexes"] is False
    assert calls[0]["write_history"] is False
    assert calls[0]["write_cache"] is False
    assert calls[0]["variant_id"] == "oracle_context"
    assert calls[0]["contexts"][0].content == "private reviewed context text"


@pytest.mark.asyncio
async def test_non_oracle_variant_cannot_reach_oracle_executor() -> None:
    calls = 0

    async def execute(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    with pytest.raises(DiagnosticOracleError, match="oracle_variant_required"):
        prepare_oracle_input(
            plan=_plan(),
            variant_id="observed",
            case_id="case-1",
            contexts=[_context()],
            reviewed_route=None,
            runtime_manifest_sha256="b" * 64,
            runtime_authorization_scope_sha256="c" * 64,
        )
    assert calls == 0
