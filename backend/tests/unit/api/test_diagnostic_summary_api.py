"""API-boundary tests for additive diagnostic summaries."""

from __future__ import annotations

from api.routers.evaluation_pkg.release_workflows import _release_attempt_response


def _attempt(metrics: dict) -> dict:
    return {
        "attempt_id": "era-1",
        "stage": "baseline",
        "attempt_number": 1,
        "status": "succeeded",
        "reason_code": None,
        "metrics": metrics,
        "started_at": None,
        "finished_at": None,
    }


def test_diagnostic_summary_is_allowlisted_bounded_and_additive() -> None:
    response = _release_attempt_response(
        _attempt(
            {
                "run_id": "run-1",
                "diagnostic_summary": {
                    "schema_version": "category-aware-diagnostic-summary-v1",
                    "identities": {
                        "category_policy_version": "policy-v1",
                        "category_policy_sha256": "a" * 64,
                        "variant_plan_sha256": "b" * 64,
                        "response_snapshot_sha256": "c" * 64,
                        "credential": "must-not-leak",
                    },
                    "categories": {
                        "fully_answerable": {"count": 10, "rate": 0.8, "raw_question": "secret"}
                    },
                    "stages": {"retrieval": {"coverage": 0.75}},
                    "metric_coverage": {"faithfulness": {"scored": 8, "failed": 2}},
                    "route_transitions": {"answered->refused": 2},
                    "hard_gates": ["authorization_leakage"],
                    "case_ids": [f"case-{index}" for index in range(30)],
                    "root_cause": {
                        "decision": "qualification_or_routing",
                        "reason_code": "oracle_route_material_repair",
                        "paired_count": 30,
                        "effect_direction": "positive",
                        "recommend_embedding_change": False,
                        "compared_variants": ["oracle_context", "oracle_route"],
                        "supporting_hashes": ["d" * 64],
                        "raw_context": "private context",
                    },
                    "raw_answer": "private answer",
                },
            }
        )
    )

    payload = response.model_dump()
    summary = payload["diagnostic_summary"]
    assert payload["metrics"] == {"run_id": "run-1"}
    assert summary["availability"] == "available"
    assert summary["categories"]["fully_answerable"] == {
        "count": 10,
        "rate": 0.8,
    }
    assert len(summary["case_ids"]) == 20
    assert summary["root_cause"]["decision"] == "qualification_or_routing"
    rendered = repr(payload)
    for secret in ("must-not-leak", "secret", "private context", "private answer"):
        assert secret not in rendered


def test_historical_and_unknown_diagnostic_schema_are_distinguishable() -> None:
    historical = _release_attempt_response(_attempt({})).diagnostic_summary
    unknown = _release_attempt_response(
        _attempt(
            {
                "diagnostic_summary": {
                    "schema_version": "future-diagnostic-v9",
                    "raw_context": "private",
                }
            }
        )
    ).diagnostic_summary

    assert historical.availability == "not_available"
    assert historical.schema_version is None
    assert unknown.availability == "unsupported_schema"
    assert unknown.schema_version == "future-diagnostic-v9"
    assert unknown.categories == {}
