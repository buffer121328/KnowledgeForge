"""Acceptance tests for evidence-gate metrics and promotion decisions."""

from __future__ import annotations

import pytest

from evaluation.evidence_gate.metrics import (
    REQUIRED_RUN_IDENTITY_FIELDS,
    aggregate_category_custom_metrics,
    aggregate_evidence_gate_metrics,
    evaluate_category_custom_metrics,
    evaluate_evidence_gate_promotion,
)
from evaluation.benchmarks.category_metric_policy import get_category_metric_contract


def _identity() -> dict[str, object]:
    return {
        field: (
            ["dense_bm25_graph"]
            if field == "retrieval_modes"
            else "identity-v1"
        )
        for field in REQUIRED_RUN_IDENTITY_FIELDS
    }


def _record(
    benchmark_id: str,
    expected: str,
    actual: str,
    *,
    citation_expected: str | None = None,
    citation_actual: str | None = None,
    grounded: bool | None = None,
) -> dict[str, object]:
    claims = []
    citations = []
    if actual in {"answered", "partially_answered"}:
        claims = [
            {
                "claim_id": f"claim-{benchmark_id}",
                "material": True,
                "citation_ids": (
                    [f"citation-{benchmark_id}"] if citation_actual else []
                ),
            }
        ]
        if citation_actual:
            citations = [
                {
                    "citation_id": f"citation-{benchmark_id}",
                    "context_id": citation_actual,
                }
            ]
    return {
        "benchmark_id": benchmark_id,
        "status": "succeeded",
        "expected_response_status": expected,
        "response_status": actual,
        "expected_evidence_states": ["direct_evidence"],
        "evidence_state": "direct_evidence",
        "expected_citation_context_ids": (
            [citation_expected] if citation_expected else []
        ),
        "claims": claims,
        "citations": citations,
        "grounding_result": None if grounded is None else {"passed": grounded},
    }


def test_metrics_cover_refusal_routes_partial_conflict_citations_and_grounding() -> None:
    records = [
        _record("answer-ok", "answered", "answered", citation_expected="ctx-a", citation_actual="ctx-a", grounded=True),
        _record("answer-refused", "answered", "insufficient_evidence"),
        _record("no-answer-ok", "insufficient_evidence", "insufficient_evidence"),
        _record("no-answer-hallucinated", "insufficient_evidence", "answered", citation_expected=None, citation_actual="ctx-x", grounded=False),
        _record("partial-ok", "partially_answered", "partially_answered", citation_expected="ctx-p", citation_actual="ctx-p", grounded=True),
        _record("partial-missed", "partially_answered", "insufficient_evidence"),
        _record("conflict-ok", "conflicting_evidence", "conflicting_evidence"),
        _record("conflict-missed", "conflicting_evidence", "answered", citation_actual="ctx-z", grounded=False),
        _record("citation-missing", "answered", "answered", citation_expected="ctx-c", grounded=False),
        _record("citation-wrong", "answered", "answered", citation_expected="ctx-good", citation_actual="ctx-bad", grounded=True),
    ]

    report = aggregate_evidence_gate_metrics(records, run_identity=_identity())
    metrics = report["metrics"]

    assert metrics["refusal_precision"] == 0.5
    assert metrics["refusal_recall"] == 0.5
    assert metrics["no_answer_hallucination_rate"] == 0.5
    assert metrics["answerable_false_refusal_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["partial_answer_recognition_rate"] == 0.5
    assert metrics["conflict_recognition_rate"] == 0.5
    assert metrics["claim_citation_coverage"] == pytest.approx(5 / 6, abs=1e-6)
    assert metrics["citation_correctness"] == 0.4
    assert metrics["groundedness_pass_rate"] == 0.5
    assert report["failure_sample_ids"]["no_answer_hallucination_rate"] == [
        "no-answer-hallucinated",
        "conflict-missed",
    ]


def test_promotion_fails_on_every_required_threshold_and_missing_identity() -> None:
    report = aggregate_evidence_gate_metrics(
        [
            _record(
                "bad-no-answer",
                "insufficient_evidence",
                "answered",
                citation_actual="ctx-x",
                grounded=False,
            )
        ],
        run_identity={"dataset_version": "v1"},
    )

    decision = evaluate_evidence_gate_promotion(
        report,
        thresholds={
            "refusal_recall": 0.99,
            "no_answer_hallucination_rate": 0.01,
            "groundedness_pass_rate": 0.99,
        },
    )

    assert decision["decision"] == "blocked"
    fields = {reason["field"] for reason in decision["reasons"]}
    assert {
        "metrics.refusal_recall",
        "metrics.no_answer_hallucination_rate",
        "metrics.groundedness_pass_rate",
        "run_identity.answer_model",
        "run_identity.grounding_policy_version",
    } <= fields
    hallucination = next(
        reason
        for reason in decision["reasons"]
        if reason["field"] == "metrics.no_answer_hallucination_rate"
    )
    assert hallucination["sample_ids"] == ["bad-no-answer"]


def test_promotion_passes_only_when_all_metrics_and_identity_pass() -> None:
    report = aggregate_evidence_gate_metrics(
        [
            _record("answer", "answered", "answered", citation_expected="ctx", citation_actual="ctx", grounded=True),
            _record("refusal", "insufficient_evidence", "insufficient_evidence"),
            _record("partial", "partially_answered", "partially_answered", citation_expected="ctx-p", citation_actual="ctx-p", grounded=True),
            _record("conflict", "conflicting_evidence", "conflicting_evidence"),
        ],
        run_identity=_identity(),
    )

    decision = evaluate_evidence_gate_promotion(
        report,
        thresholds={
            "refusal_precision": 1.0,
            "refusal_recall": 1.0,
            "no_answer_hallucination_rate": 0.0,
            "answerable_false_refusal_rate": 0.0,
            "partial_answer_recognition_rate": 1.0,
            "conflict_recognition_rate": 1.0,
            "claim_citation_coverage": 1.0,
            "citation_correctness": 1.0,
            "groundedness_pass_rate": 1.0,
        },
    )

    assert decision == {"decision": "passed", "reasons": []}


def _category_record(category: str) -> dict[str, object]:
    routes = {
        "fully_answerable": ("answered", "direct_evidence", ["direct_support"]),
        "partially_answerable": ("partially_answered", "partial_evidence", ["partial_support"]),
        "conflicting": ("conflicting_evidence", "conflicting_evidence", ["material_conflict"]),
        "background_only": ("insufficient_evidence", "relevant_background", ["background_only"]),
        "missing_version_or_date": ("needs_clarification", "relevant_background", ["missing_question_detail"]),
        "missing_business_record": ("insufficient_evidence", "relevant_background", ["background_only"]),
        "completely_unanswerable": ("insufficient_evidence", "insufficient_evidence", ["zero_results"]),
        "authorization_filtered": ("insufficient_evidence", "insufficient_evidence", ["authorized_contexts_empty"]),
        "single_branch_unavailable": ("answered", "direct_evidence", ["direct_support", "partial_dependency_unavailable"]),
        "all_branches_unavailable": ("source_unavailable", "insufficient_evidence", ["all_dependencies_unavailable"]),
        "prompt_injection": ("human_review_required", "invalid_provenance", ["prompt_injection_detected"]),
    }
    response_status, evidence_state, reasons = routes[category]
    answer = response_status in {"answered", "partially_answered"}
    expected_contexts = ["ctx-a", "ctx-b"] if category == "conflicting" else ["ctx-a"]
    if category in {"all_branches_unavailable", "completely_unanswerable", "prompt_injection"}:
        expected_contexts = []
    missing = {
        "partially_answerable": ["business_record"],
        "background_only": ["business_record"],
        "missing_version_or_date": ["effective_date"],
        "missing_business_record": ["business_record"],
    }.get(category, [])
    expected_branches = {"dense": "available", "bm25": "available", "graph": "available"}
    if category == "single_branch_unavailable":
        expected_branches["bm25"] = "unavailable"
    if category == "all_branches_unavailable":
        expected_branches = {key: "unavailable" for key in expected_branches}
    citation_contexts = expected_contexts if answer else []
    claims = [{"claim_id": "claim-a", "material": True, "citation_ids": ["cite-a"]}] if answer else []
    citations = [{"citation_id": "cite-a", "context_id": citation_contexts[0]}] if citation_contexts else []
    return {
        "benchmark_id": f"case-{category}",
        "category": category,
        "status": "succeeded",
        "expected_response_status": response_status,
        "response_status": response_status,
        "expected_evidence_states": [evidence_state],
        "evidence_state": evidence_state,
        "expected_reason_codes": reasons,
        "observed_reason_codes": reasons,
        "expected_missing_information_fields": missing,
        "observed_missing_information_fields": missing,
        "expected_citation_context_ids": citation_contexts,
        "expected_evidence_context_ids": expected_contexts,
        "retrieved_chunk_ids": expected_contexts,
        "expected_source_document_ids": ["protected-doc"] if category == "authorization_filtered" else ["doc-a"],
        "retrieved_source_document_ids": (
            []
            if category in {"authorization_filtered", "prompt_injection"}
            else ["doc-a"]
        ),
        "expected_branch_availability": expected_branches,
        "stages": {"retrieval": {"branch_availability": expected_branches}},
        "claims": claims,
        "citations": citations,
        "grounding_result": {"passed": True, "accepted_claim_ids": ["claim-a"], "rejected_claim_ids": []} if answer else None,
        "security_actions": ["prompt_injection"] if category == "prompt_injection" else [],
    }


@pytest.mark.parametrize(
    "category",
    [
        "fully_answerable", "partially_answerable", "conflicting", "background_only",
        "missing_version_or_date", "missing_business_record", "completely_unanswerable",
        "authorization_filtered", "single_branch_unavailable", "all_branches_unavailable",
        "prompt_injection",
    ],
)
def test_each_category_scores_its_exact_custom_metric_contract(category: str) -> None:
    outcomes = evaluate_category_custom_metrics(_category_record(category))
    expected = get_category_metric_contract(category)

    assert {item["metric"] for item in outcomes} == {
        item.metric_id for item in expected.custom
    }
    assert all(item["status"] == "scored" for item in outcomes)
    assert all(
        item["score"] == (1.0 if item["direction"] == "minimum" else 0.0)
        for item in outcomes
    )


def test_unsupported_claim_authorization_and_injection_leakage_are_visible() -> None:
    partial = _category_record("partially_answerable")
    partial["grounding_result"] = {
        "passed": False,
        "accepted_claim_ids": [],
        "rejected_claim_ids": ["claim-a"],
    }
    authorization = _category_record("authorization_filtered")
    authorization["retrieved_source_document_ids"] = ["protected-doc"]
    authorization["citations"] = [{"citation_id": "cite-a", "context_id": "protected-ctx"}]
    injection = _category_record("prompt_injection")
    injection.update(response_status="answered", claims=[{"claim_id": "claim-a"}], citations=[])

    partial_scores = {item["metric"]: item["score"] for item in evaluate_category_custom_metrics(partial)}
    auth_scores = {item["metric"]: item["score"] for item in evaluate_category_custom_metrics(authorization)}
    injection_scores = {item["metric"]: item["score"] for item in evaluate_category_custom_metrics(injection)}

    assert partial_scores["unsupported_claim_rate"] == 1.0
    assert auth_scores["unauthorized_citation_rate"] == 1.0
    assert auth_scores["cross_scope_leakage_rate"] == 1.0
    assert injection_scores["instruction_data_leakage_rate"] == 1.0


def test_prompt_injection_internal_retrieval_is_not_reported_as_user_visible_leakage() -> None:
    injection = _category_record("prompt_injection")
    injection["retrieved_source_document_ids"] = ["internal-doc"]
    injection["retrieved_chunk_ids"] = ["internal-chunk"]

    scores = {
        item["metric"]: item["score"]
        for item in evaluate_category_custom_metrics(injection)
    }

    assert scores["instruction_data_leakage_rate"] == 0.0


def test_skipped_generation_and_grounding_are_not_fabricated_as_zero_scores() -> None:
    record = _category_record("fully_answerable")
    record["claims"] = []
    record["citations"] = []
    record["grounding_result"] = None
    record["stages"] = {
        **record["stages"],
        "generation": {"status": "skipped"},
        "grounding": {"status": "skipped"},
    }

    outcomes = {
        item["metric"]: item for item in evaluate_category_custom_metrics(record)
    }

    assert outcomes["exact_evidence_retrieval"]["status"] == "scored"
    assert outcomes["response_route_correctness"]["status"] == "scored"
    assert outcomes["citation_correctness"]["status"] == "not_applicable"
    assert outcomes["citation_correctness"]["score"] is None
    assert outcomes["citation_correctness"]["reason_code"] == "evaluation_stage_disabled"
    assert outcomes["grounding_correctness"]["status"] == "not_applicable"

    report = aggregate_category_custom_metrics([record])
    assert report["metrics"]["citation_correctness"]["scored"] == 0
    assert report["metrics"]["citation_correctness"]["not_applicable"] == 1
    assert report["metrics"]["citation_correctness"]["mean"] is None
    assert report["metrics"]["citation_correctness"]["not_applicable_reasons"] == {
        "evaluation_stage_disabled": 1
    }


def test_category_custom_metric_aggregation_is_reportable() -> None:
    rows = [_category_record("fully_answerable"), _category_record("prompt_injection")]
    report = aggregate_category_custom_metrics(rows)

    assert report["schema_version"] == "category-custom-metrics-v1"
    assert report["case_count"] == 2
    assert report["outcome_count"] == 7
    assert report["metrics"]["response_route_correctness"]["mean"] == 1.0
    assert report["metrics"]["injection_detection"]["category_means"]["prompt_injection"] == 1.0
    assert report["metrics"]["injection_detection"]["direction"] == "minimum"
    assert report["metrics"]["instruction_data_leakage_rate"]["direction"] == "maximum"
    assert report["metrics"]["instruction_data_leakage_rate"]["hard_gate"] is True
