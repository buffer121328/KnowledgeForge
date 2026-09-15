"""Acceptance tests for deterministic offline evaluation reports and gates."""

from __future__ import annotations

import json
import stat
from pathlib import Path

from evaluation.release.quality_gates import (
    QUALITY_REPORT_VERSION,
    aggregate_quality_report,
    evaluate_release_gate,
    validate_quality_report,
)
from evaluation.scripts.run_quality_gate import main as run_quality_gate_main


DOC_A = "7560dc54-81cb-4d28-bbb1-205f060ad678"
DOC_B = "08bb9178-7daa-40d3-b8df-601920a6df2c"
DOC_C = "a61cfe75-666a-422f-a32f-7a28c40a2621"


def _metadata() -> dict[str, object]:
    return {
        "run_id": "20260726T000000Z-ab12cd34",
        "benchmark_sha256": "a" * 64,
        "benchmark_source": "benchmark.jsonl",
        "code_revision": "b" * 40,
        "retrieval_modes": ["vector", "hybrid"],
        "started_at": "2026-07-26T00:00:00+00:00",
    }


def _record(
    benchmark_id: str,
    mode: str,
    *,
    category: str = "single_document_fact",
    retrieved: list[str] | None = None,
    reference: list[str] | None = None,
    expected_refusal: bool = False,
    refused: bool = False,
    status: str = "succeeded",
    latency_ms: float = 100.0,
) -> dict[str, object]:
    return {
        "benchmark_id": benchmark_id,
        "retrieval_mode": mode,
        "category": category,
        "status": status,
        "retrieved_context_ids": retrieved if retrieved is not None else [DOC_A],
        "reference_context_ids": reference if reference is not None else [DOC_A],
        "expected_refusal": expected_refusal,
        "refused": refused,
        "latency_ms": latency_ms,
        "exception": "RuntimeError" if status == "failed" else None,
    }


def test_aggregates_metrics_and_strata_deterministically() -> None:
    records = [
        _record("fact-1", "vector", retrieved=[DOC_A, DOC_B], latency_ms=100),
        _record("fact-1", "hybrid", retrieved=[DOC_A], latency_ms=300),
        _record(
            "refusal-1",
            "vector",
            category="insufficient_evidence",
            expected_refusal=True,
            refused=True,
            retrieved=[DOC_C],
            reference=[DOC_A],
            latency_ms=200,
        ),
    ]

    report = aggregate_quality_report(_metadata(), records, smoke_threshold=1)
    repeated = aggregate_quality_report(_metadata(), records, smoke_threshold=1)

    assert report == repeated
    assert report["schema_version"] == QUALITY_REPORT_VERSION
    assert report["run_classification"] == "baseline"
    assert report["counts"] == {
        "total": 3,
        "succeeded": 3,
        "failed": 0,
        "invalid_provenance": 0,
        "completed": 3,
        "id_scored": 3,
    }
    assert report["metrics"]["id_based_context_precision"] == 0.5
    assert report["metrics"]["id_based_context_recall"] == 0.666667
    assert report["metrics"]["refusal"]["f1"] == 1.0
    assert report["latency_ms"]["p50"] == 200.0
    assert report["latency_ms"]["p95"] == 290.0
    assert set(report["by_retrieval_mode"]) == {"vector", "hybrid"}
    assert "question" not in json.dumps(report)


def test_invalid_provenance_is_counted_but_not_scored() -> None:
    records = [
        _record("ok", "vector"),
        _record("invalid", "vector", status="invalid_provenance", retrieved=[]),
    ]
    report = aggregate_quality_report(_metadata(), records, smoke_threshold=1)

    assert report["counts"]["invalid_provenance"] == 1
    assert report["counts"]["id_scored"] == 1
    assert report["metrics"]["id_based_context_precision"] == 1.0
    assert report["metrics"]["id_based_context_recall"] == 1.0


def test_incompatible_legacy_id_namespaces_are_not_scored_as_zero() -> None:
    record = _record("mixed", "vector", retrieved=[DOC_A], reference=[f"{DOC_A}#chunk-1"])
    record["id_namespaces"] = {
        "retrieved_context_ids": "source_document",
        "reference_context_ids": "context",
    }

    report = aggregate_quality_report(_metadata(), [record], smoke_threshold=1)

    assert report["counts"]["id_scored"] == 0
    assert report["metrics"]["id_based_context_precision"] is None
    assert report["metrics"]["id_based_context_recall"] is None


def test_smoke_report_has_no_significance_and_gate_blocks() -> None:
    report = aggregate_quality_report(_metadata(), [_record("one", "vector")], smoke_threshold=30)

    assert report["run_classification"] == "smoke_only"
    assert "significance" not in report
    decision = evaluate_release_gate(report)
    assert decision["decision"] == "blocked"
    assert "smoke_only" in {item["code"] for item in decision["reasons"]}


def test_gate_fails_closed_for_missing_metadata_and_failed_records() -> None:
    report = aggregate_quality_report(
        _metadata(),
        [_record("failed", "vector", status="failed")],
        smoke_threshold=1,
    )
    report["metadata"].pop("benchmark_sha256")

    errors = validate_quality_report(report)
    decision = evaluate_release_gate(report)

    assert any(error["field"] == "metadata.benchmark_sha256" for error in errors)
    assert decision["decision"] == "blocked"
    assert {item["code"] for item in decision["reasons"]} >= {"missing_metadata", "failed_records"}


def test_gate_accepts_complete_non_smoke_report_with_required_modes() -> None:
    records = [_record(f"sample-{index}", "vector") for index in range(2)]
    records.extend(_record(f"sample-{index}", "hybrid") for index in range(2))
    report = aggregate_quality_report(_metadata(), records, smoke_threshold=2)

    decision = evaluate_release_gate(report, required_modes=["vector", "hybrid"])

    assert decision == {"decision": "passed", "reasons": []}


def test_report_does_not_include_sensitive_raw_values(tmp_path: Path) -> None:
    secret_question = "question-with-secret-very-private"
    records = [_record("safe", "vector")]
    records[0].update(
        {
            "question": secret_question,
            "response": "raw answer",
            "contexts": [{"content": "raw context", "source": "/private/secret.pdf"}],
            "exception": "internal endpoint https://provider.internal/token",
        }
    )
    report = aggregate_quality_report(_metadata(), records, smoke_threshold=1)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    serialized = path.read_text(encoding="utf-8")
    assert secret_question not in serialized
    assert "raw context" not in serialized
    assert "/private/secret.pdf" not in serialized
    assert "provider.internal" not in serialized


def test_cli_writes_report_and_blocks_failed_records(tmp_path: Path) -> None:
    metadata_path = tmp_path / "run_metadata.json"
    responses_path = tmp_path / "responses.jsonl"
    output_path = tmp_path / "quality-report.json"
    metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")
    responses_path.write_text(json.dumps(_record("failed", "vector", status="failed")) + "\n", encoding="utf-8")

    exit_code = run_quality_gate_main(
        [
            "--metadata-json",
            str(metadata_path),
            "--responses-jsonl",
            str(responses_path),
            "--output-json",
            str(output_path),
            "--smoke-threshold",
            "1",
            "--release-gate",
        ]
    )

    assert exit_code == 3
    assert output_path.is_file()
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600


NATIVE_STRATEGIES = ["dense", "dense_graph", "dense_bm25", "dense_bm25_graph"]


def _native_metadata() -> dict[str, object]:
    """Return complete four-way comparison metadata and promotion evidence."""
    metadata = _metadata()
    metadata.update(
        {
            "retrieval_modes": list(NATIVE_STRATEGIES),
            "corpus_revision": "company-demo-v1",
            "evidence_class": "production_service_benchmark",
            "knowledge_revision": 7,
            "embedding_model": "text-embedding-v4",
            "answer_model": "deepseek-v4-flash",
            "prompt_id": "qa-generation-v1",
            "retrieval_config_id": "native-rrf-v1",
            "top_k": 8,
            "cache_policy": "disabled",
            "authorization_scope_fixture": "tenant-department-v1",
            "bm25": {
                "schema_version": "native-bm25-v1",
                "tokenizer_version": "unicode-cjk-bigram-v1",
                "k1": 1.5,
                "b": 0.75,
                "index_generation": "generation-1",
            },
            "release_evidence": {
                "tenant_isolation": True,
                "department_isolation": True,
                "sparse_lifecycle": {
                    "add": True,
                    "update": True,
                    "delete": True,
                    "rebuild": True,
                },
                "degradation": {
                    "unavailable_snapshot": True,
                    "corrupt_snapshot": True,
                },
                "rollback": {
                    "configuration_only": True,
                    "target_strategy": "dense_graph",
                },
                "cost": {"observed": 1.25, "budget": 2.0, "unit": "usd"},
            },
        }
    )
    return metadata


def _native_records(*, latency_ms: float = 100.0) -> list[dict[str, object]]:
    """Return a non-smoke, provenance-complete record set for every strategy."""
    return [
        _record(f"sample-{index}", mode, latency_ms=latency_ms)
        for mode in NATIVE_STRATEGIES
        for index in range(2)
    ]


def test_four_way_report_preserves_frozen_variable_and_bm25_metadata() -> None:
    """Aggregation keeps only bounded comparison metadata needed for auditability."""
    report = aggregate_quality_report(
        _native_metadata(),
        _native_records(),
        smoke_threshold=2,
    )

    assert report["retrieval_modes"] == NATIVE_STRATEGIES
    assert set(report["by_retrieval_mode"]) == set(NATIVE_STRATEGIES)
    assert report["metadata"]["corpus_revision"] == "company-demo-v1"
    assert report["metadata"]["knowledge_revision"] == 7
    assert report["metadata"]["authorization_scope_fixture"] == "tenant-department-v1"
    assert report["metadata"]["bm25"] == {
        "schema_version": "native-bm25-v1",
        "tokenizer_version": "unicode-cjk-bigram-v1",
        "k1": 1.5,
        "b": 0.75,
        "index_generation": "generation-1",
    }


def test_bm25_promotion_gate_blocks_missing_isolation_lifecycle_and_rollback() -> None:
    """Quality gain cannot compensate for missing governance or lifecycle evidence."""
    metadata = _native_metadata()
    evidence = metadata["release_evidence"]
    assert isinstance(evidence, dict)
    evidence["tenant_isolation"] = False
    evidence["sparse_lifecycle"] = {"add": True, "update": True, "delete": False}
    evidence["degradation"] = {"unavailable_snapshot": True}
    evidence["rollback"] = {"configuration_only": False, "target_strategy": "dense_graph"}
    evidence["cost"] = {"observed": 3.0, "budget": 2.0, "unit": "usd"}
    records = _native_records(latency_ms=900.0)
    records[0]["status"] = "invalid_provenance"

    report = aggregate_quality_report(metadata, records, smoke_threshold=2)
    decision = evaluate_release_gate(
        report,
        required_modes=NATIVE_STRATEGIES,
        thresholds={
            "provenance_completeness": 1.0,
            "id_based_context_recall": 0.9,
            "p95_latency_ms": 500.0,
        },
    )

    assert decision["decision"] == "blocked"
    assert {reason["code"] for reason in decision["reasons"]} >= {
        "invalid_provenance",
        "tenant_isolation_failed",
        "missing_sparse_lifecycle_evidence",
        "missing_degradation_evidence",
        "missing_rollback_evidence",
        "cost_budget_exceeded",
        "latency_budget_exceeded",
    }


def test_bm25_promotion_gate_passes_only_with_complete_evidence() -> None:
    """A non-smoke four-way report passes only when every BM25 gate is evidenced."""
    report = aggregate_quality_report(
        _native_metadata(),
        _native_records(),
        smoke_threshold=2,
    )

    decision = evaluate_release_gate(
        report,
        required_modes=NATIVE_STRATEGIES,
        thresholds={
            "provenance_completeness": 1.0,
            "id_based_context_recall": 0.9,
            "p95_latency_ms": 500.0,
        },
    )

    assert decision == {"decision": "passed", "reasons": []}


def test_local_fixture_evidence_never_promotes_bm25_default() -> None:
    """Deterministic local fixtures remain distinct from production-service evidence."""
    metadata = _native_metadata()
    metadata["evidence_class"] = "local_deterministic_fixture"
    report = aggregate_quality_report(metadata, _native_records(), smoke_threshold=2)

    decision = evaluate_release_gate(
        report,
        required_modes=NATIVE_STRATEGIES,
        thresholds={"provenance_completeness": 1.0, "p95_latency_ms": 500.0},
    )

    assert decision["decision"] == "blocked"
    assert {reason["code"] for reason in decision["reasons"]} >= {
        "non_production_evidence"
    }
