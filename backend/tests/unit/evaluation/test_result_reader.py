"""Acceptance tests for the read-only evaluation result reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.benchmarks.result_reader import (
    EvaluationResultError,
    list_records,
    list_runs,
    load_run,
)


def _write_run(
    root: Path,
    run_id: str,
    *,
    metadata: dict | None = None,
    responses: list[dict] | None = None,
    quality_report: dict | None = None,
    ragas_summaries: dict[str, dict] | None = None,
) -> Path:
    """Create one result directory with optional artifacts."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        (run_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
    if responses is not None:
        (run_dir / "responses.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in responses) + "\n",
            encoding="utf-8",
        )
    if quality_report is not None:
        (run_dir / "quality_report.json").write_text(
            json.dumps(quality_report, ensure_ascii=False), encoding="utf-8"
        )
    for name, summary in (ragas_summaries or {}).items():
        (run_dir / f"{name}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False), encoding="utf-8"
        )
    return run_dir


def _metadata(*, run_id: str = "ragas-20260801T120000Z-abcd1234", started_at: str = "2026-08-01T12:00:00+00:00") -> dict:
    return {
        "run_id": run_id,
        "started_at": started_at,
        "retrieval_modes": ["vector", "hybrid"],
        "benchmark_source": "/private/tmp/benchmark.jsonl",
        "benchmark_sha256": "abc123",
        "code_revision": "deadbeef",
    }


def _response_record(*, benchmark_id: str, mode: str = "vector") -> dict:
    return {
        "run_id": "ragas-20260801T120000Z-abcd1234",
        "benchmark_id": benchmark_id,
        "retrieval_mode": mode,
        "category": "single_document_fact",
        "expected_refusal": False,
        "refused": False,
        "question": "公司人力资源管理制度的主要管理范围是什么？",
        "response": "制度共十八章，覆盖招聘、劳动合同……" + "很长的回答内容" * 50,
        "contexts": [
            {
                "rank": 1,
                "content": "完整未截断的检索文本" * 20,
                "source": "056公司人力资源管理制度",
                "score": 0.9,
                "retrieval_type": "vector",
                "source_document_id": "5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9",
                "metadata": {"source_document_id": "5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9"},
            }
        ],
        "retrieved_context_ids": ["5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9"],
        "reference_context_ids": ["5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9"],
        "status": "succeeded",
        "latency_ms": 12.3,
        "exception": None,
    }


class TestListRuns:
    def test_missing_or_empty_root_returns_empty_list(self, tmp_path: Path) -> None:
        assert list_runs(tmp_path / "missing") == []
        (tmp_path / "results").mkdir()
        assert list_runs(tmp_path / "results") == []

    def test_lists_runs_newest_first_with_bounded_summaries(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        _write_run(
            root,
            "ragas-20260801T120000Z-aaaa1111",
            metadata=_metadata(run_id="ragas-20260801T120000Z-aaaa1111"),
        )
        _write_run(
            root,
            "ragas-20260802T120000Z-bbbb2222",
            metadata=_metadata(
                run_id="ragas-20260802T120000Z-bbbb2222",
                started_at="2026-08-02T12:00:00+00:00",
            ),
        )

        runs = list_runs(root)

        assert [run["run_id"] for run in runs] == [
            "ragas-20260802T120000Z-bbbb2222",
            "ragas-20260801T120000Z-aaaa1111",
        ]
        assert runs[0]["retrieval_modes"] == ["vector", "hybrid"]
        assert runs[0]["incomplete"] is False
        assert "benchmark_source" not in json.dumps(runs)

    def test_metadata_missing_run_is_listed_as_incomplete(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        (root / "ragas-20260801T120000Z-cccc3333").mkdir(parents=True)

        runs = list_runs(root)

        assert runs[0]["incomplete"] is True
        assert runs[0]["started_at"] is None

    def test_non_run_directories_are_ignored(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        root.mkdir()
        (root / ".gitkeep").write_text("", encoding="utf-8")
        (root / "not-a-run-dir").mkdir()

        assert list_runs(root) == []


class TestResolveRunDir:
    def test_path_traversal_and_invalid_ids_are_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        (root / "ragas-20260801T120000Z-aaaa1111").mkdir(parents=True)
        (root / "outer").mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        for run_id in ("../outside", "a/b", "..", ".", "a" * 129, "ragas$bad", "ragas-20260801T120000Z-aaaa1111/x"):
            with pytest.raises(EvaluationResultError):
                load_run(root, run_id)

    def test_missing_run_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        root.mkdir()
        with pytest.raises(EvaluationResultError):
            load_run(root, "ragas-20260801T120000Z-aaaa1111")


class TestLoadRun:
    def test_loads_metadata_report_and_ragas_summaries(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        _write_run(
            root,
            "ragas-20260801T120000Z-aaaa1111",
            metadata=_metadata(run_id="ragas-20260801T120000Z-aaaa1111"),
            quality_report={"run_classification": "smoke_only", "counts": {"completed": 60}},
            ragas_summaries={
                "ragas_faithfulness": {"metrics": {"faithfulness": {"mean": 0.81}}},
                "ragas_factual_correctness": {"metrics": {"factual_correctness": {"mean": 0.72}}},
            },
        )

        detail = load_run(root, "ragas-20260801T120000Z-aaaa1111")

        assert detail["incomplete"] is False
        assert detail["metadata"]["run_id"] == "ragas-20260801T120000Z-aaaa1111"
        assert detail["metadata"]["benchmark_source"] == "benchmark.jsonl"
        assert "tmp" not in json.dumps(detail)
        assert detail["quality_report"]["run_classification"] == "smoke_only"
        assert sorted(detail["ragas_summaries"]) == [
            "ragas_factual_correctness",
            "ragas_faithfulness",
        ]

    def test_missing_metadata_marks_incomplete(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        _write_run(root, "ragas-20260801T120000Z-aaaa1111", responses=[])

        detail = load_run(root, "ragas-20260801T120000Z-aaaa1111")

        assert detail["incomplete"] is True
        assert detail["metadata"] == {}
        assert detail["quality_report"] is None


class TestListRecords:
    def test_paginates_and_truncates_response_without_context_content(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "results"
        run_id = "ragas-20260801T120000Z-aaaa1111"
        _write_run(
            root,
            run_id,
            metadata=_metadata(),
            responses=[_response_record(benchmark_id=f"company-demo-{i:02d}") for i in range(1, 26)],
            ragas_summaries={"ragas_faithfulness": {}},
        )
        (root / run_id / "ragas_faithfulness.jsonl").write_text(
            json.dumps(
                {
                    "benchmark_id": "company-demo-01",
                    "retrieval_mode": "vector",
                    "metric": "faithfulness",
                    "score": 0.85,
                    "status": "scored",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        page = list_records(root, run_id, page=1, page_size=10)

        assert page["total"] == 25
        assert page["page"] == 1
        assert len(page["records"]) == 10
        first = page["records"][0]
        assert first["benchmark_id"] == "company-demo-01"
        assert first["response"].endswith("…")
        assert len(first["response"]) <= 201
        assert first["contexts"][0]["source_document_id"] == "5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9"
        assert "content" not in first["contexts"][0]
        assert first["ragas_scores"] == {"faithfulness": 0.85}

        page_two = list_records(root, run_id, page=3, page_size=10)
        assert len(page_two["records"]) == 5

    def test_damaged_lines_are_skipped_and_counted(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        run_id = "ragas-20260801T120000Z-aaaa1111"
        run_dir = _write_run(root, run_id, metadata=_metadata(), responses=[])
        (run_dir / "responses.jsonl").write_text(
            json.dumps(_response_record(benchmark_id="company-demo-01")) + "\n"
            + "not-json\n"
            + json.dumps(_response_record(benchmark_id="company-demo-02")) + "\n"
            + "[1,2,3]\n",
            encoding="utf-8",
        )

        page = list_records(root, run_id)

        assert page["total"] == 2
        assert page["skipped_lines"] == 2
        assert [r["benchmark_id"] for r in page["records"]] == [
            "company-demo-01",
            "company-demo-02",
        ]

    def test_missing_responses_file_returns_empty_page(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        run_id = "ragas-20260801T120000Z-aaaa1111"
        _write_run(root, run_id, metadata=_metadata())

        page = list_records(root, run_id)

        assert page["total"] == 0
        assert page["records"] == []

    def test_invalid_pagination_parameters_are_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "results"
        _write_run(root, "ragas-20260801T120000Z-aaaa1111", metadata=_metadata())

        with pytest.raises(EvaluationResultError):
            list_records(root, "ragas-20260801T120000Z-aaaa1111", page=0)
        with pytest.raises(EvaluationResultError):
            list_records(root, "ragas-20260801T120000Z-aaaa1111", page_size=101)


def _write_release_run(
    root: Path,
    *,
    workflow_id: str,
    stage_dir: str,
    mode: str,
    run_id: str,
) -> Path:
    """Create one server-owned release-stage runner artifact directory."""
    run_dir = _write_run(
        root / "release-workflows" / workflow_id / stage_dir / mode,
        run_id,
        metadata=_metadata(run_id=run_id),
        responses=[_response_record(benchmark_id="release-case-01")],
        quality_report={"run_classification": "baseline"},
    )
    (run_dir / "quality_report.json").rename(run_dir / "quality-report.json")
    return run_dir


def test_release_stage_runs_are_listed_loaded_and_paginated_without_paths(tmp_path: Path) -> None:
    root = tmp_path / "results"
    baseline_id = "20260816T090043Z-fb09fc37"
    shadow_id = "20260816T090137Z-1fc23afa"
    workflow_id = "erw_19099eaa8cfe43ffab8720f9fdf1d8fb"
    _write_release_run(
        root,
        workflow_id=workflow_id,
        stage_dir="baseline-1",
        mode="off",
        run_id=baseline_id,
    )
    _write_release_run(
        root,
        workflow_id=workflow_id,
        stage_dir="shadow-1",
        mode="shadow",
        run_id=shadow_id,
    )

    runs = list_runs(root)
    assert [item["run_id"] for item in runs] == [shadow_id]

    detail = load_run(root, baseline_id)
    assert detail["metadata"]["run_id"] == baseline_id
    assert detail["quality_report"]["run_classification"] == "baseline"
    assert "release-workflows" not in json.dumps(detail)

    records = list_records(root, shadow_id)
    assert records["total"] == 1
    assert records["records"][0]["benchmark_id"] == "release-case-01"


def test_release_baseline_run_stays_resolvable_after_list_filtering(tmp_path: Path) -> None:
    """列表隐藏 baseline run 后，深链访问（详情/逐条记录）仍然可用。"""
    root = tmp_path / "results"
    baseline_id = "20260816T090043Z-fb09fc37"
    workflow_id = "erw_19099eaa8cfe43ffab8720f9fdf1d8fb"
    _write_release_run(
        root,
        workflow_id=workflow_id,
        stage_dir="baseline-1",
        mode="off",
        run_id=baseline_id,
    )

    assert list_runs(root) == []

    detail = load_run(root, baseline_id)
    assert detail["metadata"]["run_id"] == baseline_id

    records = list_records(root, baseline_id, page=1, page_size=50)
    assert records["total"] == 1
    assert records["records"][0]["benchmark_id"] == "release-case-01"


def test_standard_offline_runs_are_listed_alongside_release_shadow(tmp_path: Path) -> None:
    """过滤只作用于发布 baseline 阶段，不影响根目录标准离线 run 与 shadow run。"""
    root = tmp_path / "results"
    offline_id = "ragas-20260815T120000Z-cccc4444"
    shadow_id = "20260816T090137Z-1fc23afa"
    workflow_id = "erw_19099eaa8cfe43ffab8720f9fdf1d8fb"
    _write_run(root, offline_id, metadata=_metadata(run_id=offline_id))
    _write_release_run(
        root,
        workflow_id=workflow_id,
        stage_dir="shadow-1",
        mode="shadow",
        run_id=shadow_id,
    )

    runs = list_runs(root)
    assert {item["run_id"] for item in runs} == {offline_id, shadow_id}


def test_ambiguous_or_path_like_release_run_ids_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run_id = "20260816T090043Z-fb09fc37"
    _write_run(root, run_id, metadata=_metadata(run_id=run_id))
    _write_release_run(
        root,
        workflow_id="erw_19099eaa8cfe43ffab8720f9fdf1d8fb",
        stage_dir="baseline-1",
        mode="off",
        run_id=run_id,
    )

    assert list_runs(root) == []
    with pytest.raises(EvaluationResultError, match="ambiguous"):
        load_run(root, run_id)
    with pytest.raises(EvaluationResultError, match="invalid"):
        load_run(root, "../release-workflows")
