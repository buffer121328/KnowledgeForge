"""Acceptance coverage for offline chunking and Ragas CLI composition."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluation.ragas.adapter import DEFAULT_RAGAS_CONCURRENCY
from evaluation.scripts import run_chunking_benchmark, run_ragas_benchmark


def test_ragas_cli_keeps_faithfulness_default_and_accepts_context_metrics() -> None:
    """Existing invocations stay compatible while context metrics are selectable."""
    default = run_ragas_benchmark.parse_args(["--responses-jsonl", "responses.jsonl"])
    selected = run_ragas_benchmark.parse_args(
        [
            "--responses-jsonl",
            "responses.jsonl",
            "--metrics",
            "context_precision",
            "context_recall",
        ]
    )

    assert default.metrics == ["faithfulness"]
    assert default.concurrency == DEFAULT_RAGAS_CONCURRENCY
    assert selected.metrics == ["context_precision", "context_recall"]


def test_ragas_cli_rejects_unsafe_concurrency() -> None:
    """The small evaluation host must not accept unbounded Judge fan-out."""
    with pytest.raises(SystemExit):
        run_ragas_benchmark.parse_args(
            [
                "--responses-jsonl",
                "responses.jsonl",
                "--concurrency",
                "5",
            ]
        )


@pytest.mark.asyncio
async def test_ragas_cli_scores_saved_contexts_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The context CLI composes adapters without rerunning retrieval."""
    responses = tmp_path / "chunking_snapshot.jsonl"
    responses.write_text(
        json.dumps(
            {
                "benchmark_id": "case-1",
                "chunking_strategy": "recursive",
                "status": "succeeded",
                "question": "问题",
                "reference": "参考",
                "contexts": [{"content": "完整上下文"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    async def fake_context_evaluator(sample: dict[str, Any]) -> float:
        return 0.75 if sample["retrieved_contexts"] else 0.0

    monkeypatch.setattr(
        run_ragas_benchmark,
        "build_openai_context_evaluators",
        lambda metrics: {metric: fake_context_evaluator for metric in metrics},
    )
    args = run_ragas_benchmark.parse_args(
        [
            "--responses-jsonl",
            str(responses),
            "--metrics",
            "context_precision",
            "context_recall",
        ]
    )

    output = await run_ragas_benchmark.run(args)

    assert output.name == "ragas_context_quality.jsonl"
    outcomes = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [outcome["metric"] for outcome in outcomes] == [
        "context_precision",
        "context_recall",
    ]
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["context_precision"]["mean"] == 0.75
    assert summary["metrics"]["context_recall"]["scored"] == 1


@pytest.mark.asyncio
async def test_chunking_cli_uses_explicit_paths_and_safe_error_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI passes controlled options and reports only validation messages."""
    observed: dict[str, Any] = {}

    class FakeRunner:
        """Capture runner arguments without constructing external clients."""

        async def run(self, **kwargs: Any) -> Any:
            """Return a minimal run handle."""
            observed.update(kwargs)
            return SimpleNamespace(snapshot_path=tmp_path / "snapshot.jsonl")

    monkeypatch.setattr(
        run_chunking_benchmark,
        "build_runner",
        lambda results_root: FakeRunner(),
    )
    args = run_chunking_benchmark.parse_args(
        [
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--benchmark",
            str(tmp_path / "benchmark.jsonl"),
            "--results-root",
            str(tmp_path / "results"),
            "--strategies",
            "fixed",
            "recursive",
            "--top-k",
            "7",
        ]
    )

    output = await run_chunking_benchmark.run(args)

    assert output == tmp_path / "snapshot.jsonl"
    assert observed["strategies"] == ("fixed", "recursive")
    assert observed["top_k"] == 7
