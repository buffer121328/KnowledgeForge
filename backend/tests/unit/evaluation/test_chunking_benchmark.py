"""Acceptance coverage for controlled offline document chunking comparison."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agents.document_parser import DocParserAgent
from evaluation.benchmarks.chunking_benchmark import (
    ChunkingBenchmarkRunner,
    ChunkingBenchmarkValidationError,
)


DOC_A = "7560dc54-81cb-4d28-bbb1-205f060ad678"
DOC_B = "08bb9178-7daa-40d3-b8df-601920a6df2c"


class FakeEmbeddings:
    """Return deterministic finite vectors while recording external-work boundaries."""

    def __init__(self) -> None:
        """Initialize recorded embedding calls."""
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        """Encode simple Chinese keyword features into a non-zero vector."""
        return [
            float(text.count("制度") + 1),
            float(text.count("流程") + 1),
            float(len(text) % 17 + 1),
        ]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Record and embed document or semantic-window texts."""
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        """Record and embed one benchmark question."""
        self.query_calls.append(text)
        return self._vector(text)


class ZeroEmbeddings(FakeEmbeddings):
    """Return invalid vectors to exercise fail-closed validation."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one zero vector for every requested text."""
        self.document_calls.append(list(texts))
        return [[0.0, 0.0] for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        """Return an invalid zero query vector."""
        self.query_calls.append(text)
        return [0.0, 0.0]


class TieEmbeddings(FakeEmbeddings):
    """Return identical vectors to verify stable source/chunk tie ordering."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        """Return one identical non-zero vector."""
        del text
        return [1.0, 1.0]


def _sha256(path: Path) -> str:
    """Return a fixture file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_corpus(
    root: Path,
    *,
    text_a: str | None = None,
    text_b: str | None = None,
) -> tuple[Path, Path]:
    """Create a validated normalized corpus and one reviewed benchmark record."""
    normalized = root / "normalized"
    normalized.mkdir(parents=True)
    file_a = normalized / "a.txt"
    file_b = normalized / "b.txt"
    file_a.write_text(
        text_a
        or ("制度总则说明适用范围。招聘流程从需求开始！审批完成后发布岗位？" * 30),
        encoding="utf-8",
    )
    file_b.write_text(
        text_b
        or ("财务制度要求凭证完整。报销流程包括申请；复核完成后付款。" * 30),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "corpus_id": "fixture-zh-v1",
        "documents": [
            {
                "source_document_id": DOC_A,
                "title": "人力制度",
                "normalized_path": "normalized/a.txt",
                "normalized_sha256": _sha256(file_a),
            },
            {
                "source_document_id": DOC_B,
                "title": "财务制度",
                "normalized_path": "normalized/b.txt",
                "normalized_sha256": _sha256(file_b),
            },
        ],
    }
    manifest_path = root / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    benchmark = {
        "id": "fixture-01",
        "question": "招聘流程和制度是什么？",
        "reference": "招聘流程从需求开始，并按制度完成审批。",
        "required_doc_ids": [DOC_A],
        "reference_context_ids": [DOC_A],
        "evidence": [
            {
                "source_document_id": DOC_A,
                "page": 1,
                "section": "招聘流程",
                "claim": "描述招聘流程。",
            }
        ],
        "category": "single_document_fact",
        "expected_refusal": False,
    }
    benchmark_path = root / "benchmark.jsonl"
    benchmark_path.write_text(
        json.dumps(benchmark, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path, benchmark_path


def _chunker(embeddings: Any) -> DocParserAgent:
    """Create the production chunker with network-free collaborators."""
    with patch("agents.document_format_parsers.ChatOpenAI"):
        return DocParserAgent(embeddings=embeddings, embedding_cache=None)


@pytest.mark.asyncio
async def test_generates_controlled_three_strategy_snapshot_and_owner_only_artifacts(
    tmp_path: Path,
) -> None:
    """The same corpus/query budget must produce auditable records for all strategies."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    manifest_path, benchmark_path = _write_corpus(corpus_root)
    embeddings = FakeEmbeddings()

    run = await ChunkingBenchmarkRunner(
        embeddings=embeddings,
        chunker=_chunker(embeddings),
        results_root=tmp_path / "results",
    ).run(
        corpus_root=corpus_root,
        manifest_path=manifest_path,
        benchmark_path=benchmark_path,
        top_k=3,
    )

    assert [record["chunking_strategy"] for record in run.records] == [
        "fixed",
        "recursive",
        "semantic",
    ]
    assert all(record["status"] == "succeeded" for record in run.records)
    assert all(len(record["contexts"]) == 3 for record in run.records)
    context = run.records[0]["contexts"][0]
    assert set(
        (
            "content",
            "source_document_id",
            "char_start",
            "char_end",
            "requested_strategy",
            "effective_strategy",
            "score",
            "rank",
        )
    ).issubset(context)
    assert run.summary["parameters"] == {
        "chunk_size_characters": 400,
        "chunk_overlap_characters": 128,
        "semantic_percentile": 95.0,
        "semantic_min_characters": 200,
        "semantic_max_characters": 512,
        "top_k": 3,
    }
    assert set(run.summary["strategies"]) == {"fixed", "recursive", "semantic"}
    assert run.summary["strategies"]["fixed"]["chunk_count"] > 0
    assert run.summary["strategies"]["fixed"]["mean_characters"] > 0
    assert stat.S_IMODE(run.run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(run.snapshot_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(run.summary_path.stat().st_mode) == 0o600
    assert embeddings.query_calls == ["招聘流程和制度是什么？"] * 3


@pytest.mark.asyncio
async def test_semantic_snapshot_records_recursive_effective_fallback(tmp_path: Path) -> None:
    """A short semantic source keeps requested and effective strategies distinct."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    manifest_path, benchmark_path = _write_corpus(
        corpus_root,
        text_a="第一句。第二句。第三句。",
        text_b="另一份短文。",
    )
    embeddings = FakeEmbeddings()

    run = await ChunkingBenchmarkRunner(
        embeddings=embeddings,
        chunker=_chunker(embeddings),
        results_root=tmp_path / "results",
    ).run(
        corpus_root=corpus_root,
        manifest_path=manifest_path,
        benchmark_path=benchmark_path,
        strategies=("semantic",),
        top_k=2,
    )

    assert run.records[0]["chunking_strategy"] == "semantic"
    assert all(
        context["requested_strategy"] == "semantic"
        and context["effective_strategy"] == "recursive"
        for context in run.records[0]["contexts"]
    )
    assert run.summary["strategies"]["semantic"]["effective_strategy_counts"] == {
        "recursive": 2
    }


@pytest.mark.asyncio
async def test_missing_normalized_file_fails_before_embedding(tmp_path: Path) -> None:
    """Manifest/file validation must finish before any external embedding request."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    manifest_path, benchmark_path = _write_corpus(corpus_root)
    (corpus_root / "normalized" / "a.txt").unlink()
    embeddings = FakeEmbeddings()

    with pytest.raises(ChunkingBenchmarkValidationError, match="normalized.*does not exist"):
        await ChunkingBenchmarkRunner(
            embeddings=embeddings,
            chunker=_chunker(embeddings),
            results_root=tmp_path / "results",
        ).run(
            corpus_root=corpus_root,
            manifest_path=manifest_path,
            benchmark_path=benchmark_path,
        )

    assert embeddings.document_calls == []
    assert embeddings.query_calls == []


@pytest.mark.asyncio
async def test_manifest_path_escape_is_rejected_before_reading(tmp_path: Path) -> None:
    """Normalized paths cannot escape the selected corpus root."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    manifest_path, benchmark_path = _write_corpus(corpus_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["normalized_path"] = "../outside.txt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    embeddings = FakeEmbeddings()

    with pytest.raises(ChunkingBenchmarkValidationError, match="escape"):
        await ChunkingBenchmarkRunner(
            embeddings=embeddings,
            chunker=_chunker(embeddings),
            results_root=tmp_path / "results",
        ).run(
            corpus_root=corpus_root,
            manifest_path=manifest_path,
            benchmark_path=benchmark_path,
        )

    assert embeddings.document_calls == []


@pytest.mark.asyncio
async def test_invalid_vectors_fail_without_quality_artifacts(tmp_path: Path) -> None:
    """Zero vectors cannot yield retrieval snapshots that look valid."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    manifest_path, benchmark_path = _write_corpus(corpus_root)
    embeddings = ZeroEmbeddings()
    results_root = tmp_path / "results"

    with pytest.raises(ChunkingBenchmarkValidationError, match="zero"):
        await ChunkingBenchmarkRunner(
            embeddings=embeddings,
            chunker=_chunker(embeddings),
            results_root=results_root,
        ).run(
            corpus_root=corpus_root,
            manifest_path=manifest_path,
            benchmark_path=benchmark_path,
            strategies=("fixed",),
        )

    assert not results_root.exists()


@pytest.mark.asyncio
async def test_cosine_ties_keep_manifest_and_chunk_order(tmp_path: Path) -> None:
    """Equal similarity scores use stable source and local chunk order."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    manifest_path, benchmark_path = _write_corpus(corpus_root)
    embeddings = TieEmbeddings()

    run = await ChunkingBenchmarkRunner(
        embeddings=embeddings,
        chunker=_chunker(embeddings),
        results_root=tmp_path / "results",
    ).run(
        corpus_root=corpus_root,
        manifest_path=manifest_path,
        benchmark_path=benchmark_path,
        strategies=("fixed",),
        top_k=3,
    )

    assert [context["source_document_id"] for context in run.records[0]["contexts"]] == [
        DOC_A,
        DOC_A,
        DOC_A,
    ]
    assert [context["char_start"] for context in run.records[0]["contexts"]] == [
        0,
        272,
        544,
    ]
