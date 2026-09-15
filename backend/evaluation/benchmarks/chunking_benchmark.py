"""Controlled offline retrieval comparison for document chunking strategies."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

import numpy as np

from agents.document_parser import DocParserAgent
from shared.config import settings

from .benchmark import BenchmarkDataset, load_benchmark, sha256_file
from ..runner import _write_jsonl, _write_text_secure

ChunkingStrategy = Literal["fixed", "recursive", "semantic"]
SUPPORTED_CHUNKING_STRATEGIES: tuple[ChunkingStrategy, ...] = (
    "fixed",
    "recursive",
    "semantic",
)


class EmbeddingsProtocol(Protocol):
    """Describe the async embedding surface required by the offline runner."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""

    async def aembed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""


class ChunkingBenchmarkValidationError(ValueError):
    """Raised when a chunking benchmark cannot produce trustworthy results."""


@dataclass(frozen=True)
class _CorpusDocument:
    """Represent one validated normalized document without exposing its path."""

    source_document_id: str
    title: str
    text: str
    sha256: str


@dataclass(frozen=True)
class _EvaluationChunk:
    """Represent one traceable chunk before retrieval ranking."""

    content: str
    source_document_id: str
    source: str
    char_start: int
    char_end: int
    requested_strategy: ChunkingStrategy
    effective_strategy: str
    ordinal: int


@dataclass(frozen=True)
class ChunkingBenchmarkRun:
    """Expose the immutable artifacts and in-memory records of one comparison."""

    run_id: str
    run_dir: Path
    snapshot_path: Path
    summary_path: Path
    records: list[dict[str, Any]]
    summary: dict[str, Any]


def _required_string(value: Any, field_name: str) -> str:
    """Return a non-empty manifest string or raise a safe validation error."""
    if not isinstance(value, str) or not value.strip():
        raise ChunkingBenchmarkValidationError(
            f"manifest {field_name} must be a non-empty string"
        )
    return value.strip()


def _safe_normalized_path(corpus_root: Path, value: Any) -> Path:
    """Resolve one normalized path while preventing absolute and traversal escapes."""
    relative = Path(_required_string(value, "documents.normalized_path"))
    if relative.is_absolute():
        raise ChunkingBenchmarkValidationError(
            "normalized path must not escape the corpus root"
        )
    resolved_root = corpus_root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ChunkingBenchmarkValidationError(
            "normalized path must not escape the corpus root"
        ) from error
    return resolved


def _load_corpus(
    corpus_root: Path,
    manifest_path: Path,
) -> tuple[str, list[_CorpusDocument], str]:
    """Load and hash normalized corpus inputs before any embedding work begins."""
    if not manifest_path.is_file():
        raise ChunkingBenchmarkValidationError(
            f"manifest file does not exist: {manifest_path.name}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ChunkingBenchmarkValidationError("manifest must be valid UTF-8 JSON") from error
    if not isinstance(manifest, dict):
        raise ChunkingBenchmarkValidationError("manifest must be a JSON object")
    corpus_id = _required_string(manifest.get("corpus_id"), "corpus_id")
    raw_documents = manifest.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ChunkingBenchmarkValidationError(
            "manifest documents must be a non-empty list"
        )

    documents: list[_CorpusDocument] = []
    document_ids: set[str] = set()
    corpus_digest = hashlib.sha256()
    for index, raw_document in enumerate(raw_documents):
        if not isinstance(raw_document, dict):
            raise ChunkingBenchmarkValidationError(
                f"manifest document {index} must be an object"
            )
        source_document_id = _required_string(
            raw_document.get("source_document_id"),
            f"documents[{index}].source_document_id",
        )
        if source_document_id in document_ids:
            raise ChunkingBenchmarkValidationError(
                f"manifest contains duplicate source_document_id: {source_document_id}"
            )
        document_ids.add(source_document_id)
        normalized_path = _safe_normalized_path(
            corpus_root,
            raw_document.get("normalized_path"),
        )
        if not normalized_path.is_file():
            raise ChunkingBenchmarkValidationError(
                "normalized document does not exist; prepare the normalized corpus first"
            )
        expected_sha256 = _required_string(
            raw_document.get("normalized_sha256"),
            f"documents[{index}].normalized_sha256",
        ).lower()
        actual_sha256 = sha256_file(normalized_path)
        if actual_sha256 != expected_sha256:
            raise ChunkingBenchmarkValidationError(
                f"normalized document hash mismatch for source_document_id {source_document_id}"
            )
        try:
            text = normalized_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ChunkingBenchmarkValidationError(
                f"normalized document is not readable UTF-8: {source_document_id}"
            ) from error
        if not text.strip():
            raise ChunkingBenchmarkValidationError(
                f"normalized document is empty: {source_document_id}"
            )
        title_value = raw_document.get("title")
        title = (
            title_value.strip()
            if isinstance(title_value, str) and title_value.strip()
            else normalized_path.name
        )
        documents.append(
            _CorpusDocument(
                source_document_id=source_document_id,
                title=title,
                text=text,
                sha256=actual_sha256,
            )
        )
        corpus_digest.update(source_document_id.encode("utf-8"))
        corpus_digest.update(actual_sha256.encode("ascii"))
    return corpus_id, documents, corpus_digest.hexdigest()


def _validate_benchmark_documents(
    benchmark: BenchmarkDataset,
    documents: Sequence[_CorpusDocument],
) -> None:
    """Ensure every reviewed source ID exists in the selected corpus."""
    available = {document.source_document_id for document in documents}
    for sample in benchmark:
        missing = [
            document_id
            for document_id in sample.required_doc_ids
            if document_id not in available
        ]
        if missing:
            raise ChunkingBenchmarkValidationError(
                f"benchmark {sample.id} references a document absent from the manifest"
            )


def _validate_strategies(
    strategies: Sequence[str],
) -> tuple[ChunkingStrategy, ...]:
    """Validate and preserve a non-empty unique strategy order."""
    if not strategies:
        raise ChunkingBenchmarkValidationError("at least one chunking strategy is required")
    resolved: list[ChunkingStrategy] = []
    for strategy in strategies:
        if strategy not in SUPPORTED_CHUNKING_STRATEGIES:
            raise ChunkingBenchmarkValidationError(
                f"unsupported chunking strategy: {strategy}"
            )
        typed_strategy = strategy
        if typed_strategy in resolved:
            raise ChunkingBenchmarkValidationError(
                f"duplicate chunking strategy: {strategy}"
            )
        resolved.append(typed_strategy)
    return tuple(resolved)


def _fixed_spans(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    """Split text only by character position using a controlled overlap."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ChunkingBenchmarkValidationError("fixed chunk size and overlap are invalid")
    spans: list[tuple[str, int, int]] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        content = text[start:end]
        if content.strip():
            spans.append((content, start, end))
        if end == len(text):
            break
    return spans


def _coerce_matrix(
    values: Any,
    *,
    expected_rows: int,
    label: str,
) -> np.ndarray:
    """Validate an embedding matrix for deterministic cosine retrieval."""
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ChunkingBenchmarkValidationError(
            f"{label} embedding vectors are invalid"
        ) from error
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
        raise ChunkingBenchmarkValidationError(
            f"{label} embedding vector count or dimensions are invalid"
        )
    if not np.isfinite(matrix).all():
        raise ChunkingBenchmarkValidationError(
            f"{label} embedding vectors contain non-finite values"
        )
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ChunkingBenchmarkValidationError(
            f"{label} embedding vectors contain a zero vector"
        )
    return matrix / norms[:, None]


def _coerce_query(value: Any, *, expected_dimension: int) -> np.ndarray:
    """Validate and normalize one query embedding."""
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ChunkingBenchmarkValidationError("query embedding vector is invalid") from error
    if vector.ndim != 1 or vector.shape[0] != expected_dimension:
        raise ChunkingBenchmarkValidationError(
            "query embedding vector dimension is invalid"
        )
    if not np.isfinite(vector).all():
        raise ChunkingBenchmarkValidationError(
            "query embedding vector contains non-finite values"
        )
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ChunkingBenchmarkValidationError("query embedding vector is zero")
    return vector / norm


def _strategy_summary(chunks: Sequence[_EvaluationChunk]) -> dict[str, Any]:
    """Summarize fragmentation and fallback behavior without document content."""
    lengths = [len(chunk.content) for chunk in chunks]
    effective_counts = Counter(chunk.effective_strategy for chunk in chunks)
    return {
        "chunk_count": len(chunks),
        "min_characters": min(lengths),
        "mean_characters": round(sum(lengths) / len(lengths), 3),
        "max_characters": max(lengths),
        "effective_strategy_counts": dict(sorted(effective_counts.items())),
    }


class ChunkingBenchmarkRunner:
    """Build controlled chunking snapshots without using a production vector store."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingsProtocol,
        chunker: DocParserAgent,
        results_root: Path,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize injected chunking, embedding, and artifact collaborators."""
        self._embeddings = embeddings
        self._chunker = chunker
        self._results_root = Path(results_root)
        self._embedding_model = embedding_model or settings.embedding_model

    async def _build_chunks(
        self,
        documents: Sequence[_CorpusDocument],
        strategy: ChunkingStrategy,
    ) -> list[_EvaluationChunk]:
        """Build one strategy's chunks while preserving production fallback metadata."""
        chunks: list[_EvaluationChunk] = []
        for document in documents:
            if strategy == "fixed":
                raw_spans = _fixed_spans(
                    document.text,
                    settings.chunk_size,
                    settings.chunk_overlap,
                )
                spans = [
                    (content, start, end, "fixed")
                    for content, start, end in raw_spans
                ]
            elif strategy == "recursive":
                recursive_spans = self._chunker._recursive_spans(
                    document.text,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                spans = [
                    (span.content, span.start, span.end, "recursive")
                    for span in recursive_spans
                ]
            else:
                semantic_spans, effective_strategy = (
                    await self._chunker._semantic_spans_with_fallback(document.text)
                )
                spans = [
                    (span.content, span.start, span.end, effective_strategy)
                    for span in semantic_spans
                ]
            for content, start, end, effective_strategy in spans:
                chunks.append(
                    _EvaluationChunk(
                        content=content,
                        source_document_id=document.source_document_id,
                        source=document.title,
                        char_start=start,
                        char_end=end,
                        requested_strategy=strategy,
                        effective_strategy=effective_strategy,
                        ordinal=len(chunks),
                    )
                )
        if not chunks:
            raise ChunkingBenchmarkValidationError(
                f"chunking strategy produced no chunks: {strategy}"
            )
        return chunks

    async def _records_for_strategy(
        self,
        benchmark: BenchmarkDataset,
        chunks: Sequence[_EvaluationChunk],
        strategy: ChunkingStrategy,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Embed chunks and queries, then build complete deterministic retrieval records."""
        provided = await self._embeddings.aembed_documents(
            [chunk.content for chunk in chunks]
        )
        matrix = _coerce_matrix(
            provided,
            expected_rows=len(chunks),
            label=f"{strategy} chunk",
        )
        records: list[dict[str, Any]] = []
        for sample in benchmark:
            query = _coerce_query(
                await self._embeddings.aembed_query(sample.question),
                expected_dimension=matrix.shape[1],
            )
            scores = matrix @ query
            ranked_indices = sorted(
                range(len(chunks)),
                key=lambda index: (-float(scores[index]), chunks[index].ordinal),
            )[: min(top_k, len(chunks))]
            contexts: list[dict[str, Any]] = []
            retrieved_context_ids: list[str] = []
            for rank, index in enumerate(ranked_indices, start=1):
                chunk = chunks[index]
                if chunk.source_document_id not in retrieved_context_ids:
                    retrieved_context_ids.append(chunk.source_document_id)
                contexts.append(
                    {
                        "rank": rank,
                        "content": chunk.content,
                        "source": chunk.source,
                        "source_document_id": chunk.source_document_id,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "requested_strategy": chunk.requested_strategy,
                        "effective_strategy": chunk.effective_strategy,
                        "score": round(float(scores[index]), 12),
                    }
                )
            records.append(
                {
                    "benchmark_id": sample.id,
                    "chunking_strategy": strategy,
                    "status": "succeeded",
                    "question": sample.question,
                    "reference": sample.reference,
                    "reference_context_ids": list(sample.reference_context_ids),
                    "retrieved_context_ids": retrieved_context_ids,
                    "category": sample.category,
                    "expected_refusal": sample.expected_refusal,
                    "contexts": contexts,
                }
            )
        return records

    async def run(
        self,
        *,
        corpus_root: str | Path,
        manifest_path: str | Path,
        benchmark_path: str | Path,
        strategies: Sequence[str] = SUPPORTED_CHUNKING_STRATEGIES,
        top_k: int = 5,
    ) -> ChunkingBenchmarkRun:
        """Validate inputs, execute the comparison, and persist auditable artifacts."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ChunkingBenchmarkValidationError("top_k must be a positive integer")
        resolved_strategies = _validate_strategies(strategies)
        resolved_corpus_root = Path(corpus_root)
        resolved_manifest_path = Path(manifest_path)
        resolved_benchmark_path = Path(benchmark_path)

        corpus_id, documents, corpus_sha256 = _load_corpus(
            resolved_corpus_root,
            resolved_manifest_path,
        )
        try:
            benchmark = load_benchmark(resolved_benchmark_path)
        except ValueError as error:
            raise ChunkingBenchmarkValidationError(str(error)) from error
        _validate_benchmark_documents(benchmark, documents)

        records: list[dict[str, Any]] = []
        strategy_summaries: dict[str, Any] = {}
        for strategy in resolved_strategies:
            chunks = await self._build_chunks(documents, strategy)
            strategy_summaries[strategy] = _strategy_summary(chunks)
            records.extend(
                await self._records_for_strategy(
                    benchmark,
                    chunks,
                    strategy,
                    top_k,
                )
            )

        run_id = f"chunking-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        summary: dict[str, Any] = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "corpus_id": corpus_id,
            "corpus_sha256": corpus_sha256,
            "manifest_sha256": sha256_file(resolved_manifest_path),
            "benchmark_sha256": benchmark.sha256,
            "embedding_model": self._embedding_model,
            "benchmark_count": len(benchmark),
            "document_count": len(documents),
            "record_count": len(records),
            "parameters": {
                "chunk_size_characters": settings.chunk_size,
                "chunk_overlap_characters": settings.chunk_overlap,
                "semantic_percentile": settings.semantic_chunk_percentile,
                "semantic_min_characters": settings.semantic_chunk_min_size,
                "semantic_max_characters": settings.semantic_chunk_max_size,
                "top_k": top_k,
            },
            "strategies": strategy_summaries,
        }

        self._results_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self._results_root, 0o700)
        run_dir = self._results_root / run_id
        run_dir.mkdir(mode=0o700)
        snapshot_path = run_dir / "chunking_snapshot.jsonl"
        summary_path = run_dir / "chunking_summary.json"
        _write_jsonl(snapshot_path, records)
        _write_text_secure(
            summary_path,
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return ChunkingBenchmarkRun(
            run_id=run_id,
            run_dir=run_dir,
            snapshot_path=snapshot_path,
            summary_path=summary_path,
            records=records,
            summary=summary,
        )


__all__ = [
    "ChunkingBenchmarkRun",
    "ChunkingBenchmarkRunner",
    "ChunkingBenchmarkValidationError",
    "ChunkingStrategy",
    "EmbeddingsProtocol",
    "SUPPORTED_CHUNKING_STRATEGIES",
]
