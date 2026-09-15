"""Validated, PDF-grounded benchmark records for offline RAG evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

SUPPORTED_CATEGORIES = frozenset(
    {
        "single_document_fact",
        "multi_document_ranking",
        "comparison",
        "multi_hop",
        "table_metric",
        "insufficient_evidence",
    }
)


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark cannot be safely used for an evaluation run."""


@dataclass(frozen=True)
class Evidence:
    """Represent evidence."""
    source_document_id: str
    page: int
    section: str
    claim: str


@dataclass(frozen=True)
class BenchmarkSample:
    """Represent benchmark sample."""
    id: str
    question: str
    reference: str
    required_doc_ids: tuple[str, ...]
    reference_context_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    category: str
    expected_refusal: bool
    notes: str = ""


class BenchmarkDataset(list[BenchmarkSample]):
    """A benchmark list carrying the source file and immutable file hash."""

    def __init__(self, samples: Iterable[BenchmarkSample], *, source_path: Path, sha256: str) -> None:
        """Initialize the benchmark dataset."""
        super().__init__(samples)
        self.source_path = source_path
        self.sha256 = sha256


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a benchmark file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_error(line_number: int, message: str) -> BenchmarkValidationError:
    """Return the line error."""
    return BenchmarkValidationError(f"benchmark line {line_number}: {message}")


def _non_empty_string(value: Any, field_name: str, line_number: int) -> str:
    """Return the non empty string."""
    if not isinstance(value, str) or not value.strip():
        raise _line_error(line_number, f"{field_name} must be a non-empty string")
    return value.strip()


def _document_ids(value: Any, field_name: str, line_number: int) -> tuple[str, ...]:
    """Return the document IDs."""
    if not isinstance(value, list) or not value:
        raise _line_error(line_number, f"{field_name} must be a non-empty list")

    values: list[str] = []
    for item in value:
        document_id = _non_empty_string(item, field_name, line_number)
        try:
            UUID(document_id)
        except ValueError as error:
            raise _line_error(line_number, f"{field_name} contains an invalid UUID: {document_id}") from error
        values.append(document_id)
    return tuple(values)


def _evidence(value: Any, required_doc_ids: tuple[str, ...], line_number: int) -> tuple[Evidence, ...]:
    """Return the evidence."""
    if not isinstance(value, list) or not value:
        raise _line_error(line_number, "evidence must be a non-empty list")

    evidence_items: list[Evidence] = []
    for item in value:
        if not isinstance(item, dict):
            raise _line_error(line_number, "evidence must contain objects")
        document_id = _non_empty_string(item.get("source_document_id"), "evidence.source_document_id", line_number)
        if document_id not in required_doc_ids:
            raise _line_error(line_number, "evidence.source_document_id must be included in required_doc_ids")
        page = item.get("page")
        if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
            raise _line_error(line_number, "evidence.page must be a positive integer")
        evidence_items.append(
            Evidence(
                source_document_id=document_id,
                page=page,
                section=_non_empty_string(item.get("section"), "evidence.section", line_number),
                claim=_non_empty_string(item.get("claim"), "evidence.claim", line_number),
            )
        )
    return tuple(evidence_items)


def _parse_record(record: Any, line_number: int) -> BenchmarkSample:
    """Parse the record."""
    if not isinstance(record, dict):
        raise _line_error(line_number, "record must be a JSON object")

    required_doc_ids = _document_ids(record.get("required_doc_ids"), "required_doc_ids", line_number)
    reference_context_ids = _document_ids(record.get("reference_context_ids"), "reference_context_ids", line_number)
    if any(document_id not in required_doc_ids for document_id in reference_context_ids):
        raise _line_error(line_number, "reference_context_ids must be included in required_doc_ids")

    category = _non_empty_string(record.get("category"), "category", line_number)
    if category not in SUPPORTED_CATEGORIES:
        raise _line_error(line_number, f"category is unsupported: {category}")

    expected_refusal = record.get("expected_refusal")
    if not isinstance(expected_refusal, bool):
        raise _line_error(line_number, "expected_refusal must be a boolean")

    notes = record.get("notes", "")
    if not isinstance(notes, str):
        raise _line_error(line_number, "notes must be a string when provided")

    return BenchmarkSample(
        id=_non_empty_string(record.get("id"), "id", line_number),
        question=_non_empty_string(record.get("question"), "question", line_number),
        reference=_non_empty_string(record.get("reference"), "reference", line_number),
        required_doc_ids=required_doc_ids,
        reference_context_ids=reference_context_ids,
        evidence=_evidence(record.get("evidence"), required_doc_ids, line_number),
        category=category,
        expected_refusal=expected_refusal,
        notes=notes.strip(),
    )


def load_benchmark(path: str | Path) -> BenchmarkDataset:
    """Load a safe JSONL benchmark without treating it as a retrieval document."""
    source_path = Path(path)
    if "documents" in source_path.parts:
        raise BenchmarkValidationError("benchmark file must stay outside the runtime documents upload directory")
    if not source_path.is_file():
        raise BenchmarkValidationError(f"benchmark file does not exist: {source_path}")

    samples: list[BenchmarkSample] = []
    sample_ids: set[str] = set()
    for line_number, raw_line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            raw_record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise _line_error(line_number, "must contain valid JSON") from error
        sample = _parse_record(raw_record, line_number)
        if sample.id in sample_ids:
            raise _line_error(line_number, f"duplicate id: {sample.id}")
        sample_ids.add(sample.id)
        samples.append(sample)

    if not samples:
        raise BenchmarkValidationError("benchmark must contain at least one JSONL record")
    return BenchmarkDataset(samples, source_path=source_path, sha256=sha256_file(source_path))


__all__ = [
    "BenchmarkDataset",
    "BenchmarkSample",
    "BenchmarkValidationError",
    "Evidence",
    "SUPPORTED_CATEGORIES",
    "load_benchmark",
    "sha256_file",
]
