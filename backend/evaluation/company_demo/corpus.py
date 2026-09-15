"""Validate the curated Chinese three-department company Demo corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from ..benchmarks.benchmark import BenchmarkDataset, load_benchmark

ALLOWED_DEPARTMENTS = frozenset({"human_resources", "finance", "procurement_warehouse", "administration"})
ALLOWED_SOURCE_FORMATS = frozenset({"docx", "doc", "xlsx", "xls", "wps", "pdf", "txt"})
REQUIRED_DOCUMENT_FIELDS = frozenset(
    {
        "source_document_id",
        "department",
        "document_type",
        "title",
        "source_filename",
        "repository_path",
        "source_format",
        "authority",
        "status",
        "sensitivity",
        "sha256",
    }
)


class CompanyDemoValidationError(ValueError):
    """Raised when the curated company Demo corpus is unsafe or incomplete."""


@dataclass(frozen=True)
class CompanyDemoReport:
    """Summarize a validated company Demo corpus."""

    root: Path
    document_count: int
    benchmark_count: int
    departments: set[str]
    benchmark_departments: set[str]
    benchmark_sha256: str


@dataclass(frozen=True)
class _ManifestDocument:
    """Represent the validated subset of a manifest document record."""

    source_document_id: str
    department: str
    repository_path: str
    source_format: str
    sha256: str
    authority: str
    status: str


def _fail(message: str) -> CompanyDemoValidationError:
    """Create a consistent profile validation error."""
    return CompanyDemoValidationError(message)


def _read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file and convert parse failures to profile errors."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise _fail(f"missing required file: {path}") from error
    except json.JSONDecodeError as error:
        raise _fail(f"invalid JSON in {path}: {error.msg}") from error


def _sha256(path: Path) -> str:
    """Compute a source file digest without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _non_empty_string(record: dict[str, Any], field: str, *, context: str) -> str:
    """Read a required non-empty string from a JSON object."""
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{context}.{field} must be a non-empty string")
    return value.strip()


def _load_manifest(root: Path) -> tuple[list[_ManifestDocument], set[str]]:
    """Load and validate manifest documents and their declared department set."""
    manifest_path = root / "corpus_manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise _fail("corpus_manifest.json must contain a JSON object")
    if manifest.get("schema_version") != 1:
        raise _fail("corpus_manifest.json schema_version must be 1")

    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise _fail("corpus_manifest.json selection must be an object")
    declared_departments = selection.get("departments")
    if not isinstance(declared_departments, list) or set(declared_departments) != ALLOWED_DEPARTMENTS:
        raise _fail("selection.departments must contain exactly the four company Demo departments")

    raw_documents = manifest.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise _fail("corpus_manifest.json documents must be a non-empty list")

    documents: list[_ManifestDocument] = []
    seen_ids: set[str] = set()
    for index, raw_document in enumerate(raw_documents, start=1):
        context = f"documents[{index}]"
        if not isinstance(raw_document, dict):
            raise _fail(f"{context} must be an object")
        missing_fields = REQUIRED_DOCUMENT_FIELDS - raw_document.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise _fail(f"{context} missing fields: {missing}")

        document_id = _non_empty_string(raw_document, "source_document_id", context=context)
        try:
            UUID(document_id)
        except ValueError as error:
            raise _fail(f"{context}.source_document_id must be a UUID: {document_id}") from error
        if document_id in seen_ids:
            raise _fail(f"duplicate source_document_id: {document_id}")
        seen_ids.add(document_id)

        department = _non_empty_string(raw_document, "department", context=context)
        if department not in ALLOWED_DEPARTMENTS:
            raise _fail(f"{context}.department is unsupported: {department}")
        repository_path = _non_empty_string(raw_document, "repository_path", context=context)
        relative_path = Path(repository_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise _fail(f"{context}.repository_path must stay inside the Demo corpus: {repository_path}")
        if not repository_path.startswith("documents/"):
            raise _fail(f"{context}.repository_path must be under documents/: {repository_path}")

        source_format = _non_empty_string(raw_document, "source_format", context=context).lower().lstrip(".")
        if source_format not in ALLOWED_SOURCE_FORMATS:
            raise _fail(f"{context}.source_format is unsupported: {source_format}")
        actual_suffix = relative_path.suffix.lower().lstrip(".")
        if actual_suffix != source_format:
            raise _fail(f"{context}.source_format does not match repository_path: {repository_path}")

        expected_sha = _non_empty_string(raw_document, "sha256", context=context).lower()
        if len(expected_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_sha):
            raise _fail(f"{context}.sha256 must be a lowercase SHA-256 digest")
        source_path = root / relative_path
        if not source_path.is_file():
            raise _fail(f"manifest source file does not exist: {repository_path}")
        actual_sha = _sha256(source_path)
        if actual_sha != expected_sha:
            raise _fail(f"sha256 mismatch for {repository_path}: expected {expected_sha}, got {actual_sha}")

        authority = _non_empty_string(raw_document, "authority", context=context)
        status = _non_empty_string(raw_document, "status", context=context)
        source_name = _non_empty_string(raw_document, "source_filename", context=context)
        lower_source_name = source_name.lower()
        template_markers = ("范本", "模板", "参考")
        if any(marker in source_name for marker in template_markers) and authority == "formal":
            raise _fail(f"{context} template/reference source cannot be marked formal: {source_name}")
        if source_path.name != source_name:
            raise _fail(f"{context}.source_filename does not match repository_path: {repository_path}")
        if lower_source_name != source_name.lower():
            raise _fail(f"{context}.source_filename is invalid")

        documents.append(
            _ManifestDocument(
                source_document_id=document_id,
                department=department,
                repository_path=repository_path,
                source_format=source_format,
                sha256=expected_sha,
                authority=authority,
                status=status,
            )
        )
    return documents, set(declared_departments)


def _validate_benchmark(root: Path, documents: list[_ManifestDocument]) -> tuple[BenchmarkDataset, set[str]]:
    """Load the benchmark and ensure every provenance ID maps to a manifest document."""
    benchmark = load_benchmark(root / "benchmark.jsonl")
    documents_by_id = {document.source_document_id: document for document in documents}
    covered_departments: set[str] = set()
    for sample in benchmark:
        referenced_ids = set(sample.required_doc_ids) | set(sample.reference_context_ids)
        referenced_ids.update(evidence.source_document_id for evidence in sample.evidence)
        unknown_ids = referenced_ids - documents_by_id.keys()
        if unknown_ids:
            unknown = ", ".join(sorted(unknown_ids))
            raise _fail(f"benchmark {sample.id} references unknown document ID: {unknown}")
        covered_departments.update(documents_by_id[document_id].department for document_id in referenced_ids)
    return benchmark, covered_departments


def validate_company_demo(root: str | Path) -> CompanyDemoReport:
    """Validate a company Demo profile without contacting external services."""
    corpus_root = Path(root).resolve()
    if not corpus_root.is_dir():
        raise _fail(f"company Demo root does not exist: {corpus_root}")
    documents, declared_departments = _load_manifest(corpus_root)
    benchmark, benchmark_departments = _validate_benchmark(corpus_root, documents)
    missing_departments = declared_departments - benchmark_departments
    if missing_departments:
        missing = ", ".join(sorted(missing_departments))
        raise _fail(f"benchmark coverage is missing departments: {missing}")
    return CompanyDemoReport(
        root=corpus_root,
        document_count=len(documents),
        benchmark_count=len(benchmark),
        departments={document.department for document in documents},
        benchmark_departments=benchmark_departments,
        benchmark_sha256=benchmark.sha256,
    )


__all__ = ["ALLOWED_DEPARTMENTS", "CompanyDemoReport", "CompanyDemoValidationError", "validate_company_demo"]
