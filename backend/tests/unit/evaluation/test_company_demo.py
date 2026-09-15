"""Acceptance tests for the four-department Chinese company Demo corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from evaluation.company_demo.corpus import CompanyDemoValidationError, validate_company_demo


DEPARTMENTS = ("human_resources", "finance", "procurement_warehouse", "administration")


def _write_fixture(root: Path, *, benchmark_document_id: str | None = None) -> dict[str, str]:
    """Create a minimal offline corpus fixture with all required departments."""
    document_ids: dict[str, str] = {}
    documents: list[dict[str, str]] = []
    for department in DEPARTMENTS:
        document_id = str(uuid4())
        document_ids[department] = document_id
        relative_path = Path("documents") / department / f"{department}.txt"
        document_path = root / relative_path
        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(f"{department} demo policy\n", encoding="utf-8")
        documents.append(
            {
                "source_document_id": document_id,
                "department": department,
                "department_label": department,
                "document_type": "policy",
                "title": f"{department} policy",
                "source_filename": document_path.name,
                "repository_path": relative_path.as_posix(),
                "source_format": "txt",
                "authority": "formal_candidate",
                "status": "needs_review",
                "sensitivity": "internal_demo",
                "sha256": hashlib.sha256(document_path.read_bytes()).hexdigest(),
            }
        )

    benchmark_records = []
    for index, department in enumerate(DEPARTMENTS, start=1):
        source_id = benchmark_document_id or document_ids[department]
        benchmark_records.append(
            {
                "id": f"demo-{index}",
                "question": f"{department} 的制度是什么？",
                "reference": f"{department} 的 Demo 制度。",
                "required_doc_ids": [source_id],
                "reference_context_ids": [source_id],
                "evidence": [
                    {
                        "source_document_id": source_id,
                        "page": 1,
                        "section": "Demo",
                        "claim": f"{department} 文档包含制度内容。",
                    }
                ],
                "category": "single_document_fact",
                "expected_refusal": False,
            }
        )

    (root / "benchmark.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in benchmark_records),
        encoding="utf-8",
    )
    (root / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus_id": "company-demo-zh-v1",
                "language": "zh-CN",
                "selection": {
                    "departments": list(DEPARTMENTS),
                    "runtime_ingestion_policy": "documents-only",
                },
                "documents": documents,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return document_ids


def test_company_demo_fixture_passes_offline_validation(tmp_path: Path) -> None:
    """Given a healthy fixture, validation covers all four departments offline."""
    _write_fixture(tmp_path)

    report = validate_company_demo(tmp_path)

    assert report.document_count == 4
    assert report.benchmark_count == 4
    assert report.departments == set(DEPARTMENTS)


def test_company_demo_rejects_hash_drift(tmp_path: Path) -> None:
    """Given a changed source file, validation fails closed with the file path."""
    _write_fixture(tmp_path)
    source = tmp_path / "documents" / "finance" / "finance.txt"
    source.write_text("changed\n", encoding="utf-8")

    with pytest.raises(CompanyDemoValidationError, match="sha256"):
        validate_company_demo(tmp_path)


def test_company_demo_rejects_unknown_benchmark_document(tmp_path: Path) -> None:
    """Given a benchmark ID outside the manifest, validation rejects provenance."""
    _write_fixture(tmp_path, benchmark_document_id=str(uuid4()))

    with pytest.raises(CompanyDemoValidationError, match="unknown document ID"):
        validate_company_demo(tmp_path)


def test_company_demo_rejects_missing_department_coverage(tmp_path: Path) -> None:
    """Given no benchmark question for one department, validation rejects the profile."""
    _write_fixture(tmp_path)
    benchmark_path = tmp_path / "benchmark.jsonl"
    records = [json.loads(line) for line in benchmark_path.read_text(encoding="utf-8").splitlines()]
    benchmark_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records[:3]),
        encoding="utf-8",
    )

    with pytest.raises(CompanyDemoValidationError, match="benchmark coverage"):
        validate_company_demo(tmp_path)


def test_company_demo_rejects_unknown_department(tmp_path: Path) -> None:
    """A manifest cannot replace administration with an unknown department."""
    _write_fixture(tmp_path)
    manifest_path = tmp_path / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"]["departments"][-1] = "sales"
    manifest["documents"][-1]["department"] = "sales"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(CompanyDemoValidationError, match="four company Demo departments"):
        validate_company_demo(tmp_path)
