"""Create deterministic normalized text from the company Demo DOCX corpus."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree
from zipfile import ZipFile

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

PLACEHOLDER_PATTERNS = (
    re.compile(r"×{2,}(?:公司|集团|有限责任公司)?", re.IGNORECASE),
    re.compile(r"\bX{2,}(?:公司|集团|有限责任公司)?\b", re.IGNORECASE),
    re.compile(r"_{2,}"),
    re.compile(r"(?:待定|待填写|年月日)"),
)
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HORIZONTAL_WHITESPACE = re.compile(r"[\t \u3000]+")


class CleaningValidationError(ValueError):
    """Raised when the company Demo cleaning run is incomplete or unsafe."""


@dataclass(frozen=True)
class CleanedDocumentRecord:
    """Describe one normalized document and its provenance."""

    source_document_id: str
    department: str
    input_path: str
    output_path: str
    input_sha256: str
    output_sha256: str
    input_bytes: int
    output_characters: int
    paragraph_count: int
    table_row_count: int
    placeholder_hits: int
    status: str


@dataclass(frozen=True)
class CleaningReport:
    """Summarize a successful company Demo cleaning run."""

    root: Path
    document_count: int
    output_characters: int
    paragraph_count: int
    table_row_count: int
    placeholder_hits: int
    documents: tuple[CleanedDocumentRecord, ...]


def _sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest for in-memory bytes."""
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for a file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_line(value: str) -> str:
    """Normalize Unicode and horizontal whitespace while preserving wording."""
    normalized = unicodedata.normalize("NFC", value.replace("\u00a0", " "))
    normalized = CONTROL_CHARACTERS.sub("", normalized)
    return HORIZONTAL_WHITESPACE.sub(" ", normalized).strip()


def _iter_blocks(document: DocumentType) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in their original document order."""
    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _count_placeholder_types(text: str) -> int:
    """Count distinct placeholder pattern types present in normalized text."""
    return sum(bool(pattern.search(text)) for pattern in PLACEHOLDER_PATTERNS)


def _extract_xml_paragraphs(path: Path) -> list[str]:
    """Recover text stored in DOCX drawings or text boxes when python-docx exposes no blocks."""
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    try:
        with ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except Exception as error:
        raise CleaningValidationError(f"DOCX cleaning failed for {path.name}: invalid document XML") from error

    lines: list[str] = []
    seen: set[str] = set()
    for paragraph in root.findall(f".//{{{namespace}}}p"):
        text = "".join(node.text or "" for node in paragraph.findall(f".//{{{namespace}}}t"))
        normalized = _normalize_line(text)
        if normalized and normalized not in seen:
            seen.add(normalized)
            lines.append(normalized)

    if len(lines) > 5 and len(lines[0]) > 200:
        embedded_following_lines = sum(line in lines[0] for line in lines[1:11])
        if embedded_following_lines >= 5:
            lines.pop(0)
    return lines


def _extract_docx(path: Path) -> tuple[list[str], int, int, int]:
    """Extract normalized paragraph and table lines from one DOCX file."""
    try:
        document = Document(path)
    except Exception as error:
        raise CleaningValidationError(f"DOCX cleaning failed for {path.name}: invalid document") from error

    lines: list[str] = []
    paragraph_count = 0
    table_row_count = 0
    for block in _iter_blocks(document):
        if isinstance(block, Paragraph):
            text = _normalize_line(block.text or "")
            if not text:
                continue
            paragraph_count += 1
            if not lines or lines[-1] != text:
                lines.append(text)
            continue

        for row in block.rows:
            cells = [_normalize_line((cell.text or "").replace("\n", " ")) for cell in row.cells]
            while cells and not cells[-1]:
                cells.pop()
            if not any(cells):
                continue
            table_row_count += 1
            table_line = f"表格行: {' | '.join(cells)}"
            if not lines or lines[-1] != table_line:
                lines.append(table_line)

    if not lines:
        lines = _extract_xml_paragraphs(path)
        paragraph_count = len(lines)
    if not lines:
        raise CleaningValidationError(f"DOCX cleaning failed for {path.name}: no extractable text")
    text = "\n".join(lines)
    return lines, paragraph_count, table_row_count, _count_placeholder_types(text)


def _required_string(record: dict[str, Any], field: str, index: int) -> str:
    """Read one required manifest string for cleaning."""
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CleaningValidationError(f"manifest documents[{index}].{field} must be a non-empty string")
    return value.strip()


def _write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text atomically inside the workspace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _report_path(path: Path, corpus_root: Path) -> str:
    """Return a corpus-relative report path when possible, otherwise a resolved path."""
    try:
        return path.relative_to(corpus_root).as_posix()
    except ValueError:
        return path.as_posix()


def _write_report(path: Path, corpus_id: str, records: list[CleanedDocumentRecord], *, status: str) -> None:
    """Write the cleaning report with deterministic document records."""
    payload = {
        "schema_version": 1,
        "corpus_id": corpus_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "document_count": len(records),
            "output_characters": sum(item.output_characters for item in records),
            "paragraph_count": sum(item.paragraph_count for item in records),
            "table_row_count": sum(item.table_row_count for item in records),
            "placeholder_hits": sum(item.placeholder_hits for item in records),
        },
        "documents": [asdict(item) for item in records],
    }
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def clean_company_demo(
    root: str | Path,
    *,
    normalized_root: str | Path | None = None,
    report_path: str | Path | None = None,
) -> CleaningReport:
    """Normalize every manifest DOCX into TXT and return an offline cleaning report."""
    corpus_root = Path(root).resolve()
    manifest_path = corpus_root / "corpus_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CleaningValidationError(f"missing manifest: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise CleaningValidationError(f"invalid manifest JSON: {error.msg}") from error
    if not isinstance(manifest, dict):
        raise CleaningValidationError("manifest must be a JSON object")

    corpus_id = manifest.get("corpus_id")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise CleaningValidationError("manifest corpus_id must be a non-empty string")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise CleaningValidationError("manifest documents must be a non-empty list")

    output_root = Path(normalized_root).resolve() if normalized_root is not None else corpus_root / "normalized"
    output_report = Path(report_path).resolve() if report_path is not None else corpus_root / "cleaning_report.json"
    records: list[CleanedDocumentRecord] = []
    failures: list[str] = []

    for index, raw_document in enumerate(documents, start=1):
        if not isinstance(raw_document, dict):
            failures.append(f"documents[{index}] is not an object")
            continue
        try:
            source_document_id = _required_string(raw_document, "source_document_id", index)
            department = _required_string(raw_document, "department", index)
            repository_path = _required_string(raw_document, "repository_path", index)
            source_format = _required_string(raw_document, "source_format", index).lower().lstrip(".")
            expected_sha256 = _required_string(raw_document, "sha256", index).lower()
            if source_format != "docx":
                raise CleaningValidationError(f"unsupported cleaning source format: {source_format}")
            source_path = corpus_root / repository_path
            if not source_path.is_file():
                raise CleaningValidationError(f"source file does not exist: {repository_path}")
            before_sha256 = _sha256_file(source_path)
            if before_sha256 != expected_sha256:
                raise CleaningValidationError(f"source sha256 mismatch: {repository_path}")

            lines, paragraphs, table_rows, placeholder_hits = _extract_docx(source_path)
            header = [
                f"文档标题: {_required_string(raw_document, 'title', index)}",
                f"来源文档ID: {source_document_id}",
                f"部门: {department}",
                "---",
            ]
            output_text = "\n".join(header + lines).strip() + "\n"
            output_path = output_root / department / f"{source_path.stem}.txt"
            _write_text_atomic(output_path, output_text)
            after_sha256 = _sha256_file(source_path)
            if after_sha256 != before_sha256:
                raise CleaningValidationError(f"source file changed during cleaning: {repository_path}")
            output_bytes = output_text.encode("utf-8")
            records.append(
                CleanedDocumentRecord(
                    source_document_id=source_document_id,
                    department=department,
                    input_path=source_path.relative_to(corpus_root).as_posix(),
                    output_path=_report_path(output_path, corpus_root),
                    input_sha256=before_sha256,
                    output_sha256=_sha256_bytes(output_bytes),
                    input_bytes=source_path.stat().st_size,
                    output_characters=len(output_text),
                    paragraph_count=paragraphs,
                    table_row_count=table_rows,
                    placeholder_hits=placeholder_hits,
                    status="succeeded",
                )
            )
        except CleaningValidationError as error:
            failures.append(str(error))

    if failures:
        _write_report(output_report, corpus_id, records, status="failed")
        raise CleaningValidationError(f"company Demo cleaning failed: {'; '.join(failures)}")

    _write_report(output_report, corpus_id, records, status="succeeded")
    return CleaningReport(
        root=corpus_root,
        document_count=len(records),
        output_characters=sum(item.output_characters for item in records),
        paragraph_count=sum(item.paragraph_count for item in records),
        table_row_count=sum(item.table_row_count for item in records),
        placeholder_hits=sum(item.placeholder_hits for item in records),
        documents=tuple(records),
    )


__all__ = ["CleaningReport", "CleaningValidationError", "CleanedDocumentRecord", "clean_company_demo"]
