"""Tenant-partitioned native BM25 snapshots over canonical document chunks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from domain.documents import DocumentChunk

BM25_SCHEMA_VERSION = "native-bm25-v1"
BM25_TOKENIZER_VERSION = "unicode-cjk-bigram-v1"
_WORD_OR_CJK = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+")


class BM25IndexError(RuntimeError):
    """Base error for the native sparse index."""


class BM25IndexUnavailable(BM25IndexError):
    """No readable native sparse snapshot is available."""


class BM25IndexContractError(BM25IndexError):
    """The stored sparse snapshot is incompatible or malformed."""


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize mixed Latin/CJK text deterministically for lexical retrieval."""

    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    tokens: list[str] = []
    for match in _WORD_OR_CJK.finditer(normalized):
        value = match.group(0)
        if not value:
            continue
        if "\u3400" <= value[0] <= "\u9fff":
            chars = list(value)
            tokens.extend(chars)
            tokens.extend("".join(chars[index : index + 2]) for index in range(len(chars) - 1))
        else:
            tokens.append(value[:128])
    return tokens


@dataclass(frozen=True)
class _SparseRecord:
    record_id: str
    content: str
    source: str
    metadata: dict[str, Any]
    term_frequencies: dict[str, int]
    length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "term_frequencies": self.term_frequencies,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "_SparseRecord":
        try:
            record_id = str(value["record_id"])
            content = str(value["content"])
            source = str(value.get("source") or "")
            metadata = dict(value["metadata"])
            term_frequencies = {
                str(term): int(count)
                for term, count in dict(value["term_frequencies"]).items()
                if int(count) > 0
            }
            length = int(value["length"])
        except (KeyError, TypeError, ValueError) as error:
            raise BM25IndexContractError("invalid sparse record") from error
        if not record_id or length < 0 or length != sum(term_frequencies.values()):
            raise BM25IndexContractError("invalid sparse record statistics")
        return cls(
            record_id=record_id,
            content=content,
            source=source,
            metadata=metadata,
            term_frequencies=term_frequencies,
            length=length,
        )


class NativeBM25Index:
    """Persist and query small deterministic BM25 snapshots partitioned by tenant."""

    def __init__(
        self,
        root_path: str | os.PathLike[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        schema_version: str = BM25_SCHEMA_VERSION,
        tokenizer_version: str = BM25_TOKENIZER_VERSION,
    ) -> None:
        if k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")
        self.root_path = Path(root_path)
        self.k1 = float(k1)
        self.b = float(b)
        self.schema_version = schema_version
        self.tokenizer_version = tokenizer_version
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, tenant_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(tenant_id, threading.RLock())

    def _tenant_directory(self, tenant_id: str) -> Path:
        digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        return self.root_path / digest

    def _current_path(self, tenant_id: str) -> Path:
        return self._tenant_directory(tenant_id) / "current.json"

    @staticmethod
    def _record_id(tenant_id: str, doc_id: str, chunk_id: str) -> str:
        return hashlib.sha256(f"{tenant_id}\0{doc_id}\0{chunk_id}".encode("utf-8")).hexdigest()

    def _record_from_chunk(self, chunk: DocumentChunk, tenant_id: str) -> _SparseRecord:
        metadata = dict(chunk.metadata or {})
        source = str(metadata.get("source") or "")
        file_name = str(
            metadata.get("file_name")
            or metadata.get("uploaded_filename")
            or (Path(source).name if source else "")
            or chunk.doc_id
        )
        canonical_metadata = {
            **metadata,
            "tenant_id": tenant_id,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "doc_type": chunk.doc_type.value,
            "source": source,
            "file_name": file_name,
        }
        tokens = tokenize_bm25(chunk.content)
        frequencies = dict(sorted(Counter(tokens).items()))
        return _SparseRecord(
            record_id=self._record_id(tenant_id, chunk.doc_id, chunk.chunk_id),
            content=chunk.content,
            source=source,
            metadata=canonical_metadata,
            term_frequencies=frequencies,
            length=len(tokens),
        )

    def _empty_payload(self, tenant_id: str) -> dict[str, Any]:
        return self._payload_from_records(tenant_id, [])

    def _payload_from_records(
        self,
        tenant_id: str,
        records: Iterable[_SparseRecord],
    ) -> dict[str, Any]:
        ordered = sorted(records, key=lambda item: item.record_id)
        document_frequency: Counter[str] = Counter()
        for record in ordered:
            document_frequency.update(record.term_frequencies)
        average_length = (
            sum(record.length for record in ordered) / len(ordered) if ordered else 0.0
        )
        generation_seed = "\n".join(
            f"{record.record_id}:{hashlib.sha256(record.content.encode('utf-8')).hexdigest()}"
            for record in ordered
        )
        generation = hashlib.sha256(generation_seed.encode("utf-8")).hexdigest()[:16]
        return {
            "manifest": {
                "schema_version": self.schema_version,
                "tokenizer_version": self.tokenizer_version,
                "tenant_id": tenant_id,
                "generation": generation,
                "created_at": datetime.now(UTC).isoformat(),
                "k1": self.k1,
                "b": self.b,
                "record_count": len(ordered),
                "average_document_length": average_length,
                "document_frequency": dict(sorted(document_frequency.items())),
            },
            "records": [record.to_dict() for record in ordered],
        }

    def _validate_payload(self, payload: dict[str, Any], tenant_id: str) -> list[_SparseRecord]:
        try:
            manifest = dict(payload["manifest"])
            records_value = list(payload["records"])
        except (KeyError, TypeError, ValueError) as error:
            raise BM25IndexContractError("invalid sparse snapshot") from error
        if manifest.get("schema_version") != self.schema_version:
            raise BM25IndexContractError("incompatible sparse schema version")
        if manifest.get("tokenizer_version") != self.tokenizer_version:
            raise BM25IndexContractError("incompatible sparse tokenizer version")
        if str(manifest.get("tenant_id") or "") != tenant_id:
            raise BM25IndexContractError("sparse snapshot tenant mismatch")
        try:
            stored_k1 = float(manifest.get("k1"))
            stored_b = float(manifest.get("b"))
            record_count = int(manifest.get("record_count"))
        except (TypeError, ValueError) as error:
            raise BM25IndexContractError("invalid sparse manifest parameters") from error
        if stored_k1 != self.k1 or stored_b != self.b:
            raise BM25IndexContractError("incompatible sparse scoring parameters")
        records = [_SparseRecord.from_dict(item) for item in records_value]
        if record_count != len(records):
            raise BM25IndexContractError("sparse manifest record count mismatch")
        for record in records:
            if str(record.metadata.get("tenant_id") or "") != tenant_id:
                raise BM25IndexContractError("sparse record tenant mismatch")
        return records

    def _load_payload(self, tenant_id: str, *, allow_missing: bool = False) -> dict[str, Any]:
        path = self._current_path(tenant_id)
        if not path.exists():
            if allow_missing:
                return self._empty_payload(tenant_id)
            raise BM25IndexUnavailable(f"no sparse snapshot for tenant {tenant_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BM25IndexUnavailable("sparse snapshot cannot be read") from error
        self._validate_payload(payload, tenant_id)
        return payload

    def _write_snapshot_file(self, target: Path, payload: dict[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with target.open("w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

    def _promote_payload(self, tenant_id: str, payload: dict[str, Any]) -> None:
        self._validate_payload(payload, tenant_id)
        current_path = self._current_path(tenant_id)
        temporary_path = current_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            self._write_snapshot_file(temporary_path, payload)
            written = json.loads(temporary_path.read_text(encoding="utf-8"))
            self._validate_payload(written, tenant_id)
            os.replace(temporary_path, current_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _add_chunks_sync(self, chunks: list[DocumentChunk]) -> int:
        groups: dict[str, list[DocumentChunk]] = {}
        for chunk in chunks:
            tenant_id = str(chunk.tenant_id or chunk.metadata.get("tenant_id") or "")
            if not tenant_id:
                raise ValueError("native BM25 chunks require tenant_id")
            groups.setdefault(tenant_id, []).append(chunk)
        for tenant_id, tenant_chunks in groups.items():
            with self._lock_for(tenant_id):
                current = self._load_payload(tenant_id, allow_missing=True)
                records = {
                    record.record_id: record
                    for record in self._validate_payload(current, tenant_id)
                }
                for chunk in tenant_chunks:
                    record = self._record_from_chunk(chunk, tenant_id)
                    records[record.record_id] = record
                self._promote_payload(tenant_id, self._payload_from_records(tenant_id, records.values()))
        return len(chunks)

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Idempotently add or replace canonical chunks by provenance identity."""

        if not chunks:
            return 0
        return await asyncio.to_thread(self._add_chunks_sync, chunks)

    def _delete_sync(self, doc_id: str, tenant_id: str) -> int:
        if not tenant_id:
            raise ValueError("native BM25 deletion requires tenant_id")
        with self._lock_for(tenant_id):
            payload = self._load_payload(tenant_id, allow_missing=True)
            records = self._validate_payload(payload, tenant_id)
            kept = [record for record in records if str(record.metadata.get("doc_id")) != doc_id]
            deleted = len(records) - len(kept)
            if deleted:
                self._promote_payload(tenant_id, self._payload_from_records(tenant_id, kept))
            return deleted

    async def delete_by_doc_id(self, doc_id: str, *, tenant_id: str) -> int:
        """Delete only the specified tenant-owned document's sparse records."""

        return await asyncio.to_thread(self._delete_sync, doc_id, tenant_id)

    def _rebuild_sync(self, tenant_id: str, chunks: list[DocumentChunk]) -> int:
        if not tenant_id:
            raise ValueError("native BM25 rebuild requires tenant_id")
        records = []
        for chunk in chunks:
            chunk_tenant = str(chunk.tenant_id or chunk.metadata.get("tenant_id") or "")
            if chunk_tenant != tenant_id:
                raise ValueError("native BM25 rebuild cannot mix tenants")
            records.append(self._record_from_chunk(chunk, tenant_id))
        deduplicated = {record.record_id: record for record in records}
        with self._lock_for(tenant_id):
            self._promote_payload(
                tenant_id,
                self._payload_from_records(tenant_id, deduplicated.values()),
            )
        return len(deduplicated)

    async def rebuild(self, tenant_id: str, chunks: list[DocumentChunk]) -> int:
        """Atomically replace one tenant's sparse snapshot from canonical chunks."""

        return await asyncio.to_thread(self._rebuild_sync, tenant_id, chunks)

    @staticmethod
    def _is_governance_visible(metadata: dict[str, Any]) -> bool:
        if metadata.get("is_visible") is False:
            return False
        return str(metadata.get("review_status") or "").lower() not in {"rejected", "deleted"}

    def _search_sync(
        self,
        query: str,
        tenant_id: str,
        visible_department_ids: tuple[str, ...],
        top_k: int,
    ) -> list[tuple[dict[str, Any], float]]:
        if not tenant_id:
            raise ValueError("native BM25 search requires tenant_id")
        if top_k <= 0:
            return []
        with self._lock_for(tenant_id):
            payload = self._load_payload(tenant_id)
            records = self._validate_payload(payload, tenant_id)
        visible_departments = frozenset(visible_department_ids)
        authorized = [
            record
            for record in records
            if self._is_governance_visible(record.metadata)
            and (
                not visible_departments
                or str(record.metadata.get("department_id") or "") in visible_departments
            )
        ]
        query_terms = tokenize_bm25(query)
        if not query_terms or not authorized:
            return []

        document_frequency: Counter[str] = Counter()
        for record in authorized:
            document_frequency.update(record.term_frequencies)
        average_length = sum(record.length for record in authorized) / len(authorized)
        query_frequency = Counter(query_terms)
        total = len(authorized)
        scored: list[tuple[_SparseRecord, float]] = []
        for record in authorized:
            score = 0.0
            normalization = 1 - self.b + self.b * (
                record.length / average_length if average_length else 0.0
            )
            for term, query_count in query_frequency.items():
                frequency = record.term_frequencies.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                score += query_count * idf * (
                    frequency * (self.k1 + 1)
                    / (frequency + self.k1 * normalization)
                )
            if score > 0:
                scored.append((record, score))
        scored.sort(key=lambda item: (-item[1], item[0].record_id))
        return [
            (
                {
                    "content": record.content,
                    "source": record.source,
                    "metadata": dict(record.metadata),
                },
                score,
            )
            for record, score in scored[:top_k]
        ]

    async def search(
        self,
        query: str,
        *,
        tenant_id: str,
        visible_department_ids: tuple[str, ...] | list[str] | None = None,
        top_k: int = 5,
    ) -> list[tuple[dict[str, Any], float]]:
        """Search one authorized tenant snapshot and return vector-compatible records."""

        return await asyncio.to_thread(
            self._search_sync,
            query,
            tenant_id,
            tuple(visible_department_ids or ()),
            top_k,
        )

    async def get_status(self, tenant_id: str) -> dict[str, Any]:
        """Return bounded health and identity metadata for one tenant snapshot."""

        path = self._current_path(tenant_id)
        try:
            payload = await asyncio.to_thread(self._load_payload, tenant_id)
        except BM25IndexError as error:
            return {
                "available": False,
                "tenant_id": tenant_id,
                "schema_version": self.schema_version,
                "tokenizer_version": self.tokenizer_version,
                "path": str(path),
                "error_type": type(error).__name__,
            }
        manifest = payload["manifest"]
        return {
            "available": True,
            "tenant_id": tenant_id,
            "schema_version": manifest["schema_version"],
            "tokenizer_version": manifest["tokenizer_version"],
            "generation": manifest["generation"],
            "record_count": manifest["record_count"],
            "path": str(path),
            "k1": manifest["k1"],
            "b": manifest["b"],
        }
