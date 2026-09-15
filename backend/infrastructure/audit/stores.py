"""Audit-log persistence boundaries with verified, rotating hash-chain files."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict
from pathlib import Path

import fcntl

from domain.audit import (
    AuditIntegrityError,
    AuditLog,
    AuditStore,
    AuditStoreError,
    AuditWriteError,
)
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryAuditStore:
    """Persist and retrieve memory audit data."""
    def __init__(self) -> None:
        """Initialize the memory audit store."""
        self._logs: list["AuditLog"] = []
        self._lock = threading.Lock()

    def append(self, log: "AuditLog") -> None:
        """Append a record to the memory audit store."""
        with self._lock:
            self._logs.append(log)

    def query(
        self,
        user_id: str | None = None,
        action: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        org_id: str | None = None,
        offset: int = 0,
    ) -> list["AuditLog"]:
        """Query records through the memory audit store."""
        with self._lock:
            result = []
            skipped = 0
            for log in reversed(self._logs):
                if org_id is not None and log.org_id != org_id:
                    continue
                if user_id and log.user_id != user_id:
                    continue
                if action and log.action != action:
                    continue
                if start and log.timestamp < start:
                    continue
                if end and log.timestamp > end:
                    continue
                if skipped < offset:
                    skipped += 1
                    continue
                result.append(log)
                if len(result) >= limit:
                    break
            return result


class FileAuditStore:
    """Append-only JSONL audit store with verified, retained hash-chain files."""

    def __init__(
        self,
        file_path: str,
        *,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 7,
    ) -> None:
        """Initialize the file audit store."""
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if backup_count < 0:
            raise ValueError("backup_count must be non-negative")
        self._file_path = str(Path(file_path))
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        os.makedirs(os.path.dirname(self._file_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._lock_path = f"{self._file_path}.lock"

    def _retained_paths_oldest_first(self) -> list[str]:
        """List retained audit files from oldest to newest."""
        paths = [
            f"{self._file_path}.{generation}"
            for generation in range(self._backup_count, 0, -1)
            if os.path.exists(f"{self._file_path}.{generation}")
        ]
        if os.path.exists(self._file_path):
            paths.append(self._file_path)
        return paths

    def _open_lock_file(self):
        """Open the audit lock file with restrictive permissions."""
        descriptor = os.open(
            self._lock_path,
            os.O_CREAT | os.O_RDWR,
            0o600,
        )
        os.chmod(self._lock_path, 0o600)
        return os.fdopen(descriptor, "a+", encoding="utf-8")

    def _read_verified(self) -> list["AuditLog"]:
        """Read the verified."""
        records: list[AuditLog] = []
        previous_hash: str | None = None
        for path in self._retained_paths_oldest_first():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    lines = handle.readlines()
            except OSError as error:
                raise AuditIntegrityError("audit history could not be read") from error

            for line_number, line in enumerate(lines, start=1):
                try:
                    data = json.loads(line)
                    record = AuditLog(**data)
                except (json.JSONDecodeError, TypeError) as error:
                    raise AuditIntegrityError(
                        f"invalid audit record at {Path(path).name}:{line_number}"
                    ) from error
                if previous_hash is not None and record.prev_hash != previous_hash:
                    raise AuditIntegrityError(
                        f"broken audit predecessor at {Path(path).name}:{line_number}"
                    )
                if not record.hash or record.hash != self._compute_hash(record):
                    raise AuditIntegrityError(
                        f"invalid audit hash at {Path(path).name}:{line_number}"
                    )
                records.append(record)
                previous_hash = record.hash
        return records

    def _rotate(self) -> None:
        """Rotate audit files when the active file exceeds its limit."""
        if not os.path.exists(self._file_path):
            return
        if self._backup_count == 0:
            os.unlink(self._file_path)
            return
        oldest = f"{self._file_path}.{self._backup_count}"
        if os.path.exists(oldest):
            os.unlink(oldest)
        for generation in range(self._backup_count - 1, 0, -1):
            source = f"{self._file_path}.{generation}"
            if os.path.exists(source):
                os.replace(source, f"{self._file_path}.{generation + 1}")
        os.replace(self._file_path, f"{self._file_path}.1")

    def append(self, log: "AuditLog") -> None:
        """Append a record to the file audit store."""
        with self._lock:
            try:
                with self._open_lock_file() as lock_handle:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    records = self._read_verified()
                    log.prev_hash = records[-1].hash if records else ""
                    log.hash = self._compute_hash(log)
                    serialized = json.dumps(asdict(log), default=str) + "\n"
                    encoded_size = len(serialized.encode("utf-8"))
                    active_size = (
                        os.path.getsize(self._file_path)
                        if os.path.exists(self._file_path)
                        else 0
                    )
                    if active_size and active_size + encoded_size > self._max_bytes:
                        self._rotate()
                    descriptor = os.open(
                        self._file_path,
                        os.O_CREAT | os.O_APPEND | os.O_WRONLY,
                        0o600,
                    )
                    os.chmod(self._file_path, 0o600)
                    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                        handle.write(serialized)
                        handle.flush()
                        os.fsync(handle.fileno())
            except Exception as error:
                if isinstance(error, (AuditWriteError, AuditIntegrityError)):
                    raise
                raise AuditWriteError("audit record could not be persisted") from error

    @staticmethod
    def _compute_hash(log: "AuditLog") -> str:
        """Compute the hash."""
        payload = asdict(log)
        # Existing records were hashed while this field still held its empty
        # default. Keeping that canonical form preserves on-disk compatibility.
        payload["hash"] = ""
        serialized = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def query(
        self,
        user_id: str | None = None,
        action: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        org_id: str | None = None,
        offset: int = 0,
    ) -> list["AuditLog"]:
        """Query records through the file audit store."""
        with self._lock:
            try:
                with self._open_lock_file() as lock_handle:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
                    records = self._read_verified()
            except AuditIntegrityError:
                raise
            except Exception as error:
                raise AuditIntegrityError("audit history could not be verified") from error

        result: list["AuditLog"] = []
        skipped = 0
        for record in reversed(records):
            if org_id is not None and record.org_id != org_id:
                continue
            if user_id and record.user_id != user_id:
                continue
            if action and record.action != action:
                continue
            if start and record.timestamp < start:
                continue
            if end and record.timestamp > end:
                continue
            if skipped < offset:
                skipped += 1
                continue
            result.append(record)
            if len(result) >= limit:
                break
        return result


__all__ = [
    "AuditIntegrityError",
    "AuditLog",
    "AuditStore",
    "AuditStoreError",
    "AuditWriteError",
    "FileAuditStore",
    "MemoryAuditStore",
]
