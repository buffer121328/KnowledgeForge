"""Validated Agent benchmark records and deterministic trace checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from domain.trace import TRACE_OPERATIONS, AgentTrace
from domain.knowledge import QAResult

SUPPORTED_SCHEMA_VERSION = 1


class AgentBenchmarkValidationError(ValueError):
    """Raised when an Agent benchmark cannot be safely evaluated."""


@dataclass(frozen=True)
class AgentBenchmarkSample:
    """One reviewed Agent/workflow benchmark record."""

    id: str
    user_input: str
    reference: str
    reference_topics: tuple[str, ...]
    expected_refusal: bool
    expected_operations: tuple[str, ...]
    notes: str = ""
    schema_version: int = SUPPORTED_SCHEMA_VERSION


class AgentBenchmarkDataset(list[AgentBenchmarkSample]):
    """A typed dataset carrying source provenance."""

    def __init__(self, samples: Iterable[AgentBenchmarkSample], *, source_path: Path, sha256: str) -> None:
        super().__init__(samples)
        self.source_path = source_path
        self.sha256 = sha256


def sha256_file(path: Path) -> str:
    """Return a benchmark file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_error(line_number: int, message: str) -> AgentBenchmarkValidationError:
    return AgentBenchmarkValidationError(f"agent benchmark line {line_number}: {message}")


def _required_string(value: Any, field_name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _line_error(line_number, f"{field_name} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, field_name: str, line_number: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise _line_error(line_number, f"{field_name} must be a non-empty list")
    values: list[str] = []
    for item in value:
        values.append(_required_string(item, field_name, line_number))
    return tuple(values)


def _operations(value: Any, line_number: int) -> tuple[str, ...]:
    operations = _string_list(value, "expected_operations", line_number)
    unknown = [operation for operation in operations if operation not in TRACE_OPERATIONS]
    if unknown:
        raise _line_error(line_number, f"unknown operation: {unknown[0]}")
    if len(set(operations)) != len(operations):
        raise _line_error(line_number, "expected_operations must not contain duplicates")
    return operations


def _parse_record(record: Any, line_number: int) -> AgentBenchmarkSample:
    if not isinstance(record, dict):
        raise _line_error(line_number, "record must be a JSON object")
    schema_version = record.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise _line_error(line_number, f"unsupported schema_version: {schema_version}")
    expected_refusal = record.get("expected_refusal")
    if not isinstance(expected_refusal, bool):
        raise _line_error(line_number, "expected_refusal must be a boolean")
    notes = record.get("notes", "")
    if not isinstance(notes, str):
        raise _line_error(line_number, "notes must be a string when provided")
    return AgentBenchmarkSample(
        id=_required_string(record.get("id"), "id", line_number),
        user_input=_required_string(record.get("user_input"), "user_input", line_number),
        reference=_required_string(record.get("reference"), "reference", line_number),
        reference_topics=_string_list(record.get("reference_topics"), "reference_topics", line_number),
        expected_refusal=expected_refusal,
        expected_operations=_operations(record.get("expected_operations"), line_number),
        notes=notes.strip(),
        schema_version=schema_version,
    )


def load_agent_benchmark(path: str | Path) -> AgentBenchmarkDataset:
    """Load the Agent benchmark as offline-only metadata."""
    source_path = Path(path)
    if "documents" in source_path.parts:
        raise AgentBenchmarkValidationError(
            "Agent benchmark must stay outside the runtime documents upload directory"
        )
    if not source_path.is_file():
        raise AgentBenchmarkValidationError(f"Agent benchmark file does not exist: {source_path}")
    samples: list[AgentBenchmarkSample] = []
    sample_ids: set[str] = set()
    for line_number, raw_line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise _line_error(line_number, "must contain valid JSON") from error
        sample = _parse_record(record, line_number)
        if sample.id in sample_ids:
            raise _line_error(line_number, f"duplicate id: {sample.id}")
        sample_ids.add(sample.id)
        samples.append(sample)
    if not samples:
        raise AgentBenchmarkValidationError("Agent benchmark must contain at least one JSONL record")
    return AgentBenchmarkDataset(samples, source_path=source_path, sha256=sha256_file(source_path))


def _trace_operation_counts(trace: AgentTrace) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in trace.events:
        counts[event.operation] = counts.get(event.operation, 0) + 1
    return counts


def _trace_is_valid(trace: AgentTrace) -> bool:
    if trace.schema_version != "v1" or not trace.trace_id:
        return False
    previous_event_ids: set[str] = set()
    for event in trace.events:
        if not event.event_id or event.event_id in previous_event_ids:
            return False
        if event.operation not in TRACE_OPERATIONS:
            return False
        if event.status not in {"succeeded", "failed", "skipped"}:
            return False
        if event.duration_ms < 0 or event.error_message is not None:
            return False
        previous_event_ids.add(event.event_id)
    return True


def evaluate_agent_trace(sample: AgentBenchmarkSample, result: QAResult) -> dict[str, Any]:
    """Evaluate a trace using only deterministic workflow-contract checks."""
    trace = result.trace
    base = {
        "benchmark_id": sample.id,
        "trace_id": trace.trace_id if trace else None,
        "expected_refusal": sample.expected_refusal,
        "actual_refusal": result.degradation_code == "insufficient_verified_evidence",
    }
    if trace is None:
        return {**base, "status": "invalid_trace", "reason_code": "missing_trace"}
    if not _trace_is_valid(trace):
        return {**base, "status": "invalid_trace", "reason_code": "malformed_trace"}

    counts = _trace_operation_counts(trace)
    missing_operations = [operation for operation in sample.expected_operations if not counts.get(operation)]
    refusal_mismatch = base["expected_refusal"] != base["actual_refusal"]
    failed_operations = [
        event.operation for event in trace.events if event.status == "failed"
    ]
    status = "passed" if not missing_operations and not refusal_mismatch and not failed_operations else "failed"
    return {
        **base,
        "status": status,
        "missing_operations": missing_operations,
        "failed_operations": failed_operations,
        "refusal_match": not refusal_mismatch,
        "operation_count": len(trace.events),
    }


__all__ = [
    "AgentBenchmarkDataset",
    "AgentBenchmarkSample",
    "AgentBenchmarkValidationError",
    "SUPPORTED_SCHEMA_VERSION",
    "evaluate_agent_trace",
    "load_agent_benchmark",
    "sha256_file",
]
