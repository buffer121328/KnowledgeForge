"""Offline execution runner that captures complete QA evidence for Ragas."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from domain.knowledge import QAResult, RetrievedContext
from shared.utils.logging import sanitize_for_observability

from .benchmarks.benchmark import BenchmarkDataset, BenchmarkSample
from .benchmarks.candidate_stage_capture import (
    CANDIDATE_STAGE_SCHEMA_VERSION,
    CandidateStageRecorder,
)
from .diagnostic.preregistration import DIAGNOSTIC_RETRIEVAL_K_VALUES
from .release.quality_gates import DEFAULT_SMOKE_THRESHOLD

RetrievalMode = Literal[
    "vector",
    "hybrid",
    "dense",
    "dense_graph",
    "dense_bm25",
    "dense_bm25_graph",
]
SUPPORTED_RETRIEVAL_MODES = frozenset(
    {"vector", "hybrid", "dense", "dense_graph", "dense_bm25", "dense_bm25_graph"}
)
QAExecutor = Callable[..., Awaitable[QAResult]]


@dataclass(frozen=True)
class BenchmarkRun:
    """Represent benchmark run."""
    run_id: str
    run_dir: Path
    metadata_path: Path
    responses_path: Path
    invalid_provenance_path: Path
    candidate_stages_path: Path
    candidate_stages_sha256: str
    records: list[dict[str, Any]]


def _utc_now() -> str:
    """Return the utc now."""
    return datetime.now(UTC).isoformat()


def _git_revision() -> str | None:
    """Return the git revision."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _ragas_version() -> str | None:
    """Return the ragas version."""
    try:
        return importlib_metadata.version("ragas")
    except importlib_metadata.PackageNotFoundError:
        return None


def _context_record(context: RetrievedContext, rank: int) -> dict[str, Any]:
    """Return the context record."""
    metadata = dict(context.metadata)
    source_document_id = metadata.get("source_document_id")
    return {
        "rank": rank,
        "content": context.content,
        "source": context.source,
        "score": context.score,
        "retrieval_type": context.retrieval_type,
        "source_document_id": source_document_id,
        "metadata": metadata,
    }


def _valid_source_document_id(value: Any) -> str | None:
    """Return a normalized source document UUID or None when invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        UUID(value)
    except ValueError:
        return None
    return value


def _provenance(contexts: Sequence[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Return the provenance."""
    retrieved_ids: list[str] = []
    unmapped_contexts: list[dict[str, Any]] = []
    for context in contexts:
        source_document_id = _valid_source_document_id(context.get("source_document_id"))
        if source_document_id is None:
            unmapped_contexts.append(context)
            continue
        if source_document_id not in retrieved_ids:
            retrieved_ids.append(source_document_id)
    return retrieved_ids, unmapped_contexts


def _retrieved_identity_fields(contexts: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Return bounded context/content identities separately from document provenance."""
    context_ids: list[str] = []
    content_sha256s: list[str] = []
    for context in contexts:
        metadata = context.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        context_id = next(
            (
                value
                for value in (metadata.get("chunk_id"), metadata.get("context_id"))
                if isinstance(value, str) and 0 < len(value) <= 128
            ),
            None,
        )
        content = context.get("content")
        content_sha256 = (
            hashlib.sha256(content.encode("utf-8")).hexdigest()
            if isinstance(content, str)
            else None
        )
        if context_id and context_id not in context_ids:
            context_ids.append(context_id)
        if (
            isinstance(content_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", content_sha256)
            and content_sha256 not in content_sha256s
        ):
            content_sha256s.append(content_sha256)
    return {
        "retrieved_chunk_ids": context_ids,
        "retrieved_content_sha256s": content_sha256s,
    }


def _sample_source_document_ids(sample: Any) -> list[str]:
    """Resolve reviewed source-document identities without reusing chunk IDs."""
    for name in ("expected_source_document_ids", "required_doc_ids", "required_document_ids"):
        values = getattr(sample, name, ())
        if values:
            return list(dict.fromkeys(str(value) for value in values if isinstance(value, str) and value))
    return []


def _enum_value(value: Any) -> str | None:
    """Return a bounded enum/string value for replay artifacts."""
    candidate = getattr(value, "value", value)
    if isinstance(candidate, str) and 0 < len(candidate) <= 128:
        return candidate
    return None


def _sample_reference(sample: Any) -> str:
    """Return an inline reviewed reference when this benchmark type carries one."""
    value = getattr(sample, "reference", "")
    return value if isinstance(value, str) else ""


def _rss_peak_mb() -> float:
    """Return the process high-water RSS normalized to MiB."""
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_value = maximum if sys.platform == "darwin" else maximum * 1024
    return round(max(0.0, bytes_value / (1024 * 1024)), 3)


def _branch_availability(trace: dict[str, Any] | None) -> dict[str, str]:
    """Extract bounded branch availability from request-local trace events."""
    branches = {"dense": "not_configured", "bm25": "not_configured", "graph": "not_configured"}
    operations = {
        "retrieval.vector": "dense",
        "retrieval.bm25": "bm25",
        "retrieval.graph": "graph",
    }
    events = trace.get("events") if isinstance(trace, dict) else None
    if not isinstance(events, list):
        return branches
    for event in events:
        if not isinstance(event, dict):
            continue
        branch = operations.get(event.get("operation"))
        if branch is None:
            continue
        metadata = event.get("metadata")
        available = metadata.get("available") if isinstance(metadata, dict) else None
        unavailable = metadata.get("unavailable") if isinstance(metadata, dict) else None
        if event.get("status") == "failed" or available is False or unavailable is True:
            branches[branch] = "unavailable"
        else:
            branches[branch] = "available"
    return branches


def _candidate_stages(contexts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist only candidate identifiers, ranks, and branch names."""
    candidates: list[dict[str, Any]] = []
    for context in contexts[:50]:
        metadata = context.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        candidate_id = next(
            (
                value
                for value in (
                    metadata.get("context_id"),
                    metadata.get("chunk_id"),
                    context.get("source_document_id"),
                )
                if isinstance(value, str) and 0 < len(value) <= 128
            ),
            None,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "rank": context.get("rank"),
                "branch": _enum_value(context.get("retrieval_type")),
            }
        )
    return candidates


def _bounded_result_artifacts(result: QAResult, trace: dict[str, Any] | None) -> dict[str, Any]:
    """Build scoring fields and replay stages without raw question/source/claim text."""
    assessment = result.evidence_assessment
    structured = result.structured_answer
    grounding = result.grounding_result
    response_status = _enum_value(result.response_status)
    if response_status is None:
        response_status = (
            "insufficient_evidence"
            if result.degradation_code == "insufficient_verified_evidence"
            else "answered"
        )
    evidence_state = (
        _enum_value(assessment.primary_state) if assessment is not None else None
    )
    claims = []
    citations = []
    observed_missing_information_fields: set[str] = set()
    generation_stage: dict[str, Any] = {"status": "skipped"}
    if structured is not None:
        observed_missing_information_fields.update(
            item.field for item in structured.missing_information
        )
        citation_ids = {citation.citation_id for citation in structured.citations}
        claims = [
            {
                "claim_id": claim.claim_id,
                "material": claim.material,
                "citation_ids": list(claim.citation_ids),
            }
            for claim in structured.claims
        ]
        citations = [
            {"citation_id": citation.citation_id, "context_id": citation.context_id}
            for citation in structured.citations
        ]
        generation_stage = {
            "schema_version": structured.schema_version,
            "claim_count": len(structured.claims),
            "material_claim_count": sum(claim.material for claim in structured.claims),
            "citation_count": len(structured.citations),
            "citation_validation_passed": all(
                set(claim.citation_ids).issubset(citation_ids)
                for claim in structured.claims
            ),
        }
    grounding_value = None
    grounding_stage: dict[str, Any] = {"status": "skipped"}
    if grounding is not None:
        grounding_value = {
            "passed": grounding.passed,
            "accepted_claim_ids": list(grounding.accepted_claim_ids),
            "rejected_claim_ids": list(grounding.rejected_claim_ids),
            "reason_codes": [reason.value for reason in grounding.reason_codes],
            "policy_version": grounding.policy_version,
        }
        grounding_stage = dict(grounding_value)
    qualification_stage: dict[str, Any] = {"status": "skipped"}
    if assessment is not None:
        observed_missing_information_fields.update(
            item.field for item in assessment.missing_information
        )
        qualification_stage = {
            "evidence_states": [state.value for state in assessment.states],
            "response_status": assessment.response_status.value,
            "reason_codes": [reason.value for reason in assessment.reason_codes],
            "evaluated_count": len(assessment.evaluated_context_ids),
            "supporting_count": len(assessment.supporting_context_ids),
            "policy_version": assessment.policy_version,
            "calibration_version": assessment.calibration_version,
        }
    return {
        "response_status": response_status,
        "evidence_state": evidence_state,
        "claims": claims,
        "citations": citations,
        "grounding_result": grounding_value,
        "observed_missing_information_fields": sorted(
            observed_missing_information_fields
        ),
        "observed_reason_codes": (
            [reason.value for reason in assessment.reason_codes]
            if assessment is not None
            else []
        ),
        "security_actions": sorted(
            {
                str(getattr(action, "value", action))
                for action in result.security_actions
                if isinstance(getattr(action, "value", action), str)
                and 0 < len(str(getattr(action, "value", action))) <= 96
            }
        ),
        "stages": {
            "retrieval": {"branch_availability": _branch_availability(trace)},
            "qualification": qualification_stage,
            "generation": generation_stage,
            "grounding": grounding_stage,
        },
    }


def _run_identity(
    *,
    run_id: str,
    started_at: str,
    benchmark_sha256: str,
    retrieval_modes: Sequence[str],
    metadata: dict[str, Any],
    samples: Sequence[Any],
    code_revision: str | None,
) -> dict[str, Any]:
    """Build the bounded immutable identities required to replay one run."""
    fields = (
        "dataset_version",
        "dataset_manifest_sha256",
        "answer_model",
        "embedding_model",
        "retrieval_config_id",
        "threshold_version",
        "candidate_budget_version",
        "reranker_mode",
        "reranker_version",
        "structured_answer_schema_version",
        "grounding_policy_version",
    )
    identity = {
        field: metadata.get(field)
        for field in fields
        if isinstance(metadata.get(field), str)
        and 0 < len(metadata[field]) <= 200
    }
    dataset_version = getattr(samples, "dataset_version", None)
    manifest_sha256 = getattr(samples, "manifest_sha256", None)
    if "dataset_version" not in identity and isinstance(dataset_version, str):
        identity["dataset_version"] = dataset_version
    if "dataset_manifest_sha256" not in identity and isinstance(manifest_sha256, str):
        identity["dataset_manifest_sha256"] = manifest_sha256
    identity.update(
        {
            "run_id": run_id,
            "run_date": started_at[:10],
            "benchmark_sha256": benchmark_sha256,
            "retrieval_modes": list(retrieval_modes),
            "code_revision": code_revision,
        }
    )
    return identity


from evaluation.io import write_jsonl as _write_jsonl  # noqa: E402
from evaluation.io import write_text_secure as _write_text_secure  # noqa: E402


class BenchmarkRunner:
    """Run a validated benchmark through an injected application-layer QA executor."""

    def __init__(
        self,
        *,
        qa_executor: QAExecutor,
        results_root: str | Path,
        candidate_budget: int = max(DIAGNOSTIC_RETRIEVAL_K_VALUES),
        candidate_k_values: Sequence[int] = DIAGNOSTIC_RETRIEVAL_K_VALUES,
    ) -> None:
        """Initialize the benchmark runner."""
        if (
            isinstance(candidate_budget, bool)
            or not isinstance(candidate_budget, int)
            or not 1 <= candidate_budget <= 50
        ):
            raise ValueError("candidate_budget must be between 1 and 50")
        k_values = tuple(candidate_k_values)
        if (
            not k_values
            or len(k_values) != len(set(k_values))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value > candidate_budget
                for value in k_values
            )
        ):
            raise ValueError("candidate_k_values must be unique and within budget")
        self.qa_executor = qa_executor
        self.results_root = Path(results_root)
        self.candidate_budget = candidate_budget
        self.candidate_k_values = tuple(sorted(k_values))

    async def run(
        self,
        samples: Sequence[BenchmarkSample],
        *,
        tenant_id: str | None,
        retrieval_modes: Sequence[RetrievalMode] = ("vector", "hybrid"),
        run_metadata: dict[str, Any] | None = None,
        smoke_threshold: int = DEFAULT_SMOKE_THRESHOLD,
    ) -> BenchmarkRun:
        """Run the benchmark runner operation."""
        if not samples:
            raise ValueError("benchmark must contain at least one sample")
        if smoke_threshold <= 0:
            raise ValueError("smoke_threshold must be positive")
        modes = tuple(retrieval_modes)
        if not modes or any(mode not in SUPPORTED_RETRIEVAL_MODES for mode in modes):
            supported = ", ".join(sorted(SUPPORTED_RETRIEVAL_MODES))
            raise ValueError(f"retrieval_modes contains an unsupported mode; expected: {supported}")

        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        self.results_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.results_root, 0o700)
        run_dir = self.results_root / run_id
        run_dir.mkdir(parents=False, exist_ok=False, mode=0o700)
        os.chmod(run_dir, 0o700)
        run_started_at = _utc_now()
        responses_path = run_dir / "responses.jsonl"
        invalid_provenance_path = run_dir / "invalid_provenance.jsonl"
        candidate_stages_path = run_dir / "candidate-stages.jsonl"
        metadata_path = run_dir / "run_metadata.json"
        raw_metadata = dict(run_metadata or {})
        benchmark_sha256 = getattr(samples, "sha256", None)
        if not benchmark_sha256:
            benchmark_sha256 = self._samples_hash(samples)
        code_revision = _git_revision()
        run_identity = _run_identity(
            run_id=run_id,
            started_at=run_started_at,
            benchmark_sha256=benchmark_sha256,
            retrieval_modes=modes,
            metadata=raw_metadata,
            samples=samples,
            code_revision=code_revision,
        )

        records: list[dict[str, Any]] = []
        invalid_provenance: list[dict[str, Any]] = []
        candidate_stage_records: list[dict[str, Any]] = []
        diagnostic_variant = raw_metadata.get("diagnostic_variant", "observed")
        if not isinstance(diagnostic_variant, str) or not 0 < len(diagnostic_variant) <= 64:
            raise ValueError("diagnostic_variant must be a bounded string")
        for sample in samples:
            for mode in modes:
                started = time.perf_counter()
                cpu_started = time.process_time()
                candidate_recorder = CandidateStageRecorder(
                    candidate_budget=self.candidate_budget
                )
                try:
                    result = await self.qa_executor(
                        question=sample.question,
                        tenant_id=tenant_id,
                        user_id="",
                        retrieval_mode=mode,
                        evaluation_candidate_recorder=candidate_recorder,
                    )
                    contexts = [_context_record(context, rank) for rank, context in enumerate(result.contexts, start=1)]
                    retrieved_source_document_ids, unmapped_contexts = _provenance(contexts)
                    retrieved_identities = _retrieved_identity_fields(contexts)
                    status = "invalid_provenance" if unmapped_contexts else "succeeded"
                    trace = result.trace.to_dict() if result.trace is not None else None
                    replay = _bounded_result_artifacts(result, trace)
                    latency_ms = round((time.perf_counter() - started) * 1000, 3)
                    record = {
                        "run_id": run_id,
                        "evaluation_case_count": len(samples),
                        "benchmark_id": sample.id,
                        "retrieval_mode": mode,
                        "category": sample.category,
                        "expected_refusal": sample.expected_refusal,
                        "expected_response_status": getattr(
                            sample, "expected_response_status", None
                        ),
                        "expected_evidence_states": list(
                            getattr(sample, "expected_evidence_states", ())
                        ),
                        "expected_reason_codes": list(
                            getattr(sample, "expected_reason_codes", ())
                        ),
                        "expected_missing_information_fields": list(
                            getattr(sample, "expected_missing_information_fields", ())
                        ),
                        "expected_evidence_context_ids": list(
                            getattr(sample, "expected_evidence_context_ids", ())
                        ),
                        "expected_source_document_ids": list(
                            getattr(sample, "expected_source_document_ids", ())
                        ),
                        "expected_citation_context_ids": list(
                            getattr(
                                sample,
                                "expected_citation_context_ids",
                                sample.reference_context_ids,
                            )
                        ),
                        "expected_branch_availability": dict(
                            getattr(sample, "expected_branch_availability", {})
                        ),
                        "refused": replay["response_status"] not in {
                            "answered",
                            "partially_answered",
                        },
                        "question": sample.question,
                        # Formal evidence-gate cases bind their reviewed reference
                        # later, in the release RAGAS preparation step. Do not turn
                        # a successful QA result into a failed record merely because
                        # this runtime sample intentionally has no inline reference.
                        "reference": _sample_reference(sample),
                        "response": result.answer,
                        "trace": trace,
                        "contexts": contexts,
                        # Legacy field retained for readers written before identity
                        # namespaces were explicit. Its namespace is declared below.
                        "retrieved_context_ids": retrieved_source_document_ids,
                        "reference_context_ids": list(sample.reference_context_ids),
                        "retrieved_source_document_ids": retrieved_source_document_ids,
                        "reference_source_document_ids": _sample_source_document_ids(sample),
                        "reference_chunk_ids": list(
                            getattr(sample, "expected_evidence_context_ids", sample.reference_context_ids)
                        ),
                        **retrieved_identities,
                        "id_namespaces": {
                            "retrieved_context_ids": "source_document",
                            "reference_context_ids": "context_or_legacy",
                            "retrieved_source_document_ids": "source_document",
                            "reference_source_document_ids": "source_document",
                            "retrieved_chunk_ids": "context",
                            "reference_chunk_ids": "context",
                            "retrieved_content_sha256s": "content_sha256",
                        },
                        "status": status,
                        "latency_ms": latency_ms,
                        "resources": {
                            "latency_ms": latency_ms,
                            "cpu_ms": round(
                                max(0.0, time.process_time() - cpu_started) * 1000, 3
                            ),
                            "rss_peak_mb": _rss_peak_mb(),
                        },
                        "run_identity": run_identity,
                        "exception": None,
                        **replay,
                    }
                    record["stages"]["retrieval"]["candidates"] = _candidate_stages(
                        contexts
                    )
                    if unmapped_contexts:
                        invalid_provenance.append(
                            {
                                "run_id": run_id,
                                "benchmark_id": sample.id,
                                "retrieval_mode": mode,
                                "unmapped_contexts": unmapped_contexts,
                            }
                        )
                except Exception as error:
                    latency_ms = round((time.perf_counter() - started) * 1000, 3)
                    record = {
                        "run_id": run_id,
                        "evaluation_case_count": len(samples),
                        "benchmark_id": sample.id,
                        "retrieval_mode": mode,
                        "category": sample.category,
                        "expected_refusal": sample.expected_refusal,
                        "expected_response_status": getattr(
                            sample, "expected_response_status", None
                        ),
                        "expected_evidence_states": list(
                            getattr(sample, "expected_evidence_states", ())
                        ),
                        "expected_reason_codes": list(
                            getattr(sample, "expected_reason_codes", ())
                        ),
                        "expected_missing_information_fields": list(
                            getattr(sample, "expected_missing_information_fields", ())
                        ),
                        "expected_evidence_context_ids": list(
                            getattr(sample, "expected_evidence_context_ids", ())
                        ),
                        "expected_source_document_ids": list(
                            getattr(sample, "expected_source_document_ids", ())
                        ),
                        "expected_citation_context_ids": list(
                            getattr(
                                sample,
                                "expected_citation_context_ids",
                                sample.reference_context_ids,
                            )
                        ),
                        "expected_branch_availability": dict(
                            getattr(sample, "expected_branch_availability", {})
                        ),
                        "response_status": None,
                        "evidence_state": None,
                        "claims": [],
                        "citations": [],
                        "grounding_result": None,
                        "observed_missing_information_fields": [],
                        "observed_reason_codes": [],
                        "security_actions": [],
                        "stages": {
                            "retrieval": {
                                "branch_availability": {
                                    "dense": "unavailable",
                                    "bm25": "unavailable",
                                    "graph": "unavailable",
                                },
                                "candidates": [],
                            },
                            "qualification": {"status": "skipped"},
                            "generation": {"status": "skipped"},
                            "grounding": {"status": "skipped"},
                        },
                        "refused": False,
                        "question": sample.question,
                        "response": "",
                        "trace": None,
                        "contexts": [],
                        "retrieved_context_ids": [],
                        "reference_context_ids": list(sample.reference_context_ids),
                        "status": "failed",
                        "latency_ms": latency_ms,
                        "resources": {
                            "latency_ms": latency_ms,
                            "cpu_ms": round(
                                max(0.0, time.process_time() - cpu_started) * 1000, 3
                            ),
                            "rss_peak_mb": _rss_peak_mb(),
                        },
                        "run_identity": run_identity,
                        "exception": type(error).__name__,
                    }
                records.append(record)
                candidate_stage_records.extend(
                    {
                        "schema_version": CANDIDATE_STAGE_SCHEMA_VERSION,
                        "run_id": run_id,
                        "benchmark_id": sample.id,
                        "retrieval_mode": mode,
                        "variant": diagnostic_variant,
                        **stage,
                    }
                    for stage in candidate_recorder.snapshot()
                )

        _write_jsonl(responses_path, records)
        _write_jsonl(invalid_provenance_path, invalid_provenance)
        _write_jsonl(candidate_stages_path, candidate_stage_records)
        candidate_stages_sha256 = hashlib.sha256(
            candidate_stages_path.read_bytes()
        ).hexdigest()
        supplied_metadata = sanitize_for_observability(raw_metadata)
        if not isinstance(supplied_metadata, dict):
            supplied_metadata = {}
        # Generic observability sanitization treats model/scope/tokenizer keys as
        # sensitive. These explicitly named values are non-secret reproducibility
        # identifiers, so restore only bounded allowlisted fields—not arbitrary
        # caller metadata or credentials.
        scalar_fields = (
            "answer_model",
            "embedding_model",
            "evaluator_model",
            "evidence_class",
            "corpus_revision",
            "knowledge_revision",
            "prompt_id",
            "retrieval_config_id",
            "top_k",
            "cache_policy",
            "authorization_scope_fixture",
            "dataset_version",
            "dataset_manifest_sha256",
            "threshold_version",
            "candidate_budget_version",
            "reranker_mode",
            "reranker_version",
            "structured_answer_schema_version",
            "grounding_policy_version",
        )
        for field in scalar_fields:
            value = raw_metadata.get(field)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ) or (isinstance(value, str) and 0 < len(value) <= 200):
                supplied_metadata[field] = value
        bm25 = raw_metadata.get("bm25")
        if isinstance(bm25, dict):
            supplied_metadata["bm25"] = {
                field: bm25[field]
                for field in (
                    "schema_version",
                    "tokenizer_version",
                    "k1",
                    "b",
                    "index_generation",
                )
                if field in bm25
            }
        release_evidence = raw_metadata.get("release_evidence")
        if isinstance(release_evidence, dict):
            def bounded(fields: tuple[str, ...], value: Any) -> dict[str, Any]:
                return (
                    {field: value[field] for field in fields if field in value}
                    if isinstance(value, dict)
                    else {}
                )

            supplied_metadata["release_evidence"] = {
                "tenant_isolation": release_evidence.get("tenant_isolation"),
                "department_isolation": release_evidence.get("department_isolation"),
                "sparse_lifecycle": bounded(
                    ("add", "update", "delete", "rebuild"),
                    release_evidence.get("sparse_lifecycle"),
                ),
                "degradation": bounded(
                    ("unavailable_snapshot", "corrupt_snapshot"),
                    release_evidence.get("degradation"),
                ),
                "rollback": bounded(
                    ("configuration_only", "target_strategy"),
                    release_evidence.get("rollback"),
                ),
                "cost": bounded(
                    ("observed", "budget", "unit"),
                    release_evidence.get("cost"),
                ),
            }
        metadata = {
            **supplied_metadata,
            "run_id": run_id,
            "started_at": run_started_at,
            "benchmark_sha256": benchmark_sha256,
            "benchmark_source": Path(
                getattr(samples, "source_path", "")
            ).name,
            "retrieval_modes": list(modes),
            "tenant_id": tenant_id,
            "code_revision": code_revision,
            "run_identity": run_identity,
            "resource_summary": {
                "cpu_ms": round(
                    sum(record["resources"]["cpu_ms"] for record in records), 3
                ),
                "rss_peak_mb": max(
                    (record["resources"]["rss_peak_mb"] for record in records),
                    default=0.0,
                ),
            },
            "ragas_version": _ragas_version(),
            "smoke_threshold": smoke_threshold,
            "records": len(records),
            "candidate_stages": {
                "artifact": candidate_stages_path.name,
                "schema_version": CANDIDATE_STAGE_SCHEMA_VERSION,
                "k_values": list(self.candidate_k_values),
                "candidate_budget": self.candidate_budget,
                "records": len(candidate_stage_records),
                "sha256": candidate_stages_sha256,
            },
            "succeeded": sum(record["status"] == "succeeded" for record in records),
            "invalid_provenance": len(invalid_provenance),
            "failed": sum(record["status"] == "failed" for record in records),
        }
        _write_text_secure(
            metadata_path,
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
        )
        return BenchmarkRun(
            run_id=run_id,
            run_dir=run_dir,
            metadata_path=metadata_path,
            responses_path=responses_path,
            invalid_provenance_path=invalid_provenance_path,
            candidate_stages_path=candidate_stages_path,
            candidate_stages_sha256=candidate_stages_sha256,
            records=records,
        )

    @staticmethod
    def _samples_hash(samples: Sequence[BenchmarkSample]) -> str:
        """Compute a stable hash for the benchmark samples."""
        serialized = [
            {
                "id": sample.id,
                "question": sample.question,
                "reference": _sample_reference(sample),
                "required_doc_ids": getattr(
                    sample,
                    "required_doc_ids",
                    getattr(sample, "expected_source_document_ids", ()),
                ),
                "reference_context_ids": sample.reference_context_ids,
                "category": sample.category,
                "expected_refusal": sample.expected_refusal,
            }
            for sample in samples
        ]
        return __import__("hashlib").sha256(
            json.dumps(serialized, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


__all__ = [
    "BenchmarkRun",
    "BenchmarkRunner",
    "QAExecutor",
    "RetrievalMode",
    "SUPPORTED_RETRIEVAL_MODES",
]
