"""ATDD for durable company release workflow completion and immutable evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select

from evaluation.release.workflow import (
    PostgreSQLReleaseWorkflowRepository,
    ReleaseWorkflowError,
)
from infrastructure.postgres.database import DatabaseService
from infrastructure.postgres.models import metadata


@pytest.fixture()
def repository() -> PostgreSQLReleaseWorkflowRepository:
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    return PostgreSQLReleaseWorkflowRepository(DatabaseService.from_engine(engine))


@pytest.fixture(autouse=True)
def configured_ragas_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep isolated Judge configuration deterministic for workflow unit tests."""
    monkeypatch.setenv("EVAL_OPENAI_API_KEY", "judge-test-key")
    monkeypatch.setenv("EVAL_OPENAI_BASE_URL", "https://ark.example/v3")
    monkeypatch.setenv("EVAL_OPENAI_MODEL", "deepseek-v4-flash")


def test_runtime_evidence_lineage_distinguishes_exact_hash_from_document_title() -> None:
    from evaluation.release.runtime import ReleaseWorkflowService

    sample = SimpleNamespace(
        id="case-1",
        category="fully_answerable",
        expected_source_document_ids=("reviewed-doc",),
        expected_evidence_context_ids=("reviewed-doc#chunk-1",),
    )
    reviewed_hash = hashlib.sha256(b"reviewed chunk").hexdigest()
    record = {
        "benchmark_id": "case-1",
        "retrieved_source_document_ids": ["runtime-doc"],
        "contexts": [{
            "content": "drifted chunk",
            "metadata": {"context_id": "reviewed-doc#chunk-1"},
        }],
    }

    drift = ReleaseWorkflowService._runtime_evidence_lineage_summary(
        records=[record],
        dataset=[sample],
        reviewed_content_hashes={"reviewed-doc#chunk-1": reviewed_hash},
    )
    matched = ReleaseWorkflowService._runtime_evidence_lineage_summary(
        records=[{
            **record,
            "contexts": [{
                "content": "reviewed chunk",
                "metadata": {"context_id": "reviewed-doc#chunk-1"},
            }],
        }],
        dataset=[sample],
        reviewed_content_hashes={"reviewed-doc#chunk-1": reviewed_hash},
    )

    assert drift == {
        "checked": 1,
        "resolved": 0,
        "document_resolved": 0,
        "unresolved": 1,
        "unresolved_by_category": {"fully_answerable": 1},
    }
    assert matched["resolved"] == 1


def test_diagnostic_report_is_flattened_for_the_admin_attempt_api() -> None:
    from evaluation.release.runtime import _bounded_diagnostic_summary

    summary = _bounded_diagnostic_summary({
        "diagnostic": {
            "run_plan": {"category_policy": {
                "category_policy_version": "policy-v1",
                "category_policy_sha256": "a" * 64,
            }},
            "routing": {"by_category": {
                "fully_answerable": {
                    "count": 2,
                    "confusion": {"answered->answered": 1, "answered->insufficient_evidence": 1},
                    "failing_case_ids": ["case-2"],
                },
            }},
            "retrieval": {"stages": {
                "dense": {
                    "exact_evidence": {"scored": 2, "mean_recall_at_k": {"5": 0.5}, "mean_mrr": 0.75},
                    "source_document": {"scored": 2, "mean_recall_at_k": {"5": 1.0}, "mean_mrr": 1.0},
                },
            }},
            "answer": {"metrics": {
                "response_route_correctness": {
                    "coverage": {"scored": 2, "not_applicable": 0, "failed": 0, "unsupported": 0},
                    "micro_mean": 0.5,
                    "category_macro_mean": 0.5,
                },
            }},
            "safety": {
                "metrics": {},
                "hard_gates": {"invalid_provenance": {"passed": False}},
            },
        },
    })

    assert summary is not None
    assert summary["categories"]["fully_answerable"] == {"count": 2, "failed": 1}
    assert summary["stages"]["dense"] == {
        "scored": 2,
        "exact_recall_at_5": 0.5,
        "source_recall_at_5": 1.0,
        "mrr": 0.75,
    }
    assert summary["metric_coverage"]["response_route_correctness"]["scored"] == 2
    assert summary["hard_gates"] == ["invalid_provenance"]
    assert summary["case_ids"] == ["case-2"]


def test_runtime_evidence_lineage_fails_before_judge_with_bounded_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.release import runtime as release_workflow_runtime
    from evaluation.release.runtime import RagasCoverageError, ReleaseWorkflowService

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"context_catalog_file": "contexts.jsonl"}), encoding="utf-8")
    reviewed_hash = hashlib.sha256(b"reviewed chunk").hexdigest()
    (tmp_path / "contexts.jsonl").write_text(
        json.dumps({"context_id": "doc#chunk-1", "content_sha256": reviewed_hash}) + "\n",
        encoding="utf-8",
    )
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        json.dumps({
            "benchmark_id": "case-1",
            "contexts": [{
                "content": "drifted chunk",
                "metadata": {"context_id": "doc#chunk-1"},
            }],
        }) + "\n",
        encoding="utf-8",
    )
    dataset = [SimpleNamespace(
        id="case-1", category="fully_answerable",
        expected_source_document_ids=("doc",),
        expected_evidence_context_ids=("doc#chunk-1",),
    )]
    monkeypatch.setattr(release_workflow_runtime, "load_evidence_gate_benchmark", lambda *_args, **_kwargs: dataset)
    service = object.__new__(ReleaseWorkflowService)

    with pytest.raises(RagasCoverageError, match="formal_runtime_evidence_mismatch") as captured:
        service._validate_runtime_evidence_lineage(
            responses_path=responses,
            manifest_path=manifest,
            expected_manifest_sha256="a" * 64,
        )

    assert captured.value.metrics["formal_evidence_lineage"]["unresolved"] == 1


def test_preflight_rejects_formal_bundle_missing_reviewed_reference_answer(tmp_path: Path) -> None:
    from evaluation.release.runtime import ReleaseWorkflowService

    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "case-1", "category": "fully_answerable"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"cases_file": "cases.jsonl"}), encoding="utf-8")

    with pytest.raises(ReleaseWorkflowError, match="ragas_reference_missing"):
        ReleaseWorkflowService._validate_formal_reference_answers(
            manifest_path=manifest_path,
            formal_case_count=1,
        )


def test_runtime_evidence_lineage_accepts_valid_empty_evidence_routes() -> None:
    from evaluation.release.runtime import ReleaseWorkflowService

    samples = [
        SimpleNamespace(id="auth", category="authorization_filtered"),
        SimpleNamespace(id="outage", category="all_branches_unavailable"),
        SimpleNamespace(id="absent", category="completely_unanswerable"),
        SimpleNamespace(id="injection", category="prompt_injection"),
    ]
    records = [
        {
            "benchmark_id": sample.id,
            "contexts": [],
            "retrieved_source_document_ids": [],
            "retrieved_chunk_ids": [],
            "retrieved_content_sha256s": [],
        }
        for sample in samples
    ]

    summary = ReleaseWorkflowService._runtime_evidence_lineage_summary(
        records=records,
        dataset=samples,
        reviewed_content_hashes={},
    )

    assert summary["checked"] == 4
    assert summary["resolved"] == 4
    assert summary["unresolved"] == 0


def test_runtime_evidence_lineage_treats_rejected_candidates_as_quality_outcome() -> None:
    from evaluation.release.runtime import ReleaseWorkflowService

    summary = ReleaseWorkflowService._runtime_evidence_lineage_summary(
        records=[{"benchmark_id": "auth", "retrieved_source_document_ids": ["restricted-doc"]}],
        dataset=[SimpleNamespace(id="auth", category="authorization_filtered")],
        reviewed_content_hashes={},
    )

    assert summary["resolved"] == 1
    assert summary["unresolved"] == 0
    assert summary["unresolved_by_category"] == {}


def test_runtime_evidence_lineage_allows_gold_recall_miss_for_scoring() -> None:
    from evaluation.release.runtime import ReleaseWorkflowService

    summary = ReleaseWorkflowService._runtime_evidence_lineage_summary(
        records=[{
            "benchmark_id": "case-1",
            "contexts": [{
                "content": "a different valid candidate",
                "metadata": {"context_id": "other-doc#chunk-2"},
            }],
        }],
        dataset=[SimpleNamespace(
            id="case-1",
            category="fully_answerable",
            expected_source_document_ids=("reviewed-doc",),
            expected_evidence_context_ids=("reviewed-doc#chunk-1",),
        )],
        reviewed_content_hashes={
            "reviewed-doc#chunk-1": hashlib.sha256(b"reviewed chunk").hexdigest(),
        },
    )

    assert summary["resolved"] == 1
    assert summary["document_resolved"] == 0
    assert summary["unresolved"] == 0


def test_runtime_evidence_lineage_rejects_invalid_approved_review_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.release import runtime as release_workflow_runtime
    from evaluation.release.runtime import RagasCoverageError, ReleaseWorkflowService

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"review_status": "approved", "context_catalog_file": "contexts.jsonl"}),
        encoding="utf-8",
    )
    (tmp_path / "contexts.jsonl").write_text("", encoding="utf-8")
    responses = tmp_path / "responses.jsonl"
    responses.write_text('{"benchmark_id":"bad-case"}\n', encoding="utf-8")
    dataset = [SimpleNamespace(
        id="bad-case",
        category="fully_answerable",
        expected_response_status="insufficient_evidence",
        expected_evidence_states=("insufficient_evidence",),
        expected_reason_codes=("zero_results",),
        expected_source_document_ids=(),
        expected_evidence_context_ids=(),
        expected_citation_context_ids=(),
        expected_missing_information_fields=(),
        expected_branch_availability={},
        required_fixture="standard",
    )]
    monkeypatch.setattr(
        release_workflow_runtime,
        "load_evidence_gate_benchmark",
        lambda *_args, **_kwargs: dataset,
    )
    service = object.__new__(ReleaseWorkflowService)

    with pytest.raises(RagasCoverageError, match="formal_reviewed_contract_invalid") as captured:
        service._validate_runtime_evidence_lineage(
            responses_path=responses,
            manifest_path=manifest,
            expected_manifest_sha256="a" * 64,
        )

    metrics = captured.value.metrics["formal_reviewed_contract"]
    assert metrics["invalid_by_category"] == {"fully_answerable": 1}


def _create(repository: PostgreSQLReleaseWorkflowRepository) -> dict:
    return repository.create(
        company_namespace="company-internal",
        dataset_id="evidence-gates-v1",
        version="20260814T010203Z-r42",
        manifest_sha256="a" * 64,
        initiated_by="company-admin-a",
    )


def test_repository_keeps_single_active_workflow_and_immutable_attempt_history(repository):
    first = _create(repository)
    duplicate = _create(repository)
    assert duplicate["workflow_id"] == first["workflow_id"]
    assert duplicate["status"] == "queued"

    first_attempt = repository.append_attempt(first["workflow_id"], stage="preflight", status="failed", reason_code="dependency_unavailable")
    retry_attempt = repository.append_attempt(first["workflow_id"], stage="preflight", status="succeeded")
    assert first_attempt["attempt_number"] == 1
    assert retry_attempt["attempt_number"] == 2
    assert [item["status"] for item in repository.attempts(first["workflow_id"])] == ["failed", "succeeded"]

    with pytest.raises(ReleaseWorkflowError, match="workflow_revision_conflict"):
        repository.transition(first["workflow_id"], expected_revision=99, status="running", stage="preflight")



def _write_baseline_template(root: Path) -> None:
    (root / "baseline-template.json").write_text(
        '{"dataset":{"manifest_sha256":"placeholder"}}\n',
        encoding="utf-8",
    )


def _release_quality_report(*, gate_mode: str) -> dict[str, object]:
    metrics: dict[str, object] = {
        "refusal_precision": 1.0,
        "refusal_recall": 1.0,
        "no_answer_hallucination_rate": 0.0,
        "answerable_false_refusal_rate": 0.0,
        "partial_answer_recognition_rate": 1.0,
        "conflict_recognition_rate": 1.0,
        "claim_citation_coverage": 1.0,
        "citation_correctness": 1.0,
        "groundedness_pass_rate": 1.0,
    }
    if gate_mode == "off":
        metrics["claim_citation_coverage"] = None
        metrics["citation_correctness"] = None
    return {
        "counts": {"total": 100, "succeeded": 100, "failed": 0, "invalid_provenance": 0},
        "run_classification": "baseline",
        "evidence_gate": {
            "counts": {"completed": 100},
            "metrics": metrics,
            "run_identity": {"run_id": f"run-{gate_mode}", "dataset_version": "v1"},
        }
    }


class _FrozenWorkspace:
    def __init__(self, root: Path, version: str = "20260814T010203Z-r42") -> None:
        self.root = root
        self.version = version
        self.draft_id = "ccd_" + "a" * 32
        self.snapshot_sha256 = "b" * 64
        self.bundle = root / "versions" / version
        self.bundle.mkdir(parents=True)
        (self.bundle / "manifest.json").write_text('{"dataset_version":"v"}\n', encoding="utf-8")
        (self.bundle / "fixture-profile.json").write_text('{"status":"ready","fixtures":{}}\n', encoding="utf-8")

    def _paths(self, _company: str, _dataset: str):
        return {"versions": self.root / "versions"}

    def _version_root(self, _paths, version: str):
        if version != self.version:
            raise RuntimeError("version_not_found")
        return self.bundle

    def get_dataset(self, _company: str, _dataset: str):
        return {
            "source_type": "current_corpus",
            "current_corpus_draft_id": self.draft_id,
            "current_corpus_snapshot_sha256": self.snapshot_sha256,
        }

    def get_version(self, _company: str, _dataset: str, version: str):
        from hashlib import sha256

        if version != self.version:
            raise RuntimeError("version_not_found")
        return {
            "version": version,
            "manifest_sha256": sha256((self.bundle / "manifest.json").read_bytes()).hexdigest(),
            "current_corpus_draft_id": self.draft_id,
            "current_corpus_snapshot_sha256": self.snapshot_sha256,
        }


class _SuccessfulFixtureValidation:
    """Runtime-alignment double; new workflows do not create Fixture lineage."""

    def __init__(self, run_id: str = "fvr_" + "c" * 32) -> None:
        self.run_id = run_id
        self.manifest_sha256: str | None = None

    def latest_success(self, **_kwargs):
        raise AssertionError("new release starts must not query a 12-case Fixture run")

    def get(self, **kwargs):
        assert kwargs["run_id"] == self.run_id
        return {
            "run_id": self.run_id,
            "status": "succeeded",
            "manifest_sha256": self.manifest_sha256,
        }

    def assert_runtime_corpus_aligned(self, **kwargs):
        self.manifest_sha256 = kwargs["manifest_sha256"]


def _start_release(service, workspace: _FrozenWorkspace, *, dataset_id: str = "evidence-gates-v1", evaluation_case_count: int = 100) -> dict:
    return service.start(
        company_namespace="company",
        dataset_id=dataset_id,
        version=workspace.version,
        actor_id="company-admin-a",
        evaluation_dataset_id="evidence-gates-v1",
        evaluation_version=workspace.version,
        evaluation_manifest_sha256=workspace.get_version("company", "evidence-gates-v1", workspace.version)["manifest_sha256"],
        evaluation_case_count=evaluation_case_count,
    )


def _stub_ragas(monkeypatch):
    from evaluation.release.runtime import _FORMAL_EVALUATION_METRICS

    monkeypatch.setattr(
        "evaluation.release.runtime.validate_ragas_configuration",
        lambda: {"model": "test-judge", "source": "test"},
    )

    async def run_ragas(args):
        responses = Path(args.responses_jsonl)
        output = responses.with_name("ragas_scores.jsonl")
        output.write_text('{"metric":"faithfulness","status":"scored","score":0.9}\n', encoding="utf-8")
        output.with_suffix(".summary.json").write_text(
            json.dumps({"metrics": {
                metric: {
                    "total": 100,
                    "scored": 100,
                    "not_applicable": 0,
                    "skipped": 0,
                    "failed": 0,
                    "mean": 1.0 if metric == "evidence_gate_contract" else 0.9,
                }
                for metric in _FORMAL_EVALUATION_METRICS
            }}),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("evaluation.release.runtime.run_ragas_benchmark", run_ragas)


def test_worker_runs_preflight_baseline_shadow_and_calibration_without_subprocess(repository, tmp_path, monkeypatch):
    """Worker invokes Python evaluation APIs and only enters approval after calibration."""
    from evaluation.release.runtime import ReleaseWorkflowService
    workspace = _FrozenWorkspace(tmp_path / "workspace")
    service = ReleaseWorkflowService(
        repository, workspace, source_bundle=tmp_path, output_root=tmp_path / "results",
        env_file=tmp_path / "env", fixture_validation_service=_SuccessfulFixtureValidation(),
    )
    work = _start_release(service, workspace)
    observed: list[str] = []

    _write_baseline_template(tmp_path)

    async def execute(**kwargs):
        gate_mode = kwargs["gate_modes"][0]
        observed.append(gate_mode)
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "responses.jsonl").write_text('{"benchmark_id":"case-1"}\n', encoding="utf-8")
        (root / "quality-report.json").write_text(
            json.dumps(_release_quality_report(gate_mode=gate_mode)),
            encoding="utf-8",
        )
        return {"status": "measured_pending_calibration"}

    monkeypatch.setattr("evaluation.release.runtime.execute_evidence_gate_benchmark", execute)
    _stub_ragas(monkeypatch)
    result = service.run(work["workflow_id"], company_namespace="company")
    assert result["status"] == "pending_approval"
    assert result["calibration_version"] == f"cal-{workspace.version}"
    assert observed == ["off", "shadow"]
    baseline_snapshot = json.loads(
        (tmp_path / "results" / work["workflow_id"] / "baseline-template.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline_snapshot["dataset"]["manifest_sha256"] == work["manifest_sha256"]
    assert baseline_snapshot["dataset"]["version"] == workspace.version
    attempts = repository.attempts(work["workflow_id"])
    assert [item["stage"] for item in attempts] == ["preflight", "baseline", "shadow", "calibration"]
    assert attempts[-1]["metrics"]["calibration_version"] == f"cal-{workspace.version}"
    # Duplicate delivery is idempotent once approval is pending.
    assert service.run(work["workflow_id"], company_namespace="company")["revision"] == result["revision"]


def test_smoke_workflow_runs_fail_closed_acceptance_after_shadow(repository, tmp_path, monkeypatch):
    from evaluation.release.runtime import ReleaseWorkflowService

    workspace = _FrozenWorkspace(tmp_path / "workspace")
    service = ReleaseWorkflowService(
        repository, workspace, source_bundle=tmp_path, output_root=tmp_path / "results",
        env_file=tmp_path / "env", fixture_validation_service=_SuccessfulFixtureValidation(),
    )
    work = _start_release(service, workspace, evaluation_case_count=12)
    _write_baseline_template(tmp_path)

    observed: list[tuple[str, str]] = []

    def measurement(*, work, stage, gate_mode, **_kwargs):
        observed.append((stage, gate_mode))
        service.repository.start_attempt(work["workflow_id"], stage=stage)
        service.repository.finish_attempt(
            work["workflow_id"],
            stage=stage,
            status="succeeded",
            metrics={
                "evaluation_case_count": work["evaluation_case_count"],
                "gate_mode": gate_mode,
                "quality_report_sha256": f"{stage}-report",
                "run_id": f"{stage}-run",
                "evidence_gate_contract_coverage": {
                    "total": 12,
                    "scored": 12,
                    "failed": 0,
                    "not_applicable": 0,
                    "skipped": 0,
                    "mean": 1.0,
                    "mismatch_by_category": {},
                },
                "ragas_coverage": {
                    "factual_correctness": {
                        "total": 12,
                        "scored": 3,
                        "failed": 0,
                        "not_applicable": 9,
                        "skipped": 0,
                        "mean": 0.8,
                        "not_applicable_reasons": {"audited_no_answer_route": 9},
                    },
                },
                "diagnostic_summary": {"hard_gates": []},
            },
        )

    monkeypatch.setattr(service, "_run_measurement", measurement)
    result = service.run(work["workflow_id"], company_namespace="company")
    assert result["status"] == "completed"
    assert result["calibration_version"] == "smoke-acceptance-v1"
    assert observed == [("baseline", "off"), ("shadow", "shadow")]
    attempts = repository.attempts(work["workflow_id"])
    assert attempts[-1]["stage"] == "calibration"
    assert attempts[-1]["status"] == "succeeded"
    assert attempts[-1]["metrics"]["acceptance_version"] == "smoke-acceptance-v1"
    assert set(attempts[-1]["metrics"]["source_identities"]) == {"baseline", "shadow"}


def test_smoke_workflow_fails_when_contract_or_reference_coverage_is_incomplete(
    repository, tmp_path, monkeypatch,
):
    from evaluation.release.runtime import ReleaseWorkflowService

    workspace = _FrozenWorkspace(tmp_path / "workspace")
    service = ReleaseWorkflowService(
        repository, workspace, source_bundle=tmp_path, output_root=tmp_path / "results",
        env_file=tmp_path / "env", fixture_validation_service=_SuccessfulFixtureValidation(),
    )
    work = _start_release(service, workspace, evaluation_case_count=12)
    _write_baseline_template(tmp_path)

    def measurement(*, work, stage, gate_mode, **_kwargs):
        service.repository.start_attempt(work["workflow_id"], stage=stage)
        service.repository.finish_attempt(
            work["workflow_id"], stage=stage, status="succeeded", metrics={
                "evaluation_case_count": 12,
                "gate_mode": gate_mode,
                "quality_report_sha256": f"{stage}-report",
                "run_id": f"{stage}-run",
                "evidence_gate_contract_coverage": {
                    "total": 12, "scored": 12, "failed": 0,
                    "not_applicable": 0, "skipped": 0, "mean": 11 / 12,
                    "mismatch_by_category": {"partially_answerable": 1},
                },
                "ragas_coverage": {
                    "factual_correctness": {
                        "total": 12, "scored": 0, "failed": 0,
                        "not_applicable": 12, "skipped": 0, "mean": None,
                        "not_applicable_reasons": {"smoke_reference_not_required": 3},
                    },
                },
                "diagnostic_summary": {"hard_gates": []},
            },
        )

    monkeypatch.setattr(service, "_run_measurement", measurement)
    result = service.run(work["workflow_id"], company_namespace="company")

    assert result["status"] == "failed"
    assert result["stage"] == "calibration"
    assert result["reason_code"] == "smoke_acceptance_contract_mismatch"
    attempts = repository.attempts(work["workflow_id"])
    assert attempts[-1]["stage"] == "calibration"
    assert attempts[-1]["status"] == "failed"


def test_retry_legacy_smoke_calibration_failure_rewinds_to_baseline(repository, tmp_path):
    from evaluation.release.runtime import ReleaseWorkflowService

    workspace = _FrozenWorkspace(tmp_path / "workspace")
    service = ReleaseWorkflowService(
        repository, workspace, source_bundle=tmp_path, output_root=tmp_path / "results",
        env_file=tmp_path / "env", fixture_validation_service=_SuccessfulFixtureValidation(),
    )
    work = _start_release(service, workspace, evaluation_case_count=100)
    repository.transition(
        work["workflow_id"], expected_revision=1, status="running", stage="calibration",
    )
    failed = repository.transition(
        work["workflow_id"], expected_revision=2, status="failed", stage="calibration",
        reason_code="calibration_metrics_incomplete",
    )
    # Simulate an old row whose smoke suite was identified by reusing the corpus dataset.
    with repository.database.session() as session:
        from sqlalchemy import update
        from infrastructure.postgres.models import evidence_release_workflows
        session.execute(update(evidence_release_workflows).where(
            evidence_release_workflows.c.id == work["workflow_id"]
        ).values(evaluation_dataset_id=work["dataset_id"], evaluation_case_count=None))
    retried = service.retry(work["workflow_id"], expected_revision=failed["revision"])
    assert retried["stage"] == "baseline"
    assert retried["status"] == "queued"


def test_worker_fails_closed_for_manifest_mismatch_and_incomplete_calibration(repository, tmp_path, monkeypatch):
    from evaluation.release.runtime import ReleaseWorkflowService
    workspace = _FrozenWorkspace(tmp_path / "workspace", version="20260814T010203Z-r43")
    service = ReleaseWorkflowService(
        repository, workspace, source_bundle=tmp_path, output_root=tmp_path / "results",
        env_file=tmp_path / "env", fixture_validation_service=_SuccessfulFixtureValidation(),
    )
    work = _start_release(service, workspace)
    (workspace.bundle / "manifest.json").write_text('{"tampered":true}\n', encoding="utf-8")
    failed = service.run(work["workflow_id"], company_namespace="company")
    assert failed["status"] == "failed"
    assert failed["reason_code"] == "manifest_hash_mismatch"

    workspace = _FrozenWorkspace(tmp_path / "workspace-2", version="20260814T010203Z-r44")
    service = ReleaseWorkflowService(
        repository, workspace, source_bundle=tmp_path, output_root=tmp_path / "results-2",
        env_file=tmp_path / "env", fixture_validation_service=_SuccessfulFixtureValidation(),
    )
    work = _start_release(service, workspace, dataset_id="evidence-gates-v2")
    _write_baseline_template(tmp_path)

    async def execute(**kwargs):
        root = Path(kwargs["output_root"]); root.mkdir(parents=True, exist_ok=True)
        (root / "responses.jsonl").write_text('{"benchmark_id":"case-1"}\n', encoding="utf-8")
        (root / "quality-report.json").write_text(
            json.dumps(_release_quality_report(gate_mode=kwargs["gate_modes"][0])),
            encoding="utf-8",
        )
        return {"status": "measured_pending_calibration"}

    monkeypatch.setattr("evaluation.release.runtime.execute_evidence_gate_benchmark", execute)
    monkeypatch.setattr(
        "evaluation.release.runtime.build_calibration_draft",
        lambda *_args, **_kwargs: {"status": "blocked_missing_measurements"},
    )
    _stub_ragas(monkeypatch)
    failed = service.run(work["workflow_id"], company_namespace="company")
    assert failed["status"] == "failed"
    assert failed["reason_code"] == "calibration_metrics_incomplete"


def test_retry_resumes_failed_stage_as_a_new_attempt_without_overwriting_prior_evidence(
    repository, tmp_path, monkeypatch,
):
    """A retry resumes from the failed stage and writes a distinct attempt directory."""
    from evaluation.release.runtime import ReleaseWorkflowService

    workspace = _FrozenWorkspace(tmp_path / "workspace", version="20260814T010203Z-r45")
    _write_baseline_template(tmp_path)
    service = ReleaseWorkflowService(
        repository,
        workspace,
        source_bundle=tmp_path,
        output_root=tmp_path / "results",
        env_file=tmp_path / "env",
        fixture_validation_service=_SuccessfulFixtureValidation(),
    )
    work = _start_release(service, workspace)
    observed: list[str] = []

    async def execute(**kwargs):
        mode = kwargs["gate_modes"][0]
        observed.append(mode)
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "responses.jsonl").write_text('{"benchmark_id":"case-1"}\n', encoding="utf-8")
        if mode == "shadow" and observed.count("shadow") == 1:
            return {"status": "blocked_service_probe_failed", "reason_codes": ["service_probe_failed"]}
        (root / "quality-report.json").write_text(
            json.dumps(_release_quality_report(gate_mode=mode)),
            encoding="utf-8",
        )
        return {"status": "measured_pending_calibration"}

    def calibration(_reports, *, output_dir, calibration_version, source_evidence=None):
        assert source_evidence is not None
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "calibration-record.json").write_text("{}", encoding="utf-8")
        return {"status": "completed", "calibration_version": calibration_version}

    monkeypatch.setattr("evaluation.release.runtime.execute_evidence_gate_benchmark", execute)
    monkeypatch.setattr("evaluation.release.runtime.build_calibration_draft", calibration)
    _stub_ragas(monkeypatch)

    failed = service.run(work["workflow_id"], company_namespace="company")
    assert failed["status"] == "failed"
    assert failed["stage"] == "shadow"
    assert observed == ["off", "shadow"]
    assert (tmp_path / "results" / work["workflow_id"] / "baseline-1").is_dir()
    assert (tmp_path / "results" / work["workflow_id"] / "shadow-1").is_dir()

    queued = service.retry(failed["workflow_id"], expected_revision=failed["revision"])
    completed = service.run(
        queued["workflow_id"],
        company_namespace="company"
    )
    assert completed["status"] == "pending_approval"
    assert observed == ["off", "shadow", "shadow"]
    assert (tmp_path / "results" / work["workflow_id"] / "shadow-2").is_dir()
    attempts = repository.attempts(work["workflow_id"])
    assert [(item["stage"], item["attempt_number"], item["status"]) for item in attempts] == [
        ("preflight", 1, "succeeded"),
        ("baseline", 1, "succeeded"),
        ("shadow", 1, "failed"),
        ("shadow", 2, "succeeded"),
        ("calibration", 1, "succeeded"),
    ]


def test_stale_running_attempt_is_closed_then_retried_as_new_evidence(repository):
    """Recovery never overwrites an orphaned running stage attempt."""
    work = _create(repository)
    running = repository.transition(
        work["workflow_id"],
        expected_revision=work["revision"],
        status="running",
        stage="baseline",
    )
    repository.start_attempt(work["workflow_id"], stage="baseline")

    recovered = repository.recover_orphaned(
        work["workflow_id"],
        expected_revision=running["revision"],
        minimum_running_seconds=0,
    )
    assert recovered["status"] == "queued"
    assert recovered["stage"] == "baseline"
    assert recovered["reason_code"] == "orphaned_worker_attempt"
    attempts = repository.attempts(work["workflow_id"])
    assert len(attempts) == 1
    assert attempts[0]["stage"] == "baseline"
    assert attempts[0]["attempt_number"] == 1
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["reason_code"] == "orphaned_worker_attempt"
    assert attempts[0]["finished_at"]


def test_release_response_uses_company_scoped_initiator_display_name(monkeypatch) -> None:
    """The API presentation layer never exposes an internal initiator ID."""
    from api.routers.evaluation_pkg import release_workflows as router
    from types import SimpleNamespace

    class _Users:
        def __init__(self, record): self.record = record
        def list_users(self, **kwargs):
            assert kwargs["org_id"] == "company"
            return ([{**self.record, "user_id": "user-internal-id"}] if self.record else [])

    work = {
        "workflow_id": "erw-label", "dataset_id": "evidence-gates-v1", "version": "v1",
        "manifest_sha256": "a" * 64, "status": "failed", "stage": "baseline",
        "revision": 1, "initiated_by": "user-internal-id",
    }
    viewer = SimpleNamespace(org_id="company")
    monkeypatch.setattr("api.routers.evaluation_pkg._common.USER_SERVICE", _Users({"display_name": "王小明", "username": "wang"}))
    assert router._release_workflow_response(viewer, work).initiated_by_display_name == "王小明"

    monkeypatch.setattr("api.routers.evaluation_pkg._common.USER_SERVICE", _Users({"display_name": "", "username": "wang"}))
    assert router._release_workflow_response(viewer, work).initiated_by_display_name == "wang"

    monkeypatch.setattr("api.routers.evaluation_pkg._common.USER_SERVICE", _Users(None))
    assert router._release_workflow_response(viewer, work).initiated_by_display_name == "历史用户"
    assert router._release_workflow_response(viewer, work).initiated_by == "user-internal-id"


def test_release_start_does_not_require_fixture_validation(repository, tmp_path: Path) -> None:
    """The historical 12-case Fixture run is not a new-release prerequisite."""
    from evaluation.release.runtime import ReleaseWorkflowService

    workspace = _FrozenWorkspace(tmp_path / "workspace-fixture-gate")

    class _Validation:
        def latest_success(self, **_kwargs):
            raise AssertionError("release start must not query Fixture history")

    service = ReleaseWorkflowService(
        repository,
        workspace,
        source_bundle=tmp_path,
        output_root=tmp_path / "results-fixture-gate",
        env_file=None,
        fixture_validation_service=_Validation(),
    )
    workflow = _start_release(service, workspace)
    assert workflow["status"] == "queued"
    assert workflow["fixture_validation_run_id"] is None


def test_release_start_persists_current_corpus_and_fixture_lineage(repository, tmp_path: Path) -> None:
    from evaluation.release.runtime import ReleaseWorkflowService

    workspace = _FrozenWorkspace(tmp_path / "workspace-fixture-success")
    validation = _SuccessfulFixtureValidation()
    service = ReleaseWorkflowService(
        repository,
        workspace,
        source_bundle=tmp_path,
        output_root=tmp_path / "results-fixture-success",
        env_file=None,
        fixture_validation_service=validation,
    )

    workflow = _start_release(service, workspace)

    assert workflow["status"] == "queued"
    assert workflow["current_corpus_draft_id"] is None
    assert workflow["current_corpus_snapshot_sha256"] is None
    assert workflow["fixture_validation_run_id"] is None


def test_release_worker_fails_closed_when_the_current_corpus_snapshot_changes(repository, tmp_path: Path) -> None:
    """A queued release never evaluates a corpus different from its reviewed draft."""
    from evaluation.release.runtime import ReleaseWorkflowService

    workspace = _FrozenWorkspace(tmp_path / "workspace-stale-corpus")
    service = ReleaseWorkflowService(
        repository,
        workspace,
        source_bundle=tmp_path,
        output_root=tmp_path / "results-stale-corpus",
        env_file=None,
        fixture_validation_service=_SuccessfulFixtureValidation(),
    )
    workflow = _start_release(service, workspace)
    (workspace.bundle / "manifest.json").write_text('{"dataset_version":"changed"}\n', encoding="utf-8")

    failed = service.run(
        workflow["workflow_id"],
        company_namespace="company"
    )

    assert failed["status"] == "failed"
    assert failed["reason_code"] == "manifest_hash_mismatch"
    assert repository.attempts(workflow["workflow_id"]) == []


def test_release_start_uses_frozen_current_corpus_and_rejects_mismatched_legacy_lineage(repository, tmp_path: Path) -> None:
    """Service-level callers launch from the frozen version; legacy lineage is only checked when supplied."""
    from evaluation.release.runtime import ReleaseWorkflowService

    workspace = _FrozenWorkspace(tmp_path / "workspace-lineage")
    service = ReleaseWorkflowService(
        repository,
        workspace,
        source_bundle=tmp_path,
        output_root=tmp_path / "results-lineage",
        env_file=None,
        fixture_validation_service=_SuccessfulFixtureValidation(),
    )

    started = service.start(
        company_namespace="company",
        dataset_id="evidence-gates-v1",
        version=workspace.version,
        actor_id="company-admin-a",
        evaluation_dataset_id="evidence-gates-v1",
        evaluation_version=workspace.version,
        evaluation_manifest_sha256=workspace.get_version("company", "evidence-gates-v1", workspace.version)["manifest_sha256"],
        evaluation_case_count=100,
    )
    assert started["current_corpus_draft_id"] is None
    with pytest.raises(ReleaseWorkflowError, match="current_corpus_snapshot_mismatch"):
        service.start(
            company_namespace="company",
            dataset_id="evidence-gates-v1",
            version=workspace.version,
            actor_id="company-admin-a",
            current_corpus_draft_id=workspace.draft_id,
            current_corpus_snapshot_sha256="d" * 64,
            evaluation_dataset_id="evidence-gates-v1",
            evaluation_version=workspace.version,
            evaluation_manifest_sha256=workspace.get_version("company", "evidence-gates-v1", workspace.version)["manifest_sha256"],
            evaluation_case_count=100,
        )


def test_repository_lists_bounded_company_scoped_history_and_persists_formal_suite(repository):
    """Historical workflow discovery cannot cross company boundaries."""
    first = repository.create(
        company_namespace="company-a",
        dataset_id="current-corpus-a",
        version="20260816T010203Z-r1",
        manifest_sha256="a" * 64,
        initiated_by="admin-a",
        evaluation_dataset_id="evidence-gates-v1",
        evaluation_version="20260815T010203Z-r19",
        evaluation_manifest_sha256="b" * 64,
        evaluation_case_count=100,
    )
    repository.create(
        company_namespace="company-b",
        dataset_id="current-corpus-b",
        version="20260816T010203Z-r1",
        manifest_sha256="c" * 64,
        initiated_by="admin-b",
        evaluation_dataset_id="evidence-gates-v1",
        evaluation_version="20260815T010203Z-r19",
        evaluation_manifest_sha256="d" * 64,
        evaluation_case_count=100,
    )

    history = repository.list(company_namespace="company-a", limit=100)
    assert [item["workflow_id"] for item in history] == [first["workflow_id"]]
    assert history[0]["evaluation_dataset_id"] == "evidence-gates-v1"
    assert history[0]["evaluation_case_count"] == 100


def test_formal_suite_runs_ragas_and_persists_coverage(repository, tmp_path, monkeypatch):
    """A selected full suite creates immutable RAGAS outcomes for both stages."""
    from hashlib import sha256

    from evaluation.release.runtime import ReleaseWorkflowService
    from evaluation.release.runtime import _FORMAL_EVALUATION_METRICS
    from shared.config import settings

    workspace = _FrozenWorkspace(tmp_path / "workspace", version="20260816T010203Z-r2")
    service = ReleaseWorkflowService(
        repository, workspace, source_bundle=tmp_path, output_root=tmp_path / "results",
        env_file=tmp_path / "env", fixture_validation_service=_SuccessfulFixtureValidation(),
    )
    manifest_sha256 = sha256((workspace.bundle / "manifest.json").read_bytes()).hexdigest()
    work = service.start(
        company_namespace="company",
        dataset_id="current-corpus-under-test",
        version=workspace.version,
        actor_id="company-admin-a",
        current_corpus_draft_id=workspace.draft_id,
        current_corpus_snapshot_sha256=workspace.snapshot_sha256,
        evaluation_dataset_id="evidence-gates-v1",
        evaluation_version=workspace.version,
        evaluation_manifest_sha256=manifest_sha256,
        evaluation_case_count=100,
    )
    _write_baseline_template(tmp_path)

    async def execute(**kwargs):
        gate_mode = kwargs["gate_modes"][0]
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "responses.jsonl").write_text('{"benchmark_id":"case-1"}\n', encoding="utf-8")
        (root / "quality-report.json").write_text(
            json.dumps(_release_quality_report(gate_mode=gate_mode)), encoding="utf-8"
        )
        return {"status": "measured_pending_calibration"}

    observed_args: list[SimpleNamespace] = []

    async def run_ragas(args):
        responses = Path(args.responses_jsonl)
        observed_args.append(args)
        output = responses.with_name("ragas_scores.jsonl")
        output.write_text('{"metric":"faithfulness","status":"scored","score":0.9}\n', encoding="utf-8")
        output.with_suffix(".summary.json").write_text(
            json.dumps({"metrics": {
                metric: {
                    "total": 100,
                    "scored": 100 if metric == "evidence_gate_contract" else 80,
                    "not_applicable": 0 if metric == "evidence_gate_contract" else 20,
                    "skipped": 0,
                    "failed": 0,
                    "mean": 1.0 if metric == "evidence_gate_contract" else 0.9,
                }
                for metric in _FORMAL_EVALUATION_METRICS
            }}),
            encoding="utf-8",
        )
        return output

    monkeypatch.setenv("DEEPSEEK_API_KEY", "ark-test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://ark.example/v3")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.delenv("EVAL_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("evaluation.release.runtime.execute_evidence_gate_benchmark", execute)
    monkeypatch.setattr("evaluation.release.runtime.run_ragas_benchmark", run_ragas)

    completed = service.run(work["workflow_id"], company_namespace="company")
    assert completed["status"] == "pending_approval"
    assert len(observed_args) == 2
    # 正式评分集已收窄为五项核心指标（砍掉冗余的 semantic_similarity
    # 与短样本上区分度差的 noise_sensitivity）。
    assert all(set(args.metrics) == set(_FORMAL_EVALUATION_METRICS) for args in observed_args)
    assert all(args.embedding_model == settings.embedding_model for args in observed_args)
    attempts = repository.attempts(work["workflow_id"])
    measured = [item for item in attempts if item["stage"] in {"baseline", "shadow"}]
    assert all(item["metrics"]["evaluation_case_count"] == 100 for item in measured)
    assert all(item["metrics"]["ragas_coverage"]["faithfulness"]["scored"] == 80 for item in measured)


def test_formal_suite_fails_before_baseline_when_ragas_judge_is_unconfigured(
    repository, tmp_path, monkeypatch,
):
    """Missing Judge configuration must not spend a 100-case baseline run."""
    from hashlib import sha256

    from evaluation.release.runtime import ReleaseWorkflowService

    workspace = _FrozenWorkspace(tmp_path / "workspace-ragas-preflight", version="20260816T010203Z-r3")
    service = ReleaseWorkflowService(
        repository,
        workspace,
        source_bundle=tmp_path,
        output_root=tmp_path / "results-ragas-preflight",
        env_file=None,
        fixture_validation_service=_SuccessfulFixtureValidation(),
    )
    manifest_sha256 = sha256((workspace.bundle / "manifest.json").read_bytes()).hexdigest()
    workflow = service.start(
        company_namespace="company",
        dataset_id="current-corpus-under-test",
        version=workspace.version,
        actor_id="company-admin-a",
        current_corpus_draft_id=workspace.draft_id,
        current_corpus_snapshot_sha256=workspace.snapshot_sha256,
        evaluation_dataset_id="evidence-gates-v1",
        evaluation_version=workspace.version,
        evaluation_manifest_sha256=manifest_sha256,
        evaluation_case_count=100,
    )
    _write_baseline_template(tmp_path)
    monkeypatch.delenv("EVAL_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVAL_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("EVAL_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    async def execute(**_kwargs):
        raise AssertionError("baseline must not start without a RAGAS Judge")

    monkeypatch.setattr("evaluation.release.runtime.execute_evidence_gate_benchmark", execute)
    failed = service.run(workflow["workflow_id"], company_namespace="company")

    assert failed["status"] == "failed"
    assert failed["stage"] == "preflight"
    assert failed["reason_code"] == "ragas_configuration_missing"
    assert [item["stage"] for item in repository.attempts(workflow["workflow_id"])] == ["preflight"]


def test_ragas_failure_reason_is_bounded_and_actionable() -> None:
    """Persist only safe stable reason codes, never provider error text."""
    import errno

    from evaluation.release.runtime import _ragas_failure_code

    assert _ragas_failure_code(OSError(errno.ENOSPC, "disk full")) == "ragas_temporary_storage_full"
    assert _ragas_failure_code(TimeoutError("provider timed out")) == "ragas_judge_timeout"
    assert _ragas_failure_code(
        ValueError("collection adapter type mismatch"), phase="evaluate"
    ) == "ragas_evaluator_initialization_failed"
    assert _ragas_failure_code(RuntimeError("provider response contains sensitive details")) == "ragas_evaluation_failed"


def test_ragas_soft_time_limit_has_a_precise_reason_code() -> None:
    from evaluation.release.runtime import _ragas_failure_code

    class SoftTimeLimitExceeded(Exception):
        pass

    assert _ragas_failure_code(SoftTimeLimitExceeded()) == "ragas_execution_timeout"


def test_quality_latency_metrics_keep_only_numeric_percentiles(tmp_path) -> None:
    from evaluation.release.runtime import ReleaseWorkflowService

    report = tmp_path / "quality-report.json"
    report.write_text(json.dumps({"latency_ms": {"p50": 4180.415, "p95": 7468.176, "bad": "x"}}), encoding="utf-8")

    assert ReleaseWorkflowService._quality_latency_metrics(report) == {
        "p50": 4180.415,
        "p95": 7468.176,
    }


def test_formal_qa_snapshot_fails_before_ragas_when_any_record_is_incomplete(
    tmp_path: Path,
) -> None:
    """A doomed formal QA report must not spend Judge calls before failing."""
    from evaluation.release.runtime import ReleaseWorkflowService

    report = tmp_path / "quality-report.json"
    report.write_text(
        json.dumps({
            "run_classification": "baseline",
            "counts": {
                "total": 100, "succeeded": 99, "failed": 1,
                "invalid_provenance": 0,
            },
        }),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseWorkflowError, match="formal_qa_snapshot_incomplete"):
        ReleaseWorkflowService._assert_formal_qa_snapshot(
            report,
            formal_case_count=100,
        )


def test_smoke_qa_snapshot_with_twelve_completed_records_is_ragas_eligible(tmp_path: Path) -> None:
    """The 12-case smoke suite is complete even though it is classified smoke_only."""
    from evaluation.release.runtime import ReleaseWorkflowService

    report = tmp_path / "quality-report.json"
    report.write_text(json.dumps({
        "run_classification": "smoke_only",
        "counts": {"total": 12, "succeeded": 12, "failed": 0, "invalid_provenance": 0},
    }), encoding="utf-8")

    ReleaseWorkflowService._assert_formal_qa_snapshot(report, formal_case_count=12)


def test_failed_stage_snapshot_is_available_for_ragas_retry(repository, tmp_path) -> None:
    """A retry may reuse complete benchmark artifacts from a timed-out RAGAS pass."""
    from evaluation.release.runtime import ReleaseWorkflowService

    import hashlib

    work = _create(repository)
    report = (
        tmp_path / "results" / work["workflow_id"] / "baseline-1" / "off" / "run-1" / "quality-report.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    responses = report.parent / "responses.jsonl"
    responses.write_text('{"benchmark_id":"case-1"}\n', encoding="utf-8")
    repository.append_attempt(
        work["workflow_id"], stage="baseline", status="failed",
        metrics={
            "quality_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "responses_sha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
        },
    )
    service = ReleaseWorkflowService(
        repository, object(), source_bundle=tmp_path, output_root=tmp_path / "results", env_file=None,
    )

    assert service._latest_resumable_report(
        workflow_id=work["workflow_id"], stage="baseline", run_root=tmp_path / "results" / work["workflow_id"],
    ) == (report, 1)
    # A changed artifact can no longer be reused even though its old attempt failed
    # after the benchmark output was written.
    report.write_text('{"altered":true}', encoding="utf-8")
    assert service._latest_resumable_report(
        workflow_id=work["workflow_id"], stage="baseline", run_root=tmp_path / "results" / work["workflow_id"],
    ) is None


def test_failed_smoke_only_stage_snapshot_is_not_reused_for_ragas_retry(repository, tmp_path) -> None:
    """A fully failed QA run must execute QA again rather than loop on empty RAGAS input."""
    from evaluation.release.runtime import ReleaseWorkflowService

    import hashlib

    work = _create(repository)
    report = (
        tmp_path / "results" / work["workflow_id"] / "baseline-1" / "off" / "run-1" / "quality-report.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({
            "run_classification": "smoke_only",
            "counts": {"total": 100, "succeeded": 0, "failed": 100, "invalid_provenance": 0},
        }),
        encoding="utf-8",
    )
    responses = report.parent / "responses.jsonl"
    responses.write_text('{"benchmark_id":"case-1","status":"failed"}\n', encoding="utf-8")
    repository.append_attempt(
        work["workflow_id"],
        stage="baseline",
        status="failed",
        metrics={
            "quality_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "responses_sha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
        },
    )
    service = ReleaseWorkflowService(
        repository, object(), source_bundle=tmp_path, output_root=tmp_path / "results", env_file=None,
    )

    assert service._latest_resumable_report(
        workflow_id=work["workflow_id"],
        stage="baseline",
        run_root=tmp_path / "results" / work["workflow_id"],
    ) is None


def test_partial_qa_snapshot_is_not_reused_for_ragas_retry(repository, tmp_path) -> None:
    """A calibration-ineligible QA snapshot must rerun QA instead of only RAGAS."""
    import hashlib

    from evaluation.release.runtime import ReleaseWorkflowService

    work = _create(repository)
    report = (
        tmp_path / "results" / work["workflow_id"] / "baseline-1" / "off" / "run-1"
        / "quality-report.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({
            "run_classification": "baseline",
            "counts": {
                "total": 100, "succeeded": 82, "failed": 18,
                "invalid_provenance": 0,
            },
        }),
        encoding="utf-8",
    )
    responses = report.parent / "responses.jsonl"
    responses.write_text(
        '{"benchmark_id":"case-1","retrieval_mode":"hybrid","status":"succeeded"}\n',
        encoding="utf-8",
    )
    repository.append_attempt(
        work["workflow_id"],
        stage="baseline",
        status="failed",
        metrics={
            "quality_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "responses_sha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
        },
    )
    service = ReleaseWorkflowService(
        repository, object(), source_bundle=tmp_path,
        output_root=tmp_path / "results", env_file=None,
    )

    assert service._latest_resumable_report(
        workflow_id=work["workflow_id"],
        stage="baseline",
        run_root=tmp_path / "results" / work["workflow_id"],
    ) is None


def test_resumed_ragas_attempt_derives_run_id_from_saved_report_directory(tmp_path) -> None:
    """A resumable attempt has no fresh benchmark result object to read from."""
    from evaluation.release.runtime import ReleaseWorkflowService

    valid = tmp_path / "off" / "20260818T030609Z-f6dbfd03" / "quality-report.json"
    invalid = tmp_path / "off" / "unsafe.run" / "quality-report.json"

    assert ReleaseWorkflowService._artifact_run_id(valid) == "20260818T030609Z-f6dbfd03"
    assert ReleaseWorkflowService._artifact_run_id(invalid) is None


@pytest.mark.parametrize(
    ("summary_metric", "reason_code"),
    [
        ({"total": 99, "scored": 99, "not_applicable": 0, "skipped": 0, "failed": 0, "mean": 0.9}, "ragas_coverage_incomplete"),
        ({"total": 100, "scored": 99, "not_applicable": 0, "skipped": 0, "failed": 1, "mean": 0.9}, "ragas_required_score_failed"),
        ({"total": 100, "scored": 0, "not_applicable": 0, "skipped": 100, "failed": 0, "mean": None}, "ragas_coverage_contains_skips"),
    ],
)
def test_formal_ragas_coverage_fails_closed_for_incomplete_or_unusable_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, summary_metric: dict[str, object], reason_code: str,
) -> None:
    """Release evidence cannot progress with partial, failed, or legacy-skipped scores."""
    from evaluation.release.runtime import ReleaseWorkflowService
    from evaluation.release.runtime import (
        RagasCoverageError,
        _FORMAL_EVALUATION_METRICS,
    )

    report = tmp_path / "quality-report.json"
    report.write_text("{}", encoding="utf-8")
    responses = report.parent / "responses.jsonl"
    responses.write_text('{"benchmark_id":"case-1"}\n', encoding="utf-8")
    output = report.parent / "ragas_scores.jsonl"
    output.write_text("{}\n", encoding="utf-8")

    async def run_ragas(_args):
        output.with_suffix(".summary.json").write_text(
            json.dumps({"metrics": {
                metric: (
                    summary_metric if metric == "faithfulness" else {
                        "total": 100, "scored": 100, "not_applicable": 0,
                        "skipped": 0, "failed": 0, "mean": 1.0 if metric == "evidence_gate_contract" else 0.9,
                    }
                )
                for metric in _FORMAL_EVALUATION_METRICS
            }}),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("evaluation.release.runtime.run_ragas_benchmark", run_ragas)
    service = ReleaseWorkflowService(
        object(), object(), source_bundle=tmp_path, output_root=tmp_path, env_file=None,
    )

    with pytest.raises(RagasCoverageError, match=reason_code) as error:
        service._run_ragas(report=report, formal_case_count=100)

    assert error.value.metrics["ragas_coverage"]["faithfulness"]["failed"] == summary_metric["failed"]
    assert error.value.metrics["ragas_outcomes_sha256"]
    assert error.value.metrics["ragas_summary_sha256"]


def test_formal_release_requires_every_behavior_contract_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No RAGAS applicability accounting can hide an unscored formal case."""
    from evaluation.release.runtime import ReleaseWorkflowService
    from evaluation.release.runtime import (
        RagasCoverageError,
        _FORMAL_EVALUATION_METRICS,
    )

    report = tmp_path / "quality-report.json"
    report.write_text("{}", encoding="utf-8")
    responses = tmp_path / "responses.jsonl"
    responses.write_text('{"benchmark_id":"case-1"}\n', encoding="utf-8")
    output = tmp_path / "ragas_scores.jsonl"
    output.write_text("{}\n", encoding="utf-8")

    async def run_ragas(_args):
        output.with_suffix(".summary.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        metric: {
                            "total": 100,
                            "scored": 99 if metric == "evidence_gate_contract" else 80,
                            "not_applicable": 0 if metric == "evidence_gate_contract" else 20,
                            "skipped": 0,
                            "failed": 1 if metric == "evidence_gate_contract" else 0,
                            "mean": 0.9,
                        }
                        for metric in _FORMAL_EVALUATION_METRICS
                    }
                }
            ),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("evaluation.release.runtime.run_ragas_benchmark", run_ragas)
    service = ReleaseWorkflowService(
        object(), object(), source_bundle=tmp_path, output_root=tmp_path, env_file=None,
    )

    with pytest.raises(RagasCoverageError, match="evidence_gate_contract_incomplete") as error:
        service._run_ragas(report=report, formal_case_count=100)

    assert error.value.metrics["evidence_gate_contract_coverage"]["failed"] == 1


def test_formal_ragas_coverage_rejects_duplicate_case_ids_even_when_summary_counts_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate totals cannot substitute for the frozen per-case identity check."""
    from types import SimpleNamespace

    from evaluation.release.runtime import ReleaseWorkflowService
    from evaluation.release.runtime import _FORMAL_EVALUATION_METRICS

    expected_cases = [SimpleNamespace(id=f"case-{index:03d}") for index in range(100)]
    monkeypatch.setattr(
        "evaluation.release.runtime.load_evidence_gate_benchmark",
        lambda *_args, **_kwargs: expected_cases,
    )
    report = tmp_path / "quality-report.json"
    report.write_text("{}", encoding="utf-8")
    responses = report.with_name("responses.jsonl")
    responses.write_text(
        "".join(
            json.dumps(
                {
                    "benchmark_id": case.id,
                    "retrieval_mode": "dense_bm25_graph",
                    "status": "succeeded",
                    "question": "问题",
                    "reference": "参考答案",
                    "response": "回答",
                    "contexts": [{"content": "上下文"}],
                },
                ensure_ascii=False,
            )
            + "\n"
            for case in expected_cases
        ),
        encoding="utf-8",
    )

    async def run_ragas(args):
        output = Path(args.responses_jsonl).with_name("ragas_scores.jsonl")
        # Replace case-099 with a second case-000. Totals remain 100 but the
        # frozen suite is not fully represented.
        duplicated_ids = [case.id for case in expected_cases[:-1]] + [expected_cases[0].id]
        output.write_text(
            "".join(
                json.dumps(
                    {
                        "benchmark_id": case_id,
                        "retrieval_mode": "dense_bm25_graph",
                        "metric": metric,
                        "status": "scored",
                        "score": 0.9,
                    }
                )
                + "\n"
                for metric in _FORMAL_EVALUATION_METRICS
                for case_id in duplicated_ids
            ),
            encoding="utf-8",
        )
        output.with_suffix(".summary.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        metric: {
                            "total": 100,
                            "scored": 100,
                            "not_applicable": 0,
                            "skipped": 0,
                            "failed": 0,
                            "mean": 0.9,
                        }
                        for metric in _FORMAL_EVALUATION_METRICS
                    }
                }
            ),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("evaluation.release.runtime.run_ragas_benchmark", run_ragas)
    service = ReleaseWorkflowService(
        object(), object(), source_bundle=tmp_path, output_root=tmp_path, env_file=None,
    )

    with pytest.raises(ReleaseWorkflowError, match="ragas_coverage_incomplete"):
        service._run_ragas(
            report=report,
            formal_case_count=100,
            manifest_path=tmp_path / "manifest.json",
            expected_manifest_sha256="a" * 64,
            output_root=tmp_path / "retry",
        )


def test_release_retry_enriches_legacy_ragas_inputs_without_mutating_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reviewed evidence excerpts make reference metrics eligible without mutating history."""
    import hashlib
    from types import SimpleNamespace

    from evaluation.release.runtime import ReleaseWorkflowService

    source_dir = tmp_path / "baseline-1" / "off" / "run-1"
    source_dir.mkdir(parents=True)
    source_responses = source_dir / "responses.jsonl"
    source_responses.write_text(
        json.dumps({
            "benchmark_id": "case-1", "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded", "question": "问题", "response": "回答",
            "reference": "", "contexts": [{"content": "上下文"}],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    source_scores = source_dir / "ragas_scores.jsonl"
    source_scores.write_text(
        json.dumps({
            "benchmark_id": "case-1", "retrieval_mode": "dense_bm25_graph",
            "metric": "faithfulness", "score": 0.91, "status": "scored",
            "exception": None,
        }) + "\n",
        encoding="utf-8",
    )
    source_scores.with_suffix(".resume.json").write_text(
        json.dumps({
            "responses_sha256": hashlib.sha256(source_responses.read_bytes()).hexdigest(),
            "metrics": [
                "faithfulness", "factual_correctness", "context_precision", "context_recall",
            ],
        }) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "context-catalog.jsonl").write_text(
        json.dumps({
            "context_id": "context-1",
            "content_excerpt": "冻结审核证据摘要",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"context_catalog_file": "context-catalog.jsonl"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evaluation.release.runtime.load_evidence_gate_benchmark",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                id="case-1",
                reference_answer="冻结审核参考答案",
                expected_evidence_context_ids=("context-1",),
            ),
        ],
    )
    service = ReleaseWorkflowService(
        object(), object(), source_bundle=tmp_path, output_root=tmp_path, env_file=None,
    )
    retry_root = tmp_path / "baseline-2"

    prepared = service._prepare_ragas_inputs(
        responses_path=source_responses,
        manifest_path=tmp_path / "manifest.json",
        expected_manifest_sha256="a" * 64,
        output_root=retry_root,
    )

    source_record = json.loads(source_responses.read_text(encoding="utf-8"))
    assert source_record["reference"] == ""
    prepared_record = json.loads(prepared.read_text(encoding="utf-8"))
    assert prepared_record["reference"] == "冻结审核参考答案"
    seeded = [json.loads(line) for line in (retry_root / "ragas_scores.jsonl").read_text().splitlines()]
    assert seeded[0]["metric"] == "faithfulness"
    assert seeded[0]["score"] == 0.91


def test_release_retry_does_not_reuse_scores_from_a_different_response_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-ID score from another QA snapshot must never seed a new RAGAS run."""
    import hashlib
    from types import SimpleNamespace

    from evaluation.release.runtime import ReleaseWorkflowService

    source_dir = tmp_path / "baseline-2" / "off" / "run-2"
    source_dir.mkdir(parents=True)
    source_responses = source_dir / "responses.jsonl"
    source_responses.write_text(
        json.dumps({
            "benchmark_id": "case-1", "retrieval_mode": "dense_bm25_graph",
            "status": "succeeded", "question": "新问题", "response": "新回答",
            "reference": "", "contexts": [{"content": "新上下文"}],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stale_output = tmp_path / "baseline-1" / "ragas_scores.jsonl"
    stale_output.parent.mkdir(parents=True)
    stale_output.write_text(
        json.dumps({
            "benchmark_id": "case-1", "retrieval_mode": "dense_bm25_graph",
            "metric": "faithfulness", "score": 0.99, "status": "scored",
        }) + "\n",
        encoding="utf-8",
    )
    stale_output.with_suffix(".resume.json").write_text(
        json.dumps({
            "responses_sha256": hashlib.sha256(b"different snapshot").hexdigest(),
            "metrics": [
                "faithfulness", "factual_correctness", "context_precision", "context_recall",
            ],
        }) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "context-catalog.jsonl").write_text(
        json.dumps({"context_id": "context-1", "content_excerpt": "审核证据"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"context_catalog_file": "context-catalog.jsonl"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evaluation.release.runtime.load_evidence_gate_benchmark",
        lambda *_args, **_kwargs: [
            SimpleNamespace(id="case-1", expected_evidence_context_ids=("context-1",)),
        ],
    )
    service = ReleaseWorkflowService(
        object(), object(), source_bundle=tmp_path, output_root=tmp_path, env_file=None,
    )

    prepared = service._prepare_ragas_inputs(
        responses_path=source_responses,
        manifest_path=tmp_path / "manifest.json",
        expected_manifest_sha256="a" * 64,
        output_root=tmp_path / "baseline-3",
    )

    assert prepared.is_file()
    assert not (tmp_path / "baseline-3" / "ragas_scores.jsonl").exists()


def test_release_retry_reuses_all_finite_scores_from_the_same_prepared_snapshot(
    tmp_path: Path,
) -> None:
    """Reference-based scores may resume only after the enriched input hash matches."""
    import hashlib

    from evaluation.release.runtime import ReleaseWorkflowService
    from evaluation.release.runtime import _RAGAS_METRICS

    original = tmp_path / "source" / "responses.jsonl"
    original.parent.mkdir()
    original.write_text(
        '{"benchmark_id":"case-1","retrieval_mode":"hybrid","reference":""}\n',
        encoding="utf-8",
    )
    prepared = tmp_path / "current" / "ragas-inputs.jsonl"
    prepared.parent.mkdir()
    prepared.write_text(
        '{"benchmark_id":"case-1","retrieval_mode":"hybrid","reference":"审核证据"}\n',
        encoding="utf-8",
    )
    prior = tmp_path / "prior" / "ragas_scores.jsonl"
    prior.parent.mkdir()
    prior.write_text(
        "".join(
            json.dumps({
                "benchmark_id": "case-1", "retrieval_mode": "hybrid",
                "metric": metric, "score": 0.8, "status": "scored",
            }) + "\n"
            for metric in _RAGAS_METRICS
        ),
        encoding="utf-8",
    )
    prior.with_suffix(".resume.json").write_text(
        json.dumps({
            "responses_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
            "metrics": list(_RAGAS_METRICS),
        }) + "\n",
        encoding="utf-8",
    )

    ReleaseWorkflowService._seed_ragas_checkpoint(
        source_outputs=[prior],
        original_responses=original,
        prepared_responses=prepared,
    )

    seeded = [
        json.loads(line)
        for line in (prepared.parent / "ragas_scores.jsonl").read_text().splitlines()
    ]
    assert {item["metric"] for item in seeded} == set(_RAGAS_METRICS)


def test_ragas_setup_failure_logs_only_phase_and_exception_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Startup diagnostics stay actionable without logging provider or evaluation content."""
    from evaluation.release.runtime import ReleaseWorkflowService

    report = tmp_path / "quality-report.json"
    report.write_text("{}", encoding="utf-8")
    (tmp_path / "responses.jsonl").write_text(
        '{"benchmark_id":"case-1","retrieval_mode":"hybrid"}\n',
        encoding="utf-8",
    )
    service = ReleaseWorkflowService(
        object(), object(), source_bundle=tmp_path, output_root=tmp_path, env_file=None,
    )

    def fail_prepare(**_kwargs):
        raise AttributeError("sensitive question and provider payload")

    monkeypatch.setattr(service, "_prepare_ragas_inputs", fail_prepare)
    with pytest.raises(ReleaseWorkflowError, match="ragas_input_preparation_failed"):
        service._run_ragas(
            report=report,
            formal_case_count=1,
            manifest_path=tmp_path / "manifest.json",
            expected_manifest_sha256="a" * 64,
            output_root=tmp_path / "baseline-1",
        )

    assert "phase=prepare_inputs" in caplog.text
    assert "exception_type=AttributeError" in caplog.text
    assert "sensitive question" not in caplog.text
    assert "provider payload" not in caplog.text


def test_repository_maker_checker_promotion_and_history_rollback(repository) -> None:
    """Approval, one-step activation and rollback retain a durable audit chain."""
    from evaluation.release.runtime import ReleaseWorkflowService
    from infrastructure.postgres.models import (
        evidence_gate_configuration_history,
        evidence_release_approvals,
    )

    pending = repository.transition(
        _create(repository)["workflow_id"],
        expected_revision=1,
        status="pending_approval",
        stage="approval",
    )
    with pytest.raises(ReleaseWorkflowError, match="release_self_approval_forbidden"):
        repository.review(
            pending["workflow_id"], expected_revision=pending["revision"],
            reviewer_id="company-admin-a", decision="approved",
            reason="发起人不能自行批准", target_mode="shadow",
        )

    service = ReleaseWorkflowService(
        repository, object(), source_bundle=Path("."), output_root=Path("."), env_file=None,
    )
    with pytest.raises(ReleaseWorkflowError, match="release_checker_unavailable"):
        service.review(
            pending["workflow_id"], expected_revision=pending["revision"],
            reviewer_id="company-admin-b", decision="approved", reason="双人复核可用",
            target_mode="shadow", active_organization_admin_count=1,
        )

    approved = service.review(
        pending["workflow_id"], expected_revision=pending["revision"],
        reviewer_id="company-admin-b", decision="approved", reason="离线证据复核通过",
        target_mode="shadow", active_organization_admin_count=2,
    )
    assert approved["status"] == "approved"
    assert approved["target_mode"] == "shadow"
    with repository.database.session() as session:
        approvals = session.execute(select(evidence_release_approvals)).mappings().all()
    assert [(row["reviewer_id"], row["decision"], row["reason"])
            for row in approvals] == [("company-admin-b", "approved", "离线证据复核通过")]

    promoted = service.promote(
        approved["workflow_id"], expected_revision=approved["revision"], actor_id="company-admin-c"
    )
    assert promoted["status"] == "promoted"
    assert repository.gate()["mode"] == "shadow"
    assert repository.gate()["revision"] == 1

    rolled_back = service.rollback(
        expected_revision=1, actor_id="company-admin-c", reason="观察窗口内回退"
    )
    assert rolled_back["mode"] == "off"
    assert rolled_back["revision"] == 2
    with repository.database.session() as session:
        history = session.execute(
            select(evidence_gate_configuration_history).order_by(
                evidence_gate_configuration_history.c.configuration_revision
            )
        ).mappings().all()
    assert [(row["action"], row["previous_mode"], row["mode"], row["workflow_id"])
            for row in history] == [
        ("promote", "off", "shadow", promoted["workflow_id"]),
        ("rollback", "shadow", "off", promoted["workflow_id"]),
    ]
    with pytest.raises(ReleaseWorkflowError, match="gate_revision_conflict"):
        service.rollback(expected_revision=1, actor_id="company-admin-c", reason="陈旧版本")


def test_repository_rejects_direct_off_to_enforce_after_valid_approval(repository) -> None:
    """An approval records intent but can never skip the mandatory shadow stage."""
    pending = repository.transition(
        _create(repository)["workflow_id"], expected_revision=1,
        status="pending_approval", stage="approval",
    )
    approved = repository.review(
        pending["workflow_id"], expected_revision=pending["revision"],
        reviewer_id="company-admin-b", decision="approved", reason="复核通过",
        target_mode="enforce",
    )
    with pytest.raises(ReleaseWorkflowError, match="gate_transition_invalid"):
        repository.promote(
            approved["workflow_id"], expected_revision=approved["revision"],
            actor_id="company-admin-c",
        )
    assert repository.gate()["mode"] == "off"


def _formal_release_work(repository: PostgreSQLReleaseWorkflowRepository) -> dict:
    """Create a minimal formal release workflow for failed-RAGAS persistence tests."""
    return repository.create(
        company_namespace="company-internal",
        dataset_id="current-corpus-fixture",
        version="20260815T173052Z-r5",
        manifest_sha256="a" * 64,
        initiated_by="company-admin-a",
        evaluation_dataset_id="evidence-gates-v1",
        evaluation_version="20260815T092331Z-r19",
        evaluation_manifest_sha256="b" * 64,
        evaluation_case_count=100,
    )


def _complete_formal_report(root: Path) -> Path:
    """Write a complete formal QA snapshot that is eligible for RAGAS resume."""
    report = root / "run-1" / "quality-report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(_release_quality_report(gate_mode="off")), encoding="utf-8")
    (report.parent / "responses.jsonl").write_text(
        '{"benchmark_id":"case-001","retrieval_mode":"dense_bm25_graph"}\n',
        encoding="utf-8",
    )
    return report


def _failed_ragas_metrics() -> dict[str, object]:
    """Return only the bounded diagnostics permitted on a failed formal attempt."""
    return {
        "ragas_outcomes_sha256": "c" * 64,
        "ragas_summary_sha256": "d" * 64,
        "ragas_coverage": {
            "faithfulness": {
                "total": 100,
                "scored": 99,
                "skipped": 0,
                "failed": 1,
                "mean": 0.9,
                "skip_reasons": {},
                "failure_reasons": {"ragas_judge_timeout": 1},
                "failure_exceptions": {"APITimeoutError": 1},
                "failure_finish_reasons": {"length": 1},
                "batch_fallbacks": 1,
            },
        },
    }


def test_formal_ragas_coverage_preserves_bounded_failure_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validated exception/finish/fallback counters survive fail-closed release scoring."""
    from evaluation.release.runtime import ReleaseWorkflowService
    from evaluation.release.runtime import RagasCoverageError, _FORMAL_EVALUATION_METRICS

    report = tmp_path / "quality-report.json"
    report.write_text("{}", encoding="utf-8")
    (tmp_path / "responses.jsonl").write_text('{"benchmark_id":"case-1"}\n', encoding="utf-8")
    output = tmp_path / "ragas_scores.jsonl"
    output.write_text("{}\n", encoding="utf-8")

    async def run_ragas(_args):
        output.with_suffix(".summary.json").write_text(
            json.dumps({"metrics": {
                metric: {
                    "total": 100,
                    "scored": 99 if metric == "faithfulness" else 100,
                    "not_applicable": 0,
                    "skipped": 0,
                    "failed": 1 if metric == "faithfulness" else 0,
                    "failure_reasons": {"judge_json_object_invalid": 1} if metric == "faithfulness" else {},
                    "failure_exceptions": {"RagasJudgeJsonObjectError": 1} if metric == "faithfulness" else {},
                    "failure_finish_reasons": {"length": 1} if metric == "faithfulness" else {},
                    "batch_fallbacks": 2 if metric == "faithfulness" else 0,
                    "mean": 0.9,
                }
                for metric in _FORMAL_EVALUATION_METRICS
            }}),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("evaluation.release.runtime.run_ragas_benchmark", run_ragas)
    service = ReleaseWorkflowService(
        object(), object(), source_bundle=tmp_path, output_root=tmp_path, env_file=None,
    )

    with pytest.raises(RagasCoverageError) as error:
        service._run_ragas(report=report, formal_case_count=100)

    metric = error.value.metrics["ragas_coverage"]["faithfulness"]
    assert metric["failure_exceptions"] == {"RagasJudgeJsonObjectError": 1}
    assert metric["failure_finish_reasons"] == {"length": 1}
    assert metric["batch_fallbacks"] == 2


def test_failed_formal_ragas_persists_bounded_diagnostics_before_attempt_finishes(
    repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fail-closed score stage keeps safe retry diagnostics in the same attempt."""
    from evaluation.release.runtime import ReleaseWorkflowService
    from evaluation.release.runtime import RagasCoverageError

    work = _formal_release_work(repository)
    report = _complete_formal_report(tmp_path)
    service = ReleaseWorkflowService(
        repository, object(), source_bundle=tmp_path, output_root=tmp_path / "results", env_file=None,
    )
    monkeypatch.setattr(
        service,
        "_latest_resumable_report",
        lambda **_kwargs: (report, 1),
    )
    expected_metrics = _failed_ragas_metrics()

    def fail_ragas(**_kwargs):
        raise RagasCoverageError("ragas_required_score_failed", metrics=expected_metrics)

    monkeypatch.setattr(service, "_run_ragas", fail_ragas)

    with pytest.raises(RagasCoverageError, match="ragas_required_score_failed"):
        service._run_measurement(
            work=work,
            company_namespace="company-internal",
            manifest_path=tmp_path / "manifest.json",
            baseline_path=tmp_path / "baseline-template.json",
            run_root=tmp_path / "results" / work["workflow_id"],
            stage="baseline",
            gate_mode="off",
        )

    service._fail(
        work["workflow_id"], stage="baseline", code="ragas_required_score_failed",
    )
    attempt = repository.attempts(work["workflow_id"])[-1]
    assert attempt["status"] == "failed"
    assert attempt["reason_code"] == "ragas_required_score_failed"
    assert attempt["metrics"]["ragas_coverage"] == expected_metrics["ragas_coverage"]
    assert attempt["metrics"]["ragas_outcomes_sha256"] == "c" * 64
    assert attempt["metrics"]["ragas_summary_sha256"] == "d" * 64
    assert "question" not in attempt["metrics"]
    assert "response" not in attempt["metrics"]
    assert "contexts" not in attempt["metrics"]


def test_invalid_ragas_summary_does_not_persist_unvalidated_diagnostics(
    repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed score summaries fail without making an untrusted coverage map visible."""
    from evaluation.release.runtime import ReleaseWorkflowService

    work = _formal_release_work(repository)
    report = _complete_formal_report(tmp_path)
    service = ReleaseWorkflowService(
        repository, object(), source_bundle=tmp_path, output_root=tmp_path / "results", env_file=None,
    )
    monkeypatch.setattr(
        service,
        "_latest_resumable_report",
        lambda **_kwargs: (report, 1),
    )

    def invalid_summary(**_kwargs):
        raise ReleaseWorkflowError("ragas_summary_invalid")

    monkeypatch.setattr(service, "_run_ragas", invalid_summary)

    with pytest.raises(ReleaseWorkflowError, match="ragas_summary_invalid"):
        service._run_measurement(
            work=work,
            company_namespace="company-internal",
            manifest_path=tmp_path / "manifest.json",
            baseline_path=tmp_path / "baseline-template.json",
            run_root=tmp_path / "results" / work["workflow_id"],
            stage="baseline",
            gate_mode="off",
        )

    service._fail(work["workflow_id"], stage="baseline", code="ragas_summary_invalid")
    attempt = repository.attempts(work["workflow_id"])[-1]
    assert attempt["status"] == "failed"
    assert attempt["reason_code"] == "ragas_summary_invalid"
    assert "ragas_coverage" not in attempt["metrics"]
    assert "ragas_outcomes_sha256" not in attempt["metrics"]
    assert "ragas_summary_sha256" not in attempt["metrics"]
