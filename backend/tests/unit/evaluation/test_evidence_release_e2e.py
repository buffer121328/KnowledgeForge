"""Isolated fixture end-to-end check for the frozen evidence release journey."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from evaluation.evidence_gate.metrics import SUPPORTED_EVIDENCE_GATE_METRICS
from evaluation.evidence_gate.review_workspace import EvidenceReviewConflict, EvidenceReviewWorkspace
from evaluation.release.workflow import PostgreSQLReleaseWorkflowRepository
from evaluation.release.runtime import ReleaseWorkflowService
from infrastructure.postgres.database import DatabaseService
from infrastructure.postgres.models import metadata


SOURCE_BUNDLE = Path(__file__).resolve().parents[3] / "evaluation" / "data" / "evidence-gates"


def _freeze_reviewed_dataset(workspace: EvidenceReviewWorkspace) -> dict:
    """Approve the deterministic local fixture with separate maker and checker IDs."""
    dataset = workspace.get_dataset("company", "evidence-gates-v1")
    revision = dataset["revision"]
    cases = workspace.list_cases("company", "evidence-gates-v1", page=1, page_size=200)["cases"]
    for case in cases:
        submitted = workspace.submit_case(
            "company",
            "evidence-gates-v1",
            case["id"],
            expected_revision=revision,
            actor_id="maker-a",
            actor_department_id=case["department_id"],
            reviewer_id="checker-b",
        )
        approved = workspace.review_case(
            "company",
            "evidence-gates-v1",
            case["id"],
            decision="approve",
            reason="fixture checked",
            expected_revision=submitted["revision"],
            actor_id="checker-b",
            actor_department_id=case["department_id"],
            actor_is_department_manager=True,
        )
        revision = approved["revision"]
    return workspace.freeze(
        "company",
        "evidence-gates-v1",
        expected_revision=revision,
        actor_id="company-admin-a",
    )


def test_local_release_journey_freezes_locks_derives_runs_and_enters_review_without_gate_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the company-admin journey without external services or deployment control."""
    workspace = EvidenceReviewWorkspace(tmp_path / "workspace", SOURCE_BUNDLE)
    frozen = _freeze_reviewed_dataset(workspace)

    versions = workspace.list_versions("company", "evidence-gates-v1")
    assert versions[0]["version"] == frozen["version"]
    assert versions[0]["manifest_sha256"] == frozen["manifest_sha256"]
    with pytest.raises(EvidenceReviewConflict, match="dataset_frozen"):
        workspace.freeze(
            "company",
            "evidence-gates-v1",
            expected_revision=frozen["revision"],
            actor_id="company-admin-a",
        )

    draft = workspace.create_draft_from_version(
        "company",
        "evidence-gates-v1",
        frozen["version"],
        expected_revision=frozen["revision"],
        actor_id="company-admin-a",
    )
    assert draft["status"] == "authoring"
    assert draft["source_frozen_version"] == frozen["version"]

    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    repository = PostgreSQLReleaseWorkflowRepository(DatabaseService.from_engine(engine))
    service = ReleaseWorkflowService(
        repository,
        workspace,
        source_bundle=SOURCE_BUNDLE,
        output_root=tmp_path / "results",
        env_file=tmp_path / "not-read-by-fixture.env",
    )
    observed_modes: list[str] = []

    async def fixture_benchmark(**kwargs):
        mode = kwargs["gate_modes"][0]
        observed_modes.append(mode)
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True, exist_ok=True)
        report = root / "quality-report.json"
        report.write_text(
            json.dumps(
                {
                    "evidence_gate": {
                        "metrics": {name: 1.0 for name in SUPPORTED_EVIDENCE_GATE_METRICS},
                        "run_identity": {"run_id": f"fixture-{mode}"},
                        "counts": {"completed": 100},
                    }
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "measured_pending_calibration",
            "quality_metrics": [{"quality_report": str(report), "run_id": f"fixture-{mode}"}],
        }

    monkeypatch.setattr(
        "evaluation.release.runtime.execute_evidence_gate_benchmark",
        fixture_benchmark,
    )
    # This row simulates a historical workflow created before formal-suite
    # identity was persisted; it remains executable for audit/recovery.
    workflow = repository.create(
        company_namespace="company",
        dataset_id="evidence-gates-v1",
        version=frozen["version"],
        manifest_sha256=frozen["manifest_sha256"],
        initiated_by="company-admin-a",
    )
    completed = service.run(
        workflow["workflow_id"],
        company_namespace="company"
    )
    assert completed["status"] == "pending_approval"
    assert observed_modes == ["off", "shadow"]
    attempts = repository.attempts(workflow["workflow_id"])
    assert [attempt["stage"] for attempt in attempts] == ["preflight", "baseline", "shadow", "calibration"]
    assert attempts[1]["metrics"]["run_id"] == "fixture-off"
    assert "quality_report" not in attempts[1]["metrics"]

    assert repository.gate()["mode"] == "off"
