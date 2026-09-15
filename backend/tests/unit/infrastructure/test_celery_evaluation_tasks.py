"""Worker contract tests for company-admin release orchestration."""
from __future__ import annotations

from typing import Any


def test_release_worker_marks_completed_workflow_succeeded(
    monkeypatch,
) -> None:
    """The worker accepts enum-backed roles and never marks a live duplicate failed."""
    from infrastructure.tasks import celery_evaluation_tasks as tasks

    observed: dict[str, Any] = {}
    lifecycle: list[str] = []

    class Service:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, workflow_id: str, *, company_namespace: str):
            observed.update(
                workflow_id=workflow_id,
                company_namespace=company_namespace,
            )
            return {"status": "completed"}

    monkeypatch.setattr(tasks, "EvidenceReviewWorkspace", lambda *_args: object())
    monkeypatch.setattr(tasks, "PostgreSQLReleaseWorkflowRepository", lambda *_args: object())
    monkeypatch.setattr(tasks, "ReleaseWorkflowService", Service)
    monkeypatch.setattr(tasks, "get_database_service", lambda: object())
    monkeypatch.setattr(tasks, "update_task_state_safely", lambda _id, state: lifecycle.append(state))

    result = tasks.evidence_release_workflow_task.run("erw-test", "company-internal")

    assert result == {"status": "completed"}
    assert observed == {
        "workflow_id": "erw-test",
        "company_namespace": "company-internal",
    }
    assert lifecycle == ["started", "succeeded"]


def test_release_worker_marks_pending_approval_workflow_succeeded(
    monkeypatch,
) -> None:
    """A workflow waiting for human approval is finished, not still executing."""
    from infrastructure.tasks import celery_evaluation_tasks as tasks

    lifecycle: list[str] = []

    class Service:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, *_args, **_kwargs):
            return {"status": "pending_approval", "stage": "approval"}

    monkeypatch.setattr(tasks, "EvidenceReviewWorkspace", lambda *_args: object())
    monkeypatch.setattr(tasks, "PostgreSQLReleaseWorkflowRepository", lambda *_args: object())
    monkeypatch.setattr(tasks, "ReleaseWorkflowService", Service)
    monkeypatch.setattr(tasks, "get_database_service", lambda: object())
    monkeypatch.setattr(tasks, "update_task_state_safely", lambda _id, state: lifecycle.append(state))

    result = tasks.evidence_release_workflow_task.run("erw-pending", "company-internal")

    assert result["status"] == "pending_approval"
    assert lifecycle == ["started", "succeeded"]


def test_fixture_validation_worker_runs_only_the_server_bound_run(monkeypatch) -> None:
    from infrastructure.tasks import celery_evaluation_tasks as tasks

    observed: dict[str, Any] = {}

    class Service:
        def __init__(self, *_args, **kwargs) -> None:
            observed["finance_department_id"] = kwargs["finance_department_id"]
            observed["corpus_probe"] = kwargs["corpus_probe"]

        def run(self, *, org_id: str, dataset_id: str, version: str, run_id: str):
            observed.update(org_id=org_id, dataset_id=dataset_id, version=version, run_id=run_id)
            return {"run_id": run_id, "status": "succeeded"}

    corpus_probe = object()
    monkeypatch.setattr(tasks, "EvidenceReviewWorkspace", lambda *_args: object())
    monkeypatch.setattr(tasks, "FixtureValidationService", Service)
    monkeypatch.setattr(tasks, "get_database_service", lambda: object())
    monkeypatch.setattr(tasks, "PostgreSQLDocumentCatalogRepository", lambda _database: object())
    monkeypatch.setattr(tasks, "build_catalog_fixture_corpus_probe", lambda _catalog: corpus_probe)
    monkeypatch.setattr(tasks.settings, "evaluation_fixture_finance_department_id", "finance")

    result = tasks.evidence_fixture_validation_task.run("fvr-1", "org-a", "evidence-gates-v1", "v1")

    assert result == {"run_id": "fvr-1", "status": "succeeded"}
    assert observed == {
        "finance_department_id": "finance",
        "corpus_probe": corpus_probe,
        "org_id": "org-a",
        "dataset_id": "evidence-gates-v1",
        "version": "v1",
        "run_id": "fvr-1",
    }


def test_release_evaluation_worker_reserves_extended_timeout_window() -> None:
    """Formal RAGAS scoring gets 90 minutes plus a 10-minute cleanup margin."""
    from infrastructure.celery_app import celery_app

    soft_limit = celery_app.conf.task_soft_time_limit
    hard_limit = celery_app.conf.task_time_limit

    assert soft_limit == 90 * 60
    assert hard_limit == 100 * 60
    assert hard_limit - soft_limit >= 10 * 60


def test_release_evaluation_task_has_a_dedicated_twelve_hour_budget() -> None:
    """Long RAGAS release runs must not change generic worker task limits."""
    from infrastructure.celery_app import celery_app

    limits = celery_app.conf.task_annotations[
        "infrastructure.celery_tasks.evidence_release_workflow_task"
    ]

    assert limits == {
        "soft_time_limit": 12 * 60 * 60,
        "time_limit": 12 * 60 * 60 + 30 * 60,
    }
