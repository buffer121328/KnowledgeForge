"""Celery worker entrypoint for durable evidence release workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.evidence_gate.review_workspace import EvidenceReviewWorkspace
from shared.paths import BACKEND_ROOT, COMPANY_DEMO_CORPUS_MANIFEST, EVIDENCE_GATES_DATA_ROOT, EVIDENCE_REVIEW_ROOT
from evaluation.fixture_validation import (
    FixtureValidationService,
    build_catalog_fixture_corpus_probe,
)
from evaluation.release.workflow import PostgreSQLReleaseWorkflowRepository
from evaluation.release.runtime import ReleaseWorkflowService
from infrastructure.celery_app import celery_app
from infrastructure.documents.catalog import PostgreSQLDocumentCatalogRepository
from infrastructure.postgres.database import get_database_service
from infrastructure.tasks.task_registry import update_task_state_safely
from shared.utils.metrics import evidence_release_workflows_total
from shared.config import settings


@celery_app.task(name="infrastructure.celery_tasks.evidence_release_workflow_task", bind=True, max_retries=1)
def evidence_release_workflow_task(self, workflow_id: str, company_namespace: str) -> dict[str, Any]:
    """在 Worker 内直接执行评测发布工作流（基准/校准），不派生 shell 子进程。

    Args:
        self: Celery 任务实例（bind=True），用于任务生命周期控制。
        workflow_id: 发布工作流 ID，用于状态登记与运行定位。
        company_namespace: 公司命名空间（组织 ID），限定工作流的数据范围。
    """
    update_task_state_safely(workflow_id, "started")
    evidence_release_workflows_total.labels("worker", "started").inc()
    root = BACKEND_ROOT
    workspace = EvidenceReviewWorkspace(EVIDENCE_REVIEW_ROOT, EVIDENCE_GATES_DATA_ROOT)
    catalog = PostgreSQLDocumentCatalogRepository(get_database_service())
    service = ReleaseWorkflowService(
        PostgreSQLReleaseWorkflowRepository(get_database_service()), workspace,
        source_bundle=EVIDENCE_GATES_DATA_ROOT,
        output_root=root / "evaluation" / "results" / "release-workflows",
        # Compose 已向 Worker 进程注入运行时配置，因此不再挂载 .env 文件。
        env_file=None,
        # 内联装配 Fixture 校验：财务部门 ID 来自服务端专用配置，
        # 语料探针基于 PostgreSQL 文档目录而非浏览器可控输入。
        fixture_validation_service=FixtureValidationService(
            workspace,
            finance_department_id=settings.evaluation_fixture_finance_department_id,
            corpus_probe=build_catalog_fixture_corpus_probe(catalog),
        ),
    )
    try:
        result = service.run(
            workflow_id,
            company_namespace=company_namespace,
        )
        workflow_status = result.get("status")
        if workflow_status in {"completed", "pending_approval", "approved", "promoted", "rejected"}:
            task_state = "succeeded"
        elif workflow_status == "failed":
            task_state = "failed"
        else:
            # 重复投递的 Celery 消息可能观察到原任务仍在运行，
            # 此时不得把生命周期证据覆盖为 failed。
            task_state = "started"
        update_task_state_safely(workflow_id, task_state)
        evidence_release_workflows_total.labels("worker", task_state).inc()
        return result
    except Exception:
        update_task_state_safely(workflow_id, "failed")
        evidence_release_workflows_total.labels("worker", "failed").inc()
        raise


@celery_app.task(
    name="infrastructure.celery_evaluation_tasks.evidence_fixture_validation_task",
    bind=True,
    max_retries=1,
)
def evidence_fixture_validation_task(
    self, run_id: str, org_id: str, dataset_id: str, version: str
) -> dict[str, Any]:
    """在 Worker 内执行固定版本 Fixture 的全部七项断言校验。

    Args:
        self: Celery 任务实例（bind=True），用于任务生命周期控制。
        run_id: 本次校验运行 ID，用于登记运行状态与结果定位。
        org_id: 组织 ID，限定校验的数据与语料范围。
        dataset_id: 待校验的评测数据集 ID。
        version: 数据集的固定版本号。
    """
    root = BACKEND_ROOT
    workspace = EvidenceReviewWorkspace(
        EVIDENCE_REVIEW_ROOT,
        EVIDENCE_GATES_DATA_ROOT,
    )
    catalog = PostgreSQLDocumentCatalogRepository(get_database_service())
    service = FixtureValidationService(
        workspace,
        finance_department_id=settings.evaluation_fixture_finance_department_id,
        corpus_probe=build_catalog_fixture_corpus_probe(catalog),
    )
    return service.run(org_id=org_id, dataset_id=dataset_id, version=version, run_id=run_id)
