"""Compatibility and discovery facade for partitioned Celery task modules."""

from __future__ import annotations

from infrastructure.tasks.celery_ingest_tasks import _run_ingest, batch_ingest_task, ingest_document_task
from infrastructure.tasks.celery_maintenance_tasks import cleanup_task, cleanup_upload_files, ping
from infrastructure.tasks.celery_evaluation_tasks import evidence_release_workflow_task
from infrastructure.tasks.celery_update_tasks import _run_update, knowledge_update_task

__all__ = [
    "_run_ingest",
    "_run_update",
    "batch_ingest_task",
    "cleanup_task",
    "cleanup_upload_files",
    "evidence_release_workflow_task",
    "ingest_document_task",
    "knowledge_update_task",
    "ping",
]
