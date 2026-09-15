"""Celery maintenance task implementations with legacy names."""

from __future__ import annotations

from typing import Any

from infrastructure.celery_app import celery_app
from infrastructure.documents.local_uploads import LocalUploadStorage
from infrastructure.tasks.task_registry import update_task_state_safely
from shared.config import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(name="infrastructure.celery_tasks.cleanup_task", bind=True)
def cleanup_task(self) -> dict[str, Any]:
    """Clean up the task."""
    update_task_state_safely(self.request.id, "started")
    logger.info("cleanup_task_started")
    update_task_state_safely(self.request.id, "succeeded")
    return {"status": "completed", "cleaned": 0}


@celery_app.task(name="infrastructure.celery_tasks.cleanup_upload_files", bind=True)
def cleanup_upload_files(self) -> dict[str, Any]:
    """Clean up the upload files."""
    update_task_state_safely(self.request.id, "started")
    logger.info("cleanup_upload_files_started")
    try:
        cleaned = LocalUploadStorage(settings.upload_dir).cleanup_all(
            staging_older_than_seconds=settings.upload_staging_retention_hours * 3600,
            quarantine_older_than_seconds=settings.upload_quarantine_retention_hours * 3600,
        )
    except Exception:
        update_task_state_safely(self.request.id, "failed")
        raise
    total = sum(cleaned.values())
    logger.info(
        "cleanup_upload_files_completed",
        cleaned_staging=cleaned["staging"],
        cleaned_quarantine=cleaned["quarantine"],
    )
    update_task_state_safely(self.request.id, "succeeded")
    return {"status": "completed", "cleaned": total, **cleaned}


@celery_app.task(name="infrastructure.celery_tasks.ping")
def ping() -> str:
    """Check whether the module is available."""
    return "pong"
