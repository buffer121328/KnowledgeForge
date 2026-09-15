"""Celery 任务队列配置

约定:
  - broker:   Redis DB 2
  - backend:  Redis DB 3
  - 任务超时: 100 分钟硬超时，90 分钟软超时
  - 重试:     最多 3 次，指数退避

启动 worker:
  celery -A infrastructure.celery_app worker --loglevel=info
"""

from __future__ import annotations

import os

from celery import Celery

# ── 配置 ───────────────────────────────────────────────────────

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/2")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/3")
# This must exceed the 100-minute hard limit to avoid redelivering a live task.
REDIS_VISIBILITY_TIMEOUT_SECONDS = 6600

celery_app = Celery(
    main="agenthub",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

celery_app.conf.update(
    # 序列化
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # 时区
    timezone="UTC",
    enable_utc=True,
    # 执行追踪
    task_track_started=True,
    # A worker interruption must return unacknowledged Redis messages promptly.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_transport_options={
        "visibility_timeout": REDIS_VISIBILITY_TIMEOUT_SECONDS,
    },
    # 超时
    task_time_limit=6000,        # 硬超时 100 分钟
    task_soft_time_limit=5400,   # 软超时 90 分钟
    # 重试
    task_default_max_retries=3,
    task_default_retry_delay=60,  # 60 秒后重试
    # 持久化
    result_extended=True,
    result_expires=7 * 24 * 3600,  # 结果保留 7 天
    # 工人抢占
    worker_prefetch_multiplier=1,  # 长任务场景下避免一个 worker 囤积多个任务
    # 任务发现
    include=["infrastructure.celery_tasks"],
    task_default_queue="default",
    task_routes={
        "infrastructure.celery_tasks.ingest_document_task": {
            "queue": "parser",
        },
        "infrastructure.celery_tasks.batch_ingest_task": {
            "queue": "parser",
        },
        "infrastructure.celery_tasks.knowledge_update_task": {
            "queue": "update",
        },
        "infrastructure.celery_tasks.cleanup_task": {
            "queue": "maintenance",
        },
        "infrastructure.celery_tasks.cleanup_upload_files": {
            "queue": "maintenance",
        },
        "infrastructure.celery_tasks.evidence_release_workflow_task": {
            "queue": "evaluation",
        },
    },
    task_annotations={
        "infrastructure.celery_tasks.ingest_document_task": {
            "time_limit": 900,
            "soft_time_limit": 840,
        },
        "infrastructure.celery_tasks.batch_ingest_task": {
            "time_limit": 1800,
            "soft_time_limit": 1740,
        },
        "infrastructure.celery_tasks.evidence_release_workflow_task": {
            "time_limit": 12 * 60 * 60 + 30 * 60,
            "soft_time_limit": 12 * 60 * 60,
        },
    },
)


def get_app() -> Celery:
    """获取 Celery app 实例"""
    return celery_app
