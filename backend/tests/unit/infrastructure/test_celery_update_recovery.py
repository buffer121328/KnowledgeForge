"""Recovery contracts for interrupted Redis-backed update tasks."""

from __future__ import annotations

import pytest

from infrastructure.celery_app import celery_app
from infrastructure.tasks import celery_update_tasks
from infrastructure.tasks.celery_update_tasks import _advance_knowledge_revision, _has_retry_remaining
from infrastructure.security.security_state import MemorySecurityStateBackend
from infrastructure.tasks.task_registry import TaskRegistry


def test_redis_broker_uses_bounded_visibility_recovery() -> None:
    """A stopped consumer must not leave an update task invisible indefinitely."""
    assert celery_app.conf.broker_transport_options["visibility_timeout"] == 6600
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True


def test_retrying_task_registry_state_is_non_terminal() -> None:
    registry = TaskRegistry(MemorySecurityStateBackend(), ttl_seconds=700_000)
    registry.reserve(
        "task-1",
        org_id="org-a",
        actor_id="user-a",
        kind="knowledge_update",
    )

    record = registry.update_state("task-1", "retrying")

    assert record is not None
    assert record.state == "retrying"


def test_retry_budget_only_exhausts_after_final_attempt() -> None:
    assert _has_retry_remaining(retries=0, max_retries=3) is True
    assert _has_retry_remaining(retries=2, max_retries=3) is True
    assert _has_retry_remaining(retries=3, max_retries=3) is False


def test_task_marks_retrying_before_requesting_another_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states: list[str] = []

    async def fail_update(*_args, **_kwargs):
        raise RuntimeError("transient failure")

    class RetryRequested(Exception):
        pass

    monkeypatch.setattr(celery_update_tasks, "_run_update", fail_update)
    monkeypatch.setattr(
        celery_update_tasks,
        "update_task_state_safely",
        lambda _task_id, state: states.append(state),
    )

    task = celery_update_tasks.knowledge_update_task
    monkeypatch.setattr(task, "retry", lambda **_kwargs: RetryRequested())
    task.push_request(id="task-1", retries=0)
    try:
        with pytest.raises(RetryRequested):
            task.run("safe.docx", "modified", "org-1")
    finally:
        task.pop_request()

    assert states == ["started", "retrying"]


def test_task_marks_failed_only_after_retry_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states: list[str] = []

    async def fail_update(*_args, **_kwargs):
        raise RuntimeError("terminal failure")

    monkeypatch.setattr(celery_update_tasks, "_run_update", fail_update)
    monkeypatch.setattr(
        celery_update_tasks,
        "update_task_state_safely",
        lambda _task_id, state: states.append(state),
    )

    task = celery_update_tasks.knowledge_update_task
    task.push_request(id="task-1", retries=3)
    try:
        with pytest.raises(RuntimeError, match="terminal failure"):
            task.run("safe.docx", "modified", "org-1")
    finally:
        task.pop_request()

    assert states == ["started", "failed"]


@pytest.mark.asyncio
async def test_revision_cache_is_closed_with_the_update_task_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    class TaskCache:
        async def close(self) -> None:
            closed.append(True)

    class RevisionStore:
        def __init__(self, cache: TaskCache) -> None:
            assert isinstance(cache, TaskCache)

        async def advance(self, tenant_id: str) -> int:
            assert tenant_id == "org-1"
            return 7

    monkeypatch.setattr(celery_update_tasks, "CacheService", lambda *_args, **_kwargs: TaskCache())
    monkeypatch.setattr(celery_update_tasks, "KnowledgeRevisionStore", RevisionStore)

    assert await _advance_knowledge_revision("org-1") == 7
    assert closed == [True]
