"""ATDD for durable task ownership and reserve-before-publish."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from api.routers.tasks import (
    TaskSubmitRequest,
    cancel_task,
    get_task_status,
    list_tasks,
    submit_ingest_task,
)
from domain.identity import Permission, UserContext, UserRole
from infrastructure.security.security_state import (
    MemorySecurityStateBackend,
)
from infrastructure.tasks.task_registry import TaskRegistry, TaskRegistryUnavailableError


def _user() -> UserContext:
    return UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.EDITOR,
        org_id="org-a",
        permissions=[Permission.DOC_WRITE],
    )


def test_registry_state_is_visible_across_clients_and_tenant_scoped() -> None:
    backend = MemorySecurityStateBackend()
    registry_a = TaskRegistry(backend, ttl_seconds=700_000)
    registry_b = TaskRegistry(backend, ttl_seconds=700_000)

    record = registry_a.reserve(
        "task-a",
        org_id="org-a",
        actor_id="user-a",
        kind="ingest",
        file_reference="sha256:file",
    )
    registry_a.update_state("task-a", "queued")

    assert registry_b.get_owned("task-a", "org-a").state == "queued"
    assert registry_b.get_owned("task-a", "org-b") is None
    assert registry_b.list_owned("org-a")[0].task_id == record.task_id


@pytest.mark.asyncio
async def test_valid_looking_unregistered_task_never_reaches_celery() -> None:
    registry = TaskRegistry(MemorySecurityStateBackend())
    celery_result = MagicMock()

    with patch("api.routers.tasks.get_task_registry", return_value=registry), patch(
        "api.routers.tasks.celery_app.AsyncResult",
        celery_result,
    ):
        with pytest.raises(HTTPException) as captured:
            await get_task_status("tsk_valid_looking", _user())

    assert captured.value.status_code == 404
    celery_result.assert_not_called()


@pytest.mark.asyncio
async def test_submit_reserves_before_publish_and_rolls_back_publish_failure() -> None:
    registry = TaskRegistry(MemorySecurityStateBackend())
    order: list[str] = []
    original_reserve = registry.reserve

    def reserve(*args, **kwargs):
        order.append("reserve")
        return original_reserve(*args, **kwargs)

    registry.reserve = reserve  # type: ignore[method-assign]

    def publish(*_args, **_kwargs):
        order.append("publish")
        raise RuntimeError("redis://broker-secret")

    with patch("api.routers.tasks.get_task_registry", return_value=registry), patch(
        "api.routers.tasks._resolve_task_file",
        return_value="/srv/uploads/tenants/org-a/accepted/a.pdf",
    ), patch(
        "api.routers.tasks.ingest_document_task.apply_async",
        side_effect=publish,
    ):
        with pytest.raises(HTTPException) as captured:
            await submit_ingest_task(TaskSubmitRequest(file_path="a.pdf"), _user())

    assert order == ["reserve", "publish"]
    assert captured.value.status_code == 503
    assert captured.value.detail["code"] == "task_dispatch_unavailable"
    assert "private" not in str(captured.value.detail)
    assert registry.list_owned("org-a") == []


@pytest.mark.asyncio
async def test_list_and_cancel_use_registry_before_celery() -> None:
    registry = TaskRegistry(MemorySecurityStateBackend())
    registry.reserve(
        "task-a",
        org_id="org-a",
        actor_id="user-a",
        kind="ingest",
        file_reference="sha256:file",
    )
    revoke = MagicMock()

    with patch("api.routers.tasks.get_task_registry", return_value=registry), patch(
        "api.routers.tasks.celery_app.control.revoke",
        revoke,
    ), patch("api.routers.tasks.celery_app.control.inspect") as inspect:
        listed = await list_tasks(limit=10, user=_user())
        cancelled = await cancel_task("task-a", _user())

    assert [item.task_id for item in listed] == ["task-a"]
    assert cancelled == {"task_id": "task-a", "status": "REVOKED"}
    assert registry.get_owned("task-a", "org-a").state == "revoked"
    inspect.assert_not_called()
    revoke.assert_called_once_with("task-a", terminate=True)


@pytest.mark.asyncio
async def test_registry_unavailable_fails_before_celery_control() -> None:
    registry = MagicMock()
    registry.get_owned.side_effect = TaskRegistryUnavailableError()
    revoke = MagicMock()

    with patch("api.routers.tasks.get_task_registry", return_value=registry), patch(
        "api.routers.tasks.celery_app.control.revoke",
        revoke,
    ):
        with pytest.raises(HTTPException) as captured:
            await cancel_task("task-a", _user())

    assert captured.value.status_code == 503
    assert captured.value.detail["code"] == "task_registry_unavailable"
    revoke.assert_not_called()
