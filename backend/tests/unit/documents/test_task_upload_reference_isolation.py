"""ATDD for tenant-owned accepted-upload references in task submission."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from api.routers.tasks import (
    BatchTaskSubmitRequest,
    TaskSubmitRequest,
    submit_batch_task,
    submit_ingest_task,
)
from domain.identity import Permission, UserContext, UserRole
from infrastructure.documents.local_uploads import LocalUploadStorage
from infrastructure.security.security_state import MemorySecurityStateBackend
from infrastructure.tasks.task_registry import TaskRegistry
from shared.config import settings


def _user(org_id: str = "org-a") -> UserContext:
    return UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.EDITOR,
        org_id=org_id,
        permissions=[Permission.DOC_WRITE],
    )


def _accepted(storage: LocalUploadStorage, org_id: str, name: str = "policy.txt") -> tuple[str, str]:
    staged = storage.stage(
        tenant_id=org_id,
        original_filename=name,
        source=BytesIO(b"policy"),
        max_file_bytes=1024,
        tenant_quota_bytes=4096,
    )
    path = storage.promote(staged)
    return storage.accepted_reference(path, org_id), path


@pytest.mark.asyncio
async def test_submit_resolves_tenant_reference_before_dispatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    storage = LocalUploadStorage(str(tmp_path))
    reference, resolved = _accepted(storage, "org-a")
    registry = TaskRegistry(MemorySecurityStateBackend())
    publish = MagicMock()

    with patch("api.routers.tasks.get_task_registry", return_value=registry), patch(
        "api.routers.tasks.ingest_document_task.apply_async", publish
    ):
        response = await submit_ingest_task(TaskSubmitRequest(file_path=reference), _user())

    assert response.file_path == reference
    dispatched = publish.call_args.kwargs
    assert dispatched["args"] == [resolved, "org-a", "user-a"]
    assert registry.get_owned(response.task_id, "org-a") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["/etc/passwd", "../policy.txt", "missing.txt"])
async def test_submit_rejects_untrusted_or_missing_reference_before_reservation(
    tmp_path, monkeypatch, reference: str
) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    registry = TaskRegistry(MemorySecurityStateBackend())
    publish = MagicMock()

    with patch("api.routers.tasks.get_task_registry", return_value=registry), patch(
        "api.routers.tasks.ingest_document_task.apply_async", publish
    ):
        with pytest.raises(HTTPException) as captured:
            await submit_ingest_task(TaskSubmitRequest(file_path=reference), _user())

    assert captured.value.status_code == 404
    assert captured.value.detail["code"] == "task_document_not_found"
    assert registry.list_owned("org-a") == []
    publish.assert_not_called()


@pytest.mark.asyncio
async def test_batch_rejects_all_when_one_reference_is_cross_tenant(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    storage = LocalUploadStorage(str(tmp_path))
    own_reference, _ = _accepted(storage, "org-a", "own.txt")
    other_reference, _ = _accepted(storage, "org-b", "other.txt")
    registry = TaskRegistry(MemorySecurityStateBackend())
    publish = MagicMock()

    with patch("api.routers.tasks.get_task_registry", return_value=registry), patch(
        "api.routers.tasks.batch_ingest_task.apply_async", publish
    ):
        with pytest.raises(HTTPException) as captured:
            await submit_batch_task(
                BatchTaskSubmitRequest(file_paths=[own_reference, other_reference]),
                _user(),
            )

    assert captured.value.detail["code"] == "task_document_not_found"
    assert registry.list_owned("org-a") == []
    publish.assert_not_called()
