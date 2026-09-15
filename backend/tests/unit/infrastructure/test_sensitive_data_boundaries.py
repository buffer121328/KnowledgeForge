"""ATDD acceptance tests for secondary sensitive-data boundaries."""

from __future__ import annotations

import logging

import pytest

from api.routers import tasks as tasks_router
from auth.token_blacklist import TokenBlacklist
from domain.documents import IngestDocumentInput
from domain.identity import UserContext, UserRole
from infrastructure.audit.log import AuditAction, AuditService
from infrastructure.audit.stores import MemoryAuditStore
from infrastructure.tasks import celery_ingest_tasks, celery_update_tasks
from infrastructure.cache.redis import CacheService
from shared.utils.task_ids import new_task_id
from shared.utils.logging import (
    SensitiveDataFilter,
    safe_file_reference,
    safe_fingerprint,
    sanitize_for_observability,
)


def test_recursive_sanitizer_redacts_fields_embedded_values_and_bounds_collections():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signaturevalue"
    raw = {
        "request_id": "req-123",
        "org_id": "org-123",
        "task_id": "tsk-123",
        "error_type": "RuntimeError",
        "count": 4,
        "password": "correct horse battery staple",
        "nested": {
            "authorization": f"Bearer {jwt}",
            "file_path": "/private/tenant/acme/secret.pdf",
            "message": f"provider rejected {jwt}",
        },
        "items": list(range(100)),
    }

    sanitized = sanitize_for_observability(raw, max_collection_items=10)
    rendered = repr(sanitized)

    assert sanitized["request_id"] == "req-123"
    assert sanitized["org_id"] == "org-123"
    assert sanitized["task_id"] == "tsk-123"
    assert sanitized["error_type"] == "RuntimeError"
    assert sanitized["count"] == 4
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["nested"]["authorization"] == "[REDACTED]"
    assert sanitized["nested"]["file_path"] == "[REDACTED]"
    assert jwt not in rendered
    assert "secret.pdf" not in rendered
    assert sanitized["items"][-1] == "[TRUNCATED]"


def test_standard_logging_filter_removes_exception_and_sensitive_arguments():
    secret = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1In0.signaturevalue"
    try:
        raise RuntimeError(f"provider failed with {secret}")
    except RuntimeError:
        exc_info = __import__("sys").exc_info()

    record = logging.LogRecord(
        name="legacy",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="operation failed: %s",
        args=(secret,),
        exc_info=exc_info,
    )
    record.stack_info = f"stack contains {secret}"

    assert SensitiveDataFilter().filter(record) is True
    assert secret not in record.getMessage()
    assert record.exc_info is None
    assert record.stack_info is None
    assert record.error_type == "RuntimeError"


def test_audit_service_sanitizes_extensible_metadata_before_store():
    store = MemoryAuditStore()
    service = AuditService(store)
    token = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1In0.signaturevalue"

    log = service.log(
        user_id="user-1",
        org_id="org-1",
        action=AuditAction.DOC_UPLOAD,
        metadata={
            "target_id": "doc-123",
            "operation": "upload",
            "request_body": {"question": "confidential acquisition"},
            "token_hint": token,
            "file_path": "/srv/private/acme.pdf",
            "connection": "postgresql://admin:password@db.internal/private",
            "exception": "provider secret detail",
        },
    )

    rendered = repr(log.metadata)
    assert log.metadata["target_id"] == "doc-123"
    assert log.metadata["operation"] == "upload"
    assert token not in rendered
    assert "confidential acquisition" not in rendered
    assert "acme.pdf" not in rendered
    assert "password@db.internal" not in rendered
    assert "provider secret detail" not in rendered


class _RecordingRedis:
    def __init__(self, *, fail: bool = False) -> None:
        self.keys: list[str] = []
        self.fail = fail

    async def get(self, key: str):
        self.keys.append(key)
        if self.fail:
            raise RuntimeError("redis://admin:secret@cache.internal/0")
        return None

    async def set(self, key: str, value: str, *, ex: int):
        self.keys.append(key)
        return True


@pytest.mark.asyncio
async def test_token_blacklist_uses_only_versioned_token_fingerprint():
    redis = _RecordingRedis()
    blacklist = TokenBlacklist(redis_client=redis)
    token = "header.payload.signature-with-sensitive-material"

    await blacklist.contains(token)
    await blacklist.add(token, 4_102_444_800)

    assert len(redis.keys) == 2
    assert redis.keys[0] == redis.keys[1]
    assert redis.keys[0].startswith("token:blacklist:v2:")
    assert token not in redis.keys[0]


@pytest.mark.asyncio
async def test_cache_failure_logs_only_key_fingerprint_and_error_type(monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, dict]] = []

    class _Logger:
        def warning(self, event: str, **kwargs):
            events.append((event, kwargs))

    cache = CacheService.__new__(CacheService)
    cache.redis = _RecordingRedis(fail=True)
    monkeypatch.setattr("infrastructure.cache.redis.logger", _Logger())
    raw_key = "qa:cache:customer-secret-question"

    assert await cache.get(raw_key) is None
    assert events == [
        (
            "cache_get_failed",
            {
                "key_fingerprint": safe_fingerprint(raw_key, namespace="cache-key"),
                "error_type": "RuntimeError",
            },
        )
    ]
    assert raw_key not in repr(events)


def test_file_reference_is_deterministic_and_secret_free():
    path = "/srv/uploads/tenants/acme/contracts/merger-secret.pdf"

    assert safe_file_reference(path) == safe_file_reference(path)
    assert safe_file_reference(path).startswith("file:v1:")
    assert "acme" not in safe_file_reference(path)
    assert "merger-secret.pdf" not in safe_file_reference(path)


def test_batch_ingest_failure_result_and_logs_are_secret_free(monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, dict]] = []
    path = "/srv/uploads/tenants/acme/contracts/merger-secret.pdf"

    class _Logger:
        def info(self, event: str, **kwargs):
            events.append((event, kwargs))

    async def _fail(_file_path: str, tenant_id: str):
        raise RuntimeError("provider token=secret and internal path")

    monkeypatch.setattr(celery_ingest_tasks, "logger", _Logger())
    monkeypatch.setattr(celery_ingest_tasks, "_run_ingest", _fail)

    result = celery_ingest_tasks.batch_ingest_task.run([path], "org-1", "user-1")

    assert result["status"] == "partial"
    assert result["results"][0]["error"] == "task_execution_failed"
    assert result["results"][0]["error_type"] == "RuntimeError"
    assert path not in repr(result)
    assert "token=secret" not in repr(result)
    assert path not in repr(events)


def test_batch_ingest_soft_timeout_recovers_active_and_unstarted_documents(monkeypatch: pytest.MonkeyPatch):
    from billiard.exceptions import SoftTimeLimitExceeded
    from dataclasses import asdict

    active = IngestDocumentInput(
        doc_id="doc-active", tenant_id="org-1", company_id="org-1", department_id="finance",
        file_path="/srv/uploads/active.docx", uploaded_filename="active.docx", display_name="active",
        provenance_source_filename="active.docx", relative_path="finance/active.docx", folder_path="finance",
        content_sha256="a",
    )
    pending = IngestDocumentInput(
        doc_id="doc-pending", tenant_id="org-1", company_id="org-1", department_id="finance",
        file_path="/srv/uploads/pending.docx", uploaded_filename="pending.docx", display_name="pending",
        provenance_source_filename="pending.docx", relative_path="finance/pending.docx", folder_path="finance",
        content_sha256="b",
    )
    recovered: list[tuple[str, bool]] = []

    class _Lifecycle:
        async def process_accepted_document(self, *_args, **_kwargs):
            raise SoftTimeLimitExceeded()

        async def recover_interrupted_document(self, document_input, *, cleanup_partial_artifacts: bool):
            recovered.append((document_input.doc_id, cleanup_partial_artifacts))
            return True

    states: list[str] = []
    monkeypatch.setattr(celery_ingest_tasks, "_worker_lifecycle", lambda **_kwargs: _Lifecycle())
    monkeypatch.setattr(celery_ingest_tasks, "update_task_state_safely", lambda _task_id, state: states.append(state))

    with pytest.raises(SoftTimeLimitExceeded):
        celery_ingest_tasks.batch_ingest_task.run(
            [active.file_path, pending.file_path],
            "org-1",
            "user-1",
            [asdict(active), asdict(pending)],
        )

    assert recovered == [("doc-active", True), ("doc-pending", False)]
    assert states == ["started", "failed"]


def test_update_retry_uses_controlled_exception_and_safe_log(monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, dict]] = []
    path = "/srv/uploads/tenants/acme/contracts/private-plan.docx"

    class _Logger:
        def info(self, event: str, **kwargs):
            events.append((event, kwargs))

        def error(self, event: str, **kwargs):
            events.append((event, kwargs))

    async def _fail(_file_path: str, _change_type: str, *, tenant_id: str):
        raise RuntimeError("postgresql://admin:secret@db.internal/private")

    monkeypatch.setattr(celery_update_tasks, "logger", _Logger())
    monkeypatch.setattr(celery_update_tasks, "_run_update", _fail)
    monkeypatch.setattr(
        celery_update_tasks.knowledge_update_task,
        "retry",
        lambda **kwargs: kwargs["exc"],
    )

    with pytest.raises(RuntimeError) as captured:
        celery_update_tasks.knowledge_update_task.run(
            path,
            "modified",
            "org-1",
            "user-1",
        )

    assert str(captured.value) == "task_execution_failed:RuntimeError"
    assert path not in repr(events)
    assert "admin:secret" not in repr(events)
    assert events[-1][1]["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_task_status_returns_opaque_failure(monkeypatch: pytest.MonkeyPatch):
    class _FailedResult:
        status = "FAILURE"
        result = RuntimeError(
            "provider token=secret at /srv/uploads/tenants/acme/private.pdf"
        )

        @staticmethod
        def ready() -> bool:
            return True

        @staticmethod
        def successful() -> bool:
            return False

    user = UserContext(
        user_id="user-1",
        username="owner",
        role=UserRole.ADMIN,
        org_id="org-1",
    )
    task_id = new_task_id(user.org_id)
    monkeypatch.setattr(tasks_router.celery_app, "AsyncResult", lambda _task_id: _FailedResult())

    response = await tasks_router.get_task_status(task_id, user)

    assert response.error == "task_execution_failed"
    assert "provider" not in repr(response)
    assert "private.pdf" not in repr(response)
