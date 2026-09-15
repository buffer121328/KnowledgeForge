from __future__ import annotations

from domain.evidence import AnswerCitation
from domain.identity import Permission, UserContext, UserRole
from auth.config import AuthSettings
from auth.jwt_service import AuthService
from infrastructure.tasks.task_registry import TaskRecord
from api.routers.admin import _summarize_request_details
from services.documents.lifecycle import DocumentLifecycleCoordinator


def test_department_manager_can_manage_only_own_department_documents() -> None:
    can_manage_document = DocumentLifecycleCoordinator.can_manage_document
    manager = UserContext(
        user_id="manager-a",
        username="manager-a",
        role=UserRole.ADMIN,
        org_id="org-a",
        permissions=[Permission.DOC_READ, Permission.DOC_DELETE],
        department_id="finance",
        is_department_manager=True,
    )

    assert can_manage_document(manager, "finance") is True
    assert can_manage_document(manager, "hr") is False
    assert can_manage_document(manager, None) is False

    organization_admin = UserContext(
        user_id="admin-a",
        username="admin-a",
        role=UserRole.ORGANIZATION_ADMIN,
        org_id="org-a",
        permissions=list(Permission),
        department_id="finance",
        is_department_manager=True,
    )
    assert can_manage_document(organization_admin, "hr") is True


def test_task_record_exposes_human_readable_description() -> None:
    record = TaskRecord(
        task_id="task-a",
        org_id="org-a",
        actor_id="user-a",
        kind="knowledge_update",
        state="queued",
        file_reference="finance/policy.docx",
        created_at="2026-08-07T00:00:00+00:00",
        updated_at="2026-08-07T00:00:00+00:00",
    )

    assert record.description == "知识更新：finance/policy.docx"


def test_department_manager_token_receives_department_document_permissions() -> None:
    service = AuthService(AuthSettings(secret_key="test-secret-key-with-enough-diversity-123456789"))
    token = service.create_access_token(
        user_id="manager-a",
        username="manager-a",
        role=UserRole.ADMIN,
        org_id="org-a",
        department_id="finance",
        is_department_manager=True,
    )
    payload = service.decode_token(token)
    assert payload is not None
    assert Permission.DOC_DELETE.value in payload.permissions
    assert Permission.DOC_WRITE.value in payload.permissions


def test_request_summary_distinguishes_ai_and_system_api() -> None:
    summary = _summarize_request_details([
        {
            "details": {
                "ai": {"count": 2, "error_count": 1, "total_latency_ms": 300, "methods": {"POST": 2}, "routes": {"/api/v1/qa/ask": {"count": 2, "error_count": 1, "last_status": 200}}},
                "system_api": {"count": 3, "error_count": 0, "total_latency_ms": 90, "methods": {"GET": 3}, "routes": {"/api/v1/docs": {"count": 3, "error_count": 0, "last_status": 200}}},
            }
        }
    ])

    assert summary["ai"]["count"] == 2
    assert summary["ai"]["routes"]["/api/v1/qa/ask"]["error_count"] == 1
    assert summary["system_api"]["methods"] == {"GET": 3}


def test_legacy_citation_provenance_fields_are_optional() -> None:
    citation = AnswerCitation(citation_id="cite-a", context_id="ctx-a", source="policy.md", content="evidence")
    assert citation.document_id == ""
    assert citation.chunk_id == ""
