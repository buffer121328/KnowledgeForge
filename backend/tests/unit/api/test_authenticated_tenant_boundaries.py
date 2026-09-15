"""ATDD coverage for authenticated, end-to-end tenant ownership."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from api.middleware.auth import JWTAuthMiddleware
from api.middleware.tenant import TenantMiddleware
from api.routers import tasks as task_router
from auth.config import AuthSettings
from auth.role_service import RoleService
from auth.jwt_service import AuthService
from auth.user_service import IdentityServiceError, UserService
from domain.identity import Permission, UserContext, UserRole
from infrastructure.tasks.celery_ingest_tasks import _run_ingest
from infrastructure.tasks.celery_update_tasks import _run_update
from infrastructure.cache.redis import QACache
from infrastructure.webhooks.models import Webhook, WebhookEvent
from infrastructure.webhooks.stores import MemoryWebhookStore
from infrastructure.webhooks.service import WebhookService
from shared.utils.task_ids import new_task_id, task_belongs_to_org


def _request(path: str = "/api/docs") -> Request:
    app = FastAPI()
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "app": app,
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )


class _UserService:
    def __init__(self, record: dict | None):
        self.record = record

    def find_by_id(self, user_id: str, *, raise_on_missing: bool = True):
        return self.record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record_org,token_org",
    [("org-b", "org-a"), ("", "")],
)
async def test_bearer_auth_rejects_stale_or_empty_organization(
    record_org: str,
    token_org: str,
) -> None:
    settings = AuthSettings(secret_key="tenant-boundary-test-secret")
    auth = AuthService(settings)
    token = auth.create_access_token(
        user_id="user-a",
        username="alice",
        role=UserRole.ADMIN,
        org_id=token_org,
    )
    middleware = JWTAuthMiddleware(
        FastAPI(),
        settings=settings,
        user_service=_UserService(
            {
                "user_id": "user-a",
                "org_id": record_org,
                "is_active": True,
                "token_version": 0,
            }
        ),
    )

    response = await middleware._handle_bearer_token(
        _request(),
        lambda _: Response(status_code=204),
        token,
    )

    assert response.status_code == 401


def test_tenant_header_is_ignored_without_authenticated_context() -> None:
    app = FastAPI()

    @app.get("/")
    async def endpoint(request: Request):
        return {"tenant_id": request.state.tenant_id}

    app.add_middleware(TenantMiddleware)
    response = TestClient(app).get("/", headers={"X-Tenant-Id": "org-forged"})

    assert response.status_code == 200
    assert response.json() == {"tenant_id": ""}


def _users() -> dict[str, dict]:
    return {
        "alice": {
            "user_id": "user-a",
            "username": "alice",
            "password_hash": "x",
            "role": UserRole.ADMIN,
            "org_id": "org-a",
            "department_id": "finance",
            "is_department_manager": True,
            "is_active": True,
            "token_version": 0,
        },
        "bob": {
            "user_id": "user-b",
            "username": "bob",
            "password_hash": "x",
            "role": UserRole.ADMIN,
            "org_id": "org-b",
            "department_id": "human_resources",
            "is_department_manager": True,
            "is_active": True,
            "token_version": 0,
        },
    }


def test_user_service_conceals_cross_org_and_org_change_invalidates_tokens() -> None:
    users = _users()
    service = UserService(users=users)

    assert [item["user_id"] for item in service.list_users(org_id="org-a")] == [
        "user-a"
    ]
    with pytest.raises(IdentityServiceError) as hidden:
        service.find_by_id("user-b", org_id="org-a")
    assert hidden.value.status_code == 404

    service.update_user("user-a", org_id="org-c")
    assert users["alice"]["token_version"] == 1


def test_custom_roles_and_role_users_are_org_scoped() -> None:
    roles = {
        "admin": {
            "role_id": "role_admin",
            "name": "admin",
            "display_name": "Admin",
            "description": "",
            "permissions": [Permission.ADMIN_USER.value],
            "is_builtin": True,
            "user_count": 0,
        }
    }
    service = RoleService(roles=roles, users=_users())
    service.create_role(
        name="reviewer",
        display_name="Reviewer",
        description="",
        permissions=[Permission.DOC_READ.value],
        org_id="org-a",
    )
    service.create_role(
        name="reviewer",
        display_name="Reviewer B",
        description="",
        permissions=[Permission.DOC_READ.value],
        org_id="org-b",
    )

    assert service.find_role("reviewer", org_id="org-a")["display_name"] == "Reviewer"
    assert service.find_role("reviewer", org_id="org-b")["display_name"] == "Reviewer B"
    assert [user["user_id"] for user in service.list_role_users("admin", org_id="org-a")] == [
        "user-a"
    ]


def _webhook(webhook_id: str, org_id: str) -> Webhook:
    return Webhook(
        id=webhook_id,
        org_id=org_id,
        url="https://receiver.example/hook",
        events=[WebhookEvent.DOC_INGESTED],
        secret="safe-test-secret",
        is_active=True,
    )


@pytest.mark.asyncio
async def test_webhook_store_and_delivery_select_only_event_org() -> None:
    store = MemoryWebhookStore()
    org_a = _webhook("wh-a", "org-a")
    org_b = _webhook("wh-b", "org-b")
    legacy = _webhook("wh-legacy", "")
    for item in (org_a, org_b, legacy):
        store.save(item)
    delivery = MagicMock()
    delivery.send_with_retry = AsyncMock(return_value=True)
    service = WebhookService(store=store)
    service._delivery = delivery

    assert service.list_webhooks("org-a") == [org_a]
    assert service.delete("wh-b", "org-a") is False
    assert await service.trigger(
        WebhookEvent.DOC_INGESTED,
        {"doc_id": "doc-a"},
        org_id="org-a",
    ) == 1
    assert delivery.send_with_retry.await_args.args[0] is org_a


def test_qa_cache_key_is_tenant_qualified() -> None:
    cache = QACache(MagicMock())
    org_a = cache._make_key("question", "user-a", "hybrid", "org-a")
    org_b = cache._make_key("question", "user-a", "hybrid", "org-b")
    assert org_a != org_b


def test_task_ids_are_org_namespaced_and_cross_org_is_rejected() -> None:
    task_id = new_task_id("org-a")
    assert task_belongs_to_org(task_id, "org-a") is True
    assert task_belongs_to_org(task_id, "org-b") is False


@pytest.mark.asyncio
async def test_task_submission_passes_tenant_and_server_task_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = MagicMock()
    queued.id = "ignored-celery-id"
    apply_async = MagicMock(return_value=queued)
    registry = MagicMock()
    monkeypatch.setattr(task_router.ingest_document_task, "apply_async", apply_async)
    monkeypatch.setattr(task_router, "get_task_registry", lambda: registry)
    monkeypatch.setattr(
        task_router,
        "_resolve_task_file",
        lambda reference, org_id: f"/accepted/{org_id}/{reference}",
    )
    user = UserContext(
        user_id="user-a",
        username="alice",
        role=UserRole.EDITOR,
        org_id="org-a",
        permissions=[Permission.DOC_WRITE],
    )

    response = await task_router.submit_ingest_task(
        task_router.TaskSubmitRequest(file_path="doc.pdf"),
        user,
    )

    assert task_belongs_to_org(response.task_id, "org-a")
    call = apply_async.call_args
    assert call.kwargs["task_id"] == response.task_id
    assert call.kwargs["args"] == ["/accepted/org-a/doc.pdf", "org-a", "user-a"]
    registry.reserve.assert_called_once()
    reserve_call = registry.reserve.call_args
    assert reserve_call.args == (response.task_id,)
    assert reserve_call.kwargs["org_id"] == "org-a"
    assert reserve_call.kwargs["actor_id"] == "user-a"
    assert reserve_call.kwargs["kind"] == "ingest"
    assert reserve_call.kwargs["file_reference"].startswith("file:v1:")


@pytest.mark.asyncio
async def test_ingest_and_update_helpers_forward_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = AsyncMock(return_value=[])
    vector_add = AsyncMock(return_value=0)
    update_batch = AsyncMock(return_value=[])
    parser_instance = MagicMock()
    parser_instance.parse = parser
    extractor_instance = MagicMock()
    extractor_instance.extract = AsyncMock(return_value=[])
    vector_instance = MagicMock()
    vector_instance.init = AsyncMock()
    vector_instance.add_chunks = vector_add
    graph_instance = MagicMock()
    graph_instance.init = AsyncMock()
    graph_instance.close = AsyncMock()
    graph_instance.upsert_entity = AsyncMock()
    graph_instance.add_relation = AsyncMock()
    updater_instance = MagicMock()
    updater_instance.process_batch = update_batch
    revision = MagicMock(advance=AsyncMock())
    monkeypatch.setattr(
        "agents.document_parser.DocParserAgent",
        MagicMock(return_value=parser_instance),
    )
    monkeypatch.setattr(
        "agents.knowledge_extractor.KnowledgeExtractAgent",
        MagicMock(return_value=extractor_instance),
    )
    monkeypatch.setattr(
        "infrastructure.retrieval.vector_store.VectorStoreService",
        MagicMock(return_value=vector_instance),
    )
    monkeypatch.setattr(
        "infrastructure.graph.neo4j_graph.KnowledgeGraphService",
        MagicMock(return_value=graph_instance),
    )
    monkeypatch.setattr(
        "agents.knowledge_updater.KnowledgeUpdateAgent",
        MagicMock(return_value=updater_instance),
    )
    monkeypatch.setattr(
        "infrastructure.tasks.celery_ingest_tasks.KnowledgeRevisionStore",
        MagicMock(return_value=revision),
    )
    monkeypatch.setattr(
        "infrastructure.tasks.celery_update_tasks.KnowledgeRevisionStore",
        MagicMock(return_value=revision),
    )

    await _run_ingest("/tmp/doc.pdf", tenant_id="org-a")
    await _run_update("/tmp/doc.pdf", "modified", tenant_id="org-a")

    parser.assert_awaited_once_with("/tmp/doc.pdf", tenant_id="org-a")
    assert update_batch.await_args.kwargs["tenant_id"] == "org-a"
    assert revision.advance.await_args_list == [call("org-a")]


@pytest.mark.asyncio
async def test_update_helper_wires_native_sparse_dependencies_before_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Celery updates own all persistence dependencies, including Native BM25."""
    from domain.tasks import ChangeType, DocumentChange, UpdateResult

    parser = MagicMock()
    extractor = MagicMock()
    vector = MagicMock(init=AsyncMock())
    graph = MagicMock(init=AsyncMock(), close=AsyncMock())
    sparse = MagicMock()
    change = DocumentChange(file_path="/tmp/doc.pdf", change_type=ChangeType.MODIFIED)
    updater = MagicMock(
        process_batch=AsyncMock(return_value=[UpdateResult(change=change)])
    )
    updater_factory = MagicMock(return_value=updater)
    revision = MagicMock(advance=AsyncMock())

    monkeypatch.setattr("agents.document_parser.DocParserAgent", MagicMock(return_value=parser))
    monkeypatch.setattr(
        "agents.knowledge_extractor.KnowledgeExtractAgent",
        MagicMock(return_value=extractor),
    )
    monkeypatch.setattr(
        "infrastructure.retrieval.vector_store.VectorStoreService", MagicMock(return_value=vector)
    )
    monkeypatch.setattr(
        "infrastructure.graph.neo4j_graph.KnowledgeGraphService", MagicMock(return_value=graph)
    )
    monkeypatch.setattr(
        "infrastructure.retrieval.bm25_index.NativeBM25Index", MagicMock(return_value=sparse)
    )
    monkeypatch.setattr("agents.knowledge_updater.KnowledgeUpdateAgent", updater_factory)
    monkeypatch.setattr(
        "infrastructure.tasks.celery_update_tasks.KnowledgeRevisionStore",
        MagicMock(return_value=revision),
    )

    await _run_update("/tmp/doc.pdf", "modified", tenant_id="org-a")

    updater_factory.assert_called_once_with(
        doc_parser=parser,
        knowledge_extractor=extractor,
        vector_store=vector,
        knowledge_graph=graph,
        sparse_index=sparse,
    )
    vector.init.assert_awaited_once()
    graph.init.assert_awaited_once()
    graph.close.assert_awaited_once()
    revision.advance.assert_awaited_once_with("org-a")


@pytest.mark.asyncio
async def test_update_helper_does_not_advance_revision_after_branch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sparse/vector/graph update failure is not published as a new revision."""
    from domain.tasks import ChangeType, DocumentChange, UpdateResult

    parser = MagicMock()
    extractor = MagicMock()
    vector = MagicMock(init=AsyncMock())
    graph = MagicMock(init=AsyncMock(), close=AsyncMock())
    sparse = MagicMock()
    change = DocumentChange(file_path="/tmp/doc.pdf", change_type=ChangeType.MODIFIED)
    updater = MagicMock(
        process_batch=AsyncMock(
            return_value=[
                UpdateResult(
                    change=change,
                    success=False,
                    error="sparse snapshot promotion failed",
                )
            ]
        )
    )
    revision = MagicMock(advance=AsyncMock())

    monkeypatch.setattr("agents.document_parser.DocParserAgent", MagicMock(return_value=parser))
    monkeypatch.setattr(
        "agents.knowledge_extractor.KnowledgeExtractAgent",
        MagicMock(return_value=extractor),
    )
    monkeypatch.setattr(
        "infrastructure.retrieval.vector_store.VectorStoreService", MagicMock(return_value=vector)
    )
    monkeypatch.setattr(
        "infrastructure.graph.neo4j_graph.KnowledgeGraphService", MagicMock(return_value=graph)
    )
    monkeypatch.setattr(
        "infrastructure.retrieval.bm25_index.NativeBM25Index", MagicMock(return_value=sparse)
    )
    monkeypatch.setattr(
        "agents.knowledge_updater.KnowledgeUpdateAgent", MagicMock(return_value=updater)
    )
    monkeypatch.setattr(
        "infrastructure.tasks.celery_update_tasks.KnowledgeRevisionStore",
        MagicMock(return_value=revision),
    )

    with pytest.raises(RuntimeError, match="knowledge_update_failed"):
        await _run_update("/tmp/doc.pdf", "modified", tenant_id="org-a")

    revision.advance.assert_not_awaited()
    graph.close.assert_awaited_once()
