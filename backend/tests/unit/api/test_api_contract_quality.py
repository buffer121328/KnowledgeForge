"""API boundary regression tests for strict inputs and typed OpenAPI contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI
from pydantic import ValidationError

from api.contracts import StrictRequestModel, sanitize_validation_errors
from api.routers.admin import admin_router, health_router
from api.routers.apikeys import (
    APIKeyCreateRequest,
    APIKeyRotateRequest,
    APIKeyScopeUpdateRequest,
    router as apikey_router,
)
from api.routers.audit import router as audit_router
from api.routers.auth import LoginRequest, RefreshRequest, RegisterRequest, router as auth_router
from api.routers.documents_ingest import ingest_router
from api.routers.documents_read import docs_router
from api.routers.evaluation_pkg import evaluation_router
from api.routers.knowledge_graph import graph_router
from api.routers.qa import qa_router
from api.routers.roles import PermissionListRequest, RoleCreateRequest, RoleUpdateRequest, router as role_router
from api.routers.tasks import BatchTaskSubmitRequest, TaskSubmitRequest, UpdateTaskRequest, router as tasks_router
from api.routers.users import (
    BatchRoleRequest,
    PasswordChangeRequest,
    PasswordResetRequest,
    UserCreateRequest,
    UserUpdateRequest,
    router as user_router,
)
from api.routers.webhooks import WebhookCreateRequest, router as webhooks_router
from api.schemas import (
    EvidenceBulkActionRequest,
    EvidenceCaseMutationRequest,
    EvidenceReviewRequest,
    EvidenceRevisionRequest,
    QAConversationCreateRequest,
    QAFeedbackRequest,
    QuestionRequest,
    SemanticCacheDecisionRequest,
    UpdateRequest,
)
from domain.identity import Permission
from infrastructure.webhooks.models import WebhookEvent


VALID_REQUESTS: list[tuple[type[StrictRequestModel], dict[str, Any]]] = [
    (LoginRequest, {"username": "cheng", "password": "safe-password"}),
    (RegisterRequest, {"username": "new_user", "password": "safe-password"}),
    (RefreshRequest, {"refresh_token": "r" * 32}),
    (UserCreateRequest, {"username": "new_user", "password": "safe-password", "role": "viewer"}),
    (UserUpdateRequest, {"display_name": "Cheng", "department_id": "dept_1"}),
    (PasswordChangeRequest, {"old_password": "old-password", "new_password": "new-password"}),
    (PasswordResetRequest, {"new_password": "new-password"}),
    (BatchRoleRequest, {"user_ids": ["user_1"], "role": "editor"}),
    (RoleCreateRequest, {"name": "qa_reviewer", "display_name": "QA Reviewer", "permissions": [Permission.QA_QUERY.value]}),
    (RoleUpdateRequest, {"description": "Reviews QA evidence"}),
    (PermissionListRequest, {"permissions": [Permission.QA_QUERY.value]}),
    (APIKeyCreateRequest, {"name": "automation", "permissions": [Permission.QA_QUERY.value], "expires_days": 30}),
    (APIKeyRotateRequest, {"expires_days": 30}),
    (APIKeyScopeUpdateRequest, {"permissions": [Permission.QA_QUERY.value]}),
    (TaskSubmitRequest, {"file_path": "tenant/uploads/document.pdf"}),
    (BatchTaskSubmitRequest, {"file_paths": ["tenant/uploads/document.pdf"]}),
    (UpdateTaskRequest, {"file_path": "tenant/uploads/document.pdf", "change_type": "modified"}),
    (QuestionRequest, {"question": "What is the policy?", "retrieval_mode": "hybrid"}),
    (QAConversationCreateRequest, {"title": "Policy review"}),
    (QAFeedbackRequest, {"rating": "up", "note": "helpful"}),
    (SemanticCacheDecisionRequest, {"question": "What is the policy?", "confirmation_token": "c" * 32}),
    (WebhookCreateRequest, {"url": "https://example.com/hook", "events": [next(iter(WebhookEvent)).value], "secret": "s" * 16}),
    (EvidenceCaseMutationRequest, {"expected_revision": 1, "case": {"case_id": "case_1"}}),
    (EvidenceBulkActionRequest, {"expected_revision": 1, "selection_mode": "selected", "case_ids": ["case_1"]}),
    (EvidenceRevisionRequest, {"expected_revision": 1}),
    (EvidenceReviewRequest, {"expected_revision": 1, "decision": "approve", "reason": "verified"}),
    (UpdateRequest, {"file_path": "tenant/uploads/document.pdf", "change_type": "modified"}),
]


@pytest.mark.parametrize(("model", "payload"), VALID_REQUESTS)
def test_json_mutation_models_accept_current_payloads_and_reject_unknown_fields(
    model: type[StrictRequestModel], payload: dict[str, Any]
) -> None:
    model.model_validate(payload)
    invalid = deepcopy(payload)
    invalid["unexpected_field"] = "must-not-be-ignored"
    with pytest.raises(ValidationError) as error:
        model.model_validate(invalid)
    assert any(item["type"] == "extra_forbidden" for item in error.value.errors())


def _contract_app() -> FastAPI:
    app = FastAPI(title="Contract audit")
    versioned = APIRouter(prefix="/api/v1")
    versioned.include_router(auth_router)
    versioned.include_router(apikey_router, prefix="/auth")
    versioned.include_router(user_router)
    versioned.include_router(role_router)
    versioned.include_router(tasks_router)
    versioned.include_router(webhooks_router)
    versioned.include_router(audit_router)
    versioned.include_router(admin_router)
    versioned.include_router(qa_router)
    versioned.include_router(ingest_router)
    versioned.include_router(docs_router)
    versioned.include_router(graph_router)
    versioned.include_router(evaluation_router)
    app.include_router(health_router)
    app.include_router(versioned)
    return app


def _contains_ref(value: Any) -> bool:
    if isinstance(value, dict):
        return "$ref" in value or any(_contains_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_ref(item) for item in value)
    return False


def _dereference(schema: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in schema:
        schema = components[schema["$ref"].rsplit("/", 1)[-1]]
    return schema


def test_openapi_contract_has_strict_mutations_typed_json_responses_and_bounded_parameters() -> None:
    document = _contract_app().openapi()
    components = document["components"]["schemas"]
    failures: list[str] = []

    for path, path_item in document["paths"].items():
        for method, operation in path_item.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue

            if method in {"post", "put", "patch", "delete"}:
                json_body = operation.get("requestBody", {}).get("content", {}).get("application/json")
                if json_body:
                    body_schema = _dereference(json_body.get("schema", {}), components)
                    if body_schema.get("type") == "object" and body_schema.get("additionalProperties") is not False:
                        failures.append(f"{method.upper()} {path}: permissive JSON body")

            for code, response in operation.get("responses", {}).items():
                if not str(code).startswith("2"):
                    continue
                json_response = response.get("content", {}).get("application/json")
                if json_response is not None and not _contains_ref(json_response.get("schema", {})):
                    failures.append(f"{method.upper()} {path}: unnamed {code} JSON response")

            parameters = [*path_item.get("parameters", []), *operation.get("parameters", [])]
            for parameter in parameters:
                if parameter.get("in") not in {"path", "query"}:
                    continue
                schema = parameter.get("schema", {})
                kind = schema.get("type")
                bounded = {
                    "string": {"minLength", "maxLength", "pattern", "enum"},
                    "integer": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum"},
                    "number": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum"},
                    "array": {"minItems", "maxItems"},
                }.get(kind)
                if bounded and not bounded.intersection(schema):
                    failures.append(
                        f"{method.upper()} {path}: unbounded {parameter['in']} parameter {parameter['name']}"
                    )

    export = document["paths"]["/api/v1/audit/logs/export"]["get"]["responses"]["200"]
    assert export["content"]["text/csv"]["schema"] == {"type": "string", "format": "binary"}
    assert failures == []


def test_validation_error_sanitization_does_not_echo_secret_values() -> None:
    secret = "secret-value-that-must-not-be-returned"
    cleaned = sanitize_validation_errors(
        [{"type": "string_too_short", "loc": ("body", "password"), "msg": "too short", "input": secret, "ctx": {"min_length": 12}}]
    )
    assert cleaned == [{"type": "string_too_short", "loc": ("body", "password"), "msg": "too short"}]
    assert secret not in repr(cleaned)


def test_evidence_bulk_action_request_rejects_ambiguous_selection() -> None:
    with pytest.raises(ValidationError):
        EvidenceBulkActionRequest.model_validate(
            {"expected_revision": 1, "selection_mode": "selected", "case_ids": []}
        )
    with pytest.raises(ValidationError):
        EvidenceBulkActionRequest.model_validate(
            {
                "expected_revision": 1,
                "selection_mode": "selected",
                "case_ids": ["case_1", "case_1"],
            }
        )
    with pytest.raises(ValidationError):
        EvidenceBulkActionRequest.model_validate(
            {
                "expected_revision": 1,
                "selection_mode": "filtered",
                "case_ids": ["case_1"],
            }
        )
