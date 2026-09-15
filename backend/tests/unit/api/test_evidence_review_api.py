"""API acceptance tests for the evidence dataset governance workbench."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_current_user, get_document_read
from api.routers.evaluation_pkg import evaluation_router
from domain.identity import Permission, ROLE_PERMISSIONS, UserContext, UserRole
from evaluation.evidence_gate.review_workspace import EvidenceReviewWorkspace


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_BUNDLE = PROJECT_ROOT / "backend/evaluation/data/evidence-gates"


def _user(
    user_id: str = "org-admin-a",
    role: UserRole = UserRole.ORGANIZATION_ADMIN,
    *,
    org_id: str = "org-a",
    department_id: str | None = "human_resources",
    is_department_manager: bool = False,
) -> UserContext:
    return UserContext(
        user_id=user_id,
        username=user_id,
        role=role,
        org_id=org_id,
        permissions=list(Permission) if role == UserRole.ORGANIZATION_ADMIN else list(ROLE_PERMISSIONS[role]),
        department_id=department_id,
        is_department_manager=is_department_manager,
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    app = FastAPI()
    app.include_router(evaluation_router)
    app.dependency_overrides[get_current_user] = lambda: _user()
    service = EvidenceReviewWorkspace(tmp_path / "reviews", SOURCE_BUNDLE)
    monkeypatch.setattr("api.routers.evaluation_pkg._common.REVIEW_WORKSPACE", service)
    class ReviewerDirectory:
        def list_users(self, **_kwargs):
            return [
                {
                    "user_id": "org-admin-a",
                    "username": "org-admin-a",
                    "display_name": "公司管理员 A",
                    "role": UserRole.ORGANIZATION_ADMIN,
                    "org_id": "org-a",
                    "department_id": "human_resources",
                    "is_department_manager": True,
                    "is_active": True,
                },
                {
                    "user_id": "manager-b",
                    "username": "manager-b",
                    "display_name": "人力资源审核人",
                    "role": UserRole.ORGANIZATION_ADMIN,
                    "org_id": "org-a",
                    "department_id": "human_resources",
                    "is_department_manager": True,
                    "is_active": True,
                },
                {
                    "user_id": "finance-manager",
                    "username": "finance-manager",
                    "display_name": "财务审核人",
                    "role": UserRole.ADMIN,
                    "org_id": "org-a",
                    "department_id": "finance",
                    "is_department_manager": False,
                    "is_active": True,
                },
            ]
    monkeypatch.setattr("api.routers.evaluation_pkg._common.USER_SERVICE", ReviewerDirectory())
    return TestClient(app)


def test_employee_cannot_access_dataset_governance(client: TestClient) -> None:
    client.app.dependency_overrides[get_current_user] = lambda: _user(role=UserRole.EDITOR)
    assert client.get("/evaluation/datasets").status_code == 403
    assert client.get("/evaluation/datasets/evidence-gates-v1/fixtures").status_code == 403


def test_organization_admin_lists_dataset_and_paginated_cases(client: TestClient) -> None:
    datasets = client.get("/evaluation/datasets")
    assert datasets.status_code == 200
    assert datasets.json()["datasets"][0]["case_count"] == 100

    cases = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 5, "review_status": "draft"},
    )
    assert cases.status_code == 200
    assert cases.json()["total"] == 100
    assert len(cases.json()["cases"]) == 5
    assert len(cases.json()["cases"][0]["context_summaries"][0]["content_excerpt"]) <= 320


def test_organization_admin_rebinds_the_formal_suite_to_tenant_runtime_chunks(
    client: TestClient,
) -> None:
    from api.routers.evaluation_pkg import _common as router
    from api.routers.evaluation_pkg.cases import review_evidence_case  # noqa: F401
    from api.schemas import EvidenceReviewRequest  # noqa: F401

    router.review_evidence_case = review_evidence_case
    router.EvidenceReviewRequest = EvidenceReviewRequest

    contexts = [
        json.loads(line)
        for line in (SOURCE_BUNDLE / "context-catalog.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    chunks_by_document: dict[str, list[dict[str, object]]] = {}
    documents: dict[str, SimpleNamespace] = {}
    for context in contexts:
        document_id = str(context["source_document_id"])
        documents.setdefault(
            document_id,
            SimpleNamespace(
                doc_id=document_id,
                display_name=context["title"],
                provenance_source_filename=context["title"],
                department_id=context["department"],
            ),
        )
        chunks_by_document.setdefault(document_id, []).append(
            {
                "chunk_id": context["context_id"],
                "chunk_index": context["chunk_index"],
                "content": context["content_excerpt"],
            }
        )

    class DocumentRead:
        def list_catalog_documents(self, *, tenant_id: str):
            assert tenant_id == "org-a"
            return list(documents.values())

        async def get_chunks(self, document_id: str, *, tenant_id: str):
            assert tenant_id == "org-a"
            return chunks_by_document[document_id]

    client.app.dependency_overrides[get_document_read] = DocumentRead
    before = client.get("/evaluation/datasets/evidence-gates-v1").json()
    response = client.post(
        "/evaluation/datasets/evidence-gates-v1/rebind-current-corpus",
        json={"expected_revision": before["revision"]},
    )

    assert response.status_code == 200
    assert response.json()["case_count"] == 100
    assert response.json()["counts"]["draft"] == 100
    workspace_path = router.REVIEW_WORKSPACE.workspace_path("org-a", "evidence-gates-v1")
    workspace = json.loads((workspace_path / "workspace.json").read_text(encoding="utf-8"))
    assert workspace["current_corpus_rebound_by"] == "org-admin-a"
    assert workspace["current_corpus_rebound_at"]


def test_non_organization_admin_cannot_rebind_before_runtime_chunks_are_read(
    client: TestClient,
) -> None:
    class DocumentRead:
        def list_catalog_documents(self, **_kwargs):
            raise AssertionError("runtime chunks must not be read for an unauthorized user")

    client.app.dependency_overrides[get_document_read] = DocumentRead
    client.app.dependency_overrides[get_current_user] = lambda: _user(role=UserRole.EDITOR)

    response = client.post(
        "/evaluation/datasets/evidence-gates-v1/rebind-current-corpus",
        json={"expected_revision": 1},
    )

    assert response.status_code == 403


def test_organization_admin_can_read_historical_fixture_profile_but_cannot_manually_validate(client: TestClient) -> None:
    profile = client.get("/evaluation/datasets/evidence-gates-v1/fixtures")
    assert profile.status_code == 200
    body = profile.json()
    assert body["status"] == "historical_only"
    assert body["revision"] >= 1
    assert "version_path" not in body
    assert "tenant_id" not in str(body)
    assert "finance_only_user_against_hr_document" in body["fixtures"]

    removed = client.put(
        "/evaluation/datasets/evidence-gates-v1/fixtures",
        json={"expected_revision": body["revision"], "status": "ready", "fixtures": body["fixtures"]},
    )
    assert removed.status_code == 405


def test_api_user_cannot_access_dataset_workbench(client: TestClient) -> None:
    client.app.dependency_overrides[get_current_user] = lambda: _user(
        role=UserRole.API_USER,
        department_id=None,
    )
    assert client.get("/evaluation/datasets").status_code == 403


def test_edit_submit_and_authenticated_maker_checker_review(client: TestClient) -> None:
    page = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 1},
    ).json()
    case = page["cases"][0]
    edited = client.put(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}",
        json={
            "expected_revision": page["revision"],
            "case": {**case, "notes": "API maker edit"},
        },
    )
    assert edited.status_code == 200
    reviewers = client.get(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}/reviewers"
    )
    assert reviewers.status_code == 200
    assert reviewers.json() == [
        {
            "user_id": "manager-b",
            "username": "manager-b",
            "display_name": "人力资源审核人",
        },
    ]
    ineligible = client.post(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}/submit",
        json={"expected_revision": edited.json()["revision"], "reviewer_id": "unknown-reviewer"},
    )
    assert ineligible.status_code == 422
    assert ineligible.json()["detail"]["code"] == "reviewer_not_eligible"
    submitted = client.post(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}/submit",
        json={"expected_revision": edited.json()["revision"], "reviewer_id": "manager-b"},
    )
    assert submitted.status_code == 200
    assert submitted.json()["case"]["reviewer_display_name"] == "人力资源审核人"

    client.app.dependency_overrides[get_current_user] = lambda: _user(
        is_department_manager=True
    )
    self_review = client.post(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}/review",
        json={
            "expected_revision": submitted.json()["revision"],
            "decision": "approve",
            "reason": "self review",
            "reviewer_id": "forged-user",
        },
    )
    assert self_review.status_code == 422

    client.app.dependency_overrides[get_current_user] = lambda: _user(
        "manager-b",
        role=UserRole.ORGANIZATION_ADMIN,
        is_department_manager=True,
    )
    approved = client.post(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}/review",
        json={
            "expected_revision": submitted.json()["revision"],
            "decision": "approve",
            "reason": "verified",
        },
    )
    assert approved.status_code == 200
    assert approved.json()["case"]["reviewer_id"] == "manager-b"
    assert "reviewer_department_id" not in approved.json()["case"]
    assert approved.json()["case"]["review_status"] == "approved"


def test_employee_and_cross_department_manager_cannot_review(client: TestClient) -> None:
    page = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 1},
    ).json()
    case = page["cases"][0]
    edited = client.put(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}",
        json={"expected_revision": page["revision"], "case": case},
    ).json()
    submitted = client.post(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}/submit",
        json={"expected_revision": edited["revision"], "reviewer_id": "manager-b"},
    ).json()

    client.app.dependency_overrides[get_current_user] = lambda: _user(
        "employee-a", role=UserRole.EDITOR
    )
    employee_review = client.post(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}/review",
        json={
            "expected_revision": submitted["revision"],
            "decision": "approve",
            "reason": "ordinary employee",
        },
    )
    assert employee_review.status_code == 403

    client.app.dependency_overrides[get_current_user] = lambda: _user(
        "hr-manager",
        role=UserRole.VIEWER,
        department_id="finance",
        is_department_manager=True,
    )
    cross_department = client.post(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}/review",
        json={
            "expected_revision": submitted["revision"],
            "decision": "approve",
            "reason": "wrong department",
        },
    )
    assert cross_department.status_code == 403


def test_stale_revision_and_invalid_context_are_structured(client: TestClient) -> None:
    page = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 1},
    ).json()
    case = page["cases"][0]
    invalid = client.put(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}",
        json={
            "expected_revision": page["revision"],
            "case": {**case, "expected_evidence_context_ids": ["missing#chunk-1"]},
        },
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "unknown_context_id"


def test_dataset_detail_safe_pagination_and_unknown_dataset(client: TestClient) -> None:
    detail = client.get("/evaluation/datasets/evidence-gates-v1")
    assert detail.status_code == 200
    assert detail.json()["context_count"] == 707
    assert detail.json()["required_category_count"] == 11

    oversized = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 101},
    )
    assert oversized.status_code == 422

    missing = client.get("/evaluation/datasets/missing-dataset")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "dataset_not_found"


def test_only_organization_admin_can_delete_current_evidence_case(
    client: TestClient,
) -> None:
    before = client.get("/evaluation/datasets/evidence-gates-v1")
    assert before.status_code == 200
    revision = before.json()["revision"]
    case_id = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 1},
    ).json()["cases"][0]["id"]

    client.app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="department-admin",
        username="department-admin",
        role=UserRole.ADMIN,
        org_id="org-a",
        permissions=[Permission.ADMIN_MANAGE],
        department_id="human_resources",
        is_department_manager=True,
    )
    denied_list = client.get("/evaluation/datasets")
    assert denied_list.status_code == 403

    denied = client.request(
        "DELETE",
        f"/evaluation/datasets/evidence-gates-v1/cases/{case_id}",
        json={"expected_revision": revision},
    )
    assert denied.status_code == 403
    assert "organization_admin" in denied.json()["detail"]

    client.app.dependency_overrides[get_current_user] = lambda: _user()
    unchanged = client.get("/evaluation/datasets/evidence-gates-v1")
    assert unchanged.json()["revision"] == revision
    assert unchanged.json()["case_count"] == 100

    client.app.dependency_overrides[get_current_user] = lambda: _user()
    deleted = client.request(
        "DELETE",
        f"/evaluation/datasets/evidence-gates-v1/cases/{case_id}",
        json={"expected_revision": revision},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {
        "revision": revision + 1,
        "deleted_case_id": case_id,
    }

    after = client.get("/evaluation/datasets/evidence-gates-v1")
    assert after.json()["status"] == "authoring"
    assert after.json()["case_count"] == 99
    cases = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 100},
    ).json()["cases"]
    assert case_id not in {case["id"] for case in cases}



def test_authoring_workspace_is_organization_isolated(client: TestClient) -> None:
    page = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 1},
    ).json()
    case = page["cases"][0]
    edited = client.put(
        f"/evaluation/datasets/evidence-gates-v1/cases/{case['id']}",
        json={
            "expected_revision": page["revision"],
            "case": {**case, "notes": "org-a-only"},
        },
    )
    assert edited.status_code == 200
    assert edited.json()["case"]["notes"] == "org-a-only"

    client.app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="admin-b",
        username="admin-b",
        role=UserRole.ORGANIZATION_ADMIN,
        org_id="org-b",
        permissions=list(Permission),
    )
    other_org = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 1},
    )
    assert other_org.status_code == 200
    assert other_org.json()["revision"] == 1
    assert other_org.json()["cases"][0].get("notes") != "org-a-only"


def test_create_and_fail_closed_freeze_endpoints(client: TestClient) -> None:
    page = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 1},
    ).json()
    template = page["cases"][0]
    created = client.post(
        "/evaluation/datasets/evidence-gates-v1/cases",
        json={
            "expected_revision": page["revision"],
            "case": {**template, "id": "api-created-case", "question": "API 创建样本"},
        },
    )
    assert created.status_code == 200
    assert created.json()["case"]["last_editor_id"] == "org-admin-a"
    assert created.json()["case"]["department_id"] == "human_resources"

    client.app.dependency_overrides[get_current_user] = lambda: _user(
        "release-admin",
        role=UserRole.ORGANIZATION_ADMIN,
    )
    blocked = client.post(
        "/evaluation/datasets/evidence-gates-v1/freeze",
        json={"expected_revision": created.json()["revision"]},
    )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "review_incomplete"
    assert "version_path" not in blocked.json()


def test_organization_admin_can_review_across_departments(monkeypatch, tmp_path):
    """Organization administrators can complete cross-department governance review."""
    from api.routers.evaluation_pkg import _common as router
    from api.routers.evaluation_pkg.cases import review_evidence_case  # noqa: F401
    from api.schemas import EvidenceReviewRequest  # noqa: F401

    router.review_evidence_case = review_evidence_case
    router.EvidenceReviewRequest = EvidenceReviewRequest
    from evaluation.evidence_gate.review_workspace import EvidenceReviewWorkspace

    workspace = EvidenceReviewWorkspace(tmp_path / "review-root", router.SOURCE_BUNDLE)
    monkeypatch.setattr(router, "REVIEW_WORKSPACE", workspace)
    dataset = workspace.get_dataset("org-a", "evidence-gates-v1")
    case = workspace.list_cases("org-a", "evidence-gates-v1", page=1, page_size=1)["cases"][0]
    user = _user(
        "reviewer-b",
        role=UserRole.ORGANIZATION_ADMIN,
        department_id="finance",
        is_department_manager=False,
    )
    edited = workspace.update_case(
        "org-a",
        "evidence-gates-v1",
        case["id"],
        case,
        expected_revision=dataset["revision"],
        actor_id="maker-a",
        actor_department_id=case["department_id"],
    )
    submitted = workspace.submit_case(
        "org-a",
        "evidence-gates-v1",
        case["id"],
        expected_revision=edited["revision"],
        actor_id="maker-a",
        actor_department_id=case["department_id"],
        reviewer_id="reviewer-b",
    )

    response = anyio.run(
        router.review_evidence_case,
        "evidence-gates-v1",
        case["id"],
        router.EvidenceReviewRequest(decision="approve", reason="部门负责人确认", expected_revision=submitted["revision"]),
        user,
    )

    assert response.case.review_status == "approved"
    assert response.case.reviewer_id == "reviewer-b"


def test_bulk_filtered_submit_and_approve_process_all_pages(client: TestClient) -> None:
    dataset = client.get("/evaluation/datasets/evidence-gates-v1").json()
    submitted = client.post(
        "/evaluation/datasets/evidence-gates-v1/cases/bulk-submit",
        json={
            "expected_revision": dataset["revision"],
            "selection_mode": "filtered",
            "case_ids": [],
            "review_status": "draft",
        },
    )

    assert submitted.status_code == 200
    assert submitted.json()["matched_count"] == 100
    assert submitted.json()["processed_count"] == 100
    assert submitted.json()["skipped_items"] == []
    second_page = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 3, "page_size": 20, "review_status": "pending_review"},
    ).json()
    assert second_page["total"] == 100
    assert all(case["reviewer_id"] == "org-admin-a" for case in second_page["cases"])

    approved = client.post(
        "/evaluation/datasets/evidence-gates-v1/cases/bulk-approve",
        json={
            "expected_revision": submitted.json()["revision"],
            "selection_mode": "filtered",
            "case_ids": [],
            "review_status": "pending_review",
        },
    )
    assert approved.status_code == 200
    assert approved.json()["processed_count"] == 100
    assert approved.json()["revision"] == submitted.json()["revision"] + 1
    assert client.get("/evaluation/datasets/evidence-gates-v1").json()["counts"][
        "approved"
    ] == 100


def test_bulk_selected_returns_maker_checker_skip_and_rejects_stale_revision(
    client: TestClient,
) -> None:
    page = client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"page": 1, "page_size": 2},
    ).json()
    maker_case, eligible_case = page["cases"]
    edited = client.put(
        f"/evaluation/datasets/evidence-gates-v1/cases/{maker_case['id']}",
        json={"expected_revision": page["revision"], "case": maker_case},
    ).json()
    payload = {
        "expected_revision": edited["revision"],
        "selection_mode": "selected",
        "case_ids": [maker_case["id"], eligible_case["id"]],
    }

    submitted = client.post(
        "/evaluation/datasets/evidence-gates-v1/cases/bulk-submit", json=payload
    )
    assert submitted.status_code == 200
    assert submitted.json()["processed_case_ids"] == [eligible_case["id"]]
    assert submitted.json()["skipped_items"] == [
        {"case_id": maker_case["id"], "code": "maker_checker_violation"}
    ]

    stale = client.post(
        "/evaluation/datasets/evidence-gates-v1/cases/bulk-approve",
        json={**payload, "case_ids": [eligible_case["id"]]},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "revision_conflict"
    assert client.get(
        "/evaluation/datasets/evidence-gates-v1/cases",
        params={"query": eligible_case["id"], "page": 1, "page_size": 1},
    ).json()["cases"][0]["review_status"] == "pending_review"


@pytest.mark.parametrize(
    "payload",
    [
        {"expected_revision": 1, "selection_mode": "selected", "case_ids": []},
        {
            "expected_revision": 1,
            "selection_mode": "selected",
            "case_ids": ["case-1", "case-1"],
        },
        {
            "expected_revision": 1,
            "selection_mode": "selected",
            "case_ids": ["invalid/id"],
        },
        {
            "expected_revision": 1,
            "selection_mode": "filtered",
            "case_ids": [],
            "reviewer_id": "forged-reviewer",
        },
    ],
)
def test_bulk_request_rejects_ambiguous_or_forged_inputs(
    client: TestClient, payload: dict[str, object]
) -> None:
    response = client.post(
        "/evaluation/datasets/evidence-gates-v1/cases/bulk-submit", json=payload
    )
    assert response.status_code == 422


def test_frozen_version_endpoints_are_safe_and_create_an_editable_draft(client: TestClient) -> None:
    """Company-admin API exposes only frozen identities and locks authoring until draft."""
    from api.routers.evaluation_pkg import _common as router
    from api.routers.evaluation_pkg.cases import review_evidence_case  # noqa: F401
    from api.schemas import EvidenceReviewRequest  # noqa: F401

    router.review_evidence_case = review_evidence_case
    router.EvidenceReviewRequest = EvidenceReviewRequest

    workspace = router.REVIEW_WORKSPACE
    dataset = workspace.get_dataset("org-a", "evidence-gates-v1")
    revision = dataset["revision"]
    cases = workspace.list_cases("org-a", "evidence-gates-v1", page=1, page_size=100)["cases"]
    for case in cases:
        submitted = workspace.submit_case(
            "org-a", "evidence-gates-v1", case["id"], expected_revision=revision,
            actor_id="maker-a", actor_department_id=case["department_id"], reviewer_id="reviewer-b",
        )
        approved = workspace.review_case(
            "org-a", "evidence-gates-v1", case["id"], decision="approve", reason="verified",
            expected_revision=submitted["revision"], actor_id="reviewer-b",
            actor_department_id=case["department_id"], actor_is_department_manager=True,
        )
        revision = approved["revision"]
    frozen = workspace.freeze("org-a", "evidence-gates-v1", expected_revision=revision, actor_id="org-admin-a")

    summary = client.get("/evaluation/datasets/evidence-gates-v1")
    assert summary.status_code == 200
    body = summary.json()
    assert body["status"] == "frozen"
    assert body["last_frozen_version"] == frozen["version"]
    assert body["last_frozen_manifest_sha256"] == frozen["manifest_sha256"]
    assert body["last_frozen_at"]
    assert body["last_frozen_by"] == "org-admin-a"
    assert "version_path" not in body

    versions = client.get("/evaluation/datasets/evidence-gates-v1/versions")
    assert versions.status_code == 200
    assert versions.json()["versions"][0]["version"] == frozen["version"]
    assert "version_path" not in versions.json()["versions"][0]
    detail = client.get(f"/evaluation/datasets/evidence-gates-v1/versions/{frozen['version']}")
    assert detail.status_code == 200
    assert detail.json()["manifest_sha256"] == frozen["manifest_sha256"]

    case_id = cases[0]["id"]
    blocked = client.request(
        "DELETE",
        f"/evaluation/datasets/evidence-gates-v1/cases/{case_id}",
        json={"expected_revision": frozen["revision"]},
    )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "dataset_frozen"

    draft = client.post(
        "/evaluation/datasets/evidence-gates-v1/drafts",
        json={"version": frozen["version"], "expected_revision": frozen["revision"]},
    )
    assert draft.status_code == 200
    assert draft.json()["status"] == "authoring"
    assert draft.json()["source_frozen_version"] == frozen["version"]
    assert draft.json()["revision"] == frozen["revision"] + 1


def _freeze_for_release(workspace: EvidenceReviewWorkspace) -> dict:
    dataset = workspace.get_dataset("org-a", "evidence-gates-v1")
    revision = dataset["revision"]
    for case in workspace.list_cases("org-a", "evidence-gates-v1", page=1, page_size=100)["cases"]:
        submitted = workspace.submit_case(
            "org-a", "evidence-gates-v1", case["id"], expected_revision=revision,
            actor_id="maker-a", actor_department_id=case["department_id"], reviewer_id="manager-b",
        )
        approved = workspace.review_case(
            "org-a", "evidence-gates-v1", case["id"], decision="approve", reason="checked",
            expected_revision=submitted["revision"], actor_id="manager-b",
            actor_department_id=case["department_id"], actor_is_department_manager=True,
        )
        revision = approved["revision"]
    return workspace.freeze("org-a", "evidence-gates-v1", expected_revision=revision, actor_id="org-admin-a")


def _configure_draft_origin_release(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Install a reviewed current-corpus lineage double for release API contracts."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from api.routers.evaluation_pkg import _common as router
    from api.routers.evaluation_pkg.cases import review_evidence_case  # noqa: F401
    from api.schemas import EvidenceReviewRequest  # noqa: F401

    router.review_evidence_case = review_evidence_case
    router.EvidenceReviewRequest = EvidenceReviewRequest
    from evaluation.release.workflow import PostgreSQLReleaseWorkflowRepository
    from evaluation.release.runtime import ReleaseWorkflowService
    from infrastructure.postgres.database import DatabaseService
    from infrastructure.postgres.models import metadata

    draft_id = "ccd_" + "a" * 32
    dataset_id = "current-corpus-authoring-dataset"
    version = "20260816T010203Z-r42"
    revision = 8
    snapshot_sha256 = "b" * 64
    manifest_sha256 = "c" * 64

    class Workspace:
        def get_dataset(self, org_id, requested_dataset_id):
            assert org_id == "org-a"
            assert requested_dataset_id == dataset_id
            return {
                "source_type": "current_corpus",
                "status": "frozen",
                "revision": revision,
                "current_corpus_draft_id": draft_id,
                "current_corpus_snapshot_sha256": snapshot_sha256,
            }

        def get_version(self, org_id, requested_dataset_id, requested_version):
            assert org_id == "org-a"
            assert requested_dataset_id == dataset_id
            assert requested_version == version
            return {
                "version": version,
                "manifest_sha256": manifest_sha256,
                "current_corpus_draft_id": draft_id,
                "current_corpus_snapshot_sha256": snapshot_sha256,
            }

        def get_frozen_evaluation_suite(self, org_id, requested_dataset_id, requested_version):
            assert org_id == "org-a"
            assert requested_dataset_id == "evidence-gates-v1"
            assert requested_version == "20260815T010203Z-r19"
            return {
                "dataset_id": requested_dataset_id,
                "version": requested_version,
                "manifest_sha256": "e" * 64,
                "case_count": 100,
            }

    class FixtureValidation:
        run_id = "fvr_" + "d" * 32

        def __init__(self):
            self.preflight_calls = []

        def assert_runtime_corpus_aligned(self, **kwargs):
            self.preflight_calls.append(kwargs)
            assert kwargs["manifest_sha256"] == manifest_sha256

        def latest_success(self, **kwargs):
            assert kwargs["manifest_sha256"] == manifest_sha256
            return {
                "run_id": self.run_id,
                "status": "succeeded",
                "manifest_sha256": manifest_sha256,
            }

        def get(self, **kwargs):
            assert kwargs["run_id"] == self.run_id
            return {
                "run_id": self.run_id,
                "status": "succeeded",
                "manifest_sha256": manifest_sha256,
            }

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    metadata.create_all(engine)
    repository = PostgreSQLReleaseWorkflowRepository(DatabaseService.from_engine(engine))
    workspace = Workspace()
    fixtures = FixtureValidation()
    release = ReleaseWorkflowService(
        repository,
        workspace,
        source_bundle=SOURCE_BUNDLE,
        output_root=tmp_path / "runs",
        env_file=tmp_path / "env",
        fixture_validation_service=fixtures,
    )
    monkeypatch.setattr(router, "REVIEW_WORKSPACE", workspace)
    monkeypatch.setattr(router, "_FIXTURE_VALIDATION_SERVICE", fixtures)
    monkeypatch.setattr(router, "_RELEASE_SERVICE", release)
    monkeypatch.setattr(router, "_dispatch_release_workflow", lambda *_args, **_kwargs: True)
    client.app.dependency_overrides[get_document_read] = lambda: SimpleNamespace(catalog=object())
    return SimpleNamespace(
        draft_id=draft_id,
        dataset_id=dataset_id,
        version=version,
        revision=revision,
        manifest_sha256=manifest_sha256,
        fixtures=fixtures,
        repository=repository,
        workspace=workspace,
    )


def test_company_admin_release_endpoint_uses_frozen_version_and_hides_post_calibration_routes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The direct frozen-version handoff remains; no release approval or Gate action is exposed."""
    context = _configure_draft_origin_release(client, monkeypatch, tmp_path)

    client.app.dependency_overrides[get_current_user] = lambda: _user(role=UserRole.EDITOR)
    assert client.post(
        f"/evaluation/datasets/{context.dataset_id}/versions/{context.version}/release-workflows",
        json={"evaluation_dataset_id": "evidence-gates-v1", "evaluation_version": "20260815T010203Z-r19"},
    ).status_code == 403
    client.app.dependency_overrides[get_current_user] = lambda: _user()

    rejected_payload = client.post(
        f"/evaluation/datasets/{context.dataset_id}/versions/{context.version}/release-workflows",
        json={"tenant_id": "forged"},
    )
    assert rejected_payload.status_code == 422

    started = client.post(
        f"/evaluation/datasets/{context.dataset_id}/versions/{context.version}/release-workflows",
        json={"evaluation_dataset_id": "evidence-gates-v1", "evaluation_version": "20260815T010203Z-r19"},
    )
    assert started.status_code == 200
    workflow = started.json()
    assert workflow["current_corpus_draft_id"] is None
    assert workflow["current_corpus_snapshot_sha256"] is None
    assert workflow["fixture_validation_run_id"] is None
    assert len(context.fixtures.preflight_calls) == 1

    duplicate = client.post(
        f"/evaluation/datasets/{context.dataset_id}/versions/{context.version}/release-workflows",
        json={"evaluation_dataset_id": "evidence-gates-v1", "evaluation_version": "20260815T010203Z-r19"},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["workflow_id"] == workflow["workflow_id"]
    assert client.post(
        f"/evaluation/current-corpus-drafts/{context.draft_id}/release-workflows",
        json={"expected_revision": context.revision, "version": context.version},
    ).status_code == 404
    assert client.get(f"/evaluation/release-workflows/{workflow['workflow_id']}").status_code == 200
    for url, payload in (
        (f"/evaluation/release-workflows/{workflow['workflow_id']}/submit-approval", {"expected_revision": workflow["revision"]}),
        (f"/evaluation/release-workflows/{workflow['workflow_id']}/review", {"expected_revision": workflow["revision"], "decision": "approve", "reason": "unused"}),
    ):
        assert client.post(url, json=payload).status_code == 404
    # Gate 控制入口已存在；尚未审批或没有历史配置时必须安全拒绝。
    assert client.post(
        f"/evaluation/release-workflows/{workflow['workflow_id']}/promote",
        json={"expected_revision": workflow["revision"]},
    ).status_code == 409
    assert client.post(
        "/evaluation/gate/rollback",
        json={"expected_revision": 1, "reason": "没有可回滚的配置"},
    ).status_code == 409
    assert client.get("/evaluation/gate").json()["mode"] == "off"
    assert context.repository.gate()["mode"] == "off"


def test_direct_release_rejects_unfrozen_target_and_non_100_formal_suite(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The direct launch contract rejects mutable targets and non-authoritative suites."""
    from api.routers.evaluation_pkg import _common as router
    from api.routers.evaluation_pkg.cases import review_evidence_case  # noqa: F401
    from api.schemas import EvidenceReviewRequest  # noqa: F401

    router.review_evidence_case = review_evidence_case
    router.EvidenceReviewRequest = EvidenceReviewRequest

    context = _configure_draft_origin_release(client, monkeypatch, tmp_path)
    original_workspace = context.workspace

    class UnfrozenWorkspace:
        def get_dataset(self, org_id, dataset_id):
            result = original_workspace.get_dataset(org_id, dataset_id)
            result["status"] = "authoring"
            return result

        def get_version(self, *args):
            return original_workspace.get_version(*args)

        def get_frozen_evaluation_suite(self, *args):
            return original_workspace.get_frozen_evaluation_suite(*args)

    monkeypatch.setattr(router, "REVIEW_WORKSPACE", UnfrozenWorkspace())
    mutable = client.post(
        f"/evaluation/datasets/{context.dataset_id}/versions/{context.version}/release-workflows",
        json={"evaluation_dataset_id": "evidence-gates-v1", "evaluation_version": "20260815T010203Z-r19"},
    )
    assert mutable.status_code == 422
    assert mutable.json()["detail"]["code"] == "frozen_version_required"

    class Non100SuiteWorkspace:
        def get_dataset(self, org_id, dataset_id):
            return original_workspace.get_dataset(org_id, dataset_id)

        def get_version(self, *args):
            return original_workspace.get_version(*args)

        def get_frozen_evaluation_suite(self, *args):
            result = original_workspace.get_frozen_evaluation_suite(*args)
            result["case_count"] = 12
            return result

    monkeypatch.setattr(router, "REVIEW_WORKSPACE", Non100SuiteWorkspace())
    non100 = client.post(
        f"/evaluation/datasets/{context.dataset_id}/versions/{context.version}/release-workflows",
        json={"evaluation_dataset_id": "evidence-gates-v1", "evaluation_version": "20260815T010203Z-r19"},
    )
    assert non100.status_code == 422
    assert non100.json()["detail"]["code"] == "formal_evaluation_suite_required"


def test_release_start_does_not_require_a_second_enabled_company_administrator(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Authenticated organization admins rely on the earlier maker-checker review."""
    from api.routers.evaluation_pkg import _common as router
    from api.routers.evaluation_pkg.cases import review_evidence_case  # noqa: F401
    from api.schemas import EvidenceReviewRequest  # noqa: F401

    router.review_evidence_case = review_evidence_case
    router.EvidenceReviewRequest = EvidenceReviewRequest

    class OneCompanyAdmin:
        def list_users(self, **_kwargs):
            return [{"user_id": "org-admin-a", "role": UserRole.ORGANIZATION_ADMIN, "is_active": True}]

    context = _configure_draft_origin_release(client, monkeypatch, tmp_path)
    monkeypatch.setattr(router, "USER_SERVICE", OneCompanyAdmin())
    response = client.post(
        f"/evaluation/datasets/{context.dataset_id}/versions/{context.version}/release-workflows",
        json={"evaluation_dataset_id": "evidence-gates-v1", "evaluation_version": "20260815T010203Z-r19"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_release_retry_and_workflow_reads_stay_in_the_authenticated_company_scope(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Retry retains the failed stage and client scope cannot be overridden."""
    context = _configure_draft_origin_release(client, monkeypatch, tmp_path)
    started = client.post(
        f"/evaluation/datasets/{context.dataset_id}/versions/{context.version}/release-workflows",
        json={"evaluation_dataset_id": "evidence-gates-v1", "evaluation_version": "20260815T010203Z-r19"},
    ).json()
    failed = context.repository.transition(
        started["workflow_id"],
        expected_revision=started["revision"],
        status="failed",
        stage="shadow",
        reason_code="service_probe_failed",
    )
    retry = client.post(
        f"/evaluation/release-workflows/{started['workflow_id']}/retry",
        json={"expected_revision": failed["revision"]},
    )
    assert retry.status_code == 200
    assert retry.json()["status"] == "queued"
    assert retry.json()["stage"] == "shadow"
    assert retry.json()["reason_code"] is None

    client.app.dependency_overrides[get_current_user] = lambda: _user(
        "other-company-admin",
        org_id="org-b",
    )
    assert client.get(f"/evaluation/release-workflows/{started['workflow_id']}").status_code == 404
    assert client.get(f"/evaluation/release-workflows/{started['workflow_id']}/attempts").status_code == 404
    assert client.post(
        f"/evaluation/release-workflows/{started['workflow_id']}/retry",
        json={"expected_revision": retry.json()["revision"]},
    ).status_code == 404


def test_release_dispatch_uses_one_stable_durable_task_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A duplicate start observes the task reservation instead of publishing twice."""
    from api.routers.evaluation_pkg import _common as router
    from api.routers.evaluation_pkg.cases import review_evidence_case  # noqa: F401
    from api.schemas import EvidenceReviewRequest  # noqa: F401

    router.review_evidence_case = review_evidence_case
    router.EvidenceReviewRequest = EvidenceReviewRequest
    from infrastructure import celery_tasks
    from infrastructure.tasks import task_registry

    calls: list[tuple[str, object]] = []

    class Record:
        kind = "evidence_release"
        state = "queued"

    class Registry:
        record: Record | None = None

        def get_owned(self, task_id, org_id):
            calls.append(("get", (task_id, org_id)))
            return self.record

        def reserve(self, task_id, **kwargs):
            calls.append(("reserve", (task_id, kwargs)))
            self.record = Record()

        def update_state(self, task_id, state):
            calls.append(("update", (task_id, state)))

        def release(self, task_id):
            calls.append(("release", task_id))
            return True

    registry = Registry()
    monkeypatch.setattr(task_registry, "get_task_registry", lambda: registry)
    monkeypatch.setattr(
        celery_tasks.evidence_release_workflow_task,
        "apply_async",
        lambda *, args, task_id: calls.append(("publish", (args, task_id))),
    )
    workflow = {"workflow_id": "erw-stable"}
    assert router._dispatch_release_workflow(
        workflow,
        user=_user(),
        dataset_id="evidence-gates-v1",
    ) is True
    assert ("publish", (("erw-stable", "org-a"), "erw-stable")) in calls
    assert router._dispatch_release_workflow(
        workflow,
        user=_user(),
        dataset_id="evidence-gates-v1",
    ) is False
    assert [name for name, _value in calls].count("publish") == 1


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("get", "/evaluation/datasets", None),
        ("get", "/evaluation/datasets/evidence-gates-v1/fixtures", None),
        ("get", "/evaluation/datasets/evidence-gates-v1/versions/v1/fixture-validations", None),
        ("post", "/evaluation/datasets/evidence-gates-v1/versions/v1/fixture-validations", {"expected_revision": 1}),
        ("post", "/evaluation/datasets/evidence-gates-v1/versions/v1/release-workflows", {"expected_revision": 1}),
        ("get", "/evaluation/runs", None),
        ("get", "/evaluation/release-workflows/erw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/attempts", None),
    ],
)
def test_non_organization_admin_is_rejected_before_any_evaluation_data_read(
    client: TestClient,
    method: str,
    path: str,
    payload: dict | None,
) -> None:
    client.app.dependency_overrides[get_current_user] = lambda: _user(
        "department-manager",
        role=UserRole.ADMIN,
        department_id="finance",
        is_department_manager=True,
    )
    response = getattr(client, method)(path, json=payload) if payload else getattr(client, method)(path)
    assert response.status_code == 403
    assert response.content in {b'{"detail":"Not enough permissions"}', b'{"detail":"Insufficient permissions"}'} or "dataset" not in response.text


def test_fixture_validation_api_accepts_only_revision_and_dispatches_a_bounded_run(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.routers.evaluation_pkg import _common as router
    from api.routers.evaluation_pkg.cases import review_evidence_case  # noqa: F401
    from api.schemas import EvidenceReviewRequest  # noqa: F401

    router.review_evidence_case = review_evidence_case
    router.EvidenceReviewRequest = EvidenceReviewRequest
    from infrastructure.tasks import celery_evaluation_tasks as tasks

    run = {
        "run_id": "fvr_" + "a" * 32,
        "dataset_id": "evidence-gates-v1",
        "version": "v1",
        "manifest_sha256": "b" * 64,
        "source_workspace_revision": 4,
        "status": "queued",
        "reason_code": None,
        "fixture_results": [],
        "created_at": "2026-08-15T00:00:00+00:00",
        "started_at": None,
        "finished_at": None,
    }
    observed: dict[str, object] = {}

    class _FixtureService:
        def start(self, **kwargs):
            observed["start"] = kwargs
            return dict(run)

        def get(self, **kwargs):
            observed["get"] = kwargs
            return dict(run)

        def list(self, **_kwargs):
            return [dict(run)]

        def retry(self, **_kwargs):
            return dict(run)

        def fail_dispatch(self, **_kwargs):
            return dict(run)

    def apply_async(*, args, task_id):
        observed["dispatch"] = {"args": args, "task_id": task_id}

    monkeypatch.setattr(router, "_FIXTURE_VALIDATION_SERVICE", _FixtureService())
    monkeypatch.setattr(tasks.evidence_fixture_validation_task, "apply_async", apply_async)

    rejected = client.post(
        "/evaluation/datasets/evidence-gates-v1/versions/v1/fixture-validations",
        json={"expected_revision": 4, "tenant_id": "browser-controlled"},
    )
    assert rejected.status_code == 422
    assert "start" not in observed

    historical = client.post(
        "/evaluation/datasets/evidence-gates-v1/versions/v1/fixture-validations",
        json={"expected_revision": 4},
    )
    assert historical.status_code == 409
    assert historical.json()["detail"]["code"] == "fixture_validation_current_corpus_required"
    assert "start" not in observed

    class CurrentCorpusWorkspace:
        def get_dataset(self, org_id, dataset_id):
            assert org_id == "org-a"
            assert dataset_id == "evidence-gates-v1"
            return {"source_type": "current_corpus"}

    monkeypatch.setattr(router, "REVIEW_WORKSPACE", CurrentCorpusWorkspace())
    started = client.post(
        "/evaluation/datasets/evidence-gates-v1/versions/v1/fixture-validations",
        json={"expected_revision": 4},
    )
    assert started.status_code == 200
    assert started.json()["run_id"] == run["run_id"]
    assert observed["start"]["org_id"] == "org-a"
    assert observed["start"]["dataset_id"] == "evidence-gates-v1"
    assert observed["start"]["version"] == "v1"
    assert observed["start"]["expected_revision"] == 4
    assert observed["start"]["initiated_by"] == "org-admin-a"
    assert callable(observed["start"]["corpus_probe"])
    assert observed["dispatch"] == {
        "args": (run["run_id"], "org-a", "evidence-gates-v1", "v1"),
        "task_id": run["run_id"],
    }


def test_release_history_is_company_scoped_and_keeps_formal_suite_identity(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Panel history may reload durable records without exposing another company."""
    context = _configure_draft_origin_release(client, monkeypatch, tmp_path)
    payload = {
        "evaluation_dataset_id": "evidence-gates-v1",
        "evaluation_version": "20260815T010203Z-r19",
    }
    started = client.post(
        f"/evaluation/datasets/{context.dataset_id}/versions/{context.version}/release-workflows",
        json=payload,
    )
    assert started.status_code == 200

    history = client.get("/evaluation/release-workflows")
    assert history.status_code == 200
    assert [item["workflow_id"] for item in history.json()["workflows"]] == [started.json()["workflow_id"]]
    assert history.json()["workflows"][0]["evaluation_dataset_id"] == "evidence-gates-v1"
    assert history.json()["workflows"][0]["evaluation_case_count"] == 100

    client.app.dependency_overrides[get_current_user] = lambda: _user(
        "other-company-admin", org_id="org-b"
    )
    assert client.get("/evaluation/release-workflows").json()["workflows"] == []


def test_company_admin_release_approval_promotion_and_rollback_api(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The administrative API enforces maker-checker then exposes durable Gate controls."""
    context = _configure_draft_origin_release(client, monkeypatch, tmp_path)
    started = client.post(
        f"/evaluation/datasets/{context.dataset_id}/versions/{context.version}/release-workflows",
        json={"evaluation_dataset_id": "evidence-gates-v1", "evaluation_version": "20260815T010203Z-r19"},
    ).json()
    pending = context.repository.transition(
        started["workflow_id"], expected_revision=started["revision"],
        status="pending_approval", stage="approval",
    )

    self_approval = client.post(
        f"/evaluation/release-workflows/{pending['workflow_id']}/approve",
        json={
            "expected_revision": pending["revision"],
            "reason": "发起人不得批准自己的发布",
            "target_mode": "shadow",
        },
    )
    assert self_approval.status_code == 409
    assert self_approval.json()["detail"]["code"] == "release_self_approval_forbidden"

    client.app.dependency_overrides[get_current_user] = lambda: _user(user_id="manager-b")
    approved = client.post(
        f"/evaluation/release-workflows/{pending['workflow_id']}/approve",
        json={
            "expected_revision": pending["revision"],
            "reason": "独立管理员复核离线评测证据",
            "target_mode": "shadow",
        },
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    promoted = client.post(
        f"/evaluation/release-workflows/{pending['workflow_id']}/promote",
        json={"expected_revision": approved.json()["revision"]},
    )
    assert promoted.status_code == 200
    assert promoted.json()["status"] == "promoted"
    gate = client.get("/evaluation/gate")
    assert gate.status_code == 200
    assert gate.json()["mode"] == "shadow"
    assert gate.json()["revision"] == 1

    rollback = client.post(
        "/evaluation/gate/rollback",
        json={"expected_revision": 1, "reason": "影子观察期结束后回退"},
    )
    assert rollback.status_code == 200
    assert rollback.json()["mode"] == "off"
    assert rollback.json()["revision"] == 2


def test_refresh_current_corpus_reports_a_missing_identity_manifest(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A broken API image reports the missing runtime manifest explicitly."""
    from api.routers.evaluation_pkg import _common as router
    from api.routers.evaluation_pkg.cases import review_evidence_case  # noqa: F401
    from api.schemas import EvidenceReviewRequest  # noqa: F401

    router.review_evidence_case = review_evidence_case
    router.EvidenceReviewRequest = EvidenceReviewRequest

    monkeypatch.setattr(router, "COMPANY_DEMO_MANIFEST", tmp_path / "missing-corpus-manifest.json")

    response = client.post("/evaluation/datasets/current-corpus-production/refresh-current-corpus")

    assert response.status_code == 503
    assert response.json()["detail"] == "41 份语料身份清单未随服务正确部署，请重新构建 API 镜像"
