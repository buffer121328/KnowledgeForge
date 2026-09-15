"""Read-only compatibility boundaries for legacy current-corpus draft records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_current_user
from api.routers.evaluation_pkg import evaluation_router
from domain.documents import DocumentRecord
from domain.identity import Permission, UserContext, UserRole
from evaluation.current_corpus_drafts import CurrentCorpusDraftService


@dataclass
class FakeCatalog:
    records: list[DocumentRecord]


def _user(role: UserRole = UserRole.ORGANIZATION_ADMIN) -> UserContext:
    return UserContext(
        user_id="org-admin-a",
        username="org-admin-a",
        role=role,
        org_id="org-a",
        permissions=list(Permission) if role == UserRole.ORGANIZATION_ADMIN else [Permission.DOC_READ],
        department_id="finance",
    )


def _record(doc_id: str, department_id: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        tenant_id="org-a",
        company_id="company-a",
        department_id=department_id,
        folder_path="",
        relative_path="",
        normalized_relative_path="",
        uploaded_filename="confidential.pdf",
        display_name="confidential",
        provenance_source_filename="confidential.pdf",
        storage_reference="not-returned-storage-reference",
        content_sha256="a" * 64,
        size_bytes=1,
        mime_type="application/pdf",
        document_type="policy",
        source_format="pdf",
        ingest_status="ingested",
        version=1,
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[TestClient, CurrentCorpusDraftService]:
    app = FastAPI()
    app.include_router(evaluation_router)
    current_user = [_user()]
    app.dependency_overrides[get_current_user] = lambda: current_user[0]
    app.state.current_user = current_user
    service = CurrentCorpusDraftService(tmp_path / "current-corpus-drafts")
    monkeypatch.setattr("api.routers.evaluation_pkg._common.CURRENT_CORPUS_DRAFT_SERVICE", service)
    monkeypatch.setattr("api.routers.evaluation_pkg._common._audit", lambda *_args, **_kwargs: None)
    return TestClient(app), service


def test_company_admin_can_read_legacy_current_corpus_drafts(
    client: tuple[TestClient, CurrentCorpusDraftService],
) -> None:
    api, service = client
    historical = service.create_from_records(
        "org-a",
        [_record("hr-1", "human_resources"), _record("admin-1", "administration")],
        actor_id="legacy-seed",
    )

    listed = api.get("/evaluation/current-corpus-drafts")
    assert listed.status_code == 200
    assert [item["draft_id"] for item in listed.json()["drafts"]] == [historical["draft_id"]]
    assert "not-returned-storage-reference" not in listed.text
    assert "storage_reference" not in listed.text
    assert "department_id" not in listed.text

    detail = api.get(f"/evaluation/current-corpus-drafts/{historical['draft_id']}")
    assert detail.status_code == 200
    assert detail.json()["draft_id"] == historical["draft_id"]


def test_legacy_current_corpus_draft_mutations_are_not_available(
    client: tuple[TestClient, CurrentCorpusDraftService],
) -> None:
    api, service = client
    historical = service.create_from_records("org-a", [_record("hr-1", "human_resources")], actor_id="legacy-seed")
    draft_id = historical["draft_id"]
    candidate_id = historical["candidates"][0]["candidate_id"]

    assert api.post("/evaluation/current-corpus-drafts").status_code == 405
    assert api.post(
        f"/evaluation/current-corpus-drafts/{draft_id}/candidates/{candidate_id}/submit",
        json={"expected_revision": historical["revision"]},
    ).status_code == 404
    assert api.post(
        f"/evaluation/current-corpus-drafts/{draft_id}/candidates/{candidate_id}/review",
        json={"expected_revision": historical["revision"], "decision": "approve"},
    ).status_code == 404
    assert api.post(
        f"/evaluation/current-corpus-drafts/{draft_id}/authoring-dataset",
        json={"expected_revision": historical["revision"]},
    ).status_code == 404

    assert service.get("org-a", draft_id)["revision"] == historical["revision"]


def test_non_admin_can_neither_read_legacy_drafts_nor_use_the_retired_collection_method(
    client: tuple[TestClient, CurrentCorpusDraftService],
) -> None:
    api, _service = client
    api.app.state.current_user[0] = _user(UserRole.EDITOR)

    assert api.get("/evaluation/current-corpus-drafts").status_code == 403
    assert api.post("/evaluation/current-corpus-drafts").status_code == 405
