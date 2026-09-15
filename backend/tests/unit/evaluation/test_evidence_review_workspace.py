"""ATDD for the evidence dataset authoring and maker-checker workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from evaluation.evidence_gate.benchmark import sha256_file
from evaluation.evidence_gate.review_workspace import (
    EvidenceReviewConflict,
    EvidenceReviewPermissionError,
    EvidenceReviewValidationError,
    EvidenceReviewWorkspace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_BUNDLE = PROJECT_ROOT / "backend/evaluation/data/evidence-gates"


def _workspace(tmp_path: Path) -> EvidenceReviewWorkspace:
    return EvidenceReviewWorkspace(tmp_path / "review-root", SOURCE_BUNDLE)


def _legacy_source_bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source-bundle"
    source.mkdir()
    current_cases = (SOURCE_BUNDLE / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    (source / "cases.jsonl").write_text(
        "\n".join(current_cases[:37]) + "\n",
        encoding="utf-8",
    )
    for name in ("context-catalog.jsonl", "fixture-profile.example.json"):
        shutil.copyfile(SOURCE_BUNDLE / name, source / name)
    (source / "manifest.json").write_text(
        '{"dataset_version":"evidence-gates-legacy-37"}\n',
        encoding="utf-8",
    )
    return source


def _publish_current_source(source: Path) -> None:
    for name in (
        "cases.jsonl",
        "context-catalog.jsonl",
        "fixture-profile.example.json",
        "manifest.json",
    ):
        shutil.copyfile(SOURCE_BUNDLE / name, source / name)


def test_workspace_initializes_from_frozen_candidates_and_isolates_orgs(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)

    first = service.get_dataset("org-a", "evidence-gates-v1")
    second = service.get_dataset("org-b", "evidence-gates-v1")

    assert first["case_count"] == 100
    assert first["counts"]["draft"] == 100
    assert first["context_count"] == 707
    assert second["case_count"] == 100
    assert all(
        case["department_id"]
        for case in service.list_cases(
            "org-a", "evidence-gates-v1", page=1, page_size=100
        )["cases"]
    )
    assert service.list_cases("org-a", "evidence-gates-v1", page=1, page_size=5)[
        "total"
    ] == 100
    assert service.workspace_path("org-a", "evidence-gates-v1") != service.workspace_path(
        "org-b", "evidence-gates-v1"
    )


def test_workspace_initializes_the_50_case_routine_suite_separately(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)

    routine = service.get_dataset("org-a", "evidence-gates-v1-routine")

    assert routine["case_count"] == 50
    assert routine["counts"]["draft"] == 50
    assert routine["category_counts"]["fully_answerable"] == 10
    assert set(routine["category_counts"]) == {
        "fully_answerable",
        "completely_unanswerable",
        "background_only",
        "partially_answerable",
        "conflicting",
        "missing_version_or_date",
        "missing_business_record",
        "authorization_filtered",
        "single_branch_unavailable",
        "all_branches_unavailable",
        "prompt_injection",
    }
    assert service.get_dataset("org-a", "evidence-gates-v1")["case_count"] == 100


@pytest.mark.parametrize(
    ("dataset_id", "expected_case_count"),
    (("evidence-gates-v1", 100), ("evidence-gates-v1-routine", 50)),
)
def test_supported_suite_rebind_replaces_only_verified_contexts_and_clears_fixture_only_bindings(
    tmp_path: Path,
    dataset_id: str,
    expected_case_count: int,
) -> None:
    service = _workspace(tmp_path)
    before = service.get_dataset("org-a", dataset_id)
    root = service.workspace_path("org-a", dataset_id)
    contexts = [
        json.loads(line)
        for line in (root / "context-catalog.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    runtime_chunks = []
    for context in contexts:
        content = str(context["content_excerpt"])
        context["content_sha256"] = "0" * 64
        runtime_chunks.append(
            {
                "chunk_id": context["context_id"],
                "chunk_index": context["chunk_index"],
                "content": content,
                "source_document_id": context["source_document_id"],
                "title": context["title"],
                "department": context["department"],
            }
        )
    (root / "context-catalog.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in contexts),
        encoding="utf-8",
    )

    rebound = service.rebind_suite_to_current_chunks(
        "org-a",
        dataset_id,
        expected_revision=before["revision"],
        actor_id="admin-a",
        current_chunks=runtime_chunks,
    )

    assert rebound["case_count"] == expected_case_count
    cases = service.list_cases("org-a", dataset_id, page=1, page_size=100)["cases"]
    assert all(case["review_status"] == "draft" for case in cases)
    fixture_only = next(case for case in cases if case["category"] == "authorization_filtered")
    assert fixture_only["expected_source_document_ids"] == []
    assert fixture_only["expected_evidence_context_ids"] == []
    positive = next(case for case in cases if case["category"] == "fully_answerable")
    assert all("#chunk-" in item for item in positive["expected_evidence_context_ids"])


def test_suite_rebind_accepts_one_unique_same_document_excerpt_after_chunk_boundary_change(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)
    before = service.get_dataset("org-a", "evidence-gates-v1-routine")
    root = service.workspace_path("org-a", "evidence-gates-v1-routine")
    cases = service.list_cases("org-a", "evidence-gates-v1-routine", page=1, page_size=100)["cases"]
    target_context_id = next(
        context_id
        for case in cases
        if case["category"] == "fully_answerable"
        for context_id in case["expected_evidence_context_ids"]
    )
    contexts = [
        json.loads(line)
        for line in (root / "context-catalog.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    runtime_chunks = []
    target_runtime_content = ""
    target_runtime_context_id = ""
    for context in contexts:
        content = str(context["content_excerpt"])
        if context["context_id"] == target_context_id:
            # Simulate a current parser chunk which has expanded around the old
            # excerpt, leaving its old content hash intentionally unmatched.
            runtime_content = f"重建分块前缀{content}重建分块后缀"
            target_runtime_content = runtime_content
            context["content_sha256"] = "0" * 64
            target_runtime_context_id = f"{context['source_document_id']}#chunk-999"
        else:
            runtime_content = content
            context["content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        runtime_chunks.append(
            {
                "chunk_id": target_runtime_context_id if context["context_id"] == target_context_id else context["context_id"],
                "chunk_index": context["chunk_index"],
                "content": runtime_content,
                "source_document_id": context["source_document_id"],
                "title": context["title"],
                "department": context["department"],
            }
        )
    (root / "context-catalog.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in contexts),
        encoding="utf-8",
    )

    service.rebind_suite_to_current_chunks(
        "org-a",
        "evidence-gates-v1-routine",
        expected_revision=before["revision"],
        actor_id="admin-a",
        current_chunks=runtime_chunks,
    )

    rebound_contexts = [
        json.loads(line)
        for line in (root / "context-catalog.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        item["context_id"] == target_runtime_context_id
        and item["content_sha256"] == hashlib.sha256(target_runtime_content.encode("utf-8")).hexdigest()
        for item in rebound_contexts
    )


def test_suite_rebind_rejects_unresolved_frozen_and_unsupported_targets_atomically(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)
    dataset_id = "evidence-gates-v1"
    before = service.get_dataset("org-a", dataset_id)
    root = service.workspace_path("org-a", dataset_id)
    cases_before = (root / "cases.jsonl").read_bytes()
    contexts_before = (root / "context-catalog.jsonl").read_bytes()
    workspace_before = (root / "workspace.json").read_bytes()
    first_context = json.loads(
        (root / "context-catalog.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    unrelated_runtime_chunk = {
        "chunk_id": f"{first_context['source_document_id']}#chunk-9999",
        "chunk_index": 9999,
        "content": "与任何正式评测证据都无关的当前分块内容" * 20,
        "source_document_id": first_context["source_document_id"],
        "title": first_context["title"],
        "department": first_context["department"],
    }

    with pytest.raises(EvidenceReviewValidationError, match="current_corpus_context_unresolved"):
        service.rebind_suite_to_current_chunks(
            "org-a",
            dataset_id,
            expected_revision=before["revision"],
            actor_id="admin-a",
            current_chunks=[unrelated_runtime_chunk],
        )

    assert (root / "cases.jsonl").read_bytes() == cases_before
    assert (root / "context-catalog.jsonl").read_bytes() == contexts_before
    assert (root / "workspace.json").read_bytes() == workspace_before

    workspace = json.loads(workspace_before)
    workspace["status"] = "frozen"
    (root / "workspace.json").write_text(
        json.dumps(workspace, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(EvidenceReviewConflict, match="dataset_frozen"):
        service.rebind_suite_to_current_chunks(
            "org-a",
            dataset_id,
            expected_revision=before["revision"],
            actor_id="admin-a",
            current_chunks=[unrelated_runtime_chunk],
        )

    with pytest.raises(EvidenceReviewValidationError, match="source_evaluation_dataset_required"):
        service.rebind_suite_to_current_chunks(
            "org-a",
            "current-corpus-dataset",
            expected_revision=1,
            actor_id="admin-a",
            current_chunks=[unrelated_runtime_chunk],
        )


def test_pristine_workspace_refreshes_when_source_manifest_changes(tmp_path: Path) -> None:
    source = _legacy_source_bundle(tmp_path)
    service = EvidenceReviewWorkspace(tmp_path / "review-root", source)

    before = service.get_dataset("org-a", "evidence-gates-v1")
    workspace_path = service.workspace_path("org-a", "evidence-gates-v1")
    created_at = json.loads(
        (workspace_path / "workspace.json").read_text(encoding="utf-8")
    )["created_at"]
    assert before["case_count"] == 37

    _publish_current_source(source)
    assert service.refresh_pristine_workspaces() == 1
    after = service.get_dataset("org-a", "evidence-gates-v1")
    workspace = json.loads(
        (workspace_path / "workspace.json").read_text(encoding="utf-8")
    )

    assert after["case_count"] == 100
    assert after["revision"] == before["revision"] + 1
    assert workspace["created_at"] == created_at
    assert workspace["source_manifest_sha256"] == sha256_file(
        source / "manifest.json"
    )


def test_governed_workspace_merges_source_and_backfills_missing_departments(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)
    before = service.get_dataset("org-a", "evidence-gates-v1")
    workspace_path = service.workspace_path("org-a", "evidence-gates-v1")
    cases_path = workspace_path / "cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    expected_department = cases[0].pop("department_id")
    cases[0]["notes"] = "preserve governed content"
    cases[0]["last_editor_id"] = "maker-a"
    cases_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    workspace_file = workspace_path / "workspace.json"
    workspace = json.loads(workspace_file.read_text(encoding="utf-8"))
    workspace["source_manifest_sha256"] = "legacy-governed-source"
    workspace_file.write_text(json.dumps(workspace) + "\n", encoding="utf-8")

    after = service.get_dataset("org-a", "evidence-gates-v1")
    migrated = service.list_cases(
        "org-a", "evidence-gates-v1", page=1, page_size=1
    )["cases"][0]
    persisted_workspace = json.loads(workspace_file.read_text(encoding="utf-8"))

    assert after["revision"] == before["revision"] + 2
    assert migrated["department_id"] == expected_department
    assert migrated["notes"] == "preserve governed content"
    assert migrated["last_editor_id"] == "maker-a"
    assert persisted_workspace["source_manifest_sha256"] == sha256_file(
        SOURCE_BUNDLE / "manifest.json"
    )


@pytest.mark.parametrize("governed_state", ["edited", "review_event"])
def test_source_refresh_appends_to_governed_workspaces(
    tmp_path: Path,
    governed_state: str,
) -> None:
    source = _legacy_source_bundle(tmp_path)
    service = EvidenceReviewWorkspace(tmp_path / "review-root", source)
    before = service.get_dataset("org-a", "evidence-gates-v1")
    workspace_path = service.workspace_path("org-a", "evidence-gates-v1")
    legacy_ids = {
        json.loads(line)["id"]
        for line in (workspace_path / "cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    existing_id = json.loads(
        (workspace_path / "cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )["id"]

    if governed_state == "edited":
        case = service.list_cases(
            "org-a", "evidence-gates-v1", page=1, page_size=1
        )["cases"][0]
        service.update_case(
            "org-a",
            "evidence-gates-v1",
            case["id"],
            {**case, "notes": "human edit"},
            expected_revision=before["revision"],
            actor_id="maker-a",
            actor_department_id=case["department_id"],
        )
    elif governed_state == "review_event":
        (workspace_path / "review-events.jsonl").write_text(
            '{"event":"submit"}\n',
            encoding="utf-8",
        )
    _publish_current_source(source)
    assert service.refresh_pristine_workspaces() == 1
    after = service.get_dataset("org-a", "evidence-gates-v1")
    cases = service.list_cases(
        "org-a", "evidence-gates-v1", page=1, page_size=100
    )["cases"]
    existing = next(case for case in cases if case["id"] == existing_id)
    added = next(case for case in cases if case["id"] not in legacy_ids)

    assert after["case_count"] == 100
    assert after["revision"] > before["revision"]
    assert added["review_status"] == "draft"
    assert added["last_editor_id"] == "system-import"
    if governed_state == "edited":
        assert existing["notes"] == "human edit"
        assert existing["last_editor_id"] == "maker-a"
    else:
        assert (workspace_path / "review-events.jsonl").read_text(
            encoding="utf-8"
        ) == '{"event":"submit"}\n'


def test_source_refresh_does_not_mutate_frozen_workspace(tmp_path: Path) -> None:
    source = _legacy_source_bundle(tmp_path)
    service = EvidenceReviewWorkspace(tmp_path / "review-root", source)
    service.get_dataset("org-a", "evidence-gates-v1")
    workspace_path = service.workspace_path("org-a", "evidence-gates-v1")
    workspace_file = workspace_path / "workspace.json"
    workspace = json.loads(workspace_file.read_text(encoding="utf-8"))
    workspace["last_frozen_version"] = "v1"
    workspace_file.write_text(
        json.dumps(workspace, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _publish_current_source(source)
    assert service.refresh_pristine_workspaces() == 0
    after = service.get_dataset("org-a", "evidence-gates-v1")

    assert after["case_count"] == 37


def test_edit_validates_context_and_revision_and_invalidates_approval(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)
    dataset = service.get_dataset("org-a", "evidence-gates-v1")
    case = service.list_cases("org-a", "evidence-gates-v1", page=1, page_size=1)[
        "cases"
    ][0]

    with pytest.raises(EvidenceReviewValidationError, match="unknown_context_id"):
        service.update_case(
            "org-a",
            "evidence-gates-v1",
            case["id"],
            {**case, "expected_evidence_context_ids": ["unknown#chunk-0"]},
            expected_revision=dataset["revision"],
            actor_id="maker-a",
            actor_department_id=case["department_id"],
        )

    updated = service.update_case(
        "org-a",
        "evidence-gates-v1",
        case["id"],
        {**case, "notes": "由前端工作台更新"},
        expected_revision=dataset["revision"],
        actor_id="maker-a",
        actor_department_id=case["department_id"],
    )
    assert updated["case"]["review_status"] == "draft"
    assert updated["case"]["last_editor_id"] == "maker-a"
    assert updated["case"]["department_id"] == case["department_id"]
    assert "reviewer_department_id" not in updated["case"]

    with pytest.raises(EvidenceReviewConflict, match="revision_conflict"):
        service.update_case(
            "org-a",
            "evidence-gates-v1",
            case["id"],
            updated["case"],
            expected_revision=dataset["revision"],
            actor_id="maker-a",
            actor_department_id=case["department_id"],
        )


def test_maker_checker_submit_approve_reject_and_self_approval(tmp_path: Path) -> None:
    service = _workspace(tmp_path)
    dataset = service.get_dataset("org-a", "evidence-gates-v1")
    case = service.list_cases("org-a", "evidence-gates-v1", page=1, page_size=1)[
        "cases"
    ][0]
    edited = service.update_case(
        "org-a",
        "evidence-gates-v1",
        case["id"],
        {**case, "notes": "maker update"},
        expected_revision=dataset["revision"],
        actor_id="maker-a",
        actor_department_id=case["department_id"],
    )
    submitted = service.submit_case(
        "org-a",
        "evidence-gates-v1",
        case["id"],
        expected_revision=edited["revision"],
        actor_id="maker-a",
        actor_department_id=case["department_id"],
        reviewer_id="reviewer-b",
    )

    with pytest.raises(EvidenceReviewConflict, match="maker_checker_violation"):
        service.review_case(
            "org-a",
            "evidence-gates-v1",
            case["id"],
            decision="approve",
            reason="self approval",
            expected_revision=submitted["revision"],
            actor_id="maker-a",
            actor_department_id=case["department_id"],
            actor_is_department_manager=True,
        )

    with pytest.raises(EvidenceReviewPermissionError, match="assigned_reviewer_required"):
        service.review_case(
            "org-a",
            "evidence-gates-v1",
            case["id"],
            decision="approve",
            reason="not assigned",
            expected_revision=submitted["revision"],
            actor_id="manager-c",
            actor_department_id=case["department_id"],
            actor_is_department_manager=True,
        )

    approved = service.review_case(
        "org-a",
        "evidence-gates-v1",
        case["id"],
        decision="approve",
        reason="checked against context",
        expected_revision=submitted["revision"],
        actor_id="reviewer-b",
        actor_department_id=case["department_id"],
        actor_is_department_manager=True,
    )
    assert approved["case"]["review_status"] == "approved"
    assert approved["case"]["reviewer_id"] == "reviewer-b"
    assert "reviewer_department_id" not in approved["case"]

    edited_again = service.update_case(
        "org-a",
        "evidence-gates-v1",
        case["id"],
        {**approved["case"], "notes": "needs another review"},
        expected_revision=approved["revision"],
        actor_id="maker-a",
        actor_department_id=case["department_id"],
    )
    submitted_again = service.submit_case(
        "org-a",
        "evidence-gates-v1",
        case["id"],
        expected_revision=edited_again["revision"],
        actor_id="maker-a",
        actor_department_id=case["department_id"],
        reviewer_id="reviewer-b",
    )
    rejected = service.review_case(
        "org-a",
        "evidence-gates-v1",
        case["id"],
        decision="reject",
        reason="context needs correction",
        expected_revision=submitted_again["revision"],
        actor_id="reviewer-b",
        actor_department_id=case["department_id"],
        actor_is_department_manager=True,
    )
    assert rejected["case"]["review_status"] == "draft"
    assert rejected["case"]["rejection_reason"] == "context needs correction"


def test_bulk_selected_transitions_are_partial_and_bump_revision_once(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)
    dataset = service.get_dataset("org-a", "evidence-gates-v1")
    cases = service.list_cases(
        "org-a", "evidence-gates-v1", page=1, page_size=2
    )["cases"]
    maker_case, eligible_case = cases
    edited = service.update_case(
        "org-a",
        "evidence-gates-v1",
        maker_case["id"],
        maker_case,
        expected_revision=dataset["revision"],
        actor_id="reviewer-b",
        actor_department_id=maker_case["department_id"],
    )

    submitted = service.bulk_submit_cases(
        "org-a",
        "evidence-gates-v1",
        expected_revision=edited["revision"],
        selection_mode="selected",
        case_ids=[maker_case["id"], eligible_case["id"]],
        actor_id="reviewer-b",
        actor_department_id=maker_case["department_id"],
        actor_is_department_manager=True,
        actor_is_organization_admin=True,
    )

    assert submitted == {
        "revision": edited["revision"] + 1,
        "matched_count": 2,
        "processed_count": 1,
        "skipped_count": 1,
        "processed_case_ids": [eligible_case["id"]],
        "skipped_items": [
            {"case_id": maker_case["id"], "code": "maker_checker_violation"}
        ],
    }
    assert service.get_case("org-a", "evidence-gates-v1", maker_case["id"])[
        "review_status"
    ] == "draft"
    assert service.get_case("org-a", "evidence-gates-v1", eligible_case["id"])[
        "reviewer_id"
    ] == "reviewer-b"

    with pytest.raises(EvidenceReviewConflict, match="revision_conflict"):
        service.bulk_approve_cases(
            "org-a",
            "evidence-gates-v1",
            expected_revision=edited["revision"],
            selection_mode="selected",
            case_ids=[eligible_case["id"]],
            actor_id="reviewer-b",
            actor_department_id=maker_case["department_id"],
            actor_is_department_manager=True,
            actor_is_organization_admin=True,
        )

    approved = service.bulk_approve_cases(
        "org-a",
        "evidence-gates-v1",
        expected_revision=submitted["revision"],
        selection_mode="selected",
        case_ids=[maker_case["id"], eligible_case["id"]],
        actor_id="reviewer-b",
        actor_department_id=maker_case["department_id"],
        actor_is_department_manager=True,
        actor_is_organization_admin=True,
    )
    assert approved["revision"] == submitted["revision"] + 1
    assert approved["processed_case_ids"] == [eligible_case["id"]]
    assert approved["skipped_items"] == [
        {"case_id": maker_case["id"], "code": "case_not_pending_review"}
    ]
    assert service.get_case("org-a", "evidence-gates-v1", eligible_case["id"])[
        "review_status"
    ] == "approved"


def test_delete_case_requires_organization_admin_and_records_authoring_event(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)
    before = service.get_dataset("org-a", "evidence-gates-v1")
    case = service.list_cases(
        "org-a", "evidence-gates-v1", page=1, page_size=1
    )["cases"][0]

    with pytest.raises(
        EvidenceReviewPermissionError, match="organization_admin_required"
    ):
        service.delete_case(
            "org-a",
            "evidence-gates-v1",
            case["id"],
            expected_revision=before["revision"],
            actor_id="department-admin",
            actor_is_organization_admin=False,
        )

    unchanged = service.get_dataset("org-a", "evidence-gates-v1")
    assert unchanged["revision"] == before["revision"]
    assert unchanged["case_count"] == 100

    deleted = service.delete_case(
        "org-a",
        "evidence-gates-v1",
        case["id"],
        expected_revision=before["revision"],
        actor_id="org-admin-a",
        actor_is_organization_admin=True,
    )

    after = service.get_dataset("org-a", "evidence-gates-v1")
    events = [
        json.loads(line)
        for line in (
            service.workspace_path("org-a", "evidence-gates-v1") / "review-events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]

    assert deleted == {
        "revision": before["revision"] + 1,
        "deleted_case_id": case["id"],
    }
    assert after["status"] == "authoring"
    assert after["case_count"] == 99
    assert case["id"] not in {
        item["id"]
        for item in service.list_cases(
            "org-a", "evidence-gates-v1", page=1, page_size=100
        )["cases"]
    }
    assert events[-1] == {
        "event": "delete",
        "dataset_id": "evidence-gates-v1",
        "case_id": case["id"],
        "actor_id": "org-admin-a",
        "department_id": case["department_id"],
        "revision": deleted["revision"],
        "recorded_at": events[-1]["recorded_at"],
    }



def test_bulk_filtered_scope_crosses_pages_and_zero_success_does_not_write(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)
    dataset = service.get_dataset("org-a", "evidence-gates-v1")

    submitted = service.bulk_submit_cases(
        "org-a",
        "evidence-gates-v1",
        expected_revision=dataset["revision"],
        selection_mode="filtered",
        case_ids=[],
        review_status="draft",
        actor_id="org-admin-a",
        actor_department_id="human_resources",
        actor_is_department_manager=False,
        actor_is_organization_admin=True,
    )

    assert submitted["matched_count"] == 100
    assert submitted["processed_count"] == 100
    assert submitted["skipped_count"] == 0
    assert len(submitted["processed_case_ids"]) == 100
    assert service.get_dataset("org-a", "evidence-gates-v1")["counts"] == {
        "approved": 0,
        "draft": 0,
        "pending_review": 100,
    }

    unchanged = service.bulk_submit_cases(
        "org-a",
        "evidence-gates-v1",
        expected_revision=submitted["revision"],
        selection_mode="filtered",
        case_ids=[],
        review_status="draft",
        actor_id="org-admin-a",
        actor_department_id="human_resources",
        actor_is_department_manager=False,
        actor_is_organization_admin=True,
    )
    assert unchanged == {
        "revision": submitted["revision"],
        "matched_count": 0,
        "processed_count": 0,
        "skipped_count": 0,
        "processed_case_ids": [],
        "skipped_items": [],
    }


def test_freeze_is_immutable_and_draft_derivation_preserves_history(tmp_path: Path) -> None:
    """Freezing publishes immutable material; all later authoring starts from a draft."""
    service = _workspace(tmp_path)
    dataset = service.get_dataset("org-a", "evidence-gates-v1")

    with pytest.raises(EvidenceReviewConflict, match="review_incomplete"):
        service.freeze(
            "org-a", "evidence-gates-v1", expected_revision=dataset["revision"], actor_id="release-admin"
        )

    revision = dataset["revision"]
    cases = service.list_cases("org-a", "evidence-gates-v1", page=1, page_size=100)["cases"]
    for case in cases:
        if case["category"] in {
            "fully_answerable", "partially_answerable", "conflicting",
            "background_only", "missing_version_or_date",
            "missing_business_record", "single_branch_unavailable",
        }:
            updated = service.update_case(
                "org-a", "evidence-gates-v1", case["id"],
                {**case, "reference_answer": "经独立审核的参考答案"},
                expected_revision=revision, actor_id="maker-a",
                actor_department_id=case["department_id"],
            )
            revision = updated["revision"]
        submitted = service.submit_case(
            "org-a", "evidence-gates-v1", case["id"], expected_revision=revision,
            actor_id="maker-a", actor_department_id=case["department_id"], reviewer_id="reviewer-b",
        )
        approved = service.review_case(
            "org-a", "evidence-gates-v1", case["id"], decision="approve", reason="verified",
            expected_revision=submitted["revision"], actor_id="reviewer-b",
            actor_department_id=case["department_id"], actor_is_department_manager=True,
        )
        revision = approved["revision"]

    frozen = service.freeze(
        "org-a", "evidence-gates-v1", expected_revision=revision, actor_id="release-admin"
    )
    version_path = Path(frozen["version_path"])
    frozen_cases = (version_path / "cases.jsonl").read_bytes()
    workspace_path = service.workspace_path("org-a", "evidence-gates-v1")
    workspace_before = (workspace_path / "workspace.json").read_bytes()

    summary = service.get_dataset("org-a", "evidence-gates-v1")
    assert summary["status"] == "frozen"
    assert summary["last_frozen_version"] == frozen["version"]
    assert summary["last_frozen_manifest_sha256"] == frozen["manifest_sha256"]
    assert summary["last_frozen_at"]
    assert summary["last_frozen_by"] == "release-admin"
    assert service.list_versions("org-a", "evidence-gates-v1") == [
        {
            "version": frozen["version"],
            "manifest_sha256": frozen["manifest_sha256"],
            "frozen_at": summary["last_frozen_at"],
            "frozen_by": "release-admin",
            "source_workspace_revision": revision,
            "current_corpus_draft_id": None,
            "current_corpus_snapshot_sha256": None,
        }
    ]
    assert service.get_version("org-a", "evidence-gates-v1", frozen["version"])["version"] == frozen["version"]
    assert "version_path" not in service.get_version("org-a", "evidence-gates-v1", frozen["version"])

    current_case = service.list_cases("org-a", "evidence-gates-v1", page=1, page_size=1)["cases"][0]
    mutations = (
        lambda: service.delete_case("org-a", "evidence-gates-v1", current_case["id"], expected_revision=frozen["revision"], actor_id="release-admin", actor_is_organization_admin=True),
        lambda: service.create_case("org-a", "evidence-gates-v1", {**current_case, "id": "frozen-new-case"}, expected_revision=frozen["revision"], actor_id="release-admin", actor_department_id=current_case["department_id"]),
        lambda: service.update_case("org-a", "evidence-gates-v1", current_case["id"], current_case, expected_revision=frozen["revision"], actor_id="release-admin", actor_department_id=current_case["department_id"]),
        lambda: service.submit_case("org-a", "evidence-gates-v1", current_case["id"], expected_revision=frozen["revision"], actor_id="release-admin", actor_department_id=current_case["department_id"], reviewer_id="reviewer-b"),
        lambda: service.review_case("org-a", "evidence-gates-v1", current_case["id"], decision="approve", reason="again", expected_revision=frozen["revision"], actor_id="reviewer-b", actor_department_id=current_case["department_id"], actor_is_department_manager=True),
        lambda: service.bulk_submit_cases("org-a", "evidence-gates-v1", expected_revision=frozen["revision"], selection_mode="selected", case_ids=[current_case["id"]], review_status=None, actor_id="release-admin", actor_department_id=current_case["department_id"], actor_is_department_manager=True, actor_is_organization_admin=True),
        lambda: service.bulk_approve_cases("org-a", "evidence-gates-v1", expected_revision=frozen["revision"], selection_mode="selected", case_ids=[current_case["id"]], review_status=None, actor_id="release-admin", actor_department_id=current_case["department_id"], actor_is_department_manager=True, actor_is_organization_admin=True),
        lambda: service.freeze("org-a", "evidence-gates-v1", expected_revision=frozen["revision"], actor_id="release-admin"),
    )
    for mutate in mutations:
        with pytest.raises(EvidenceReviewConflict, match="dataset_frozen"):
            mutate()

    assert service.get_dataset("org-a", "evidence-gates-v1")["revision"] == frozen["revision"]
    assert (workspace_path / "workspace.json").read_bytes() == workspace_before
    assert (version_path / "cases.jsonl").read_bytes() == frozen_cases

    drafted = service.create_draft_from_version(
        "org-a", "evidence-gates-v1", frozen["version"], expected_revision=frozen["revision"], actor_id="company-admin-b"
    )
    assert drafted["status"] == "authoring"
    assert drafted["source_frozen_version"] == frozen["version"]
    assert drafted["revision"] == frozen["revision"] + 1
    draft_cases = service.list_cases("org-a", "evidence-gates-v1", page=1, page_size=100)["cases"]
    assert all(case["review_status"] == "draft" and case["reviewer_id"] is None for case in draft_cases)
    assert (version_path / "cases.jsonl").read_bytes() == frozen_cases


def test_refresh_existing_current_corpus_normalizes_legacy_smoke_and_supersedes_old_pointers(
    tmp_path: Path,
) -> None:
    """An interrupted refresh restores real Chunk bindings and becomes current."""
    service = _workspace(tmp_path)
    org_id = "org-a"
    source_dataset_id = "current-corpus-source"
    template = next(
        json.loads(line)
        for line in (SOURCE_BUNDLE / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("required_fixture") == "bm25_unavailable_dense_graph_available"
    )
    source_contexts = {
        item["context_id"]: item
        for item in (
            json.loads(line)
            for line in (SOURCE_BUNDLE / "context-catalog.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    bound_contexts = [
        source_contexts[context_id]
        for context_id in template["expected_evidence_context_ids"]
    ]
    source_documents = [
        {
            "document_id": context["source_document_id"],
            "source": context["title"],
            "department": context["department"],
            "version": 1,
        }
        for context in bound_contexts
    ]
    snapshot_payload = [
        {
            "document_id": item["document_id"],
            "source": item["source"],
            "department": item["department"],
        }
        for item in source_documents
    ]
    snapshot_sha256 = hashlib.sha256(
        json.dumps(snapshot_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    target_dataset_id = f"current-corpus-{snapshot_sha256[:32]}"

    def write_workspace(dataset_id: str, *, active: bool, target: bool = False) -> None:
        root = service.workspace_path(org_id, dataset_id)
        root.mkdir(parents=True)
        workspace = {
            "schema_version": "evidence-review-workspace-v1",
            "dataset_id": dataset_id,
            "title": dataset_id,
            "org_id_hash": service._org_namespace(org_id),
            "revision": 1,
            "status": "authoring" if target else "frozen",
            "source_type": "current_corpus",
            "active": active,
            "last_frozen_version": None if target else "v1",
            "source_document_count": 1,
        }
        case = {
            **template,
            "id": "smoke-bm25",
            "expected_response_status": "partially_answered",
            "review_status": "approved",
        }
        (root / "workspace.json").write_text(json.dumps(workspace), encoding="utf-8")
        (root / "cases.jsonl").write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
        (root / "context-catalog.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in bound_contexts),
            encoding="utf-8",
        )
        (root / "fixture-profile.json").write_text(
            json.dumps(
                {
                    "status": "reviewed",
                    "fixtures": {
                        "bm25_unavailable_dense_graph_available": {
                            "source_document_ids": [
                                context["source_document_id"]
                                for context in bound_contexts
                            ],
                            "source_document_count": len(bound_contexts),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / "review-events.jsonl").write_text("", encoding="utf-8")

    write_workspace(source_dataset_id, active=True)
    write_workspace("current-corpus-older", active=True)
    write_workspace(target_dataset_id, active=True, target=True)

    refreshed = service.refresh_current_corpus_dataset(
        org_id,
        source_dataset_id,
        source_documents=source_documents,
        current_chunks=[
            {
                "chunk_id": context["context_id"],
                "chunk_index": context["chunk_index"],
                "content": context["content_excerpt"],
                "source_document_id": context["source_document_id"],
            }
            for context in bound_contexts
        ],
        actor_id="admin-a",
    )

    assert refreshed["dataset_id"] == target_dataset_id
    assert refreshed["revision"] == 2
    refreshed_case = service.list_cases(org_id, target_dataset_id, page=1, page_size=10)["cases"][0]
    assert refreshed_case["expected_response_status"] == "answered"
    assert refreshed_case["expected_evidence_context_ids"] == template["expected_evidence_context_ids"]
    assert refreshed_case["expected_citation_context_ids"] == template["expected_citation_context_ids"]
    assert service.get_dataset(org_id, target_dataset_id)["active"] is True
    for old_dataset_id in (source_dataset_id, "current-corpus-older"):
        old = service.get_dataset(org_id, old_dataset_id)
        assert old["active"] is False
        assert old["superseded_by"] == target_dataset_id


def test_snapshot_catalog_derives_unique_document_count_for_legacy_manifest(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)
    root = service.workspace_path("org-a", "current-corpus-legacy")
    version_root = root / "versions" / "legacy-v1"
    version_root.mkdir(parents=True)
    (root / "workspace.json").write_text(
        json.dumps(
            {
                "dataset_id": "current-corpus-legacy",
                "status": "frozen",
                "source_type": "current_corpus",
                "org_id_hash": service._org_namespace("org-a"),
            }
        ),
        encoding="utf-8",
    )
    (root / "cases.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "context-catalog.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "fixture-profile.json").write_text("{}\n", encoding="utf-8")
    (root / "review-events.jsonl").write_text("", encoding="utf-8")
    (version_root / "manifest.json").write_text(
        json.dumps({"dataset_version": "legacy-v1"}),
        encoding="utf-8",
    )
    (version_root / "context-catalog.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "context_id": f"doc-a#chunk-{index}",
                    "source_document_id": "doc-a",
                    "title": "制度 A",
                    "department": "finance",
                    "chunk_index": index,
                    "content_excerpt": f"内容 {index}",
                },
                ensure_ascii=False,
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )

    result = service.list_version_contexts(
        "org-a", "current-corpus-legacy", "legacy-v1"
    )

    assert result["source_document_count"] == 1
    assert len(result["contexts"]) == 1
    assert result["contexts"][0]["source_document_id"] == "doc-a"


def test_normalize_current_corpus_fixture_expectation_uses_current_contract_codes() -> None:
    prompt_injection = {
        "required_fixture": "prompt_injection_safety_fixture",
        "expected_reason_codes": ["high_risk_review"],
    }
    bm25 = {
        "required_fixture": "bm25_unavailable_dense_graph_available",
        "expected_response_status": "partially_answered",
    }

    assert EvidenceReviewWorkspace._normalize_current_corpus_fixture_expectation(prompt_injection) == 1
    assert prompt_injection["expected_reason_codes"] == ["prompt_injection_detected"]
    assert EvidenceReviewWorkspace._normalize_current_corpus_fixture_expectation(bm25) == 1
    assert bm25["expected_response_status"] == "answered"


def _current_corpus_runtime_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build authenticated current-corpus inputs from the checked-in source catalog."""

    source_contexts = [
        json.loads(line)
        for line in (SOURCE_BUNDLE / "context-catalog.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    documents: dict[str, dict[str, object]] = {}
    current_chunks: list[dict[str, object]] = []
    for context in source_contexts:
        document_id = str(context["source_document_id"])
        documents.setdefault(
            document_id,
            {
                "document_id": document_id,
                "source": str(context["title"]),
                "department": str(context["department"]),
                "version": 1,
            },
        )
        current_chunks.append(
            {
                "chunk_id": str(context["context_id"]),
                "chunk_index": int(context["chunk_index"]),
                "content": str(context["content_excerpt"]),
                "source_document_id": document_id,
            }
        )
    return list(documents.values()), current_chunks


def test_current_corpus_candidates_are_document_verified_drafts_and_never_freeze(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)
    source_documents, current_chunks = _current_corpus_runtime_inputs()

    draft = service.refresh_current_corpus_dataset(
        "org-a",
        "evidence-gates-v1-routine",
        source_documents=source_documents,
        current_chunks=current_chunks,
        actor_id="org-admin-a",
    )
    before = service.list_cases(
        "org-a", draft["dataset_id"], page=1, page_size=100
    )["cases"]
    answer_case_ids = {
        case["id"]
        for case in before
        if case["category"] in {
            "fully_answerable",
            "partially_answerable",
            "conflicting",
            "background_only",
            "missing_version_or_date",
            "missing_business_record",
            "single_branch_unavailable",
        }
    }
    assert answer_case_ids == {
        "eg-company-demo-01",
        "eg-company-demo-02",
        "eg-company-demo-06",
        "eg-company-demo-10",
        "eg-company-demo-17",
        "eg-company-demo-19",
        "eg-company-demo-69",
    }
    assert all(not case["reference_answer"] for case in before if case["id"] in answer_case_ids)

    populated = service.populate_reference_answer_candidates(
        "org-a",
        draft["dataset_id"],
        expected_revision=draft["revision"],
        actor_id="org-admin-a",
    )

    assert set(populated["populated_case_ids"]) == answer_case_ids
    assert populated["skipped"] == []
    after = service.list_cases(
        "org-a", draft["dataset_id"], page=1, page_size=100
    )["cases"]
    populated_cases = [case for case in after if case["id"] in answer_case_ids]
    assert all(case["reference_answer"] for case in populated_cases)
    assert all(case["review_status"] == "draft" for case in populated_cases)
    assert all(case["last_editor_id"] == "system-reference-answer-candidate-v1" for case in populated_cases)
    assert service.get_dataset("org-a", draft["dataset_id"])["status"] == "authoring"
    assert not (service.workspace_path("org-a", draft["dataset_id"]) / "versions").exists()


def test_candidate_population_skips_when_question_contract_no_longer_matches(
    tmp_path: Path,
) -> None:
    service = _workspace(tmp_path)
    source_documents, current_chunks = _current_corpus_runtime_inputs()
    draft = service.refresh_current_corpus_dataset(
        "org-a",
        "evidence-gates-v1-routine",
        source_documents=source_documents,
        current_chunks=current_chunks,
        actor_id="org-admin-a",
    )
    catalog = json.loads(service.reference_answer_candidates_path.read_text(encoding="utf-8"))
    catalog["candidates"][0]["question"] = "与当前冻结题干不相同的问题"
    service.reference_answer_candidates_path = tmp_path / "mismatched-candidates.json"
    service.reference_answer_candidates_path.write_text(
        json.dumps(catalog, ensure_ascii=False), encoding="utf-8"
    )

    populated = service.populate_reference_answer_candidates(
        "org-a",
        draft["dataset_id"],
        expected_revision=draft["revision"],
        actor_id="org-admin-a",
    )

    assert "eg-company-demo-01" not in populated["populated_case_ids"]
    assert {item["case_id"]: item["code"] for item in populated["skipped"]} == {
        "eg-company-demo-01": "reference_answer_candidate_contract_mismatch",
    }
    cases = service.list_cases("org-a", draft["dataset_id"], page=1, page_size=100)["cases"]
    assert next(case for case in cases if case["id"] == "eg-company-demo-01")["reference_answer"] == ""


def test_candidate_matches_current_template_by_its_stable_source_benchmark_id() -> None:
    candidate = {
        "case_id": "eg-company-demo-01",
        "question": "公司人力资源管理制度的主要管理范围是什么？",
        "category": "fully_answerable",
        "expected_response_status": "answered",
        "expected_evidence_states": ["direct_evidence"],
        "expected_reason_codes": ["direct_support"],
        "expected_source_document_ids": ["hr-policy"],
        "expected_evidence_sections": ["第一章 总则"],
        "expected_missing_information_fields": [],
        "required_fixture": "standard",
        "expected_branch_availability": {
            "dense": "available",
            "bm25": "available",
            "graph": "available",
        },
        "source_document_sha256": {"hr-policy": "0" * 64},
    }
    current_template = {
        **candidate,
        "id": "current-template-standard-fully_answerable",
        "source_benchmark_ids": ["eg-company-demo-01"],
    }

    assert EvidenceReviewWorkspace._reference_answer_candidate_matches(
        current_template, candidate
    )
    assert EvidenceReviewWorkspace._reference_answer_candidate_for_case(
        current_template,
        {"eg-company-demo-01": candidate},
    ) == candidate
    assert not EvidenceReviewWorkspace._reference_answer_candidate_matches(
        {**current_template, "question": "不同题干"}, candidate
    )
    assert EvidenceReviewWorkspace._reference_answer_candidate_for_case(
        {**current_template, "question": "不同题干"},
        {"eg-company-demo-01": candidate},
    ) is None
