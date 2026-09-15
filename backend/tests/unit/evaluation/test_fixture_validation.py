from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from evaluation.fixture_validation import (
    REQUIRED_FIXTURES,
    FinanceFixtureIdentityResolver,
    build_catalog_fixture_corpus_probe,
    FixtureAssertionRuntime,
    FixtureValidationConflict,
    FixtureValidationService,
    FixtureValidationStore,
)


@dataclass(frozen=True)
class _Case:
    required_fixture: str
    expected_branch_availability: dict[str, str]
    expected_response_status: str
    expected_source_document_ids: tuple[str, ...] = ()
    department_id: str = ""
    expected_reason_codes: tuple[str, ...] = ()


def _cases() -> list[_Case]:
    return [
        _Case("all_retrieval_branches_unavailable", {"dense": "unavailable", "bm25": "unavailable", "graph": "unavailable"}, "source_unavailable"),
        _Case("bm25_unavailable_dense_graph_available", {"dense": "available", "bm25": "unavailable", "graph": "available"}, "answered"),
        _Case("equal_authority_conflicting_documents", {}, "conflicting_evidence", ("doc-a", "doc-b")),
        _Case("finance_only_user_against_administration_document", {}, "insufficient_evidence", ("admin-doc",), "administration"),
        _Case("finance_only_user_against_hr_document", {}, "insufficient_evidence", ("hr-doc",), "human_resources"),
        _Case("frozen_corpus_absence_check", {}, "insufficient_evidence"),
        _Case("prompt_injection_safety_fixture", {}, "refused"),
    ]


def _result(name: str, *, status: str = "passed", reason_code: str = "ok", summary: str = "验证通过", **extra: Any) -> dict[str, Any]:
    return {
        "fixture": name,
        "status": status,
        "reason_code": reason_code,
        "started_at": "2026-08-15T00:00:00+00:00",
        "finished_at": "2026-08-15T00:00:01+00:00",
        "summary": summary,
        **extra,
    }


def _probes(
    cases: list[_Case],
    org_id: str,
    identity: FinanceFixtureIdentityResolver,
) -> Mapping[str, list[Mapping[str, Any]]]:
    assert org_id == "org-a"
    assert identity.resolve(org_id).identity_id == "evaluation-fixture-finance"
    observations: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        name = case.required_fixture
        if name == "all_retrieval_branches_unavailable":
            result = {
                "response_status": "source_unavailable",
                "branch_availability": {"dense": "unavailable", "bm25": "unavailable", "graph": "unavailable"},
            }
        elif name == "bm25_unavailable_dense_graph_available":
            result = {
                "response_status": "answered",
                "branch_availability": {"dense": "available", "bm25": "unavailable", "graph": "available"},
            }
        elif name == "equal_authority_conflicting_documents":
            result = {"response_status": "conflicting_evidence", "expected_source_match_count": 2}
        elif name.startswith("finance_only_user"):
            result = {
                "response_status": "insufficient_evidence",
                "expected_source_match_count": 0,
                "protected_context_disclosed": False,
                "identity_scope_valid": True,
            }
        elif name == "frozen_corpus_absence_check":
            result = {"response_status": "insufficient_evidence", "expected_source_match_count": 0}
        else:
            result = {"response_status": "refused", "safety_action_applied": True}
        observations.setdefault(name, []).append(result)
    return observations


def test_finance_fixture_identity_accepts_existing_department_identifier_formats() -> None:
    identity = FinanceFixtureIdentityResolver("123456").resolve("org-a")

    assert identity.department_id == "123456"
    assert identity.visible_department_ids == ("123456",)


@pytest.mark.parametrize("department_id", ["", "   ", "finance scope", "x" * 129])
def test_finance_fixture_identity_rejects_missing_or_unsafe_department_identifier(
    department_id: str,
) -> None:
    with pytest.raises(ValueError, match="fixture_identity_not_configured"):
        FinanceFixtureIdentityResolver(department_id).resolve("org-a")


def test_store_binds_all_seven_results_keeps_output_bounded_and_rejects_active_duplicate(tmp_path: Path) -> None:
    store = FixtureValidationStore(tmp_path)
    run = store.start(
        dataset_id="dataset",
        version="v1",
        manifest_sha256="a" * 64,
        source_workspace_revision=3,
        initiated_by="admin",
    )
    with pytest.raises(FixtureValidationConflict, match="active"):
        store.start(
            dataset_id="dataset",
            version="v1",
            manifest_sha256="a" * 64,
            source_workspace_revision=3,
            initiated_by="admin",
        )
    finished = store.finish(
        run["run_id"],
        fixture_results=[_result(name, protected_context="do not persist") for name in REQUIRED_FIXTURES],
        reason_code=None,
    )
    assert finished["status"] == "succeeded"
    assert finished["manifest_sha256"] == "a" * 64
    assert all(set(item) == {"fixture", "status", "reason_code", "started_at", "finished_at", "summary"} for item in finished["fixture_results"])
    assert "tenant_id" not in str(finished)
    assert "credential" not in str(finished).lower()
    assert "protected_context" not in str(finished)
    assert store.has_success(manifest_sha256="a" * 64)


def test_store_retains_all_results_when_one_fixture_fails_and_allows_retry(tmp_path: Path) -> None:
    store = FixtureValidationStore(tmp_path)
    first = store.start(
        dataset_id="dataset",
        version="v1",
        manifest_sha256="b" * 64,
        source_workspace_revision=3,
        initiated_by="admin",
    )
    results = [
        _result(
            name,
            status="failed" if name == "prompt_injection_safety_fixture" else "passed",
            reason_code="fixture_runtime_expectation_invalid" if name == "prompt_injection_safety_fixture" else "ok",
            summary="验证失败" if name == "prompt_injection_safety_fixture" else "验证通过",
        )
        for name in REQUIRED_FIXTURES
    ]
    finished = store.finish(first["run_id"], fixture_results=results, reason_code="fixture_runtime_expectation_invalid")
    assert finished["status"] == "failed"
    assert len(finished["fixture_results"]) == 7
    assert next(item for item in finished["fixture_results"] if item["fixture"] == "prompt_injection_safety_fixture")["status"] == "failed"
    retry = store.start(
        dataset_id="dataset",
        version="v1",
        manifest_sha256="b" * 64,
        source_workspace_revision=3,
        initiated_by="admin",
    )
    assert retry["status"] == "queued"


def test_runtime_executes_every_fixture_with_server_scope_and_preserves_no_sensitive_output(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FixtureAssertionRuntime(FinanceFixtureIdentityResolver("finance"), probe=_probes)
    import evaluation.fixture_validation.assertions as module

    monkeypatch.setattr(module, "load_evidence_gate_benchmark", lambda *args, **kwargs: _cases())
    results = runtime.run(
        manifest_path="/frozen/manifest.json",
        expected_manifest_sha256="c" * 64,
        org_id="org-a",
    )
    assert {item["fixture"] for item in results} == set(REQUIRED_FIXTURES)
    assert all(item["status"] == "passed" for item in results)
    assert all("finance" not in item["summary"] for item in results)


def test_runtime_accepts_human_review_as_bounded_denial_and_expected_safety_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        replace(
            case,
            expected_response_status=(
                "human_review_required"
                if case.required_fixture == "prompt_injection_safety_fixture"
                else case.expected_response_status
            ),
            expected_reason_codes=(
                ("high_risk_review",)
                if case.required_fixture == "prompt_injection_safety_fixture"
                else case.expected_reason_codes
            ),
        )
        for case in _cases()
    ]

    def probes(
        probe_cases: list[_Case],
        *args: Any,
        **kwargs: Any,
    ) -> Mapping[str, list[Mapping[str, Any]]]:
        observations = {
            name: list(items) for name, items in _probes(probe_cases, *args, **kwargs).items()
        }
        for name in (
            "finance_only_user_against_administration_document",
            "finance_only_user_against_hr_document",
        ):
            observations[name][0] = {
                "response_status": "human_review_required",
                "expected_source_match_count": 0,
                "protected_context_disclosed": False,
                "identity_scope_valid": True,
            }
        observations["prompt_injection_safety_fixture"][0] = {
            "response_status": "human_review_required",
            "expected_source_match_count": 0,
            "expected_reason_match_count": 1,
            "protected_context_disclosed": False,
            "safety_action_applied": False,
        }
        return observations

    runtime = FixtureAssertionRuntime(
        FinanceFixtureIdentityResolver("finance"),
        probe=probes,
    )
    import evaluation.fixture_validation.assertions as module

    monkeypatch.setattr(module, "load_evidence_gate_benchmark", lambda *args, **kwargs: cases)
    results = runtime.run(
        manifest_path="/frozen/manifest.json",
        expected_manifest_sha256="f" * 64,
        org_id="org-a",
    )

    assert all(item["status"] == "passed" for item in results)


def test_runtime_excludes_non_fixture_samples_from_live_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_cases: list[_Case] = []

    def probe(cases: list[_Case], *args: Any, **kwargs: Any) -> Mapping[str, list[Mapping[str, Any]]]:
        observed_cases.extend(cases)
        return _probes(cases, *args, **kwargs)

    runtime = FixtureAssertionRuntime(FinanceFixtureIdentityResolver("finance"), probe=probe)
    import evaluation.fixture_validation.assertions as module

    non_fixture = _Case(
        required_fixture="standard",
        expected_branch_availability={},
        expected_response_status="answered",
        department_id="",
        expected_source_document_ids=(),
    )
    monkeypatch.setattr(module, "load_evidence_gate_benchmark", lambda *args, **kwargs: [*_cases(), non_fixture])
    results = runtime.run(
        manifest_path="/frozen/manifest.json",
        expected_manifest_sha256="e" * 64,
        org_id="org-a",
    )

    assert all(case.required_fixture in REQUIRED_FIXTURES for case in observed_cases)
    assert len(observed_cases) == len(REQUIRED_FIXTURES)
    assert all(item["status"] == "passed" for item in results)


def test_runtime_fails_closed_when_frozen_source_documents_are_not_in_the_tenant_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Document:
        ingest_status = "ingested"
        department_id = ""

    class _Catalog:
        def get_document(self, document_id: str, *, tenant_id: str):
            assert tenant_id == "org-a"
            if document_id == "hr-doc":
                return None
            document = _Document()
            document.department_id = (
                "administration"
                if document_id == "admin-doc"
                else "human_resources"
                if document_id == "hr-doc"
                else "finance"
            )
            return document

    observed_cases: list[_Case] = []

    def probe(cases: list[_Case], *args: Any, **kwargs: Any) -> Mapping[str, list[Mapping[str, Any]]]:
        observed_cases.extend(cases)
        return _probes(cases, *args, **kwargs)

    runtime = FixtureAssertionRuntime(
        FinanceFixtureIdentityResolver("finance"),
        probe=probe,
        corpus_probe=build_catalog_fixture_corpus_probe(_Catalog()),
    )
    import evaluation.fixture_validation.assertions as module

    monkeypatch.setattr(module, "load_evidence_gate_benchmark", lambda *args, **kwargs: _cases())
    results = runtime.run(
        manifest_path="/frozen/manifest.json",
        expected_manifest_sha256="a" * 64,
        org_id="org-a",
    )

    failed = next(
        item
        for item in results
        if item["fixture"] == "finance_only_user_against_hr_document"
    )
    assert failed["reason_code"] == "fixture_runtime_corpus_unaligned"
    assert all(
        case.required_fixture != "finance_only_user_against_hr_document"
        for case in observed_cases
    )


def test_runtime_fails_only_the_affected_fixture_when_a_real_probe_assertion_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_probes(*args: Any, **kwargs: Any) -> Mapping[str, list[Mapping[str, Any]]]:
        observations = {name: list(items) for name, items in _probes(*args, **kwargs).items()}
        observations["finance_only_user_against_hr_document"][0] = {
            "response_status": "answered",
            "expected_source_match_count": 1,
            "protected_context_disclosed": True,
            "identity_scope_valid": True,
        }
        return observations

    runtime = FixtureAssertionRuntime(FinanceFixtureIdentityResolver("finance"), probe=failing_probes)
    import evaluation.fixture_validation.assertions as module

    monkeypatch.setattr(module, "load_evidence_gate_benchmark", lambda *args, **kwargs: _cases())
    results = runtime.run(manifest_path="/frozen/manifest.json", expected_manifest_sha256="d" * 64, org_id="org-a")
    assert next(item for item in results if item["fixture"] == "finance_only_user_against_hr_document")["status"] == "failed"
    assert sum(item["status"] == "failed" for item in results) == 1


def test_runtime_reports_live_dependency_failure_instead_of_a_false_fixture_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_dense(*args: Any, **kwargs: Any) -> Mapping[str, list[Mapping[str, Any]]]:
        observations = {name: list(items) for name, items in _probes(*args, **kwargs).items()}
        observations["bm25_unavailable_dense_graph_available"][0] = {
            "response_status": "answered",
            "branch_availability": {
                "dense": "unavailable",
                "bm25": "unavailable",
                "graph": "available",
            },
        }
        return observations

    runtime = FixtureAssertionRuntime(
        FinanceFixtureIdentityResolver("finance"), probe=unavailable_dense
    )
    import evaluation.fixture_validation.assertions as module

    monkeypatch.setattr(module, "load_evidence_gate_benchmark", lambda *args, **kwargs: _cases())
    results = runtime.run(
        manifest_path="/frozen/manifest.json",
        expected_manifest_sha256="e" * 64,
        org_id="org-a",
    )

    failed = next(
        item
        for item in results
        if item["fixture"] == "bm25_unavailable_dense_graph_available"
    )
    assert failed["status"] == "failed"
    assert failed["reason_code"] == "fixture_runtime_dependency_unavailable"


def test_service_rejects_runtime_corpus_mismatch_before_queueing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Workspace:
        def _paths(self, _org_id: str, _dataset_id: str) -> dict[str, Path]:
            return {"versions": tmp_path}

        def _version_root(self, paths: Mapping[str, Path], version: str) -> Path:
            return paths["versions"] / version

        def get_version(self, _org_id: str, _dataset_id: str, version: str) -> dict[str, Any]:
            return {"source_workspace_revision": 7, "manifest_sha256": "e" * 64, "version": version}


    monkeypatch.setattr("evaluation.fixture_validation.service.load_evidence_gate_benchmark", lambda *args, **kwargs: _cases())
    service = FixtureValidationService(
        _Workspace(),
        finance_department_id="finance",
        corpus_probe=lambda _cases, _org_id: {
            "equal_authority_conflicting_documents": "fixture_runtime_corpus_unaligned"
        },
    )

    with pytest.raises(FixtureValidationConflict, match="fixture_runtime_corpus_unaligned"):
        service.start(
            org_id="org-a",
            dataset_id="dataset",
            version="v1",
            expected_revision=7,
            initiated_by="admin",
        )

    assert not (tmp_path / "v1" / "fixture-validation-runs").exists()


def test_service_rejects_revision_or_manifest_mismatch(tmp_path: Path) -> None:
    class _Workspace:
        def _paths(self, _org_id: str, _dataset_id: str) -> dict[str, Path]:
            return {"versions": tmp_path}

        def _version_root(self, paths: Mapping[str, Path], version: str) -> Path:
            return paths["versions"] / version

        def get_version(self, _org_id: str, _dataset_id: str, version: str) -> dict[str, Any]:
            return {"source_workspace_revision": 7, "manifest_sha256": "e" * 64, "version": version}

    service = FixtureValidationService(_Workspace(), finance_department_id="finance")
    with pytest.raises(FixtureValidationConflict, match="revision_conflict"):
        service.start(org_id="org-a", dataset_id="dataset", version="v1", expected_revision=6, initiated_by="admin")

    run = service.start(org_id="org-a", dataset_id="dataset", version="v1", expected_revision=7, initiated_by="admin")
    record_path = tmp_path / "v1" / "fixture-validation-runs" / f"{run['run_id']}.json"
    body = record_path.read_text(encoding="utf-8").replace('"manifest_sha256": "' + "e" * 64 + '"', '"manifest_sha256": "' + "f" * 64 + '"')
    record_path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_mismatch"):
        service.get(org_id="org-a", dataset_id="dataset", version="v1", run_id=run["run_id"])


def test_finance_identity_fails_closed_when_not_configured() -> None:
    identity = FinanceFixtureIdentityResolver("")
    with pytest.raises(ValueError, match="not_configured"):
        identity.resolve("org-a")
