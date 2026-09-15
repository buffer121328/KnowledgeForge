"""Source-only dependency direction checks for backend architecture changes."""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[3]


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _layer_violations(layer: str, forbidden_roots: set[str]) -> list[str]:
    violations: list[str] = []
    for path in sorted((BACKEND_ROOT / layer).rglob("*.py")):
        for line, module in _imports(path):
            if module.split(".")[0] in forbidden_roots:
                violations.append(f"{path.relative_to(BACKEND_ROOT)}:{line}: {module}")
    return violations


def test_domain_does_not_depend_on_api_infrastructure_or_frameworks() -> None:
    assert _layer_violations("domain", {"api", "infrastructure", "celery", "fastapi"}) == []


def test_infrastructure_does_not_depend_on_api_boundary() -> None:
    assert _layer_violations("infrastructure", {"api"}) == []


def test_document_submission_coordinator_is_http_independent() -> None:
    modules = {module.split(".")[0] for _, module in _imports(BACKEND_ROOT / "services/documents/submission.py")}
    assert "api" not in modules
    assert "fastapi" not in modules


def test_routers_do_not_import_application_assembly() -> None:
    violations = []
    for path in sorted((BACKEND_ROOT / "api/routers").glob("*.py")):
        for line, module in _imports(path):
            if module in {"api.app", "api.main"}:
                violations.append(f"{path.relative_to(BACKEND_ROOT)}:{line}: {module}")
    assert violations == []


def test_document_router_uses_explicit_dependencies_for_document_reads() -> None:
    relative_path = "api/routers/documents_ingest.py"
    observed = {
        module
        for _, module in _imports(BACKEND_ROOT / relative_path)
        if module.startswith("infrastructure.")
    }
    source = (BACKEND_ROOT / relative_path).read_text(encoding="utf-8")

    assert observed == set()
    assert "get_document_read" in source
    assert "DocumentReadProvider" in source


def test_business_routers_do_not_locate_services_through_application_state() -> None:
    """Completed document/QA migrations leave state lookup in dependencies only."""

    for relative_path in ("api/routers/documents_ingest.py", "api/routers/documents_read.py", "api/routers/qa.py"):
        source = (BACKEND_ROOT / relative_path).read_text(encoding="utf-8")
        assert "request.app.state" not in source, relative_path


def test_qa_router_has_no_concrete_infrastructure_dependencies() -> None:
    """QA HTTP behavior receives runtime implementations through its provider."""

    observed = {
        module
        for _, module in _imports(BACKEND_ROOT / "api/routers/qa.py")
        if module == "infrastructure" or module.startswith("infrastructure.")
    }
    assert observed == set()


def test_qa_agent_does_not_assemble_production_dependencies() -> None:
    """The workflow composition root is the only production QA assembler."""

    source = (BACKEND_ROOT / "agents/qa_agent.py").read_text(encoding="utf-8")
    imports = {module for _, module in _imports(BACKEND_ROOT / "agents/qa_agent.py")}

    assert "langchain_openai" not in imports
    assert "build_runtime_safety_pipeline" not in source
    assert "build_model_provider_timeout" not in source
    assert "dependencies: QAAgentDependencies | None" not in source


def test_removed_architecture_compatibility_entries_stay_absent() -> None:
    """Deleted aggregate modules/builders cannot silently return as wrappers."""

    assert not (BACKEND_ROOT / "auth/identity_store.py").exists()
    adapter_source = (
        BACKEND_ROOT / "infrastructure/security/guardrails_adapter.py"
    ).read_text(encoding="utf-8")
    assert "build_repository_guardrails" not in adapter_source
