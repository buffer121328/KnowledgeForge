"""Structural compatibility checks for the canonical backend layout.

布局守卫（目录治理后）：
- deployment 契约模块必须保持 infrastructure 根直属文件（celery_app、
  celery_tasks、postgresql_migrations 被 compose/k8s/Dockerfile 按模块路径引用）。
- infrastructure / evaluation / services 按主题子包组织；禁止复活历史分类子包
  persistence / messaging / integrations / storage。
- agents/ 只保留 4 个真 Agent 门面；服务化组件归 services/ 主题子包。
- tests/unit 按业务域子目录组织。
"""

from __future__ import annotations


def test_deployment_contract_modules_stay_direct_files():
    from pathlib import Path
    import infrastructure.celery_app as celery_app
    import infrastructure.celery_tasks as celery_tasks
    import infrastructure.postgresql_migrations as postgresql_migrations

    backend_root = Path(__file__).resolve().parents[3]
    modules = {
        celery_app: "infrastructure/celery_app.py",
        celery_tasks: "infrastructure/celery_tasks.py",
        postgresql_migrations: "infrastructure/postgresql_migrations.py",
    }

    for module, relative_path in modules.items():
        module_file = module.__file__
        assert module_file is not None
        assert Path(module_file).resolve() == backend_root / relative_path


def test_infrastructure_modules_live_in_topic_subpackages():
    from pathlib import Path
    import infrastructure.audit.log as audit_log
    import infrastructure.cache.redis as redis_cache
    import infrastructure.documents.local_uploads as local_uploads
    import infrastructure.graph.neo4j_graph as neo4j_graph
    import infrastructure.retrieval.vector_store as vector_store
    import infrastructure.security.security_state as security_state
    import infrastructure.tasks.task_registry as task_registry
    import infrastructure.webhooks.service as webhooks

    backend_root = Path(__file__).resolve().parents[3]
    modules = {
        neo4j_graph: "infrastructure/graph/neo4j_graph.py",
        vector_store: "infrastructure/retrieval/vector_store.py",
        redis_cache: "infrastructure/cache/redis.py",
        task_registry: "infrastructure/tasks/task_registry.py",
        webhooks: "infrastructure/webhooks/service.py",
        local_uploads: "infrastructure/documents/local_uploads.py",
        audit_log: "infrastructure/audit/log.py",
        security_state: "infrastructure/security/security_state.py",
    }

    for module, relative_path in modules.items():
        module_file = module.__file__
        assert module_file is not None
        assert Path(module_file).resolve() == backend_root / relative_path


def test_services_modules_live_in_topic_subpackages():
    from pathlib import Path
    import services.documents.lifecycle as lifecycle
    import services.documents.submission as submission
    import services.evidence.qualification as qualification
    import services.qa.generation as generation
    import services.qa.retrievers as retrievers
    import services.safety.pipeline as pipeline
    import services.safety.qa_checks as qa_checks
    import services.safety.tool_policy as tool_policy

    backend_root = Path(__file__).resolve().parents[3]
    modules = {
        submission: "services/documents/submission.py",
        lifecycle: "services/documents/lifecycle.py",
        generation: "services/qa/generation.py",
        retrievers: "services/qa/retrievers.py",
        pipeline: "services/safety/pipeline.py",
        qa_checks: "services/safety/qa_checks.py",
        tool_policy: "services/safety/tool_policy.py",
        qualification: "services/evidence/qualification.py",
    }

    for module, relative_path in modules.items():
        module_file = module.__file__
        assert module_file is not None
        assert Path(module_file).resolve() == backend_root / relative_path


def test_agents_package_stays_pure_agent_facades():
    """agents/ 只允许 4 个真 Agent 门面文件，防止服务化组件回迁。"""

    from pathlib import Path

    backend_root = Path(__file__).resolve().parents[3]
    agents_root = backend_root / "agents"
    allowed_files = {
        "__init__.py",
        "document_parser.py",
        "document_format_parsers.py",
        "document_chunking.py",
        "knowledge_extractor.py",
        "knowledge_updater.py",
        "qa_agent.py",
        "qa_cache.py",
    }
    for entry in sorted(agents_root.iterdir()):
        if entry.is_file() and entry.suffix == ".py":
            assert entry.name in allowed_files, f"stray module in agents/: {entry.name}"


def test_services_never_import_agents():
    """依赖方向单向：agents -> services -> {domain, infrastructure, shared}。"""

    import ast
    from pathlib import Path

    backend_root = Path(__file__).resolve().parents[3]
    services_root = backend_root / "services"
    offenders: list[str] = []
    for path in services_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "agents" or alias.name.startswith("agents."):
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level > 0 and module.startswith("agents"):
                    offenders.append(f"{path.name}: from {'.' * node.level}{module} import")
                elif node.level == 0 and (module == "agents" or module.startswith("agents.")):
                    offenders.append(f"{path.name}: from {module} import")
    assert offenders == [], "services must not import agents: " + "; ".join(offenders)


def test_obsolete_category_packages_do_not_reappear():
    from pathlib import Path

    backend_root = Path(__file__).resolve().parents[3]
    for obsolete_directory in (
        "domain/documents",
        "domain/knowledge",
        "domain/identity",
        "domain/tasks",
        "infrastructure/persistence",
        "infrastructure/messaging",
        "infrastructure/integrations",
        "infrastructure/storage",
        "workflows/graph.py",
        "workflows/knowledge_graph.py",
        "agents/trace.py",
        "auth/models.py",
    ):
        assert not (backend_root / obsolete_directory).exists()


def test_infrastructure_has_no_stray_flat_modules():
    """infrastructure 根只允许部署契约与单一职责直属文件，防止重新平铺。"""

    from pathlib import Path

    backend_root = Path(__file__).resolve().parents[3]
    infra_root = backend_root / "infrastructure"
    allowed_files = {
        "__init__.py",
        "celery_app.py",
        "celery_tasks.py",
        "postgresql_migrations.py",
        "change_data_capture.py",
        "deployment_preflight.py",
        "readiness.py",
        "evidence_gate_configuration.py",
    }
    allowed_dirs = {
        "tasks",
        "graph",
        "postgres",
        "cache",
        "retrieval",
        "audit",
        "security",
        "webhooks",
        "documents",
        "scripts",
        "__pycache__",
    }
    for entry in sorted(infra_root.iterdir()):
        if entry.is_file() and entry.suffix == ".py":
            assert entry.name in allowed_files, f"stray flat module: {entry.name}"
        elif entry.is_dir():
            assert entry.name in allowed_dirs, f"unexpected directory: {entry.name}"


def test_canonical_domain_contracts_are_shared_by_consumers():
    from agents.document_parser import DocumentChunk as parser_chunk
    from agents.knowledge_extractor import Entity as extractor_entity
    from domain.documents import DocumentChunk
    from domain.knowledge import Entity
    from infrastructure.graph.neo4j_graph import Entity as graph_entity
    from infrastructure.retrieval.vector_store import DocumentChunk as vector_chunk

    assert parser_chunk is vector_chunk is DocumentChunk
    assert extractor_entity is graph_entity is Entity


def test_canonical_celery_module_discovers_canonical_tasks():
    from infrastructure.celery_app import celery_app

    assert "infrastructure.celery_tasks" in celery_app.conf.include


def test_lazy_auth_facade_exports_canonical_symbols():
    import auth

    from domain.identity import UserContext

    assert auth.AuthService is not None
    assert auth.UserContext is UserContext


def test_feature_routes_are_mounted_after_registration(monkeypatch):
    from shared.config import settings
    from fastapi.testclient import TestClient

    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(settings, "dashscope_api_key", "test-key")

    from api.main import app

    client = TestClient(app)
    assert client.get("/api/v1/docs").status_code != 404
    assert client.get("/api/docs").status_code == 404
