"""Acceptance coverage for explicit QA runtime dependency composition."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

from agents.qa_agent import QAAgent, QAAgentDependencies

BACKEND_ROOT = Path(__file__).resolve().parents[3]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_qa_runtime_dependencies_flow_from_workflow_composition_to_agent() -> None:
    """Keep runtime QA construction in the workflow composition direction."""

    agent_imports = _imports(BACKEND_ROOT / "agents/qa_agent.py")
    factory_source = (BACKEND_ROOT / "workflows/workflow_factory.py").read_text(
        encoding="utf-8"
    )

    assert all(not module.startswith("workflows") for module in agent_imports)
    assert "build_qa_agent_dependencies()" in factory_source
    assert "dependencies=qa_dependencies" in factory_source


def test_qa_agent_uses_explicit_dependencies_without_constructing_runtime_clients():
    dependencies = QAAgentDependencies(
        llm=object(),
        safety_pipeline=object(),
        cross_encoder_reranker=object(),
        evidence_qualifier=object(),
        grounding_verifier=object(),
    )
    agent = QAAgent(dependencies=dependencies)

    assert agent.llm is dependencies.llm
    assert agent.safety_pipeline is dependencies.safety_pipeline
    assert agent.cross_encoder_reranker is dependencies.cross_encoder_reranker
    assert agent.evidence_qualifier is dependencies.evidence_qualifier
    assert agent.grounding_verifier is dependencies.grounding_verifier


def test_remote_mode_injects_dashscope_reranker(monkeypatch) -> None:
    from infrastructure.retrieval.dashscope_reranker import DashScopeRerankAdapter
    from workflows import qa_dependencies

    monkeypatch.setattr(qa_dependencies.settings, "qa_cross_encoder_mode", "remote")
    monkeypatch.setattr(
        qa_dependencies.settings,
        "qa_cross_encoder_remote_endpoint",
        "https://dashscope.example/rerank",
    )
    monkeypatch.setattr(qa_dependencies.settings, "dashscope_api_key", "test-key")

    with patch("workflows.qa_dependencies.ChatOpenAI"), patch(
        "workflows.qa_dependencies.build_runtime_safety_pipeline",
        return_value=object(),
    ):
        dependencies = qa_dependencies.build_qa_agent_dependencies()

    assert isinstance(dependencies.cross_encoder_reranker.adapter, DashScopeRerankAdapter)
    assert dependencies.cross_encoder_reranker.adapter.model == "qwen3-rerank"
