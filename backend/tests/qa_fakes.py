"""Focused test factories for explicitly composed QA Agents."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from agents.qa_agent import QAAgentDependencies
from workflows.qa_dependencies import build_qa_agent_dependencies


def build_test_qa_agent_dependencies(**overrides) -> QAAgentDependencies:
    """Build production-shaped QA dependencies without a real model client."""

    with patch("workflows.qa_dependencies.ChatOpenAI"):
        dependencies = build_qa_agent_dependencies()
    return replace(dependencies, **overrides)


__all__ = ["build_test_qa_agent_dependencies"]
