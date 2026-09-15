"""ATDD coverage for coherent DeepSeek runtime budgets and provider probes."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from shared.config import settings
from shared.config.settings import Settings


def test_llm_timeout_budget_rejects_shorter_total_deadline() -> None:
    """The wrapper deadline cannot be shorter than one configured transport attempt."""

    with pytest.raises(ValidationError):
        Settings(
            llm_connect_timeout_seconds=10,
            llm_read_timeout_seconds=60,
            llm_total_timeout_seconds=30,
        )


def test_text_agents_use_configured_http_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """QA and extraction clients share the configured connect/read/total values."""

    monkeypatch.setattr(settings, "deepseek_api_key", "deepseek-test-key", raising=False)
    monkeypatch.setattr(settings, "deepseek_base_url", "https://deepseek.test", raising=False)
    monkeypatch.setattr(settings, "deepseek_model", "deepseek-test-model", raising=False)
    monkeypatch.setattr(settings, "llm_connect_timeout_seconds", 7.0, raising=False)
    monkeypatch.setattr(settings, "llm_read_timeout_seconds", 41.0, raising=False)
    monkeypatch.setattr(settings, "llm_total_timeout_seconds", 55.0, raising=False)
    monkeypatch.setattr(settings, "llm_max_retries", 0, raising=False)

    from agents.knowledge_extractor import KnowledgeExtractAgent
    from workflows.qa_dependencies import build_qa_agent_dependencies

    for target, factory in (
        ("workflows.qa_dependencies.ChatOpenAI", build_qa_agent_dependencies),
        ("agents.knowledge_extractor.ChatOpenAI", KnowledgeExtractAgent),
    ):
        with patch(target) as chat:
            factory()
        timeout = chat.call_args.kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect == 7.0
        assert timeout.read == 41.0
        assert timeout.write == 55.0
        assert chat.call_args.kwargs["max_retries"] == 0


class _Response:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class _ProbeClient:
    def __init__(self, responses: list[_Response], captured: list[dict], **_kwargs) -> None:
        self.responses = responses
        self.captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def request(self, method: str, url: str, **kwargs):
        self.captured.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


def test_provider_probe_reports_success_without_disclosing_secret() -> None:
    """The operator probe allowlists output and never prints its bearer credential."""

    from infrastructure.scripts.probe_model_provider import run_probe

    captured: list[dict] = []
    output: list[str] = []
    responses = [
        _Response(200, {"data": [{"id": "deepseek-v4-flash"}]}),
        _Response(200, {"model": "deepseek-v4-flash", "choices": [{"message": {"content": "OK"}}]}),
    ]
    config = SimpleNamespace(
        deepseek_api_key="super-secret-provider-key",
        deepseek_base_url="https://deepseek.test",
        deepseek_model="deepseek-v4-flash",
        llm_connect_timeout_seconds=3.0,
        llm_read_timeout_seconds=20.0,
        llm_total_timeout_seconds=25.0,
    )

    exit_code = run_probe(
        config=config,
        client_factory=lambda **kwargs: _ProbeClient(responses, captured, **kwargs),
        output=output.append,
        clock=lambda: 1.0,
    )

    assert exit_code == 0
    rendered = "\n".join(output)
    assert "super-secret-provider-key" not in rendered
    assert "Authorization" not in rendered
    assert '"status": 200' in rendered
    assert captured[0]["headers"]["Authorization"] == "Bearer super-secret-provider-key"


def test_provider_probe_fails_boundedly_without_echoing_response_secrets() -> None:
    """Provider failures return non-zero with bounded sanitized output."""

    from infrastructure.scripts.probe_model_provider import run_probe

    output: list[str] = []
    secret = "provider-secret"
    responses = [_Response(401, {"error": {"message": f"bad {secret}"}})]
    config = SimpleNamespace(
        deepseek_api_key=secret,
        deepseek_base_url="https://deepseek.test",
        deepseek_model="deepseek-v4-flash",
        llm_connect_timeout_seconds=3.0,
        llm_read_timeout_seconds=20.0,
        llm_total_timeout_seconds=25.0,
    )

    exit_code = run_probe(
        config=config,
        client_factory=lambda **kwargs: _ProbeClient(responses, [], **kwargs),
        output=output.append,
        clock=lambda: 1.0,
    )

    assert exit_code == 1
    rendered = "\n".join(output)
    assert secret not in rendered
    assert "401" in rendered
