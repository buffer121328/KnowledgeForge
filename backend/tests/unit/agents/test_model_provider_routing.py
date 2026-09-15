"""Acceptance tests for provider-specific online model routing."""

from unittest.mock import patch

from shared.config import settings
from shared.config.settings import Settings


def _configure_distinct_test_providers(monkeypatch) -> None:
    monkeypatch.setattr(settings, "deepseek_api_key", "deepseek-test-key", raising=False)
    monkeypatch.setattr(settings, "deepseek_base_url", "https://deepseek.test", raising=False)
    monkeypatch.setattr(settings, "deepseek_model", "deepseek-test-model", raising=False)
    monkeypatch.setattr(settings, "dashscope_api_key", "dashscope-test-key", raising=False)
    monkeypatch.setattr(settings, "dashscope_base_url", "https://dashscope.test/v1", raising=False)
    monkeypatch.setattr(settings, "embedding_model", "qwen-test-embedding", raising=False)
    monkeypatch.setattr(settings, "embedding_batch_size", 7, raising=False)
    monkeypatch.setattr(settings, "vision_model", "qwen-test-vision", raising=False)


def test_settings_default_to_current_mimo_and_qwen_models() -> None:
    configured = Settings(_env_file=None)

    assert configured.deepseek_base_url == "https://api.xiaomimimo.com/v1"
    assert configured.deepseek_model == "mimo-v2.5-pro"
    assert configured.dashscope_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert configured.embedding_model == "text-embedding-v4"
    assert configured.vision_model == "qwen3-vl-flash"


def test_mimo_environment_names_populate_the_legacy_text_settings(monkeypatch) -> None:
    monkeypatch.setenv("MIMO_API_KEY", "mimo-env-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setenv("MIMO_MODEL", "mimo-v2.5-pro")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    configured = Settings(_env_file=None)

    assert configured.deepseek_api_key == "mimo-env-key"
    assert configured.deepseek_base_url == "https://api.xiaomimimo.com/v1"
    assert configured.deepseek_model == "mimo-v2.5-pro"


def test_text_agents_use_deepseek_settings(monkeypatch) -> None:
    _configure_distinct_test_providers(monkeypatch)

    from agents.knowledge_extractor import KnowledgeExtractAgent
    constructors = (
        ("workflows.qa_dependencies.ChatOpenAI", _build_qa_dependencies),
        ("agents.knowledge_extractor.ChatOpenAI", lambda: KnowledgeExtractAgent()),
    )
    for target, create in constructors:
        with patch(target) as chat:
            create()

        kwargs = chat.call_args.kwargs
        assert kwargs["api_key"] == "deepseek-test-key"
        assert kwargs["base_url"] == "https://deepseek.test"
        assert kwargs["model"] == "deepseek-test-model"


def _build_qa_dependencies():
    """Exercise the composition-root model constructor."""
    from workflows.qa_dependencies import build_qa_agent_dependencies

    return build_qa_agent_dependencies()


def test_embedding_clients_use_dashscope_settings(monkeypatch) -> None:
    _configure_distinct_test_providers(monkeypatch)

    from infrastructure.retrieval.vector_store import VectorStoreService
    from workflows.ingest_document import MultimodalService

    constructors = (
        ("infrastructure.retrieval.vector_store.OpenAIEmbeddings", VectorStoreService),
        ("workflows.ingest_document.OpenAIEmbeddings", MultimodalService),
    )
    for target, create in constructors:
        with patch(target) as embeddings:
            create()

        kwargs = embeddings.call_args.kwargs
        assert kwargs["api_key"] == "dashscope-test-key"
        assert kwargs["base_url"] == "https://dashscope.test/v1"
        assert kwargs["model"] == "qwen-test-embedding"
        assert kwargs["chunk_size"] == 7


def test_document_parser_uses_qwen_vision_not_deepseek(monkeypatch) -> None:
    _configure_distinct_test_providers(monkeypatch)

    from agents.document_parser import DocParserAgent

    with patch("agents.document_format_parsers.ChatOpenAI") as chat:
        parser = DocParserAgent()

    kwargs = chat.call_args.kwargs
    assert kwargs["api_key"] == "dashscope-test-key"
    assert kwargs["base_url"] == "https://dashscope.test/v1"
    assert kwargs["model"] == "qwen-test-vision"
    assert hasattr(parser, "vision_llm")

def test_qa_runtime_factory_uses_deepseek_settings(monkeypatch) -> None:
    _configure_distinct_test_providers(monkeypatch)

    from workflows import qa_dependencies

    with patch("workflows.qa_dependencies.ChatOpenAI") as chat:
        dependencies = qa_dependencies.build_qa_agent_dependencies()

    kwargs = chat.call_args.kwargs
    assert dependencies.llm is chat.return_value
    assert kwargs["api_key"] == "deepseek-test-key"
    assert kwargs["base_url"] == "https://deepseek.test"
    assert kwargs["model"] == "deepseek-test-model"


def test_mimo_model_uses_the_openai_compatible_online_route(monkeypatch) -> None:
    """MiMo values in the legacy text-provider slot reach the QA client unchanged."""
    monkeypatch.setattr(settings, "deepseek_api_key", "mimo-test-key", raising=False)
    monkeypatch.setattr(settings, "deepseek_base_url", "https://api.xiaomimimo.com/v1", raising=False)
    monkeypatch.setattr(settings, "deepseek_model", "mimo-v2.5-pro", raising=False)

    from workflows import qa_dependencies

    with patch("workflows.qa_dependencies.ChatOpenAI") as chat:
        qa_dependencies.build_qa_agent_dependencies()

    kwargs = chat.call_args.kwargs
    assert kwargs["api_key"] == "mimo-test-key"
    assert kwargs["base_url"] == "https://api.xiaomimimo.com/v1"
    assert kwargs["model"] == "mimo-v2.5-pro"
