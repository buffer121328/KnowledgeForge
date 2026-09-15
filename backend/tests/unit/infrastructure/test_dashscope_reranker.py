"""Contract tests for the DashScope text-rerank adapter."""

from __future__ import annotations

import httpx
import pytest

from domain.knowledge import RetrievedContext
from infrastructure.retrieval.dashscope_reranker import (
    DashScopeRerankAdapter,
    DashScopeRerankError,
)


def _contexts() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            content=f"document {index}",
            source=f"source-{index}.md",
            score=0.5,
            retrieval_type="vector",
            metadata={"context_id": f"ctx_{index}"},
        )
        for index in range(2)
    ]


@pytest.mark.parametrize(
    ("endpoint", "model"),
    [
        ("http://dashscope.example/rerank", "qwen3-rerank"),
        ("https://dashscope.example/rerank", "gte-rerank-v2"),
    ],
)
def test_adapter_rejects_endpoint_or_model_for_the_qwen3_contract(
    endpoint: str, model: str
) -> None:
    with pytest.raises(ValueError):
        DashScopeRerankAdapter(endpoint=endpoint, api_key="test-key", model=model)


@pytest.mark.asyncio
async def test_dashscope_request_and_index_scores_are_mapped() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["authorization"]
        seen["payload"] = httpx.Response(200, content=await request.aread()).json()
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.12},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        scores = await DashScopeRerankAdapter(
            endpoint="https://dashscope.example/rerank",
            api_key="secret-key",
            model="qwen3-rerank",
            http_client=client,
        ).score("question", _contexts())
    finally:
        await client.aclose()

    assert seen["authorization"] == "Bearer secret-key"
    assert seen["payload"] == {
        "model": "qwen3-rerank",
        "query": "question",
        "documents": ["document 0", "document 1"],
        "top_n": 2,
        "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
    }
    assert [(score.context_id, score.score) for score in scores] == [
        ("ctx_0", 0.12),
        ("ctx_1", 0.91),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": 0.1}],
        [
            {"index": 0, "relevance_score": 0.1},
            {"index": 0, "relevance_score": 0.2},
        ],
        [
            {"index": 0, "relevance_score": 0.1},
            {"index": 2, "relevance_score": 0.2},
        ],
        [
            {"index": 0, "relevance_score": "not-a-number"},
            {"index": 1, "relevance_score": 0.2},
        ],
        [
            {"index": 0, "relevance_score": -0.1},
            {"index": 1, "relevance_score": 0.2},
        ],
        [
            {"index": 0, "relevance_score": 1.1},
            {"index": 1, "relevance_score": 0.2},
        ],
    ],
)
async def test_invalid_dashscope_results_are_rejected(results: list[dict[str, object]]) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"results": results})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(DashScopeRerankError, match="dashscope_invalid_response"):
            await DashScopeRerankAdapter(
                endpoint="https://dashscope.example/rerank",
                api_key="secret-key",
                http_client=client,
            ).score("question", _contexts())
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_dashscope_provider_error_is_bounded_without_response_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, json={"message": "secret provider detail"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(DashScopeRerankError, match="dashscope_http_error") as error:
            await DashScopeRerankAdapter(
                endpoint="https://dashscope.example/rerank",
                api_key="secret-key",
                http_client=client,
            ).score("question", _contexts())
    finally:
        await client.aclose()

    assert "secret provider detail" not in str(error.value)
    assert "question" not in str(error.value)
