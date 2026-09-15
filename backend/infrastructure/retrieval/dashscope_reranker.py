"""阿里云百炼文本重排适配器。"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlparse

import httpx

from domain.knowledge import RetrievedContext
from infrastructure.retrieval.cross_encoder import CrossEncoderScore, context_id_for


_QA_RERANK_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query."
)


class DashScopeRerankError(RuntimeError):
    """Stable internal error for malformed or unavailable provider responses."""


class DashScopeRerankAdapter:
    """Score bounded candidate texts through DashScope's text-rerank API."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str = "qwen3-rerank",
        timeout_seconds: float = 2.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        endpoint = endpoint.strip()
        api_key = api_key.strip()
        model = model.strip()
        if not endpoint:
            raise ValueError("endpoint is required")
        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.scheme != "https" or not parsed_endpoint.netloc:
            raise ValueError("endpoint must be an absolute https URL")
        if not api_key:
            raise ValueError("api_key is required")
        if not model:
            raise ValueError("model is required")
        if model != "qwen3-rerank":
            raise ValueError("only qwen3-rerank is supported by this adapter")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    async def score(
        self,
        question: str,
        candidates: list[RetrievedContext],
    ) -> list[CrossEncoderScore]:
        """Return one provider score for every candidate in input order."""
        if not candidates:
            return []
        payload = {
            "model": self.model,
            "query": question,
            "documents": [candidate.content for candidate in candidates],
            "top_n": len(candidates),
            "instruct": _QA_RERANK_INSTRUCT,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            response = await self._post(payload, headers)
        except httpx.TimeoutException as exc:
            raise TimeoutError from exc
        except httpx.HTTPError as exc:
            raise DashScopeRerankError("dashscope_http_error") from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise DashScopeRerankError("dashscope_http_error")
        try:
            body = response.json()
            results = body["results"]
        except (KeyError, TypeError, ValueError) as exc:
            raise DashScopeRerankError("dashscope_invalid_response") from exc
        if not isinstance(results, list) or len(results) != len(candidates):
            raise DashScopeRerankError("dashscope_invalid_response")

        scores_by_index: dict[int, float] = {}
        for result in results:
            if not isinstance(result, dict):
                raise DashScopeRerankError("dashscope_invalid_response")
            index = result.get("index")
            raw_score = result.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int):
                raise DashScopeRerankError("dashscope_invalid_response")
            if index < 0 or index >= len(candidates) or index in scores_by_index:
                raise DashScopeRerankError("dashscope_invalid_response")
            if isinstance(raw_score, bool):
                raise DashScopeRerankError("dashscope_invalid_response")
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise DashScopeRerankError("dashscope_invalid_response") from exc
            if not math.isfinite(score) or score < 0 or score > 1:
                raise DashScopeRerankError("dashscope_invalid_response")
            scores_by_index[index] = score

        if set(scores_by_index) != set(range(len(candidates))):
            raise DashScopeRerankError("dashscope_invalid_response")
        return [
            CrossEncoderScore(
                context_id=context_id_for(candidate, index),
                score=scores_by_index[index],
            )
            for index, candidate in enumerate(candidates)
        ]

    async def _post(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        if self._http_client is not None:
            return await self._http_client.post(
                self.endpoint,
                json=payload,
                headers=headers,
            )
        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(
                self.endpoint,
                json=payload,
                headers=headers,
            )


__all__ = ["DashScopeRerankAdapter", "DashScopeRerankError"]
