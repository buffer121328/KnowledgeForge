from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from infrastructure.webhooks.delivery import WebhookDelivery, WebhookTargetValidator
from infrastructure.webhooks.models import Webhook, WebhookEvent
from infrastructure.webhooks.stores import MemoryWebhookStore
from shared.utils.dependency_resilience import DependencyExecutionPolicy


async def public_resolver(host: str, port: int) -> set[str]:
    del host, port
    return {"8.8.8.8"}


def public_target_validator() -> WebhookTargetValidator:
    return WebhookTargetValidator(resolver=public_resolver)


def mock_http_client() -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.build_request.side_effect = httpx.Request
    return client


def mock_response(status_code: int) -> MagicMock:
    response = MagicMock(status_code=status_code)

    async def no_body(chunk_size: int | None = None):
        del chunk_size
        if False:
            yield b""

    response.aiter_bytes = no_body
    response.aclose = AsyncMock()
    return response


def webhook(*, max_failures: int = 5) -> Webhook:
    return Webhook(
        id="wh_resilience",
        url="https://receiver.example/webhook",
        events=[WebhookEvent.DOC_INGESTED],
        secret="test-secret",
        max_failures=max_failures,
    )


def policy(*, total_timeout_seconds: float = 5, max_retries: int = 1) -> DependencyExecutionPolicy:
    return DependencyExecutionPolicy(
        dependency="webhook",
        connect_timeout_seconds=1.25,
        read_timeout_seconds=2.5,
        total_timeout_seconds=total_timeout_seconds,
        max_retries=max_retries,
        base_backoff_seconds=0.2,
        max_backoff_seconds=1,
    )


class TestWebhookDeliveryBudgets:
    @pytest.mark.asyncio
    async def test_owned_http_client_uses_configured_connection_and_read_timeouts(self) -> None:
        delivery = WebhookDelivery(policy=policy(), target_validator=public_target_validator())
        try:
            assert delivery._client.timeout.connect == 1.25
            assert delivery._client.timeout.read == 2.5
            assert delivery._client.timeout.write == 5
            assert delivery._client.timeout.pool == 5
        finally:
            await delivery.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [429, 500, 503])
    async def test_retryable_http_responses_retry_with_jitter_then_succeed(self, status_code: int) -> None:
        client = mock_http_client()
        client.send.side_effect = [mock_response(status_code), mock_response(204)]
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        delivery = WebhookDelivery(
            http_client=client,
            policy=policy(),
            sleep=record_sleep,
            random_value=lambda: 0.75,
            target_validator=public_target_validator(),
        )
        store = MemoryWebhookStore()
        item = webhook()
        store.save(item)

        assert await delivery.send_with_retry(item, WebhookEvent.DOC_INGESTED, {"doc_id": "d1"}, store) is True
        assert client.send.await_count == 2
        assert delays == [pytest.approx(0.25)]
        assert item.failure_count == 0
        assert item.last_response_code == 204

    @pytest.mark.asyncio
    async def test_transport_connection_error_retries_then_succeeds(self) -> None:
        client = mock_http_client()
        client.send.side_effect = [httpx.ConnectError("temporary"), mock_response(200)]
        delivery = WebhookDelivery(
            http_client=client,
            policy=policy(),
            sleep=AsyncMock(),
            random_value=lambda: 0.5,
            target_validator=public_target_validator(),
        )
        store = MemoryWebhookStore()
        item = webhook()
        store.save(item)

        assert await delivery.send_with_retry(item, WebhookEvent.DOC_INGESTED, {}, store) is True
        assert client.send.await_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_4xx_is_not_repeated_and_increments_failure_once(self) -> None:
        client = mock_http_client()
        client.send.return_value = mock_response(400)
        sleep = AsyncMock()
        delivery = WebhookDelivery(
            http_client=client,
            policy=policy(max_retries=3),
            sleep=sleep,
            target_validator=public_target_validator(),
        )
        store = MemoryWebhookStore()
        item = webhook()
        store.save(item)

        assert await delivery.send_with_retry(item, WebhookEvent.DOC_INGESTED, {}, store) is False
        assert client.send.await_count == 1
        sleep.assert_not_awaited()
        assert item.failure_count == 1
        assert item.last_response_code == 400

    @pytest.mark.asyncio
    async def test_budget_exhaustion_avoids_extra_attempt_and_preserves_disable_threshold(self) -> None:
        client = mock_http_client()
        client.send.side_effect = httpx.ConnectError("temporary")
        sleep = AsyncMock()
        delivery = WebhookDelivery(
            http_client=client,
            policy=policy(total_timeout_seconds=0.01, max_retries=3),
            sleep=sleep,
            random_value=lambda: 0.5,
            target_validator=public_target_validator(),
        )
        store = MemoryWebhookStore()
        item = webhook(max_failures=1)
        store.save(item)

        assert await delivery.send_with_retry(item, WebhookEvent.DOC_INGESTED, {}, store) is False
        assert client.send.await_count == 1
        sleep.assert_not_awaited()
        assert item.failure_count == 1
        assert item.is_active is False
