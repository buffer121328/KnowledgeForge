"""Webhook 服务的单元测试"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from infrastructure.webhooks.delivery import WebhookTargetValidator
from infrastructure.webhooks.stores import MemoryWebhookStore
from infrastructure.webhooks.service import (
    WebhookEvent,
    WebhookService,
)


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


@pytest.fixture
def service() -> WebhookService:
    """带 mock HTTP 客户端的 WebhookService"""
    mock_client = mock_http_client()
    return WebhookService(
        store=MemoryWebhookStore(),
        http_client=mock_client,
        max_retries=2,
        target_validator=public_target_validator(),
    )


class TestWebhookEvent:
    def test_event_values(self):
        assert WebhookEvent.DOC_INGESTED.value == "doc.ingested"
        assert WebhookEvent.QA_COMPLETED.value == "qa.completed"


class TestWebhookRegister:
    def test_register_basic(self, service: WebhookService):
        """测试注册 Webhook"""
        wh = service.register(
            url="https://example.com/webhook",
            events=[WebhookEvent.DOC_INGESTED, WebhookEvent.QA_COMPLETED],
        )
        assert wh.id.startswith("wh_")
        assert wh.url == "https://example.com/webhook"
        assert WebhookEvent.DOC_INGESTED in wh.events
        assert wh.is_active is True
        assert wh.secret  # 自动生成密钥

    def test_register_with_custom_secret(self, service: WebhookService):
        """测试自定义密钥"""
        wh = service.register(
            url="https://example.com",
            events=[WebhookEvent.DOC_INGESTED],
            secret="my-secret",
        )
        assert wh.secret == "my-secret"

    def test_list_webhooks(self, service: WebhookService):
        """测试列出 Webhook"""
        service.register("https://a.com", [WebhookEvent.DOC_INGESTED])
        service.register("https://b.com", [WebhookEvent.QA_COMPLETED])
        webhooks = service.list_webhooks()
        assert len(webhooks) == 2

    def test_delete_webhook(self, service: WebhookService):
        """测试删除 Webhook"""
        wh = service.register("https://a.com", [WebhookEvent.DOC_INGESTED])
        assert service.delete(wh.id) is True
        assert service.delete(wh.id) is False  # 已删除


class TestWebhookTrigger:
    @pytest.mark.asyncio
    async def test_trigger_no_subscribers(self, service: WebhookService):
        """测试无订阅者时触发"""
        count = await service.trigger(WebhookEvent.DOC_INGESTED, {"doc_id": "123"})
        assert count == 0

    @pytest.mark.asyncio
    async def test_trigger_success(self, service: WebhookService):
        """测试成功触发"""
        service.register("https://a.com", [WebhookEvent.DOC_INGESTED])
        service._delivery._client.send.return_value = mock_response(200)

        count = await service.trigger(WebhookEvent.DOC_INGESTED, {"doc_id": "123"})
        assert count == 1
        service._delivery._client.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_with_signature(self, service: WebhookService):
        """测试 HMAC 签名"""
        wh = service.register("https://a.com", [WebhookEvent.DOC_INGESTED], secret="secret")
        service._delivery._client.send.return_value = mock_response(200)

        await service.trigger(WebhookEvent.DOC_INGESTED, {"test": True})

        request = service._delivery._client.send.call_args.args[0]
        headers = request.headers
        assert "X-Webhook-Signature" in headers
        assert headers["X-Webhook-Signature"].startswith("sha256=")
        assert headers["X-Webhook-Event"] == "doc.ingested"
        assert headers["X-Webhook-Id"] == wh.id

    @pytest.mark.asyncio
    async def test_trigger_filters_by_event(self, service: WebhookService):
        """测试按事件过滤"""
        service.register("https://a.com", [WebhookEvent.DOC_INGESTED])
        service.register("https://b.com", [WebhookEvent.QA_COMPLETED])

        service._delivery._client.send.return_value = mock_response(200)
        count = await service.trigger(WebhookEvent.DOC_INGESTED, {})
        assert count == 1  # 只有 a.com 被触发

    @pytest.mark.asyncio
    async def test_trigger_failure_increments_counter(self, service: WebhookService):
        """测试失败时增加 failure_count"""
        wh = service.register("https://a.com", [WebhookEvent.DOC_INGESTED])
        service._delivery._client.send.return_value = mock_response(500)

        await service.trigger(WebhookEvent.DOC_INGESTED, {})

        updated = service.store.get(wh.id)
        assert updated.failure_count > 0
        assert updated.is_active is True  # 未达到 max_failures

    @pytest.mark.asyncio
    async def test_trigger_disables_after_max_failures(self):
        """测试多次失败后自动禁用"""
        # 使用 max_retries=1 避免测试中的 sleep 等待
        mock_client = mock_http_client()
        mock_client.send.side_effect = Exception("connection error")
        service = WebhookService(
            store=MemoryWebhookStore(),
            http_client=mock_client,
            max_retries=1,
            target_validator=public_target_validator(),
        )

        wh = service.register("https://a.com", [WebhookEvent.DOC_INGESTED])
        wh.max_failures = 1
        service.store.save(wh)

        await service.trigger(WebhookEvent.DOC_INGESTED, {})

        updated = service.store.get(wh.id)
        assert updated.is_active is False

    @pytest.mark.asyncio
    async def test_trigger_success_resets_failure_count(self, service: WebhookService):
        """测试成功后重置 failure_count"""
        wh = service.register("https://a.com", [WebhookEvent.DOC_INGESTED])
        wh.failure_count = 2
        service.store.save(wh)

        service._delivery._client.send.return_value = mock_response(200)
        await service.trigger(WebhookEvent.DOC_INGESTED, {})

        updated = service.store.get(wh.id)
        assert updated.failure_count == 0
