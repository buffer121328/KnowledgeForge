"""Public Webhook facade retaining compatible events, models, and service API."""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime, timezone

import httpx

from infrastructure.webhooks.delivery import WebhookDelivery, WebhookTargetValidationError, WebhookTargetValidator
from infrastructure.webhooks.models import Webhook, WebhookEvent
from infrastructure.webhooks.stores import (
    FernetWebhookSecretProtector,
    PostgreSQLWebhookStore,
    WebhookStore,
)
from shared.config import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class WebhookService:
    """Webhook management facade delegating persistence and delivery concerns."""

    def __init__(
        self,
        store: WebhookStore | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
        target_validator: WebhookTargetValidator | None = None,
    ) -> None:
        """Initialize the webhook service."""
        if store is not None:
            self.store = store
        else:
            from infrastructure.postgres.database import get_database_service

            protector = FernetWebhookSecretProtector(
                settings.webhook_secret_encryption_key,
                key_reference=settings.webhook_secret_key_reference,
            )
            self.store = PostgreSQLWebhookStore(get_database_service(), protector)
        self._delivery = WebhookDelivery(
            http_client=http_client,
            max_retries=max_retries,
            target_validator=target_validator,
        )
        self._target_validator = self._delivery.target_validator
        self.max_retries = self._delivery.max_retries
        self._pending_tasks: set[asyncio.Task] = set()

    def register(
        self,
        url: str,
        events: list[WebhookEvent],
        secret: str | None = None,
        is_active: bool = True,
        org_id: str = "",
    ) -> Webhook:
        """Register the webhook service."""
        self._target_validator.validate_syntax(url)
        webhook = Webhook(
            id=f"wh_{secrets.token_hex(16)}",
            org_id=org_id,
            url=url,
            events=list(events),
            secret=secret or secrets.token_hex(16),
            is_active=is_active,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.store.save(webhook)
        return webhook

    async def validate_target(self, url: str) -> None:
        """Run DNS-aware validation before a management operation persists a target."""
        await self._target_validator.validate(url)

    def list_webhooks(self, org_id: str | None = None) -> list[Webhook]:
        """List the webhooks."""
        return self.store.list_all(org_id)

    def delete(self, webhook_id: str, org_id: str | None = None) -> bool:
        """Delete a record through the webhook service."""
        return self.store.delete(webhook_id, org_id)

    async def test(self, webhook_id: str, org_id: str | None = None) -> dict:
        """Test the webhook service."""
        webhook = self.store.get(webhook_id, org_id)
        if not webhook:
            return {"success": False, "error": "Webhook 不存在"}
        try:
            ok = await self._delivery.send(webhook, WebhookEvent.SYSTEM_ERROR, {"test": True})
            if ok:
                webhook.failure_count = 0
                webhook.last_triggered_at = datetime.now(timezone.utc).isoformat()
                self.store.update(webhook)
                return {"success": True, "message": "测试回调成功", "status_code": webhook.last_response_code}
            webhook.failure_count += 1
            self.store.update(webhook)
            return {
                "success": False,
                "error": f"回调返回非成功状态码: {webhook.last_response_code}",
                "status_code": webhook.last_response_code,
            }
        except WebhookTargetValidationError:
            raise
        except Exception as error:
            logger.warning("webhook_test_failed", webhook_id=webhook.id, error_type=type(error).__name__)
            webhook.failure_count += 1
            self.store.update(webhook)
            return {"success": False, "error": "回调发送暂时失败"}

    async def trigger(
        self,
        event: WebhookEvent,
        payload: dict,
        org_id: str | None = None,
    ) -> int:
        """Trigger the webhook service."""
        webhooks = self.store.list_by_event(event, org_id)
        if not webhooks:
            return 0
        tasks = [self._delivery.send_with_retry(webhook, event, payload, self.store) for webhook in webhooks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success = sum(1 for result in results if result is True)
        logger.info("webhook_triggered", webhook_event=event.value, total=len(webhooks), success=success)
        return success

    async def close(self) -> None:
        """Release resources held by the webhook service."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        await self._delivery.close()


_webhook_service: WebhookService | None = None


def get_webhook_service() -> WebhookService:
    """Return the webhook service."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service


__all__ = [
    "PostgreSQLWebhookStore",
    "Webhook",
    "WebhookEvent",
    "WebhookService",
    "WebhookStore",
    "get_webhook_service",
]
