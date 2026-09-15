"""Acceptance tests for SSRF-safe webhook target validation and delivery."""
from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from api.routers.webhooks import WebhookCreateRequest, register_webhook
from domain.identity import UserContext, UserRole
from infrastructure.webhooks.delivery import (
    WebhookDelivery,
    WebhookTargetValidationError,
    WebhookTargetValidator,
)
from infrastructure.webhooks.models import Webhook, WebhookEvent
from infrastructure.webhooks.stores import MemoryWebhookStore


async def public_resolver(host: str, port: int) -> set[str]:
    del host, port
    return {"8.8.8.8"}


def admin_user() -> UserContext:
    return UserContext(user_id="admin-1", username="admin", role=UserRole.ADMIN, org_id="org-1")


def webhook(url: str = "https://receiver.example/hook") -> Webhook:
    return Webhook(
        id="wh_security",
        url=url,
        events=[WebhookEvent.DOC_INGESTED],
        secret="test-secret",
    )


class TestWebhookTargetValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "http://receiver.example/hook",
            "https://user:password@receiver.example/hook",
            "https://receiver.example:8443/hook",
            "https://127.0.0.1/hook",
            "https://10.0.0.1/hook",
            "https://[::1]/hook",
            "https:///missing-host",
        ],
    )
    def test_registration_syntax_rejects_unsafe_targets(self, url: str) -> None:
        validator = WebhookTargetValidator(resolver=public_resolver)

        with pytest.raises(WebhookTargetValidationError):
            validator.validate_syntax(url)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.8", "169.254.169.254", "::1", "fc00::1"])
    async def test_dns_non_public_address_is_rejected(self, address: str) -> None:
        async def resolver(host: str, port: int) -> set[str]:
            del host, port
            return {address}

        validator = WebhookTargetValidator(resolver=resolver)

        with pytest.raises(WebhookTargetValidationError):
            await validator.validate("https://receiver.example/hook")

    @pytest.mark.asyncio
    async def test_mixed_dns_answers_and_resolution_errors_fail_closed(self) -> None:
        async def mixed_resolver(host: str, port: int) -> set[str]:
            del host, port
            return {"8.8.8.8", "10.0.0.8"}

        async def failed_resolver(host: str, port: int) -> set[str]:
            del host, port
            raise socket.gaierror("not found")

        with pytest.raises(WebhookTargetValidationError):
            await WebhookTargetValidator(resolver=mixed_resolver).validate("https://receiver.example/hook")
        with pytest.raises(WebhookTargetValidationError):
            await WebhookTargetValidator(resolver=failed_resolver).validate("https://receiver.example/hook")

    @pytest.mark.asyncio
    async def test_allowlist_requires_exact_normalized_hostname(self) -> None:
        validator = WebhookTargetValidator(allowed_hosts={"Receiver.Example"}, resolver=public_resolver)

        assert (await validator.validate("https://receiver.example/hook")).hostname == "receiver.example"
        with pytest.raises(WebhookTargetValidationError):
            await validator.validate("https://sub.receiver.example/hook")

    def test_http_is_allowed_only_for_explicit_local_development(self) -> None:
        allowed = WebhookTargetValidator(
            app_environment="development",
            allow_http_in_local_development=True,
            resolver=public_resolver,
        )
        assert allowed.validate_syntax("http://receiver.example/hook").scheme == "http"

        production = WebhookTargetValidator(
            app_environment="production",
            allow_http_in_local_development=True,
            resolver=public_resolver,
        )
        with pytest.raises(WebhookTargetValidationError):
            production.validate_syntax("http://receiver.example/hook")


class TestWebhookDeliverySafety:
    @pytest.mark.asyncio
    async def test_runtime_private_target_is_never_posted(self) -> None:
        async def private_resolver(host: str, port: int) -> set[str]:
            del host, port
            return {"169.254.169.254"}

        client = AsyncMock(spec=httpx.AsyncClient)
        client.build_request.side_effect = httpx.Request
        delivery = WebhookDelivery(
            http_client=client,
            target_validator=WebhookTargetValidator(resolver=private_resolver),
        )
        item = webhook()
        store = MemoryWebhookStore()
        store.save(item)

        assert await delivery.send_with_retry(item, WebhookEvent.DOC_INGESTED, {}, store) is False
        client.send.assert_not_awaited()
        assert item.failure_count == 1

    @pytest.mark.asyncio
    async def test_redirect_is_not_followed_and_bounded_body_is_consumed(self) -> None:
        response = MagicMock(status_code=302)
        observed_limits: list[int | None] = []

        async def chunks(chunk_size: int | None = None) -> AsyncIterator[bytes]:
            observed_limits.append(chunk_size)
            yield b"redirect-body"

        response.aiter_bytes = chunks
        response.aclose = AsyncMock()
        client = AsyncMock(spec=httpx.AsyncClient)
        client.build_request.side_effect = httpx.Request
        client.send.return_value = response
        delivery = WebhookDelivery(
            http_client=client,
            target_validator=WebhookTargetValidator(resolver=public_resolver),
            response_body_limit_bytes=4,
        )
        item = webhook()

        assert await delivery.send(item, WebhookEvent.DOC_INGESTED, {"doc_id": "d1"}) is False
        client.send.assert_awaited_once()
        assert client.send.call_args.kwargs["follow_redirects"] is False
        assert client.send.call_args.kwargs["stream"] is True
        assert observed_limits == [4]

    @pytest.mark.asyncio
    async def test_signed_delivery_uses_no_more_than_response_body_limit(self) -> None:
        response = MagicMock(status_code=204)
        consumed: list[bytes] = []

        async def chunks(chunk_size: int | None = None) -> AsyncIterator[bytes]:
            assert chunk_size == 4
            for chunk in (b"abcd", b"efgh"):
                consumed.append(chunk)
                yield chunk

        response.aiter_bytes = chunks
        response.aclose = AsyncMock()
        client = AsyncMock(spec=httpx.AsyncClient)
        client.build_request.side_effect = httpx.Request
        client.send.return_value = response
        delivery = WebhookDelivery(
            http_client=client,
            target_validator=WebhookTargetValidator(resolver=public_resolver),
            response_body_limit_bytes=4,
        )
        item = webhook()

        assert await delivery.send(item, WebhookEvent.DOC_INGESTED, {"doc_id": "d1"}) is True
        assert consumed == [b"abcd"]
        headers = client.send.call_args.args[0].headers
        assert headers["X-Webhook-Signature"].startswith("sha256=")


@pytest.mark.asyncio
async def test_registration_endpoint_returns_400_before_saving_unsafe_target() -> None:
    service = MagicMock()
    service.validate_target = AsyncMock(side_effect=WebhookTargetValidationError("unsafe target"))

    with patch("api.routers.webhooks.get_webhook_service", return_value=service):
        with pytest.raises(HTTPException) as captured:
            await register_webhook(
                WebhookCreateRequest(url="https://127.0.0.1/hook", events=[WebhookEvent.DOC_INGESTED.value]),
                admin_user(),
            )

    assert captured.value.status_code == 400
    service.register.assert_not_called()
