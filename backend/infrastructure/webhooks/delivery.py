"""Webhook HTTP delivery, signing, SSRF controls, retries, and failure-disable mechanics."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import random
import socket
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from urllib.parse import SplitResult, urlsplit

import httpx

from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service
from infrastructure.webhooks.models import Webhook, WebhookEvent
from infrastructure.webhooks.stores import WebhookStore
from shared.config.settings import settings
from shared.utils.circuit_breaker import is_transient_dependency_error
from shared.utils.dependency_resilience import (
    DependencyExecutionPolicy,
    execute_with_policy,
    get_dependency_policy,
)
from shared.utils.logging import get_logger

logger = get_logger(__name__)

Sleep = Callable[[float], Awaitable[None]]
RandomValue = Callable[[], float]
Resolver = Callable[[str, int], Awaitable[Iterable[str]]]


class WebhookTargetValidationError(ValueError):
    """Raised when a target URL is not safe for an outbound webhook request."""


@dataclass(frozen=True)
class WebhookTarget:
    """Normalized target components retained only for request-time validation."""

    url: str
    scheme: str
    hostname: str
    port: int


async def _resolve_hostname(hostname: str, port: int) -> set[str]:
    """Resolve all TCP addresses without caching a target across delivery attempts."""
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    return {record[4][0] for record in records}


def _normalized_hostname(parsed: SplitResult) -> str:
    """Return the normalized hostname."""
    hostname = parsed.hostname
    if not hostname:
        raise WebhookTargetValidationError("Webhook target must include a hostname")
    try:
        return hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as error:
        raise WebhookTargetValidationError("Webhook target hostname is invalid") from error


def _is_global_address(address: str) -> bool:
    """Report whether the global address."""
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False


class WebhookTargetValidator:
    """Fail-closed URL and DNS policy for all outbound webhook paths."""

    def __init__(
        self,
        *,
        allowed_hosts: Iterable[str] = (),
        app_environment: str = "development",
        allow_http_in_local_development: bool = False,
        resolver: Resolver = _resolve_hostname,
    ) -> None:
        """Initialize the webhook target validator."""
        self._allowed_hosts = frozenset(
            host.strip().encode("idna").decode("ascii").lower().rstrip(".")
            for host in allowed_hosts
            if host.strip()
        )
        self._app_environment = app_environment.strip().lower()
        self._allow_http_in_local_development = allow_http_in_local_development
        self._resolver = resolver

    @classmethod
    def from_settings(cls) -> "WebhookTargetValidator":
        """Create the webhook target validator from settings."""
        return cls(
            allowed_hosts=settings.webhook_allowed_host_set,
            app_environment=settings.app_environment,
            allow_http_in_local_development=settings.webhook_allow_http_in_local_development,
        )

    def validate_syntax(self, url: str) -> WebhookTarget:
        """Validate target components that do not need network I/O."""
        if not isinstance(url, str) or not url.strip():
            raise WebhookTargetValidationError("Webhook target must be a non-empty URL")

        try:
            parsed = urlsplit(url.strip())
            port = parsed.port
        except ValueError as error:
            raise WebhookTargetValidationError("Webhook target URL is malformed") from error

        scheme = parsed.scheme.lower()
        local_http_allowed = (
            scheme == "http"
            and self._allow_http_in_local_development
            and self._app_environment in {"development", "test"}
        )
        if scheme != "https" and not local_http_allowed:
            raise WebhookTargetValidationError("Webhook target must use HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise WebhookTargetValidationError("Webhook target must not contain userinfo")
        if not parsed.netloc:
            raise WebhookTargetValidationError("Webhook target must include a hostname")

        hostname = _normalized_hostname(parsed)
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise WebhookTargetValidationError("Webhook target hostname is not publicly routable")

        expected_port = 80 if local_http_allowed else 443
        effective_port = port or expected_port
        if effective_port != expected_port:
            raise WebhookTargetValidationError("Webhook target port is not allowed")
        if self._allowed_hosts and hostname not in self._allowed_hosts:
            raise WebhookTargetValidationError("Webhook target hostname is not allowlisted")
        if _is_ip_literal(hostname) and not _is_global_address(hostname):
            raise WebhookTargetValidationError("Webhook target address is not publicly routable")

        return WebhookTarget(url=url.strip(), scheme=scheme, hostname=hostname, port=effective_port)

    async def validate(self, url: str) -> WebhookTarget:
        """Validate URL syntax and all DNS answers immediately before requesting."""
        target = self.validate_syntax(url)
        if _is_ip_literal(target.hostname):
            return target

        try:
            addresses = set(await self._resolver(target.hostname, target.port))
        except Exception as error:
            raise WebhookTargetValidationError("Webhook target hostname could not be resolved") from error

        if not addresses or any(not _is_global_address(address) for address in addresses):
            raise WebhookTargetValidationError("Webhook target resolved to a non-public address")
        return target


def _is_ip_literal(hostname: str) -> bool:
    """Report whether the IP literal."""
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


class RetryableWebhookResponseError(RuntimeError):
    """Private retry signal for HTTP responses that can safely be retried."""

    def __init__(self, status_code: int) -> None:
        """Initialize the retryable webhook response error."""
        super().__init__("Webhook delivery received a retryable response")
        self.status_code = status_code


def _status_class(status_code: int) -> str:
    """Return the status class."""
    if 400 <= status_code <= 499:
        return "4xx"
    if 500 <= status_code <= 599:
        return "5xx"
    return "http"


def build_delivery_request(
    webhook: Webhook,
    event: WebhookEvent,
    payload: dict,
    timestamp: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Build the unchanged JSON payload and optional HMAC headers."""
    body = json.dumps({
        "event": event.value,
        "payload": payload,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    })
    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Event": event.value,
        "X-Webhook-Id": webhook.id,
    }
    if webhook.secret:
        signature = hmac.new(webhook.secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={signature}"
    return body, headers


class WebhookDelivery:
    """Stateful async HTTP client with SSRF-safe bounded delivery attempts."""

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | None = None,
        policy: DependencyExecutionPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
        random_value: RandomValue = random.random,
        target_validator: WebhookTargetValidator | None = None,
        response_body_limit_bytes: int | None = None,
    ) -> None:
        """Initialize the webhook delivery."""
        resolved_policy = policy or get_dependency_policy("webhook")
        if max_retries is not None:
            resolved_policy = replace(resolved_policy, max_retries=max_retries)

        self.policy = resolved_policy
        self.max_retries = resolved_policy.max_retries
        self._sleep = sleep
        self._random_value = random_value
        self.target_validator = target_validator or WebhookTargetValidator.from_settings()
        self.response_body_limit_bytes = (
            response_body_limit_bytes
            if response_body_limit_bytes is not None
            else settings.webhook_response_body_limit_bytes
        )
        self._client = http_client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(
                timeout=resolved_policy.total_timeout_seconds,
                connect=resolved_policy.connect_timeout_seconds,
                read=resolved_policy.read_timeout_seconds,
                write=resolved_policy.total_timeout_seconds,
                pool=resolved_policy.total_timeout_seconds,
            ),
        )

    async def send(self, webhook: Webhook, event: WebhookEvent, payload: dict) -> bool:
        """Handle send for the webhook delivery."""
        await self.target_validator.validate(webhook.url)
        body, headers = build_delivery_request(webhook, event, payload)
        request = self._client.build_request("POST", webhook.url, content=body, headers=headers)
        response = await self._client.send(request, stream=True, follow_redirects=False)
        try:
            await self._consume_response_body(response)
            webhook.last_response_code = response.status_code
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                raise RetryableWebhookResponseError(response.status_code)
            return 200 <= response.status_code < 300
        finally:
            await response.aclose()

    async def _consume_response_body(self, response: httpx.Response) -> None:
        """Consume at most the configured number of bytes without retaining response data."""
        limit = self.response_body_limit_bytes
        if limit <= 0:
            return
        try:
            iterator = response.aiter_bytes(chunk_size=limit)
        except (AttributeError, TypeError):
            return
        if not hasattr(iterator, "__aiter__"):
            return

        remaining = limit
        async for chunk in iterator:
            remaining -= len(chunk)
            if remaining <= 0:
                break

    async def send_with_retry(self, webhook: Webhook, event: WebhookEvent, payload: dict, store: WebhookStore) -> bool:
        """Send the with retry."""
        rejection_status: str | None = None
        try:
            delivered = await execute_with_policy(
                self.policy,
                lambda: self.send(webhook, event, payload),
                retry_predicate=is_transient_dependency_error,
                sleep=self._sleep,
                random_value=self._random_value,
            )
            if delivered:
                webhook.failure_count = 0
                webhook.last_triggered_at = datetime.now(timezone.utc).isoformat()
                store.update(webhook)
                self._record_delivery_attempt(store, webhook, event, "2xx", "succeeded")
                return True
            if webhook.last_response_code is not None:
                rejection_status = _status_class(webhook.last_response_code)
        except WebhookTargetValidationError:
            rejection_status = "validation"
        except RetryableWebhookResponseError as error:
            rejection_status = _status_class(error.status_code)
        except Exception as error:
            rejection_status = "network"
            logger.warning(
                "webhook_send_failed",
                webhook_id=webhook.id,
                error_type=type(error).__name__,
            )

        if rejection_status is not None:
            try:
                get_audit_service().log_security_event(
                    user_id="system",
                    org_id=webhook.org_id,
                    action=AuditAction.WEBHOOK_REJECTED,
                    reason_code="webhook_delivery_rejected",
                    metadata={
                        "target_id": webhook.id,
                        "status_class": rejection_status,
                    },
                    result=AuditResult.DENIED,
                    required=rejection_status == "validation",
                )
            except Exception as audit_error:
                logger.warning(
                    "security_event_audit_failed",
                    security_event="webhook_rejected",
                    error_type=type(audit_error).__name__,
                )

        webhook.failure_count += 1
        if webhook.failure_count >= webhook.max_failures:
            webhook.is_active = False
            logger.warning("webhook_disabled", webhook_id=webhook.id, failure_count=webhook.failure_count)
        store.update(webhook)
        self._record_delivery_attempt(
            store,
            webhook,
            event,
            rejection_status or "unknown",
            "failed",
        )
        return False

    @staticmethod
    def _record_delivery_attempt(
        store: WebhookStore,
        webhook: Webhook,
        event: WebhookEvent,
        status_category: str,
        final_status: str,
    ) -> None:
        """Persist only controlled delivery facts when the store supports them."""
        recorder = getattr(store, "record_delivery_attempt", None)
        if recorder is None:
            return
        try:
            recorder(
                webhook,
                event,
                attempt=1,
                status_category=status_category,
                final_status=final_status,
            )
        except Exception as error:
            logger.warning(
                "webhook_delivery_attempt_persist_failed",
                webhook_id=webhook.id,
                error_type=type(error).__name__,
            )

    async def close(self) -> None:
        """Release resources held by the webhook delivery."""
        await self._client.aclose()
