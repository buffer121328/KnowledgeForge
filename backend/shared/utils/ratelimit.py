"""Composite sensitive-route rate limiting and trusted client resolution."""

from __future__ import annotations

import hashlib
import ipaddress
import secrets
from typing import Iterable

from fastapi.responses import JSONResponse
from slowapi import Limiter
from starlette.requests import Request

from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service
from shared.config import settings
from shared.utils.logging import get_logger
from shared.utils.metrics import rate_limit_events_total

logger = get_logger(__name__)


RATE_LIMITS = {
    "auth_public": settings.rate_limit_auth_public,
    "qa_ask": settings.rate_limit_qa_ask,
    "doc_upload": settings.rate_limit_doc_upload,
    "doc_batch_upload": settings.rate_limit_doc_batch_upload,
    "password": settings.rate_limit_password,
    "api_key_mutation": settings.rate_limit_api_key_mutation,
    "webhook_test": settings.rate_limit_webhook_test,
}


def _networks(cidrs: Iterable[str]) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Return the networks."""
    return tuple(ipaddress.ip_network(value, strict=False) for value in cidrs)


def _valid_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse an IP address, returning None for invalid input."""
    try:
        return ipaddress.ip_address(value.strip())
    except ValueError:
        return None


def resolve_client_ip(
    request: Request,
    *,
    trusted_proxy_cidrs: Iterable[str] | None = None,
) -> str:
    """Resolve one client address without trusting caller-controlled forwarding."""

    peer_text = request.client.host if request.client else "0.0.0.0"
    peer = _valid_ip(peer_text)
    if peer is None:
        return "0.0.0.0"

    networks = _networks(
        settings.rate_limit_trusted_proxy_cidr_tuple
        if trusted_proxy_cidrs is None
        else trusted_proxy_cidrs
    )
    if not any(peer in network for network in networks):
        return str(peer)

    forwarded = request.headers.get("x-forwarded-for", "")
    if not forwarded:
        return str(peer)
    candidates = [item.strip() for item in forwarded.split(",")]
    parsed = [_valid_ip(item) for item in candidates]
    if not parsed or any(item is None for item in parsed):
        return str(peer)

    for candidate in reversed(parsed):
        assert candidate is not None
        if not any(candidate in network for network in networks):
            return str(candidate)
    return str(parsed[0])


def _normalized_route(path: str) -> str:
    """Return the normalized route."""
    for prefix in ("/api/v1", "/api"):
        if path == prefix:
            return "/"
        if path.startswith(f"{prefix}/"):
            return path[len(prefix):]
    return path or "/"


def _digest(*parts: str) -> str:
    """Return the digest."""
    value = "\x1f".join(("rate-limit-v1", *parts))
    return f"rl:v1:{hashlib.sha256(value.encode()).hexdigest()}"


def authenticated_composite_key(request: Request) -> str:
    """Hash authoritative credential identity together with validated client IP."""

    user = getattr(request.state, "user", None)
    if not user:
        return public_route_ip_key(request)
    credential_kind = "api_key" if getattr(user, "api_key_id", None) else "user"
    credential_id = getattr(user, "api_key_id", None) or getattr(user, "user_id", "")
    organization = getattr(user, "org_id", "")
    return _digest(
        credential_kind,
        str(organization),
        str(credential_id),
        resolve_client_ip(request),
    )


def public_route_ip_key(request: Request) -> str:
    """Build a rate-limit key from the public route and client IP."""
    return _digest("public", _normalized_route(request.url.path), resolve_client_ip(request))


# Backward-compatible aliases now use the secure composite subject.
user_key_func = authenticated_composite_key
api_key_key_func = authenticated_composite_key


def validate_rate_limit_configuration(*, environment: str, storage_uri: str) -> None:
    """Validate the rate limit configuration."""
    normalized = environment.strip().lower()
    if normalized not in {"development", "test"} and storage_uri.strip().lower().startswith("memory://"):
        raise ValueError("production rate limiting requires a shared storage backend")


def _credential_kind(request: Request) -> str:
    """Return the credential kind."""
    user = getattr(request.state, "user", None)
    if not user:
        return "public"
    return "api_key" if getattr(user, "api_key_id", None) else "user"


def _record_denial(request: Request, *, request_id: str, result: str) -> None:
    """Record the denial."""
    user = getattr(request.state, "user", None)
    credential_kind = _credential_kind(request)
    rate_limit_events_total.labels(
        result=result,
        credential_kind=credential_kind,
    ).inc()
    try:
        get_audit_service().log(
            user_id=getattr(user, "user_id", "") if user else "",
            username="",
            org_id=getattr(user, "org_id", "") if user else "",
            action=AuditAction.RATE_LIMITED,
            resource=f"route/{_normalized_route(request.url.path)}",
            result=AuditResult.DENIED if result == "exceeded" else AuditResult.FAILURE,
            metadata={
                "request_id": request_id,
                "route": _normalized_route(request.url.path),
                "method": request.method,
                "credential_kind": credential_kind,
                "result": result,
            },
        )
    except Exception as error:  # denial response must not expose audit internals
        logger.error("rate_limit_audit_failed", error_type=type(error).__name__)


def rate_limit_exceeded_handler(request: Request, _exc: Exception) -> JSONResponse:
    """Return the stable response for an exceeded rate limit."""
    request_id = secrets.token_hex(16)
    _record_denial(request, request_id=request_id, result="exceeded")
    return JSONResponse(
        status_code=429,
        content={
            "detail": {
                "code": "rate_limit_exceeded",
                "message": "请求过于频繁，请稍后重试。",
                "request_id": request_id,
            }
        },
        headers={"X-Request-ID": request_id, "Retry-After": "60"},
    )


def rate_limit_storage_error_handler(request: Request, error: Exception) -> JSONResponse:
    """Fail closed when the shared rate-limit backend is unavailable."""
    request_id = secrets.token_hex(16)
    _record_denial(request, request_id=request_id, result="storage_unavailable")
    logger.error(
        "rate_limit_storage_unavailable",
        error_type=type(error).__name__,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": {
                "code": "rate_limit_unavailable",
                "message": "请求保护服务暂不可用，请稍后重试。",
                "request_id": request_id,
            }
        },
        headers={"X-Request-ID": request_id},
    )


validate_rate_limit_configuration(
    environment=settings.app_environment,
    storage_uri=settings.rate_limit_storage_uri,
)
limiter = Limiter(
    key_func=public_route_ip_key,
    storage_uri=settings.rate_limit_storage_uri,
    headers_enabled=True,
    swallow_errors=False,
    in_memory_fallback_enabled=False,
    # The same endpoint function is mounted under /api and /api/v1. Endpoint
    # style prevents callers from doubling a budget by switching prefixes.
    key_style="endpoint",
)
