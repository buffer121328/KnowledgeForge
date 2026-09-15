"""API Key HTTP endpoints backed by the authoritative lifecycle service."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from api.contracts import BearerToken, MessageResponse, ResourceId, ShortName, StrictRequestModel, ToggleResponse
from api.dependencies import get_current_user
from auth.apikey_service import APIKeyLifecycleError, APIKeyService, get_default_service
from domain.identity import APIKey, Permission, UserContext
from shared.utils.ratelimit import RATE_LIMITS, authenticated_composite_key, limiter

router = APIRouter(prefix="/apikey", tags=["认证"])


class APIKeyCreateRequest(StrictRequestModel):
    """Represent a API key create request."""
    name: ShortName
    permissions: list[Permission] = Field(default_factory=list, max_length=len(Permission))
    expires_days: int | None = Field(default=None, ge=1, le=3650)


class APIKeyRotateRequest(StrictRequestModel):
    """Represent a API key rotate request."""
    expires_days: int | None = Field(default=None, ge=1, le=3650)


class APIKeyScopeUpdateRequest(StrictRequestModel):
    """Represent a API key scope update request."""
    permissions: list[Permission] = Field(default_factory=list, max_length=len(Permission))


class APIKeyResponse(BaseModel):
    """Represent a API key response."""
    id: str
    name: str
    key_prefix: str
    api_key: BearerToken | None = None
    scopes: list[str]
    expires_at: str | None
    created_at: str
    last_used_at: str | None
    disabled_at: str | None
    rotated_at: str | None
    rotation_count: int
    revoked_at: str | None
    is_active: bool
    status: str
    warnings: list[str]


def _to_response(
    api_key: APIKey,
    *,
    service: APIKeyService,
    raw_key: str | None = None,
) -> APIKeyResponse:
    """Convert the module to response."""
    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        api_key=raw_key,
        scopes=[
            scope.value if hasattr(scope, "value") else str(scope)
            for scope in api_key.scopes
        ],
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        disabled_at=api_key.disabled_at,
        rotated_at=api_key.rotated_at,
        rotation_count=api_key.rotation_count,
        revoked_at=api_key.revoked_at,
        is_active=api_key.is_active,
        status=service.status(api_key),
        warnings=service.warnings(api_key),
    )


def _service() -> APIKeyService:
    """Return the service."""
    return get_default_service()


def _raise_lifecycle_error(error: APIKeyLifecycleError) -> None:
    """Raise the appropriate error for the lifecycle error."""
    request_id = secrets.token_hex(8)
    raise HTTPException(
        status_code=error.status_code,
        detail={
            "code": error.code,
            "message": error.message,
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    ) from error


def _not_found() -> None:
    """Return the not found."""
    raise HTTPException(status_code=404, detail="API Key 不存在")


@router.post("/create", response_model=APIKeyResponse)
@limiter.limit(RATE_LIMITS["api_key_mutation"], key_func=authenticated_composite_key)
async def create_api_key(
    req: APIKeyCreateRequest,
    request: Request,
    response: Response,
    user: UserContext = Depends(get_current_user),
):
    """Create a bounded Key and return its raw value exactly once."""
    service = _service()
    try:
        api_key, raw_key = service.create_key(
            name=req.name,
            user_id=user.user_id,
            org_id=user.org_id,
            role=user.role,
            scopes=req.permissions,
            expires_days=req.expires_days,
            actor_permissions=user.permissions,
            actor_username=user.username,
        )
    except APIKeyLifecycleError as error:
        _raise_lifecycle_error(error)
    return _to_response(api_key, service=service, raw_key=raw_key)


@router.get("/list", response_model=list[APIKeyResponse])
async def list_api_keys(user: UserContext = Depends(get_current_user)):
    """List current-user Keys, including revoked lifecycle records."""
    service = _service()
    return [
        _to_response(api_key, service=service)
        for api_key in service.list_keys(user.user_id, user.org_id)
    ]


@router.post("/{key_id}/rotate", response_model=APIKeyResponse)
@limiter.limit(RATE_LIMITS["api_key_mutation"], key_func=authenticated_composite_key)
async def rotate_api_key(
    key_id: ResourceId,
    req: APIKeyRotateRequest,
    request: Request,
    response: Response,
    user: UserContext = Depends(get_current_user),
):
    """Atomically replace Key material and disclose the new raw value once."""
    service = _service()
    try:
        result = service.rotate(
            key_id,
            user.user_id,
            user.org_id,
            expires_days=req.expires_days,
            actor_username=user.username,
        )
    except APIKeyLifecycleError as error:
        _raise_lifecycle_error(error)
    if result is None:
        _not_found()
    api_key, raw_key = result
    return _to_response(api_key, service=service, raw_key=raw_key)


@router.patch("/{key_id}/scopes", response_model=APIKeyResponse)
@limiter.limit(RATE_LIMITS["api_key_mutation"], key_func=authenticated_composite_key)
async def update_api_key_scopes(
    key_id: ResourceId,
    req: APIKeyScopeUpdateRequest,
    request: Request,
    response: Response,
    user: UserContext = Depends(get_current_user),
):
    """Reduce a Key's explicit scopes without permitting in-place expansion."""
    service = _service()
    try:
        api_key = service.update_scopes(
            key_id,
            user.user_id,
            user.org_id,
            req.permissions,
            actor_permissions=user.permissions,
            actor_username=user.username,
        )
    except APIKeyLifecycleError as error:
        _raise_lifecycle_error(error)
    if api_key is None:
        _not_found()
    return _to_response(api_key, service=service)


@router.delete("/{key_id}", response_model=MessageResponse)
@limiter.limit(RATE_LIMITS["api_key_mutation"], key_func=authenticated_composite_key)
async def delete_api_key(
    key_id: ResourceId,
    request: Request,
    response: Response,
    user: UserContext = Depends(get_current_user),
):
    """Soft-revoke a current-user, current-organization API Key."""
    try:
        api_key = _service().revoke(
            key_id,
            user.user_id,
            user.org_id,
            actor_username=user.username,
        )
    except APIKeyLifecycleError as error:
        _raise_lifecycle_error(error)
    if api_key is None:
        _not_found()
    return {"message": "已撤销"}


@router.post("/{key_id}/toggle", response_model=ToggleResponse)
@limiter.limit(RATE_LIMITS["api_key_mutation"], key_func=authenticated_composite_key)
async def toggle_api_key(
    key_id: ResourceId,
    request: Request,
    response: Response,
    user: UserContext = Depends(get_current_user),
):
    """Suspend or resume a non-revoked, owned API Key."""
    try:
        api_key = _service().toggle(
            key_id,
            user.user_id,
            user.org_id,
            actor_username=user.username,
        )
    except APIKeyLifecycleError as error:
        _raise_lifecycle_error(error)
    if api_key is None:
        _not_found()
    status_text = "启用" if api_key.is_active else "禁用"
    return {"message": f"API Key 已{status_text}", "is_active": api_key.is_active}
