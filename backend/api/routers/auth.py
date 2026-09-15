"""Authentication HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from api.contracts import BearerToken, InvitationToken, LoginPassword, MessageResponse, Password, StrictRequestModel, Username
from auth.config import auth_settings
from auth.jwt_service import AuthService
from auth.token_blacklist import TokenBlacklist
from auth.user_service import IdentityServiceError, UserService
from domain.identity import UserContext
from api.dependencies import get_current_user
from fastapi import Depends
from shared.utils.ratelimit import RATE_LIMITS, limiter, public_route_ip_key

router = APIRouter(prefix="/auth", tags=["认证"])
auth_service = AuthService(auth_settings)
user_service = UserService(auth_service=auth_service)
token_blacklist = TokenBlacklist(auth_settings)


class LoginRequest(StrictRequestModel):
    """Represent a login request."""
    username: Username
    password: LoginPassword


class RegisterRequest(StrictRequestModel):
    """Represent a register request."""
    username: Username
    password: Password
    invitation_token: InvitationToken | None = None


class TokenResponse(BaseModel):
    """Represent a token response."""
    access_token: BearerToken
    refresh_token: BearerToken
    token_type: str = "bearer"
    expires_in: int = Field(gt=0)


class RefreshRequest(StrictRequestModel):
    """Represent a refresh request."""
    refresh_token: BearerToken


class CurrentUserResponse(BaseModel):
    """Return the authenticated identity and effective permissions."""

    user_id: str
    username: str
    display_name: str
    role: str
    org_id: str
    permissions: list[str]
    department_id: str | None = None
    is_department_manager: bool = False


def _raise_identity_error(error: IdentityServiceError) -> None:
    """Raise the appropriate error for the identity error."""
    raise HTTPException(status_code=error.status_code, detail=error.detail) from error


def _tokens_for(user: dict) -> TokenResponse:
    """Return the tokens for."""
    return TokenResponse(
        access_token=auth_service.create_access_token(
            user_id=user["user_id"],
            username=user["username"],
            role=user["role"],
            org_id=user["org_id"],
            token_version=user.get("token_version", 0),
            department_id=user.get("department_id"),
            is_department_manager=bool(user.get("is_department_manager", False)),
        ),
        refresh_token=auth_service.create_refresh_token(user_id=user["user_id"]),
        expires_in=auth_settings.access_token_expire_minutes * 60,
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit(RATE_LIMITS["auth_public"], key_func=public_route_ip_key)
async def login(req: LoginRequest, request: Request, response: Response):
    """Authenticate a user and return JWT access/refresh tokens."""
    try:
        user = user_service.authenticate(req.username, req.password)
    except IdentityServiceError as error:
        _raise_identity_error(error)
    return _tokens_for(user)


@router.post("/register", response_model=TokenResponse)
@limiter.limit(RATE_LIMITS["auth_public"], key_func=public_route_ip_key)
async def register(req: RegisterRequest, request: Request, response: Response):
    """Register a public viewer in the default organization."""
    try:
        user = user_service.register_public_user_with_invitation(
            req.username,
            req.password,
            invitation_token=req.invitation_token,
        )
    except IdentityServiceError as error:
        _raise_identity_error(error)
    return _tokens_for(user)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit(RATE_LIMITS["auth_public"], key_func=public_route_ip_key)
async def refresh_token(req: RefreshRequest, request: Request, response: Response):
    """Exchange an existing refresh token for a new access/refresh pair."""
    payload = auth_service.decode_token(req.refresh_token)
    if (
        not payload
        or payload.type != "refresh"
        or not auth_service.is_authorization_current(payload)
    ):
        raise HTTPException(status_code=401, detail="Refresh Token 无效或已过期")

    user = user_service.find_by_id(payload.sub, raise_on_missing=False)
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="账号已被禁用")
    return _tokens_for(user)


@router.post("/logout", response_model=MessageResponse)
async def logout(request: Request, user: UserContext = Depends(get_current_user)):
    """Best-effort blacklist of the current bearer token, then return success."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = auth_service.decode_token(token)
        if payload:
            await token_blacklist.add(token, payload.exp)
    return {"message": "已注销"}


@router.get("/me", response_model=CurrentUserResponse)
async def get_me(user: UserContext = Depends(get_current_user)):
    """Return the current authenticated user context."""
    record = user_service.find_by_id(user.user_id, raise_on_missing=False) or {}
    return {
        "user_id": user.user_id,
        "username": user.username,
        "display_name": str(record.get("display_name") or user.username),
        "role": user.role.value,
        "org_id": user.org_id,
        "permissions": [permission.value for permission in user.permissions],
        "department_id": user.department_id,
        "is_department_manager": user.is_department_manager,
    }


__all__ = [
    "LoginRequest",
    "RefreshRequest",
    "RegisterRequest",
    "TokenResponse",
    "auth_service",
    "router",
]
