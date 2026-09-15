"""User-management HTTP routes backed by ``auth.user_service``."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, EmailStr, Field

from api.contracts import MessageResponse, Password, ResourceId, SearchText, StrictRequestModel, ToggleResponse, Username
from api.dependencies import get_current_user, require_permission
from auth.config import auth_settings
from auth.jwt_service import AuthService
from auth.user_service import IdentityServiceError, UserService
from domain.identity import Permission, UserContext, UserRole
from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service
from shared.utils.ratelimit import RATE_LIMITS, authenticated_composite_key, limiter

router = APIRouter(prefix="/users", tags=["用户管理"])
auth_service = AuthService(auth_settings)
user_service = UserService(auth_service=auth_service)


class UserListItem(BaseModel):
    """Represent user list item."""
    user_id: str
    username: str
    display_name: str
    email: str
    role: str
    org_id: str
    department_id: str | None = None
    is_department_manager: bool = False
    is_active: bool
    created_at: str | None = None


class UserDetail(UserListItem):
    """Represent user detail."""
    last_login_at: str | None = None


class UserCreateRequest(StrictRequestModel):
    """Represent a user create request."""
    username: Username
    password: Password
    display_name: str = Field("", max_length=64)
    email: EmailStr = ""
    role: UserRole = UserRole.VIEWER
    org_id: ResourceId | None = None
    department_id: ResourceId | None = None
    is_department_manager: bool = False


class UserUpdateRequest(StrictRequestModel):
    """Represent a user update request."""
    display_name: str | None = Field(default=None, max_length=64)
    email: EmailStr | None = None
    role: UserRole | None = None
    org_id: ResourceId | None = None
    department_id: ResourceId | None = None
    is_department_manager: bool | None = None
    is_active: bool | None = None


class PasswordChangeRequest(StrictRequestModel):
    """Represent a password change request."""
    old_password: Password
    new_password: Password


class PasswordResetRequest(StrictRequestModel):
    """Represent a password reset request."""
    new_password: Password


class BatchRoleRequest(StrictRequestModel):
    """Represent a batch role request."""
    user_ids: list[ResourceId] = Field(min_length=1, max_length=100)
    role: UserRole


class UserMutationResponse(BaseModel):
    """Return a user mutation result and affected user identifier."""

    message: str
    user_id: ResourceId


class BatchRoleResponse(BaseModel):
    """Return the number of users updated by a batch role change."""

    message: str
    updated: int = Field(ge=0)


def _raise_identity_error(error: IdentityServiceError) -> None:
    """Raise the appropriate error for the identity error."""
    raise HTTPException(status_code=error.status_code, detail=error.detail) from error


def _reject_org_override(requested_org_id: str | None, user: UserContext) -> None:
    """Reject the org override."""
    if requested_org_id is not None and requested_org_id != user.org_id:
        _audit_tenant_denial(user, "user", requested_org_id, "org_override")
        raise HTTPException(status_code=400, detail="org_id 只能由当前认证组织确定")


def _audit_tenant_denial(
    actor: UserContext,
    target_type: str,
    target_id: str,
    operation: str,
) -> None:
    """Record an audit event for the tenant denial."""
    get_audit_service().log(
        user_id=actor.user_id,
        username=actor.username,
        org_id=actor.org_id,
        action=AuditAction.TENANT_ACCESS_DENIED,
        resource=f"tenant/{target_type}/{target_id}",
        result=AuditResult.DENIED,
        metadata={"target_type": target_type, "target_id": target_id, "operation": operation},
    )


def _audit_cross_org_user_if_present(
    target_user_id: str,
    actor: UserContext,
    operation: str,
) -> None:
    """Record an audit event for the cross org user if present."""
    target = user_service.find_by_id(target_user_id, raise_on_missing=False)
    if target and target.get("org_id") != actor.org_id:
        _audit_tenant_denial(actor, "user", target_user_id, operation)


def _role_value(role: UserRole | str) -> str:
    """Return the role value."""
    return role.value if isinstance(role, UserRole) else role


def _to_list_item(user: dict) -> UserListItem:
    """Convert the module to list item."""
    return UserListItem(
        user_id=user["user_id"],
        username=user["username"],
        display_name=user.get("display_name", ""),
        email=user.get("email", ""),
        role=_role_value(user["role"]),
        org_id=user["org_id"],
        department_id=user.get("department_id"),
        is_department_manager=bool(user.get("is_department_manager", False)),
        is_active=user.get("is_active", True),
        created_at=user.get("created_at"),
    )


def _to_detail(user: dict) -> UserDetail:
    """Convert the module to detail."""
    return UserDetail(**_to_list_item(user).model_dump(), last_login_at=user.get("last_login_at"))


@router.get("", response_model=list[UserListItem])
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    role: UserRole | None = None,
    org_id: ResourceId | None = None,
    search: SearchText | None = None,
    is_active: bool | None = None,
    user: UserContext = Depends(require_permission(Permission.ADMIN_USER)),
):
    """List the users."""
    _reject_org_override(org_id, user)
    return [
        _to_list_item(record)
        for record in user_service.list_users(
            page=page,
            page_size=page_size,
            role=role,
            org_id=user.org_id,
            search=search,
            is_active=is_active,
        )
    ]


@router.post("", response_model=UserDetail, status_code=201)
async def create_user(req: UserCreateRequest, user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """Create the user."""
    _reject_org_override(req.org_id, user)
    try:
        record = user_service.create_user(
            username=req.username,
            password=req.password,
            display_name=req.display_name,
            email=req.email,
            role=req.role,
            org_id=user.org_id,
            department_id=req.department_id,
            is_department_manager=req.is_department_manager,
        )
    except IdentityServiceError as error:
        _raise_identity_error(error)
    return _to_detail(record)


# Static paths must be registered before the dynamic /{user_id} routes.
@router.post("/batch/role", response_model=BatchRoleResponse)
async def batch_change_role(req: BatchRoleRequest, user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """Process the change role."""
    for target_user_id in req.user_ids:
        _audit_cross_org_user_if_present(target_user_id, user, "batch_role")
    updated = user_service.batch_change_role(
        req.user_ids,
        req.role,
        org_id=user.org_id,
        actor_user_id=user.user_id,
    )
    return {"message": f"已更新 {updated} 个用户的角色", "updated": updated}


@router.post("/me/password", response_model=MessageResponse)
@limiter.limit(RATE_LIMITS["password"], key_func=authenticated_composite_key)
async def change_own_password(
    req: PasswordChangeRequest,
    request: Request,
    response: Response,
    user: UserContext = Depends(get_current_user),
):
    """Change the own password."""
    try:
        user_service.change_own_password(user.user_id, old_password=req.old_password, new_password=req.new_password)
    except IdentityServiceError as error:
        _raise_identity_error(error)
    return {"message": "密码已修改"}


@router.get("/{user_id}", response_model=UserDetail)
async def get_user(user_id: ResourceId, user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """Return the user."""
    try:
        return _to_detail(user_service.find_by_id(user_id, org_id=user.org_id))
    except IdentityServiceError as error:
        _audit_cross_org_user_if_present(user_id, user, "get")
        _raise_identity_error(error)


@router.patch("/{user_id}", response_model=UserDetail)
async def update_user(
    user_id: ResourceId,
    req: UserUpdateRequest,
    user: UserContext = Depends(require_permission(Permission.ADMIN_USER)),
):
    """Update the user."""
    _reject_org_override(req.org_id, user)
    try:
        record = user_service.update_user(
            user_id,
            display_name=req.display_name,
            email=req.email,
            role=req.role,
            org_id=None,
            department_id=req.department_id,
            is_department_manager=req.is_department_manager,
            actor_user_id=user.user_id,
            is_active=req.is_active,
            scope_org_id=user.org_id,
        )
    except IdentityServiceError as error:
        _audit_cross_org_user_if_present(user_id, user, "update")
        _raise_identity_error(error)
    return _to_detail(record)


@router.delete("/{user_id}", response_model=UserMutationResponse)
async def delete_user(user_id: ResourceId, user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """Delete the user."""
    try:
        user_service.delete_user(
            user_id,
            actor_user_id=user.user_id,
            org_id=user.org_id,
        )
    except IdentityServiceError as error:
        _audit_cross_org_user_if_present(user_id, user, "delete")
        _raise_identity_error(error)
    return {"message": "用户已删除", "user_id": user_id}


@router.post("/{user_id}/reset-password", response_model=UserMutationResponse)
@limiter.limit(RATE_LIMITS["password"], key_func=authenticated_composite_key)
async def reset_user_password(
    user_id: ResourceId,
    req: PasswordResetRequest,
    request: Request,
    response: Response,
    user: UserContext = Depends(require_permission(Permission.ADMIN_USER)),
):
    """Reset the user password."""
    try:
        user_service.reset_password(
            user_id,
            new_password=req.new_password,
            org_id=user.org_id,
        )
    except IdentityServiceError as error:
        _audit_cross_org_user_if_present(user_id, user, "reset_password")
        _raise_identity_error(error)
    return {"message": "密码已重置", "user_id": user_id}


@router.post("/{user_id}/toggle-active", response_model=ToggleResponse)
async def toggle_user_active(user_id: ResourceId, user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """Toggle the user active."""
    try:
        target = user_service.toggle_active(
            user_id,
            actor_user_id=user.user_id,
            org_id=user.org_id,
        )
    except IdentityServiceError as error:
        _audit_cross_org_user_if_present(user_id, user, "toggle_active")
        _raise_identity_error(error)
    status_text = "启用" if target["is_active"] else "禁用"
    return {"message": f"用户已{status_text}", "is_active": target["is_active"]}


# Legacy helper retained for tests/callers that previously reached into the router.
def _find_user_by_id(user_id: str, *, raise_on_missing: bool = True) -> dict | None:
    """Find the user by ID."""
    try:
        return user_service.find_by_id(user_id, raise_on_missing=raise_on_missing)
    except IdentityServiceError as error:
        _raise_identity_error(error)
