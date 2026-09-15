"""Role-management HTTP routes backed by ``auth.role_service``."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, RootModel

from api.contracts import Description, DisplayName, ResourceId, RoleName, SearchText, StrictRequestModel
from api.dependencies import require_permission
from auth.role_service import RoleService
from auth.user_service import IdentityServiceError
from domain.identity import Permission, UserContext, UserRole
from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service

router = APIRouter(prefix="/roles", tags=["角色管理"])
role_service = RoleService()


class RoleListItem(BaseModel):
    """Represent role list item."""
    role_id: str
    name: str
    display_name: str
    description: str
    permission_count: int
    user_count: int
    is_builtin: bool


class RoleDetail(BaseModel):
    """Represent role detail."""
    role_id: str
    name: str
    display_name: str
    description: str
    permissions: list[str]
    is_builtin: bool
    user_count: int


class RoleCreateRequest(StrictRequestModel):
    """Represent a role create request."""
    name: RoleName
    display_name: DisplayName = ""
    description: Description = ""
    permissions: list[Permission] = Field(default_factory=list, max_length=len(Permission))


class RoleUpdateRequest(StrictRequestModel):
    """Represent a role update request."""
    display_name: DisplayName | None = None
    description: Description | None = None
    permissions: list[Permission] | None = Field(default=None, max_length=len(Permission))


class PermissionListRequest(StrictRequestModel):
    """Replace or extend the permission set for one role."""

    permissions: list[Permission] = Field(min_length=1, max_length=len(Permission))


class RoleUserItem(BaseModel):
    """Represent one user assigned to a role."""

    user_id: ResourceId
    username: str
    display_name: str
    org_id: ResourceId
    is_active: bool


class RoleUsersResponse(BaseModel):
    """Represent a role users response."""
    role_id: str
    role_name: RoleName
    users: list[RoleUserItem]


class PermissionItem(BaseModel):
    """Describe one assignable permission code."""

    value: Permission
    category: str
    name: str


class PermissionCodeList(RootModel[list[Permission]]):
    """Preserve the permission-array response with a reusable schema."""


class RoleMutationResponse(BaseModel):
    """Return an affected role identifier."""

    message: str
    role_name: RoleName


class PermissionMutationResponse(BaseModel):
    """Return permission mutation counts."""

    added: int | None = Field(default=None, ge=0)
    removed: int | None = Field(default=None, ge=0)
    total: int = Field(ge=0)


def _raise_identity_error(error: IdentityServiceError) -> None:
    """Raise the appropriate error for the identity error."""
    raise HTTPException(status_code=error.status_code, detail=error.detail) from error


def _audit_cross_org_role_if_present(
    role_name: RoleName,
    user: UserContext,
    operation: str,
) -> None:
    """Record an audit event for the cross org role if present."""
    target = next(
        (
            role
            for role in role_service.roles.values()
            if not role.get("is_builtin")
            and role.get("name") == role_name
            and role.get("org_id") not in {"", user.org_id}
        ),
        None,
    )
    if target:
        get_audit_service().log(
            user_id=user.user_id,
            username=user.username,
            org_id=user.org_id,
            action=AuditAction.TENANT_ACCESS_DENIED,
            resource=f"tenant/role/{role_name}",
            result=AuditResult.DENIED,
            metadata={
                "target_type": "role",
                "target_id": role_name,
                "operation": operation,
            },
        )


def _to_list_item(role: dict, org_id: str | None = None) -> RoleListItem:
    """Convert the module to list item."""
    return RoleListItem(
        role_id=role["role_id"],
        name=role["name"],
        display_name=role["display_name"],
        description=role.get("description", ""),
        permission_count=len(role["permissions"]),
        user_count=role_service.count_users_with_role(role["name"], org_id=org_id),
        is_builtin=role["is_builtin"],
    )


def _to_detail(role: dict, org_id: str | None = None) -> RoleDetail:
    """Convert the module to detail."""
    return RoleDetail(
        role_id=role["role_id"],
        name=role["name"],
        display_name=role["display_name"],
        description=role.get("description", ""),
        permissions=role["permissions"],
        is_builtin=role["is_builtin"],
        user_count=role_service.count_users_with_role(role["name"], org_id=org_id),
    )


@router.get("", response_model=list[RoleListItem])
async def list_roles(
    search: SearchText | None = None,
    is_builtin: bool | None = None,
    user: UserContext = Depends(require_permission(Permission.ADMIN_USER)),
):
    """List the roles."""
    return [
        _to_list_item(role, user.org_id)
        for role in role_service.list_roles(
            search=search,
            is_builtin=is_builtin,
            org_id=user.org_id,
        )
    ]


# Static route stays ahead of /{role_name}.
@router.get("/meta/permissions", response_model=list[PermissionItem])
async def list_permissions(user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """List the permissions."""
    return [
        {"value": permission.value, "category": permission.value.split(":")[0], "name": permission.value.split(":")[1]}
        for permission in Permission
    ]


@router.post("", response_model=RoleDetail, status_code=201)
async def create_role(req: RoleCreateRequest, user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """Create the role."""
    try:
        return _to_detail(
            role_service.create_role(
                name=req.name,
                display_name=req.display_name,
                description=req.description,
                permissions=req.permissions,
                org_id=user.org_id,
            ),
            user.org_id,
        )
    except IdentityServiceError as error:
        _raise_identity_error(error)


@router.get("/{role_name}", response_model=RoleDetail)
async def get_role(role_name: RoleName, user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """Return the role."""
    try:
        return _to_detail(
            role_service.find_role(role_name, org_id=user.org_id),
            user.org_id,
        )
    except IdentityServiceError as error:
        _audit_cross_org_role_if_present(role_name, user, "get")
        _raise_identity_error(error)


@router.patch("/{role_name}", response_model=RoleDetail)
async def update_role(
    role_name: RoleName,
    req: RoleUpdateRequest,
    user: UserContext = Depends(require_permission(Permission.ADMIN_USER)),
):
    """Update the role."""
    try:
        return _to_detail(
            role_service.update_role(
                role_name,
                display_name=req.display_name,
                description=req.description,
                permissions=req.permissions,
                org_id=user.org_id,
            ),
            user.org_id,
        )
    except IdentityServiceError as error:
        _audit_cross_org_role_if_present(role_name, user, "update")
        _raise_identity_error(error)


@router.delete("/{role_name}", response_model=RoleMutationResponse)
async def delete_role(role_name: RoleName, user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """Delete the role."""
    try:
        role_service.delete_role(role_name, org_id=user.org_id)
    except IdentityServiceError as error:
        _audit_cross_org_role_if_present(role_name, user, "delete")
        _raise_identity_error(error)
    return {"message": "角色已删除", "role_name": role_name}


@router.get("/{role_name}/permissions", response_model=PermissionCodeList)
async def get_role_permissions(role_name: RoleName, user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """Return the role permissions."""
    try:
        return role_service.get_permissions(role_name, org_id=user.org_id)
    except IdentityServiceError as error:
        _audit_cross_org_role_if_present(role_name, user, "get_permissions")
        _raise_identity_error(error)


@router.put("/{role_name}/permissions", response_model=PermissionCodeList)
async def set_role_permissions(
    role_name: RoleName,
    req: PermissionListRequest,
    user: UserContext = Depends(require_permission(Permission.ADMIN_USER)),
):
    """Store the role permissions."""
    try:
        return role_service.set_permissions(
            role_name,
            [permission.value for permission in req.permissions],
            org_id=user.org_id,
        )
    except IdentityServiceError as error:
        _audit_cross_org_role_if_present(role_name, user, "set_permissions")
        _raise_identity_error(error)


@router.post("/{role_name}/permissions", response_model=PermissionMutationResponse)
async def add_role_permissions(
    role_name: RoleName,
    req: PermissionListRequest,
    user: UserContext = Depends(require_permission(Permission.ADMIN_USER)),
):
    """Add the role permissions."""
    try:
        added, total = role_service.add_permissions(
            role_name,
            [permission.value for permission in req.permissions],
            org_id=user.org_id,
        )
    except IdentityServiceError as error:
        _audit_cross_org_role_if_present(role_name, user, "add_permissions")
        _raise_identity_error(error)
    return {"added": added, "total": total}


@router.delete("/{role_name}/permissions", response_model=PermissionMutationResponse)
async def remove_role_permissions(
    role_name: RoleName,
    permissions: list[Permission] = Query(..., min_length=1, max_length=len(Permission), description="要移除的权限列表"),
    user: UserContext = Depends(require_permission(Permission.ADMIN_USER)),
):
    """Remove the role permissions."""
    try:
        removed, total = role_service.remove_permissions(
            role_name,
            [permission.value for permission in permissions],
            org_id=user.org_id,
        )
    except IdentityServiceError as error:
        _audit_cross_org_role_if_present(role_name, user, "remove_permissions")
        _raise_identity_error(error)
    return {"removed": removed, "total": total}


@router.get("/{role_name}/users", response_model=RoleUsersResponse)
async def list_role_users(role_name: RoleName, user: UserContext = Depends(require_permission(Permission.ADMIN_USER))):
    """List the role users."""
    try:
        role = role_service.find_role(role_name, org_id=user.org_id)
        users = role_service.list_role_users(role_name, org_id=user.org_id)
    except IdentityServiceError as error:
        _audit_cross_org_role_if_present(role_name, user, "list_users")
        _raise_identity_error(error)
    return RoleUsersResponse(role_id=role["role_id"], role_name=role_name, users=users)


# Legacy helpers retained for tests/callers that previously reached into the router.
def _find_role(role_name: RoleName) -> dict:
    """Find the role."""
    try:
        return role_service.find_role(role_name)
    except IdentityServiceError as error:
        _raise_identity_error(error)


def _count_users_with_role(role_name: RoleName) -> int:
    """Count the users with role."""
    return role_service.count_users_with_role(role_name)


def _validate_permissions(permissions: list[str]) -> None:
    """Validate the permissions."""
    try:
        role_service.validate_permissions(permissions)
    except IdentityServiceError as error:
        _raise_identity_error(error)
