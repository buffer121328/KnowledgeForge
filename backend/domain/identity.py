"""Authentication and authorization domain models."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "APIKey",
    "Permission",
    "ROLE_PERMISSIONS",
    "TokenPayload",
    "UserContext",
    "UserRole",
    "builtin_role_records",
]


class UserRole(str, Enum):
    """Represent user role."""
    ADMIN = "admin"
    ORGANIZATION_ADMIN = "organization_admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    API_USER = "api_user"


class Permission(str, Enum):
    """Represent permission."""
    DOC_READ = "doc:read"
    DOC_WRITE = "doc:write"
    DOC_DELETE = "doc:delete"
    DOC_EXPORT = "doc:export"
    QA_QUERY = "qa:query"
    QA_HISTORY = "qa:history"
    QA_FEEDBACK = "qa:feedback"
    GRAPH_READ = "graph:read"
    GRAPH_EDIT = "graph:edit"
    ADMIN_MANAGE = "admin:manage"
    ADMIN_AUDIT = "admin:audit"
    ADMIN_USER = "admin:user"


ROLE_PERMISSIONS: dict[UserRole, list[Permission]] = {
    UserRole.ORGANIZATION_ADMIN: list(Permission),
    UserRole.ADMIN: [Permission.DOC_READ, Permission.DOC_WRITE, Permission.DOC_DELETE, Permission.DOC_EXPORT, Permission.QA_QUERY, Permission.QA_HISTORY, Permission.QA_FEEDBACK, Permission.GRAPH_READ, Permission.GRAPH_EDIT],
    UserRole.EDITOR: [Permission.DOC_READ, Permission.QA_QUERY, Permission.QA_HISTORY, Permission.GRAPH_READ],
    UserRole.VIEWER: [Permission.DOC_READ, Permission.QA_QUERY, Permission.QA_HISTORY, Permission.GRAPH_READ],
    UserRole.API_USER: [],
}


def builtin_role_records() -> dict[str, dict]:
    """Return independent canonical records for all immutable built-in roles."""

    records = {
        role.value: {
            "role_id": f"role_{role.value}",
            "name": role.value,
            "display_name": {
                "organization_admin": "公司管理员",
                "admin": "部门负责人",
                "editor": "员工",
                "viewer": "员工",
                "api_user": "接口账号",
            }.get(role.value, role.value),
            "description": {
                "organization_admin": "组织级治理权限角色",
                "admin": "部门负责人系统权限角色",
                "editor": "员工系统权限角色",
                "viewer": "员工系统权限角色",
                "api_user": "接口账号系统权限角色",
            }.get(role.value, f"内置{role.value}角色"),
            "permissions": [permission.value for permission in permissions],
            "is_builtin": True,
            "user_count": 0,
        }
        for role, permissions in ROLE_PERMISSIONS.items()
    }
    return copy.deepcopy(records)


@dataclass
class UserContext:
    """Represent user context."""
    user_id: str
    username: str
    role: UserRole
    org_id: str
    permissions: list[Permission] = field(default_factory=list)
    department_id: str | None = None
    is_department_manager: bool = False
    api_key_id: str | None = None
    token_version: int = 0


@dataclass
class TokenPayload:
    """Represent token payload."""
    sub: str
    name: str
    role: str
    org_id: str
    permissions: list[str]
    exp: int
    iat: int
    type: str = "access"
    token_version: int = 0
    authorization_version: int = 0
    department_id: str | None = None
    is_department_manager: bool = False


@dataclass
class APIKey:
    """Represent API key."""
    id: str
    key_prefix: str
    key_hash: str
    name: str
    user_id: str
    org_id: str
    role: UserRole
    scopes: list[Permission] = field(default_factory=list)
    expires_at: str | None = None
    created_at: str = ""
    last_used_at: str | None = None
    is_active: bool = True
    disabled_at: str | None = None
    rotated_at: str | None = None
    rotation_count: int = 0
    revoked_at: str | None = None
    revoked_by: str | None = None
