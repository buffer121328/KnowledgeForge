"""Role business rules independent of FastAPI request/response models."""

from __future__ import annotations

import secrets
from typing import Any

from auth.identity_contracts import IdentityStore
from auth.identity_provider import get_identity_store
from auth.memory_identity_store import MemoryIdentityStore
from auth.memory_accounts import ROLE_DB, USER_DB
from auth.user_service import IdentityServiceError
from domain.identity import Permission, UserRole


class RoleService:
    """Custom-role operations over an explicit identity store."""

    def __init__(
        self,
        roles: dict[str, dict[str, Any]] | None = None,
        users: dict[str, dict[str, Any]] | None = None,
        *,
        store: IdentityStore | None = None,
    ) -> None:
        """Initialize the role service."""
        if store is not None:
            self.store = store
        elif roles is not None or users is not None:
            self.store = MemoryIdentityStore(
                users if users is not None else USER_DB,
                roles if roles is not None else ROLE_DB,
            )
        else:
            self.store = get_identity_store()
        self.roles = getattr(self.store, "roles", None)
        self.users = getattr(self.store, "users", None)

    @staticmethod
    def _storage_key(role_name: str, org_id: str | None) -> str:
        """Build the storage key for a tenant-scoped role."""
        return f"{org_id}::{role_name}" if org_id else role_name

    def find_role(
        self,
        role_name: str,
        *,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Find the role."""
        role = self.store.find_role(role_name, org_id)
        if not role:
            raise IdentityServiceError(404, f"角色 '{role_name}' 不存在")
        return role

    def count_users_with_role(
        self,
        role_name: str,
        *,
        org_id: str | None = None,
    ) -> int:
        """Count the users with role."""
        return sum(
            1
            for user in self.store.list_users()
            if (user["role"].value if isinstance(user["role"], UserRole) else user["role"]) == role_name
            and (org_id is None or user.get("org_id") == org_id)
        )

    def list_roles(
        self,
        *,
        search: str | None = None,
        is_builtin: bool | None = None,
        org_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List the roles."""
        roles = [
            role
            for role in self.store.list_roles()
            if role.get("is_builtin")
            or (org_id is None and not role.get("org_id"))
            or role.get("org_id") == org_id
        ]
        if is_builtin is not None:
            roles = [role for role in roles if role["is_builtin"] == is_builtin]
        if search:
            query = search.lower()
            roles = [
                role
                for role in roles
                if query in role["name"].lower()
                or query in role["display_name"].lower()
                or query in role.get("description", "").lower()
            ]
        return roles

    def create_role(
        self,
        *,
        name: str,
        display_name: str,
        description: str,
        permissions: list[str],
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Create the role."""
        existing = self.store.find_role(name, org_id)
        if existing:
            raise IdentityServiceError(409, "角色名已存在")
        try:
            UserRole(name)
        except ValueError:
            pass
        else:
            raise IdentityServiceError(400, "角色名不可与内置角色重名")
        self.validate_permissions(permissions)
        role = {
            "role_id": f"role_{secrets.token_hex(4)}",
            "name": name,
            "display_name": display_name or name,
            "description": description,
            "permissions": permissions,
            "is_builtin": False,
            "user_count": 0,
            "org_id": org_id or "",
        }
        if not self.store.create_role(role):
            raise IdentityServiceError(409, "角色名已存在")
        return role

    def update_role(
        self,
        role_name: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        permissions: list[str] | None = None,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Update the role."""
        role = self.find_role(role_name, org_id=org_id)
        if role["is_builtin"]:
            raise IdentityServiceError(400, "内置角色不可修改，请使用自定义角色")
        if display_name is not None:
            role["display_name"] = display_name
        if description is not None:
            role["description"] = description
        if permissions is not None:
            self.validate_permissions(permissions)
            role["permissions"] = permissions
        self.store.save_role(role)
        return role

    def delete_role(
        self,
        role_name: str,
        *,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete the role."""
        role = self.find_role(role_name, org_id=org_id)
        if role["is_builtin"]:
            raise IdentityServiceError(400, "内置角色不可删除")
        user_count = self.count_users_with_role(role_name, org_id=org_id)
        if user_count > 0:
            raise IdentityServiceError(400, f"该角色下还有 {user_count} 个用户，请先迁移用户")
        self.store.delete_role(role_name, org_id)
        return role

    def get_permissions(self, role_name: str, *, org_id: str | None = None) -> list[str]:
        """Return the permissions."""
        return self.find_role(role_name, org_id=org_id)["permissions"]

    def set_permissions(
        self,
        role_name: str,
        permissions: list[str],
        *,
        org_id: str | None = None,
    ) -> list[str]:
        """Store the permissions."""
        role = self.find_role(role_name, org_id=org_id)
        if role["is_builtin"]:
            raise IdentityServiceError(400, "内置角色权限不可修改")
        self.validate_permissions(permissions)
        role["permissions"] = permissions
        self.store.save_role(role)
        return permissions

    def add_permissions(
        self,
        role_name: str,
        permissions: list[str],
        *,
        org_id: str | None = None,
    ) -> tuple[int, int]:
        """Add the permissions."""
        role = self.find_role(role_name, org_id=org_id)
        if role["is_builtin"]:
            raise IdentityServiceError(400, "内置角色权限不可修改")
        self.validate_permissions(permissions)
        existing = set(role["permissions"])
        added = [permission for permission in permissions if permission not in existing]
        role["permissions"].extend(added)
        self.store.save_role(role)
        return len(added), len(role["permissions"])

    def remove_permissions(
        self,
        role_name: str,
        permissions: list[str],
        *,
        org_id: str | None = None,
    ) -> tuple[int, int]:
        """Remove the permissions."""
        role = self.find_role(role_name, org_id=org_id)
        if role["is_builtin"]:
            raise IdentityServiceError(400, "内置角色权限不可修改")
        before = len(role["permissions"])
        role["permissions"] = [permission for permission in role["permissions"] if permission not in permissions]
        self.store.save_role(role)
        return before - len(role["permissions"]), len(role["permissions"])

    def list_role_users(
        self,
        role_name: str,
        *,
        org_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List the role users."""
        self.find_role(role_name, org_id=org_id)
        return [
            {
                "user_id": user["user_id"],
                "username": user["username"],
                "display_name": user.get("display_name", ""),
                "org_id": user["org_id"],
                "is_active": user.get("is_active", True),
            }
            for user in self.store.list_users()
            if (user["role"].value if isinstance(user["role"], UserRole) else user["role"]) == role_name
            and (org_id is None or user.get("org_id") == org_id)
        ]

    @staticmethod
    def validate_permissions(permissions: list[str]) -> None:
        """Validate the permissions."""
        valid = {permission.value for permission in Permission}
        invalid = [permission for permission in permissions if permission not in valid]
        if invalid:
            raise IdentityServiceError(
                400,
                f"无效的权限标识: {', '.join(invalid)}。可用权限: {', '.join(sorted(valid))}",
            )
