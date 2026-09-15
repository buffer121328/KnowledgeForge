"""Role and user permission resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.identity import Permission, ROLE_PERMISSIONS, UserRole

if TYPE_CHECKING:
    from domain.identity import UserContext

# ── 权限解析服务 ──────────────────────────────────────────────

class PermissionResolver:
    """
    动态权限解析器 — 支持内置角色 + 自定义角色。
    从 ROLE_DB 读取自定义角色的权限配置。
    """

    def __init__(self):
        """Initialize the permission resolver."""
        self._cache: dict[str, list[Permission]] = {}
        self._cache_version: int = 0

    def get_permissions_for_role(
        self,
        role_name: str,
        *,
        org_id: str | None = None,
    ) -> list[Permission]:
        """获取角色的所有权限（内置 + 自定义）"""
        # 尝试内置角色
        try:
            role = UserRole(role_name)
            return ROLE_PERMISSIONS.get(role, [])
        except ValueError:
            pass

        from auth.identity_provider import get_identity_store

        role_data = get_identity_store().find_role(role_name, org_id)
        if not role_data:
            return []

        return [Permission(p) for p in role_data.get("permissions", [])]

    def resolve_user_permissions(self, user: UserContext) -> list[Permission]:
        """
        解析用户的完整权限列表。
        优先使用 UserContext 中已有的权限（JWT 中携带），
        如果为空则动态解析。
        """
        # API Keys always use their explicit scope list. An intentionally empty
        # list must not fall back to the owning user's or creator's role.
        if user.api_key_id is not None:
            return user.permissions
        if user.permissions:
            return user.permissions

        # 动态解析
        role_name = (
            user.role.value if hasattr(user.role, "value") else str(user.role)
        )
        return self.get_permissions_for_role(
            role_name,
            org_id=user.org_id,
        )

    def has_permission(
        self,
        user: UserContext,
        required: list[Permission],
        require_all: bool = False,
    ) -> bool:
        """
        检查用户是否拥有指定权限。

        Args:
            user: 用户上下文
            required: 需要的权限列表
            require_all: True=需要全部权限(AND)，False=任一即可(OR)
        """
        user_perms = set(self.resolve_user_permissions(user))
        required_set = set(required)

        if require_all:
            return required_set.issubset(user_perms)
        else:
            return bool(required_set & user_perms)


# 全局权限解析器实例
permission_resolver = PermissionResolver()
