"""认证授权 — 共享存储（内存实现，生产环境替换为数据库）"""

from __future__ import annotations

from domain.identity import UserRole, builtin_role_records
from auth.jwt_service import AuthService

_auth_service = AuthService()

# ── 用户表 ────────────────────────────────────────────────────
# token_version: 每次角色变更 +1，中间件用于检测 JWT 是否已过期
# is_active: 是否启用，False 时禁止登录

USER_DB: dict[str, dict] = {
    "admin": {
        "user_id": "user_001",
        "username": "admin",
        "password_hash": _auth_service.hash_password("admin123"),
        "role": UserRole.ORGANIZATION_ADMIN,
        "org_id": "org_001",
        "display_name": "组织管理员",
        "email": "admin@example.com",
        "is_active": True,
        "token_version": 0,
    },
}

# ── 角色表 ────────────────────────────────────────────────────
# 内置角色为只读，自定义角色可增删改

ROLE_DB: dict[str, dict] = builtin_role_records()

# 角色名称 → UserRole 枚举的映射缓存（含自定义角色）
_ROLE_NAME_MAP: dict[str, UserRole] = {r.value: r for r in UserRole}


def increment_token_version(user_id: str) -> int:
    """递增用户 token_version，使所有旧 JWT 失效。返回新版本号。"""
    for u in USER_DB.values():
        if u["user_id"] == user_id:
            u["token_version"] = u.get("token_version", 0) + 1
            return u["token_version"]
    return 0


def get_token_version(user_id: str) -> int:
    """获取用户当前 token_version"""
    for u in USER_DB.values():
        if u["user_id"] == user_id:
            return u.get("token_version", 0)
    return 0
