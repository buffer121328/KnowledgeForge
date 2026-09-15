"""认证授权 — FastAPI 依赖注入"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from domain.identity import Permission, UserContext, UserRole

security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserContext:
    """
    获取当前登录用户。
    优先从 middleware 注入的 request.state.user 获取；
    若 middleware 未注入，则自行解析 Bearer Token。
    """
    # middleware 已注入
    user = getattr(request.state, "user", None)
    if user is not None:
        return user

    if not credentials:
        raise HTTPException(status_code=401, detail="未认证")

    # fallback: 手动解析（middleware 未加载时的兜底）
    from auth.jwt_service import AuthService
    from auth.user_service import UserService
    auth_service = AuthService()
    payload = auth_service.decode_token(credentials.credentials)
    if (
        not payload
        or payload.type != "access"
        or not auth_service.is_authorization_current(payload)
    ):
        raise HTTPException(status_code=401, detail="Token 无效或已过期")
    record = UserService().find_by_id(payload.sub, raise_on_missing=False)
    if (
        not record
        or not record.get("is_active", True)
        or payload.token_version != record.get("token_version", 0)
        or not record.get("org_id")
        or payload.org_id != record.get("org_id")
    ):
        raise HTTPException(status_code=401, detail="Token 无效或已过期")
    return auth_service.token_to_context(payload)


def require_permission(permission: Permission):
    """权限检查依赖 — 用法: Depends(require_permission(Permission.DOC_WRITE))"""

    async def _check(user: UserContext = Depends(get_current_user)):
        """Return the check."""
        if permission not in user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"权限不足: 需要 {permission.value}",
            )
        return user

    return _check


def require_role(*roles: UserRole):
    """角色检查依赖 — 用法: Depends(require_role(UserRole.ADMIN))"""

    async def _check(user: UserContext = Depends(get_current_user)):
        """Return the check."""
        if user.role not in roles:
            role_names = [r.value for r in roles]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"权限不足: 需要角色 {role_names}",
            )
        return user

    return _check
