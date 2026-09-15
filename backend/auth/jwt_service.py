"""认证授权 — 核心服务（JWT 签发 / 验证 / 密码哈希）"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from auth.config import AuthSettings
from domain.identity import (
    ROLE_PERMISSIONS,
    TokenPayload,
    UserContext,
    UserRole,
)


AUTHORIZATION_VERSION = 2


class AuthService:
    """无状态认证服务 — 纯函数式，不依赖外部 IO"""

    def __init__(self, settings: AuthSettings | None = None) -> None:
        """Initialize the auth service."""
        self.settings = settings or AuthSettings()
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=self.settings.bcrypt_rounds,
        )

    # ── 密码 ──────────────────────────────────────────────────

    def hash_password(self, password: str) -> str:
        """Hash the password."""
        return self.pwd_context.hash(password)

    def verify_password(self, plain: str, hashed: str) -> bool:
        """Verify the password."""
        return self.pwd_context.verify(plain, hashed)

    # ── Token 签发 ────────────────────────────────────────────

    def create_access_token(
        self,
        user_id: str,
        username: str,
        role: UserRole,
        org_id: str,
        token_version: int = 0,
        department_id: str | None = None,
        is_department_manager: bool = False,
    ) -> str:
        """Create the access token."""
        permissions = [p.value for p in ROLE_PERMISSIONS.get(role, [])]
        now = datetime.now(timezone.utc)
        payload = TokenPayload(
            sub=user_id,
            name=username,
            role=role.value,
            org_id=org_id,
            permissions=permissions,
            exp=int((now + timedelta(minutes=self.settings.access_token_expire_minutes)).timestamp()),
            iat=int(now.timestamp()),
            type="access",
            token_version=token_version,
            authorization_version=AUTHORIZATION_VERSION,
            department_id=department_id,
            is_department_manager=is_department_manager,
        )
        return self._encode(payload)

    def create_refresh_token(self, user_id: str) -> str:
        """Create the refresh token."""
        now = datetime.now(timezone.utc)
        payload = TokenPayload(
            sub=user_id,
            name="",
            role="",
            org_id="",
            permissions=[],
            exp=int((now + timedelta(days=self.settings.refresh_token_expire_days)).timestamp()),
            iat=int(now.timestamp()),
            type="refresh",
            authorization_version=AUTHORIZATION_VERSION,
        )
        return self._encode(payload)

    def _encode(self, payload: TokenPayload) -> str:
        """Encode a signed JWT with the configured algorithm."""
        data = {
            "sub": payload.sub,
            "name": payload.name,
            "role": payload.role,
            "org_id": payload.org_id,
            "permissions": payload.permissions,
            "exp": payload.exp,
            "iat": payload.iat,
            "type": payload.type,
            "token_version": payload.token_version,
            "authorization_version": payload.authorization_version,
            "department_id": payload.department_id,
            "is_department_manager": payload.is_department_manager,
        }
        return jwt.encode(data, self.settings.secret_key, algorithm=self.settings.algorithm)

    # ── Token 验证 ────────────────────────────────────────────

    def decode_token(self, token: str) -> TokenPayload | None:
        """解码并校验 JWT，失败返回 None"""
        try:
            data = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.algorithm],
            )
            return TokenPayload(
                sub=data["sub"],
                name=data.get("name", ""),
                role=data.get("role", ""),
                org_id=data.get("org_id", ""),
                permissions=data.get("permissions", []),
                exp=data["exp"],
                iat=data["iat"],
                type=data.get("type", "access"),
                token_version=data.get("token_version", 0),
                authorization_version=data.get("authorization_version", 0),
                department_id=data.get("department_id"),
                is_department_manager=bool(data.get("is_department_manager", False)),
            )
        except JWTError:
            return None

    @staticmethod
    def is_authorization_current(payload: TokenPayload) -> bool:
        """Reject tokens issued under a previous role-permission matrix."""
        return payload.authorization_version == AUTHORIZATION_VERSION

    def token_to_context(self, payload: TokenPayload) -> UserContext:
        """将 Token 载荷转换为请求上下文"""
        from domain.identity import Permission
        permissions = [Permission(p) for p in payload.permissions]
        return UserContext(
            user_id=payload.sub,
            username=payload.name,
            role=UserRole(payload.role),
            org_id=payload.org_id,
            permissions=permissions,
            token_version=payload.token_version,
            department_id=payload.department_id,
            is_department_manager=payload.is_department_manager,
        )
