"""认证授权模块 — 单元测试"""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from auth.config import AuthSettings
from domain.identity import (
    ROLE_PERMISSIONS,
    APIKey,
    Permission,
    UserContext,
    UserRole,
)
from auth.jwt_service import AuthService


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def settings():
    return AuthSettings(
        secret_key="test-secret-key-for-unit-tests",
        algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7,
    )


@pytest.fixture
def auth_service(settings):
    return AuthService(settings)


# ── 密码哈希测试 ──────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_password(self, auth_service):
        hashed = auth_service.hash_password("test_password")
        assert hashed != "test_password"
        assert hashed.startswith("$2")

    def test_verify_password_correct(self, auth_service):
        hashed = auth_service.hash_password("my_secret")
        assert auth_service.verify_password("my_secret", hashed) is True

    def test_verify_password_wrong(self, auth_service):
        hashed = auth_service.hash_password("my_secret")
        assert auth_service.verify_password("wrong_password", hashed) is False

    def test_different_hashes_for_same_password(self, auth_service):
        h1 = auth_service.hash_password("same_password")
        h2 = auth_service.hash_password("same_password")
        # bcrypt 每次生成不同的盐
        assert h1 != h2


# ── JWT Token 签发与验证 ─────────────────────────────────────

class TestJWTToken:
    def test_create_access_token(self, auth_service):
        token = auth_service.create_access_token(
            user_id="user_001",
            username="testuser",
            role=UserRole.ADMIN,
            org_id="org_001",
        )
        assert isinstance(token, str)
        assert len(token) > 50

    def test_decode_access_token(self, auth_service):
        token = auth_service.create_access_token(
            user_id="user_001",
            username="testuser",
            role=UserRole.EDITOR,
            org_id="org_001",
        )
        payload = auth_service.decode_token(token)
        assert payload is not None
        assert payload.sub == "user_001"
        assert payload.name == "testuser"
        assert payload.role == "editor"
        assert payload.org_id == "org_001"
        assert payload.type == "access"

    def test_create_refresh_token(self, auth_service):
        token = auth_service.create_refresh_token(user_id="user_001")
        payload = auth_service.decode_token(token)
        assert payload is not None
        assert payload.sub == "user_001"
        assert payload.type == "refresh"

    def test_decode_invalid_token(self, auth_service):
        payload = auth_service.decode_token("invalid.token.here")
        assert payload is None

    def test_decode_tampered_token(self, auth_service):
        token = auth_service.create_access_token(
            user_id="user_001",
            username="testuser",
            role=UserRole.VIEWER,
            org_id="org_001",
        )
        # 篡改 Token
        tampered = token[:-5] + "XXXXX"
        payload = auth_service.decode_token(tampered)
        assert payload is None

    def test_token_to_context(self, auth_service):
        token = auth_service.create_access_token(
            user_id="user_001",
            username="testuser",
            role=UserRole.ADMIN,
            org_id="org_001",
            department_id="finance",
            is_department_manager=True,
        )
        payload = auth_service.decode_token(token)
        ctx = auth_service.token_to_context(payload)
        assert isinstance(ctx, UserContext)
        assert ctx.user_id == "user_001"
        assert ctx.username == "testuser"
        assert ctx.role == UserRole.ADMIN
        assert Permission.ADMIN_MANAGE not in ctx.permissions
        assert ctx.department_id == "finance"
        assert ctx.is_department_manager is True

    def test_token_version_in_payload(self, auth_service):
        """token_version 被嵌入 JWT 并可在解码后读取"""
        token = auth_service.create_access_token(
            user_id="user_001",
            username="testuser",
            role=UserRole.ADMIN,
            org_id="org_001",
            token_version=5,
        )
        payload = auth_service.decode_token(token)
        assert payload is not None
        assert payload.token_version == 5

    def test_token_version_default_zero(self, auth_service):
        """未指定 token_version 时默认为 0"""
        token = auth_service.create_access_token(
            user_id="user_001",
            username="testuser",
            role=UserRole.VIEWER,
            org_id="org_001",
        )
        payload = auth_service.decode_token(token)
        assert payload.token_version == 0

    def test_token_version_propagates_to_context(self, auth_service):
        """token_version 从 Token 传递到 UserContext"""
        token = auth_service.create_access_token(
            user_id="user_001",
            username="testuser",
            role=UserRole.ADMIN,
            org_id="org_001",
            token_version=3,
        )
        payload = auth_service.decode_token(token)
        ctx = auth_service.token_to_context(payload)
        assert ctx.token_version == 3


# ── 角色与权限映射 ────────────────────────────────────────────

class TestRolePermissions:
    def test_organization_admin_has_all_permissions(self):
        admin_perms = ROLE_PERMISSIONS[UserRole.ORGANIZATION_ADMIN]
        for perm in Permission:
            assert perm in admin_perms

    def test_viewer_has_read_only_permissions(self):
        viewer_perms = ROLE_PERMISSIONS[UserRole.VIEWER]
        assert Permission.DOC_READ in viewer_perms
        assert Permission.QA_QUERY in viewer_perms
        assert Permission.DOC_WRITE not in viewer_perms
        assert Permission.ADMIN_MANAGE not in viewer_perms

    def test_editor_is_read_only(self):
        editor_perms = ROLE_PERMISSIONS[UserRole.EDITOR]
        assert Permission.DOC_WRITE not in editor_perms
        assert Permission.DOC_READ in editor_perms
        assert Permission.ADMIN_MANAGE not in editor_perms

    def test_api_user_has_no_default_permissions(self):
        api_user_perms = ROLE_PERMISSIONS[UserRole.API_USER]
        assert len(api_user_perms) == 0


# ── UserContext 测试 ──────────────────────────────────────────

class TestUserContext:
    def test_create_user_context(self):
        ctx = UserContext(
            user_id="u1",
            username="test",
            role=UserRole.VIEWER,
            org_id="org1",
        )
        assert ctx.user_id == "u1"
        assert ctx.permissions == []
        assert ctx.api_key_id is None
        assert ctx.token_version == 0

    def test_user_context_with_api_key(self):
        ctx = UserContext(
            user_id="u1",
            username="test",
            role=UserRole.API_USER,
            org_id="org1",
            permissions=[Permission.DOC_READ],
            api_key_id="key_001",
        )
        assert ctx.api_key_id == "key_001"
        assert Permission.DOC_READ in ctx.permissions

    def test_user_context_with_token_version(self):
        ctx = UserContext(
            user_id="u1",
            username="test",
            role=UserRole.ADMIN,
            org_id="org1",
            token_version=7,
        )
        assert ctx.token_version == 7


# ── API Key 测试 ──────────────────────────────────────────────

class TestAPIKey:
    def test_api_key_hash_is_deterministic(self):
        raw_key = "ak_test123456789"
        h1 = hashlib.sha256(raw_key.encode()).hexdigest()
        h2 = hashlib.sha256(raw_key.encode()).hexdigest()
        assert h1 == h2

    def test_api_key_model_creation(self):
        key = APIKey(
            id="key_001",
            key_prefix="ak_tes...8901",
            key_hash=hashlib.sha256("ak_test123456789012345678901234567890".encode()).hexdigest(),
            name="test-key",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
            scopes=[Permission.DOC_READ, Permission.QA_QUERY],
            is_active=True,
        )
        assert key.id == "key_001"
        assert key.is_active is True
        assert Permission.DOC_READ in key.scopes

    def test_api_key_expires_check(self):
        # 已过期的 Key
        expired_key = APIKey(
            id="key_expired",
            key_prefix="ak_exp...0001",
            key_hash="hash",
            name="expired-key",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
            expires_at=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        )
        assert datetime.fromisoformat(expired_key.expires_at) < datetime.now(timezone.utc)

        # 未过期的 Key
        valid_key = APIKey(
            id="key_valid",
            key_prefix="ak_val...0002",
            key_hash="hash",
            name="valid-key",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
            expires_at=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        )
        assert datetime.fromisoformat(valid_key.expires_at) > datetime.now(timezone.utc)

    def test_api_key_no_expiry(self):
        key = APIKey(
            id="key_forever",
            key_prefix="ak_for...0003",
            key_hash="hash",
            name="forever-key",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
            expires_at=None,
        )
        assert key.expires_at is None  # 永不过期
