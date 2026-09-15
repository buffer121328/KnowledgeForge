"""API Key 管理服务的单元测试"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from auth.apikey_service import APIKeyLifecycleError, APIKeyService
from auth.apikey_store import MemoryKeyStore
from domain.identity import Permission, UserRole


@pytest.fixture
def service() -> APIKeyService:
    return APIKeyService(store=MemoryKeyStore())


class TestAPIKeyGeneration:
    def test_generate_key_format(self, service: APIKeyService):
        """测试 Key 格式: ak_ + 随机字符"""
        full, prefix, hash_ = service.generate_key()
        assert full.startswith("ak_")
        assert len(full) > 40
        assert "..." in prefix
        assert prefix.startswith("ak_")
        assert len(hash_) == 64  # SHA-256 hex

    def test_generate_key_uniqueness(self, service: APIKeyService):
        """测试每次生成的 Key 唯一"""
        k1, _, h1 = service.generate_key()
        k2, _, h2 = service.generate_key()
        assert k1 != k2
        assert h1 != h2

    def test_hash_key_deterministic(self, service: APIKeyService):
        """测试哈希函数确定性"""
        key = "ak_test_key"
        h1 = service.hash_key(key)
        h2 = service.hash_key(key)
        assert h1 == h2


class TestAPIKeyCreate:
    def test_create_key_returns_full_key_once(self, service: APIKeyService):
        """测试创建 Key 时返回完整 Key"""
        api_key, full_key = service.create_key(
            name="test-key",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
            scopes=[Permission.DOC_READ, Permission.QA_QUERY],
            actor_permissions=[Permission.DOC_READ, Permission.QA_QUERY],
        )
        assert api_key.id.startswith("key_")
        assert api_key.name == "test-key"
        assert api_key.user_id == "user_001"
        assert api_key.org_id == "org_001"
        assert api_key.role == UserRole.API_USER
        assert Permission.DOC_READ in api_key.scopes
        assert Permission.QA_QUERY in api_key.scopes
        assert full_key.startswith("ak_")
        assert api_key.key_hash == service.hash_key(full_key)

    def test_create_key_with_expiry(self, service: APIKeyService):
        """测试带过期时间的 Key"""
        _, _ = service.create_key(
            name="temp",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
            expires_days=7,
        )
        # expires_at 应该是 ISO 格式
        # 直接查询 store
        keys = service.list_keys("user_001")
        assert len(keys) == 1
        assert keys[0].expires_at is not None

    def test_create_key_without_explicit_expiry_uses_default(self, service: APIKeyService):
        """未显式指定时使用受控默认有效期"""
        service.create_key(
            name="permanent",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
        )
        keys = service.list_keys("user_001")
        assert keys[0].expires_at is not None

    def test_create_key_normalizes_string_scopes(self, service: APIKeyService):
        """测试 scopes 字符串归一化"""
        api_key, _ = service.create_key(
            name="test",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
            scopes=["doc:read", "qa:query"],
            actor_permissions=[Permission.DOC_READ, Permission.QA_QUERY],
        )
        assert Permission.DOC_READ in api_key.scopes
        assert Permission.QA_QUERY in api_key.scopes

    def test_create_key_rejects_invalid_scope(self, service: APIKeyService):
        """任一无效 scope 会拒绝整个请求"""
        with pytest.raises(APIKeyLifecycleError) as error:
            service.create_key(
                name="test",
                user_id="user_001",
                org_id="org_001",
                role=UserRole.API_USER,
                scopes=["doc:read", "invalid:scope"],
                actor_permissions=[Permission.DOC_READ],
            )
        assert error.value.code == "invalid_api_key_scope"
        assert service.list_keys("user_001") == []


class TestAPIKeyVerify:
    def test_verify_valid_key(self, service: APIKeyService):
        """测试验证有效 Key"""
        _, full_key = service.create_key(
            name="test",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
        )
        verified = service.verify(full_key)
        assert verified is not None
        assert verified.user_id == "user_001"

    def test_verify_invalid_key_format(self, service: APIKeyService):
        """测试格式错误的 Key"""
        assert service.verify("invalid_key") is None
        assert service.verify("") is None

    def test_verify_nonexistent_key(self, service: APIKeyService):
        """测试不存在的 Key"""
        assert service.verify("ak_nonexistent_key_value_12345") is None

    def test_verify_expired_key(self, service: APIKeyService):
        """测试过期 Key"""
        api_key, full_key = service.create_key(
            name="expired",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
            expires_days=1,
        )
        # 手动将过期时间设为过去
        api_key.expires_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service.store.save(api_key)

        assert service.verify(full_key) is None

    def test_verify_updates_last_used(self, service: APIKeyService):
        """测试验证成功后更新 last_used_at"""
        _, full_key = service.create_key(
            name="test",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
        )
        assert service.verify(full_key) is not None
        keys = service.list_keys("user_001")
        assert keys[0].last_used_at is not None


class TestAPIKeyRevoke:
    def test_revoke_own_key(self, service: APIKeyService):
        """测试撤销自己的 Key"""
        api_key, _ = service.create_key(
            name="test",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
        )
        revoked = service.revoke(api_key.id, "user_001")
        assert revoked is not None
        assert revoked.revoked_at is not None
        assert service.list_keys("user_001") == [revoked]

    def test_revoke_others_key_fails(self, service: APIKeyService):
        """测试不能撤销别人的 Key"""
        api_key, _ = service.create_key(
            name="test",
            user_id="user_001",
            org_id="org_001",
            role=UserRole.API_USER,
        )
        assert service.revoke(api_key.id, "user_002") is None

    def test_revoke_nonexistent_key(self, service: APIKeyService):
        """测试撤销不存在的 Key"""
        assert service.revoke("key_nonexistent", "user_001") is None
