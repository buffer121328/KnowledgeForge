"""Redis 缓存服务的单元测试"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from infrastructure.cache.redis import CacheService, EmbeddingCache, QACache


@pytest.fixture
def mock_redis():
    """模拟 Redis 客户端"""
    redis_mock = AsyncMock()
    redis_mock.from_url = MagicMock(return_value=AsyncMock())
    return redis_mock


@pytest.fixture
def cache_service():
    """使用 mock Redis 的 CacheService"""
    # 替换 redis.from_url 返回 mock
    mock_redis = AsyncMock()
    mock_redis.from_url.return_value = mock_redis

    # 通过 monkey patch 创建
    cache = CacheService.__new__(CacheService)
    cache.redis = mock_redis
    return cache


class TestCacheService:
    @pytest.mark.asyncio
    async def test_get_existing_key(self, cache_service):
        """测试获取存在的 key"""
        cache_service.redis.get.return_value = json.dumps({"value": 42})
        result = await cache_service.get("test_key")
        assert result == {"value": 42}

    @pytest.mark.asyncio
    async def test_get_missing_key_returns_none(self, cache_service):
        """测试获取不存在的 key"""
        cache_service.redis.get.return_value = None
        result = await cache_service.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_invalid_json_returns_none(self, cache_service):
        """测试无效 JSON 返回 None 并删除 key"""
        cache_service.redis.get.return_value = "invalid json"
        result = await cache_service.get("test_key")
        assert result is None
        cache_service.redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_set_value(self, cache_service):
        """测试写入值"""
        await cache_service.set("test_key", {"data": "value"}, ttl=60)
        cache_service.redis.set.assert_called_once()
        args = cache_service.redis.set.call_args
        assert args[0][0] == "test_key"
        assert "data" in args[0][1]
        assert args[1]["ex"] == 60

    @pytest.mark.asyncio
    async def test_delete_key(self, cache_service):
        """测试删除 key"""
        await cache_service.delete("test_key")
        cache_service.redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_exists(self, cache_service):
        """测试 exists"""
        cache_service.redis.exists.return_value = 1
        assert await cache_service.exists("test_key") is True

        cache_service.redis.exists.return_value = 0
        assert await cache_service.exists("test_key") is False

    @pytest.mark.asyncio
    async def test_incr_first_call_sets_ttl(self, cache_service):
        """测试 incr 第一次调用设置 TTL"""
        cache_service.redis.incr.return_value = 1
        count = await cache_service.incr("counter", ttl=60)
        assert count == 1
        cache_service.redis.expire.assert_called_once_with("counter", 60)

    @pytest.mark.asyncio
    async def test_incr_subsequent_no_ttl(self, cache_service):
        """测试 incr 非第一次调用不设置 TTL"""
        cache_service.redis.incr.return_value = 2
        count = await cache_service.incr("counter", ttl=60)
        assert count == 2
        cache_service.redis.expire.assert_not_called()


class TestQACache:
    @pytest.mark.asyncio
    async def test_qa_cache_key_format(self):
        """测试 QA 缓存 key 格式"""
        mock_cache = AsyncMock()
        qa_cache = QACache(mock_cache, ttl=3600)
        await qa_cache.set("问题?", "user_001", {"answer": "答案"})
        args = mock_cache.set.call_args
        assert args[0][0].startswith("qa:cache:")
        assert args[1]["ttl"] == 3600

    @pytest.mark.asyncio
    async def test_qa_cache_get(self):
        """测试 QA 缓存读取"""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = {"answer": "答案"}
        qa_cache = QACache(mock_cache)
        result = await qa_cache.get("问题?", "user_001")
        assert result == {"answer": "答案"}

    @pytest.mark.asyncio
    async def test_qa_cache_same_question_different_users(self):
        """测试相同问题不同用户使用不同 key"""
        mock_cache = AsyncMock()
        qa_cache = QACache(mock_cache)
        await qa_cache.set("问题?", "user_001", {"answer": "1"})
        await qa_cache.set("问题?", "user_002", {"answer": "2"})
        keys = [call[0][0] for call in mock_cache.set.call_args_list]
        assert keys[0] != keys[1]


class TestEmbeddingCache:
    @pytest.mark.asyncio
    async def test_embedding_cache_key_uses_model(self):
        """测试 embedding 缓存 key 包含 model"""
        mock_cache = AsyncMock()
        emb_cache = EmbeddingCache(mock_cache)
        await emb_cache.set("text", [0.1, 0.2], model="gpt-4")
        await emb_cache.set("text", [0.3, 0.4], model="gpt-3.5")
        keys = [call[0][0] for call in mock_cache.set.call_args_list]
        assert keys[0] != keys[1]
        assert keys[0].startswith("embedding:")
