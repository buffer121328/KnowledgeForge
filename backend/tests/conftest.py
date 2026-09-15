"""pytest 公共 fixtures

提供:
  - mock_llm:            模拟 LLM 响应
  - mock_vector_store:   模拟向量库
  - mock_knowledge_graph: 模拟知识图谱
  - sample_document_chunk: 示例文档块
  - test_app:            FastAPI 测试客户端
  - test_client:         httpx AsyncClient
  - auth_token:          已登录的 JWT Token
  - admin_headers:       带 admin Token 的请求头
"""

from __future__ import annotations

import asyncio
import atexit
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# 将 backend 源码根加入 sys.path，便于顶层包导入
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Settings are instantiated by test-module imports. Declare an explicit, safe
# test runtime before those imports so production startup validation is exercised
# without any implicit local-development fallback.
os.environ.setdefault("APP_ENVIRONMENT", "test")
os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-key-for-testing-only")
_runtime_database_path = Path(tempfile.gettempdir()) / f"knowledgeforge-pytest-{os.getpid()}.sqlite3"
os.environ.setdefault("DATABASE_URL", f"sqlite+pysqlite:///{_runtime_database_path}")
os.environ.setdefault(
    "WEBHOOK_SECRET_ENCRYPTION_KEY",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
)
os.environ.setdefault("WEBHOOK_SECRET_KEY_REFERENCE", "pytest-fernet-v1")

# Runtime assembly is PostgreSQL-only; tests exercise the same relational
# contracts through an isolated SQLite engine and explicit schema migration.
from sqlalchemy import create_engine
from infrastructure.postgresql_migrations import upgrade

_runtime_engine = create_engine(os.environ["DATABASE_URL"])
upgrade(_runtime_engine)
_runtime_engine.dispose()
atexit.register(lambda: _runtime_database_path.unlink(missing_ok=True))


# ── Mock fixtures ──────────────────────────────────────────────

@pytest.fixture
def mock_llm() -> AsyncMock:
    """模拟 LLM 响应"""
    mock = AsyncMock()
    mock.ainvoke.return_value = MagicMock(content="模拟的LLM响应")
    return mock


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """模拟向量库"""
    store = AsyncMock()
    store.search.return_value = [
        ({"content": "测试文档内容", "source": "test.pdf"}, 0.85),
    ]
    store.add_chunks.return_value = 5
    store.get_stats.return_value = {"total_vectors": 100}
    store.init = AsyncMock()
    return store


@pytest.fixture
def mock_knowledge_graph() -> AsyncMock:
    """模拟知识图谱"""
    kg = AsyncMock()
    kg.execute_cypher.return_value = [
        {"name": "张三", "type": "Person"},
    ]
    kg.get_stats.return_value = {"nodes": 50, "edges": 100}
    kg.init = AsyncMock()
    kg.close = AsyncMock()
    return kg


@pytest.fixture
def sample_chunk() -> dict[str, Any]:
    """示例文档块"""
    return {
        "content": "张三担任腾讯公司CEO，负责微信产品线",
        "doc_id": "test_doc_001",
        "chunk_index": 0,
        "doc_type": "pdf",
        "metadata": {"source": "test.pdf", "page": 1},
    }


# ── 认证 fixtures ──────────────────────────────────────────────

@pytest.fixture
def auth_service():
    """认证服务实例"""
    from auth.config import AuthSettings
    from auth.jwt_service import AuthService
    settings = AuthSettings(
        secret_key="test-secret-key-for-testing-only",
        bcrypt_rounds=4,  # 测试用低 rounds 加速
    )
    return AuthService(settings)


@pytest.fixture
def admin_token(auth_service) -> str:
    """admin 用户的 Access Token"""
    from domain.identity import UserRole
    return auth_service.create_access_token(
        user_id="user_001",
        username="admin",
        role=UserRole.ADMIN,
        org_id="org_001",
    )


@pytest.fixture
def viewer_token(auth_service) -> str:
    """viewer 用户的 Access Token"""
    from domain.identity import UserRole
    return auth_service.create_access_token(
        user_id="user_002",
        username="viewer",
        role=UserRole.VIEWER,
        org_id="org_001",
    )


@pytest.fixture
def admin_headers(admin_token) -> dict[str, str]:
    """带 admin Token 的请求头"""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def viewer_headers(viewer_token) -> dict[str, str]:
    """带 viewer Token 的请求头"""
    return {"Authorization": f"Bearer {viewer_token}"}


# ── FastAPI 测试客户端 ─────────────────────────────────────────

@pytest_asyncio.fixture
async def test_app():
    """构造测试用 FastAPI app（跳过重依赖初始化）"""
    from fastapi import FastAPI
    app = FastAPI(title="Test App")
    # 不初始化 vector_store / knowledge_graph，避免依赖外部服务
    yield app


@pytest_asyncio.fixture
async def test_client(test_app):
    """httpx 异步测试客户端"""
    from httpx import ASGITransport, AsyncClient
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ── 环境配置 ───────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """设置测试环境变量"""
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key-for-testing-only")
    monkeypatch.setenv("AUTH_BCRYPT_ROUNDS", "4")
    # 测试环境禁用限流，避免共享 limiter 单例导致跨测试触发限流
    from shared.utils.ratelimit import limiter
    limiter.enabled = False


@pytest.fixture(autouse=True)
def reset_logging():
    """每个测试前重置 structlog 配置

    避免跨测试污染: capsys 会替换 sys.stdout，structlog 若持有旧 stdout
    引用会在后续测试中抛出 "I/O operation on closed file"。
    """
    from shared.utils.logging import setup_logging
    setup_logging(log_level="DEBUG", json_output=True, force=True)
    yield
    # 测试结束后再次重置，确保下一个测试拿到干净的 stdout 引用
    setup_logging(log_level="DEBUG", json_output=True, force=True)


# ── pytest-asyncio 配置 ────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """会话级事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
