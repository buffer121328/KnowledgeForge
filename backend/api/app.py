"""FastAPI application assembly and lifecycle wiring."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from limits.errors import StorageError
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from api.contracts import sanitize_validation_errors
from api.middleware.auth import JWTAuthMiddleware
from api.middleware.permission import PermissionMiddleware
from api.middleware.permission_registry import init_route_permissions
from api.middleware.request_trend import RequestTrendMiddleware
from api.middleware.tenant import TenantMiddleware
from api.routers import build_v1_router
from api.routers.admin import health_router
from auth.config import auth_settings
from infrastructure.retrieval.bm25_index import NativeBM25Index
from infrastructure.documents.catalog import PostgreSQLDocumentCatalogRepository
from infrastructure.graph.neo4j_graph import KnowledgeGraphService
from infrastructure.postgres.database import get_database_service
from infrastructure.readiness import ReadinessService
from infrastructure.documents.qa_history import PostgreSQLQAHistoryRepository
from infrastructure.cache.redis import KnowledgeRevisionStore, QACache, get_cache_service
from infrastructure.cache.semantic import (
    DisabledSemanticIndex,
    SemanticConfirmationService,
    SemanticQACache,
)
from infrastructure.retrieval.vector_store import VectorStoreService
from infrastructure.webhooks.service import get_webhook_service
from shared.config import settings
from infrastructure.security.security_state import SecurityStateUnavailableError
from shared.utils.logging import get_logger, setup_logging
from shared.utils.ratelimit import (
    limiter,
    rate_limit_exceeded_handler,
    rate_limit_storage_error_handler,
)
from workflows.workflow_factory import build_workflow_registry


setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    service_name=os.getenv("SERVICE_NAME", "agenthub-api"),
)
logger = get_logger(__name__)


async def request_validation_error_handler(
    _request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    """Return FastAPI-compatible 422 details without echoing secrets or payloads."""

    return JSONResponse(
        status_code=422,
        content={"detail": sanitize_validation_errors(error.errors())},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and release services used by the API routes."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    init_route_permissions()

    vector_store = app.state.vector_store
    knowledge_graph = app.state.knowledge_graph
    workflows: dict[str, Any] = app.state.workflows

    try:
        await vector_store.init()
    except Exception as e:
        logger.warning("vector_store_init_failed", error_type=type(e).__name__)
    try:
        await knowledge_graph.init()
    except Exception as e:
        logger.warning("knowledge_graph_init_failed", error_type=type(e).__name__)
    workflows.update(
        build_workflow_registry(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            sparse_index=app.state.sparse_index,
            qa_cache=app.state.qa_cache,
            revision_store=app.state.knowledge_revision,
            semantic_cache=app.state.semantic_qa_cache,
            confirmation_service=app.state.semantic_confirmation,
        )
    )
    logger.info("app_started")
    yield
    # 资源清理：按相反顺序关闭
    try:
        await app.state.webhook_service.close()
    except Exception as e:
        logger.warning("webhook_close_failed", error_type=type(e).__name__)
    await knowledge_graph.close()
    await app.state.cache_service.close()
    if app.state.database is not None:
        app.state.database.close()
    logger.info("app_stopped")


def _mount_business_routers(app: FastAPI) -> None:
    """Mount all completed child routers after their endpoints are registered."""
    app.include_router(build_v1_router())

def create_app() -> FastAPI:
    """Build the versioned FastAPI application instance."""
    auth_settings.validate_for_runtime(settings.app_environment)
    settings.validate_security_state_for_runtime(settings.app_environment)
    settings.validate_database_for_runtime(settings.app_environment)
    app = FastAPI(
        title="AgentKnowledgeHub - 多Agent企业知识管理系统",
        description="支持多模态RAG、知识图谱、增量更新、企业级安全的企业知识管理 API",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_exception_handler(RequestValidationError, request_validation_error_handler)

    @app.exception_handler(SecurityStateUnavailableError)
    async def security_state_unavailable_handler(_request, _error):
        """Handle security state unavailable handler for the module."""
        request_id = uuid4().hex
        return JSONResponse(
            status_code=503,
            content={
                "detail": {
                    "code": "security_state_unavailable",
                    "message": "安全状态服务暂不可用，请稍后重试。",
                    "request_id": request_id,
                }
            },
            headers={"X-Request-ID": request_id},
        )
    app.state.vector_store = VectorStoreService()
    app.state.knowledge_graph = KnowledgeGraphService()
    app.state.sparse_index = NativeBM25Index(
        settings.qa_bm25_index_path,
        k1=settings.qa_bm25_k1,
        b=settings.qa_bm25_b,
        schema_version=settings.qa_bm25_schema_version,
        tokenizer_version=settings.qa_bm25_tokenizer_version,
    )
    app.state.database = get_database_service()
    app.state.webhook_service = get_webhook_service()
    app.state.qa_history = PostgreSQLQAHistoryRepository(app.state.database)
    app.state.cache_service = get_cache_service()
    app.state.qa_cache = QACache(app.state.cache_service, ttl=settings.qa_cache_ttl_seconds)
    app.state.knowledge_revision = KnowledgeRevisionStore(app.state.cache_service)
    app.state.semantic_qa_cache = SemanticQACache(
        DisabledSemanticIndex(),
        mode=settings.qa_semantic_cache_mode,
        similarity_threshold=settings.qa_semantic_similarity_threshold,
        candidate_limit=settings.qa_semantic_candidate_limit,
    )
    app.state.semantic_confirmation = SemanticConfirmationService(
        app.state.cache_service,
        ttl_seconds=settings.qa_semantic_confirmation_ttl_seconds,
    )
    app.state.document_catalog = PostgreSQLDocumentCatalogRepository(app.state.database)
    app.state.workflows = {}
    app.state.readiness = ReadinessService(
        vector_store=app.state.vector_store,
        knowledge_graph=app.state.knowledge_graph,
        security_state=app.state.database,
        database=app.state.database,
        timeout_seconds=settings.readiness_probe_timeout_seconds,
    )

    # 限流
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_exception_handler(StorageError, rate_limit_storage_error_handler)
    app.add_exception_handler(RedisConnectionError, rate_limit_storage_error_handler)
    app.add_exception_handler(RedisTimeoutError, rate_limit_storage_error_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Prometheus 自动指标采集
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/metrics", "/api/health", "/api/health/live", "/api/health/ready"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    # 中间件（执行顺序: 逆序注册 -> JWT 先执行 -> Permission -> Tenant -> Trend）
    app.add_middleware(TenantMiddleware)
    app.add_middleware(PermissionMiddleware, settings=auth_settings)
    app.add_middleware(RequestTrendMiddleware)
    app.add_middleware(JWTAuthMiddleware, settings=auth_settings)

    # 健康检查保留独立路径；业务 API 仅装配到 /api/v1。
    app.include_router(health_router)
    _mount_business_routers(app)
    return app


app = create_app()
