"""HTTP API 路由模块与 /api/v1 版本化装配点。

``build_v1_router()`` 聚合全部业务 router（含管理类路由的子前缀），由
``api.app`` 在路由注册完成后调用；健康检查保持独立路径，不经由此处。
"""

from __future__ import annotations

from fastapi import APIRouter

from api.routers.admin import admin_router, health_router
from api.routers.apikeys import router as apikey_router
from api.routers.audit import router as audit_router
from api.routers.auth import router as auth_router
from api.routers.evaluation_pkg import evaluation_router
from api.routers.knowledge_graph import graph_router
from api.routers.qa import qa_router
from api.routers.roles import router as role_router
from api.routers.tasks import router as tasks_router
from api.routers.users import router as user_router
from api.routers.webhooks import router as webhooks_router

__all__ = ["build_v1_router", "health_router"]


def build_v1_router() -> APIRouter:
    """聚合全部业务路由到 /api/v1 前缀；在所有端点注册完成后调用。"""

    v1_router = APIRouter(prefix="/api/v1", tags=["API v1"])
    v1_router.include_router(auth_router)
    v1_router.include_router(apikey_router, prefix="/auth")
    v1_router.include_router(user_router)
    v1_router.include_router(role_router)
    v1_router.include_router(tasks_router)
    v1_router.include_router(webhooks_router)
    v1_router.include_router(audit_router)
    for feature_router in (
        admin_router,
        qa_router,
        evaluation_router,
    ):
        v1_router.include_router(feature_router)

    from api.routers.documents_ingest import ingest_router
    from api.routers.documents_read import docs_router

    v1_router.include_router(ingest_router)
    v1_router.include_router(docs_router)
    v1_router.include_router(graph_router)
    return v1_router
