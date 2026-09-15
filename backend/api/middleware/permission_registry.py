"""Route permission registration and matching."""

from __future__ import annotations

import logging
import re

from domain.identity import Permission


logger = logging.getLogger("api.middleware.permission")

# ── 路由权限注册表 ────────────────────────────────────────────
# 格式: (HTTP_METHOD, path_pattern) → required_permissions
# path_pattern 支持 FastAPI 风格路径参数: /api/v1/users/{user_id}
# required_permissions 是列表，用户只需拥有其中任一权限即可（OR 逻辑）

ROUTE_PERMISSIONS: dict[tuple[str, str], list[Permission]] = {}


def register_permission(
    method: str,
    path: str,
    *permissions: Permission,
    require_all: bool = False,
):
    """
    注册路由权限要求。

    Args:
        method: HTTP 方法 (GET, POST, PUT, PATCH, DELETE)
        path: 路径模式，支持 {param} 占位符
        require_all: True=需要全部权限(AND)，False=任一即可(OR)
    """
    key = (method.upper(), path)
    if require_all:
        # AND 逻辑: 存储为特殊标记
        ROUTE_PERMISSIONS[key] = list(permissions)
    else:
        # OR 逻辑: 默认
        ROUTE_PERMISSIONS[key] = list(permissions)


def _match_route(method: str, path: str) -> list[Permission] | None:
    """
    匹配请求路径对应的权限要求。
    支持精确匹配和路径参数匹配。
    """
    # 精确匹配
    key = (method, path)
    if key in ROUTE_PERMISSIONS:
        return ROUTE_PERMISSIONS[key]

    # 模式匹配（支持路径参数）
    for (m, pattern), perms in ROUTE_PERMISSIONS.items():
        if m != method:
            continue
        if _path_matches(pattern, path):
            return perms

    return None


def _path_matches(pattern: str, path: str) -> bool:
    """检查路径是否匹配模式（支持 {param} 占位符）"""
    # 将模式转换为正则表达式
    regex = re.sub(r"\{[^}]+\}", r"[^/]+", pattern)
    regex = f"^{regex}$"
    return bool(re.match(regex, path))


# ── 预注册所有路由权限 ────────────────────────────────────────

def _register_v1(method: str, resource_path: str, *permissions: Permission):
    """Register a permission rule under the canonical v1 business prefix."""
    register_permission(method, f"/api/v1{resource_path}", *permissions)


def init_route_permissions():
    """初始化路由权限注册表 — 在应用启动时调用"""
    from domain.identity import Permission as P

    # ── 认证路由 ──────────────────────────────────────────────
    _register_v1("POST", "/auth/login")      # 无需权限（公开）
    _register_v1("POST", "/auth/register")    # 无需权限（公开）
    _register_v1("POST", "/auth/refresh")     # 无需权限（公开）
    _register_v1("GET", "/auth/me")           # 已认证用户即可
    _register_v1("POST", "/auth/logout")      # 已认证用户即可

    # ── 文档入库 ──────────────────────────────────────────────
    _register_v1("POST", "/ingest/upload", P.DOC_WRITE)
    _register_v1("POST", "/ingest/batch", P.DOC_WRITE)
    _register_v1("POST", "/ingest/folder", P.DOC_WRITE)
    _register_v1("GET", "/ingest/progress/{upload_id}", P.DOC_READ)

    # ── 文档查阅 ──────────────────────────────────────────────
    _register_v1("GET", "/docs", P.DOC_READ)
    _register_v1("GET", "/docs/departments", P.DOC_READ)
    _register_v1("GET", "/docs/{doc_id}/file", P.DOC_READ)
    _register_v1("GET", "/docs/{doc_id}/chunks", P.DOC_READ)
    _register_v1("DELETE", "/docs/{doc_id}", P.DOC_DELETE)

    # ── 智能问答 ──────────────────────────────────────────────
    _register_v1("POST", "/qa/ask", P.QA_QUERY)

    # ── 知识图谱 ──────────────────────────────────────────────
    _register_v1("GET", "/graph/subgraph", P.GRAPH_READ)
    _register_v1("GET", "/graph/company-overview", P.GRAPH_READ)
    _register_v1("GET", "/graph/departments/{department_id}", P.GRAPH_READ)
    _register_v1("GET", "/graph/paths", P.GRAPH_READ)
    _register_v1("GET", "/graph/claims/{claim_id}/evidence", P.GRAPH_READ)
    _register_v1("GET", "/graph/types", P.GRAPH_READ)
    # 图谱编辑类路由（本次新增）：人工修正已抽取的实体与关系，需要更高的 GRAPH_EDIT 权限。
    _register_v1("PATCH", "/graph/entities", P.GRAPH_EDIT)
    _register_v1("PATCH", "/graph/relations", P.GRAPH_EDIT)

    # ── 系统管理 ──────────────────────────────────────────────
    _register_v1("GET", "/admin/stats", P.ADMIN_MANAGE)
    _register_v1("POST", "/admin/update", P.ADMIN_MANAGE)

    # ── 用户管理 ──────────────────────────────────────────────
    _register_v1("GET", "/users", P.ADMIN_USER)
    _register_v1("GET", "/users/{user_id}", P.ADMIN_USER)
    _register_v1("POST", "/users", P.ADMIN_USER)
    _register_v1("PATCH", "/users/{user_id}", P.ADMIN_USER)
    _register_v1("DELETE", "/users/{user_id}", P.ADMIN_USER)
    _register_v1("POST", "/users/batch/role", P.ADMIN_USER)
    _register_v1("POST", "/users/me/password")  # 自助，无需额外权限
    _register_v1("POST", "/users/{user_id}/reset-password", P.ADMIN_USER)
    _register_v1("POST", "/users/{user_id}/toggle-active", P.ADMIN_USER)

    # ── 角色管理 ──────────────────────────────────────────────
    _register_v1("GET", "/roles", P.ADMIN_USER)
    _register_v1("GET", "/roles/meta/permissions", P.ADMIN_USER)
    _register_v1("GET", "/roles/{role_name}", P.ADMIN_USER)
    _register_v1("POST", "/roles", P.ADMIN_USER)
    _register_v1("PATCH", "/roles/{role_name}", P.ADMIN_USER)
    _register_v1("DELETE", "/roles/{role_name}", P.ADMIN_USER)
    _register_v1("GET", "/roles/{role_name}/permissions", P.ADMIN_USER)
    _register_v1("PUT", "/roles/{role_name}/permissions", P.ADMIN_USER)
    _register_v1("POST", "/roles/{role_name}/permissions", P.ADMIN_USER)
    _register_v1("DELETE", "/roles/{role_name}/permissions", P.ADMIN_USER)
    _register_v1("GET", "/roles/{role_name}/users", P.ADMIN_USER)

    # ── API Key 管理 ──────────────────────────────────────────
    _register_v1("POST", "/auth/apikey/create")  # 登录用户即可
    _register_v1("GET", "/auth/apikey/list")     # 登录用户即可
    _register_v1("DELETE", "/auth/apikey/{key_id}")  # 登录用户即可
    _register_v1("POST", "/auth/apikey/{key_id}/toggle")  # 登录用户即可
    _register_v1("POST", "/auth/apikey/{key_id}/rotate")  # 登录用户即可
    _register_v1("PATCH", "/auth/apikey/{key_id}/scopes")  # 登录用户即可

    # ── 任务管理 ──────────────────────────────────────────────
    _register_v1("POST", "/tasks/submit", P.DOC_WRITE)
    _register_v1("POST", "/tasks/submit-batch", P.DOC_WRITE)
    _register_v1("POST", "/tasks/update", P.DOC_WRITE)
    _register_v1("GET", "/tasks/{task_id}")  # 登录用户即可
    _register_v1("GET", "/tasks")  # 登录用户即可
    _register_v1("DELETE", "/tasks/{task_id}", P.DOC_WRITE)
    _register_v1("POST", "/tasks/cleanup", P.ADMIN_MANAGE)

    # ── Webhook 管理 ──────────────────────────────────────────
    _register_v1("POST", "/webhooks", P.ADMIN_MANAGE)
    _register_v1("GET", "/webhooks", P.ADMIN_MANAGE)
    _register_v1("GET", "/webhooks/events", P.ADMIN_MANAGE)
    _register_v1("DELETE", "/webhooks/{webhook_id}", P.ADMIN_MANAGE)
    _register_v1("POST", "/webhooks/{webhook_id}/test", P.ADMIN_MANAGE)

    # ── 审计日志 ──────────────────────────────────────────────
    _register_v1("GET", "/audit/logs", P.ADMIN_AUDIT)
    _register_v1("GET", "/audit/actions", P.ADMIN_AUDIT)
    _register_v1("GET", "/audit/logs/export", P.ADMIN_AUDIT)
    _register_v1("GET", "/admin/audit-logs", P.ADMIN_AUDIT)

    logger.info("路由权限注册表初始化完成，共 %d 条规则", len(ROUTE_PERMISSIONS))
