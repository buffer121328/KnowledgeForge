"""API admin 域共享的请求与响应模型（自 api/schemas.py 拆分）。"""

from __future__ import annotations


from pydantic import BaseModel



class AuditLogResponse(BaseModel):
    """审计日志响应。"""
    timestamp: float  # 日志时间戳
    user_id: str  # 操作用户 ID
    username: str  # 操作用户名
    method: str  # HTTP 方法
    path: str  # 请求路径
    required_permissions: list[str]  # 要求的权限列表
    user_permissions: list[str]  # 用户拥有的权限列表
    granted: bool  # 是否放行
    reason: str  # 判定原因
