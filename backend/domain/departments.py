"""目录与图谱边界共享的稳定部门元数据。"""

from __future__ import annotations

from types import MappingProxyType

# 已知部门的稳定 ID -> 中文展示名映射；用 MappingProxyType 固化为只读，防止运行时被篡改
KNOWN_DEPARTMENT_NAMES = MappingProxyType(
    {
        "administration": "行政管理部",
        "finance": "财务部",
        "general": "默认部门",
        "human_resources": "人力资源部",
        "procurement_warehouse": "采购/仓储部",
    }
)


def department_display_name(
    department_id: str,
    explicit_name: str | None = None,
) -> str:
    """返回部门展示名：已知 ID 用规范名，且不覆盖调用方自定义名称。

    Args:
        department_id: 部门标识。
        explicit_name: 调用方显式提供的部门名，仅在 ID 未知时被采用。
    """
    # ① 归一化输入：去除首尾空白，空值回退为空串
    normalized_id = str(department_id or "").strip()
    normalized_name = str(explicit_name or "").strip()
    # ② 已知部门 ID（大小写不敏感）优先返回规范中文名
    known_name = KNOWN_DEPARTMENT_NAMES.get(normalized_id.casefold())
    if known_name is not None:
        return known_name
    # ③ 未知 ID：优先返回显式名称，否则原样返回归一化后的 ID
    return normalized_name or normalized_id


__all__ = ["KNOWN_DEPARTMENT_NAMES", "department_display_name"]  # 模块公开导出的稳定 API
