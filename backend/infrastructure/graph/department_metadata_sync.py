"""幂等地把已知部门名称同步到目录与图谱事实。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from domain.departments import KNOWN_DEPARTMENT_NAMES, department_display_name


@dataclass(frozen=True)
class DepartmentMetadataSyncReport:
    """汇总一次租户级、仅元数据的部门名称同步结果。"""

    tenant_id: str  # 本次同步的租户 ID
    discovered_departments: int  # 目录中发现的部门记录总数
    known_departments: int  # 命中已知部门名单的记录数
    catalog_updates: int  # 目录中名称被更新的条数
    graph_updates: int  # 图谱中部门名称被更新的条数
    department_names: dict[str, str]  # 部门 ID -> 规范显示名映射


async def sync_known_department_metadata(
    *,
    catalog: Any,
    graph: Any,
    tenant_id: str,
    company_id: str | None = None,
) -> DepartmentMetadataSyncReport:
    """只更新既有已知部门的显示名，不创建部门或用户

    Args:
        catalog: 目录仓储对象，需提供 list_departments 与 upsert_department。
        graph: 图谱服务对象，需提供 async 的 sync_department_names(tenant_id, names)。
        tenant_id: 目标租户 ID，必填。
        company_id: 公司 ID；为 None 时以租户 ID 兜底作为有效公司范围。
    """
    # ① 校验租户并确定有效公司范围（未提供公司时退回租户 ID）
    normalized_tenant = str(tenant_id or "").strip()
    if not normalized_tenant:
        raise ValueError("tenant_id is required")
    effective_company = str(company_id or normalized_tenant).strip()
    # ② 列出该范围内的部门记录，筛出命中已知部门名单（大小写不敏感）的子集
    departments = catalog.list_departments(
        tenant_id=normalized_tenant,
        company_id=effective_company,
    )
    known_records = [
        record
        for record in departments
        if record.department_id.casefold() in KNOWN_DEPARTMENT_NAMES
    ]
    # ③ 计算每个已知部门的规范显示名（记录名作为回退）
    names = {
        record.department_id: department_display_name(
            record.department_id,
            record.name,
        )
        for record in known_records
    }
    catalog_updates = 0
    # ④ 幂等更新目录：仅在记录名与规范名不一致时 upsert，保证重复执行无副作用
    for record in known_records:
        name = names[record.department_id]
        if record.name == name:
            continue
        catalog.upsert_department(replace(record, name=name))
        catalog_updates += 1
    # ⑤ 把名称映射同步到图谱事实并取回更新条数
    graph_updates = await graph.sync_department_names(normalized_tenant, names)
    return DepartmentMetadataSyncReport(
        tenant_id=normalized_tenant,
        discovered_departments=len(departments),
        known_departments=len(known_records),
        catalog_updates=catalog_updates,
        graph_updates=graph_updates,
        department_names=names,
    )


__all__ = ["DepartmentMetadataSyncReport", "sync_known_department_metadata"]  # 对外导出同步报告与同步入口
