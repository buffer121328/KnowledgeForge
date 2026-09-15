"""把既有已知部门的名称同步到 PostgreSQL 与 Neo4j。"""

from __future__ import annotations

import argparse
import asyncio
import json

from infrastructure.graph.department_metadata_sync import sync_known_department_metadata
from infrastructure.documents.catalog import PostgreSQLDocumentCatalogRepository
from infrastructure.graph.neo4j_graph import KnowledgeGraphService
from infrastructure.postgres.database import get_database_service


async def _run(tenant_id: str, company_id: str | None) -> int:
    """执行一次部门名称同步并以 JSON 打印同步报告

    Args:
        tenant_id: 目标租户 ID。
        company_id: 公司 ID；为 None 时以租户 ID 兜底。
    """
    # 构建目录仓储与图谱服务，并初始化图谱连接
    catalog = PostgreSQLDocumentCatalogRepository(get_database_service())
    graph = KnowledgeGraphService()
    await graph.init()
    try:
        report = await sync_known_department_metadata(
            catalog=catalog,
            graph=graph,
            tenant_id=tenant_id,
            company_id=company_id,
        )
    finally:
        # 无论同步成败都关闭图谱连接，避免泄漏
        await graph.close()
    # 以排序键 JSON 打印同步摘要，便于日志检索与对比
    print(
        json.dumps(
            {
                "catalog_updates": report.catalog_updates,
                "department_names": report.department_names,
                "discovered_departments": report.discovered_departments,
                "graph_updates": report.graph_updates,
                "known_departments": report.known_departments,
                "tenant_id": report.tenant_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    """命令行入口：解析租户参数并运行同步。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", required=True)  # 目标租户 ID
    parser.add_argument("--company", default="")  # 可选公司 ID，留空时以租户 ID 兜底
    args = parser.parse_args()
    # 空字符串统一转为 None，交由同步入口按租户兜底
    return asyncio.run(_run(args.tenant, args.company or None))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
