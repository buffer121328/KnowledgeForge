"""Public Neo4j graph-service facade and lifecycle/write boundary."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from domain.knowledge import Entity, Relation
from shared.config import settings

from infrastructure.graph.neo4j_queries import (
    build_claim_context_search_query,
    build_claim_evidence_query,
    build_company_overview_query,
    build_department_graph_query,
    build_department_metadata_sync_query,
    build_department_relation_summary_query,
    build_document_context_query,
    build_document_evidence_delete_queries,
    build_entity_lookup_query,
    build_entity_types_query,
    build_entity_update_query,
    build_paths_query,
    build_relation_claim_query,
    build_relation_claim_update_query,
    build_source_delete_query,
    normalize_relation_type,
)
from infrastructure.graph.neo4j_subgraph import get_subgraph as build_subgraph


class GraphCleanupConsistencyError(RuntimeError):
    """Signal that graph deletion completed without reaching a clean verified state."""


class KnowledgeGraphService:
    """Neo4j lifecycle/write facade delegating query and subgraph concerns."""

    def __init__(self) -> None:
        """Initialize the knowledge graph service."""
        self._driver: Any = None

    async def init(self) -> None:
        """Initialize the knowledge graph service resources."""
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        try:
            await driver.verify_connectivity()
            self._driver = driver
            await self._ensure_indexes()
        except Exception:
            await driver.close()
            self._driver = None
            raise

    async def health_check(self) -> bool:
        """Return whether the initialized graph driver is currently reachable."""
        if self._driver is None:
            return False
        await self._driver.verify_connectivity()
        return True

    async def close(self) -> None:
        """Release resources held by the knowledge graph service."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def _ensure_indexes(self) -> None:
        """Ensure the indexes."""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.source)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.tenant_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Company) ON (n.tenant_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Department) ON (n.tenant_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.doc_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:RelationClaim) ON (n.claim_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:RelationEvidence) ON (n.evidence_id)",
        ]
        async with self._driver.session() as session:
            for query in index_queries:
                await session.run(query)

    async def upsert_entity(self, entity: Entity, version: int = 1, source: str = "", tenant_id: str = "") -> None:
        """Create or update the entity."""
        cypher = """
        MERGE (e:Entity {name: $name, tenant_id: $tenant_id})
        ON CREATE SET
            e.type = $type,
            e.description = $description,
            e.version = $version,
            e.source = $source,
            e.tenant_id = $tenant_id,
            e.created_at = $now,
            e.updated_at = $now
        ON MATCH SET
            e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
            e.version = $version,
            e.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "name": entity.name, "type": entity.type, "description": entity.description,
                "version": version, "source": source, "tenant_id": tenant_id, "now": int(time.time()),
            })

    async def add_relation(self, relation: Relation, source: str = "", tenant_id: str = "") -> None:
        """Add the relation."""
        relation_type = normalize_relation_type(relation.relation)
        if not relation_type:
            return
        cypher = f"""
        MATCH (h:Entity {{name: $head, tenant_id: $tenant_id}})
        MATCH (t:Entity {{name: $tail, tenant_id: $tenant_id}})
        MERGE (h)-[r:{relation_type}]->(t)
        SET r.confidence = $confidence, r.source = $source, r.tenant_id = $tenant_id, r.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "head": relation.head, "tail": relation.tail, "confidence": relation.confidence,
                "source": source, "tenant_id": tenant_id, "now": int(time.time()),
            })

    async def upsert_document_context(self, context: dict[str, Any]) -> None:
        """Create deterministic company, department, and document ownership nodes."""

        if not self._driver:
            return
        cypher, params = build_document_context_query(context)
        await self.execute_cypher(cypher, params)

    async def sync_department_names(
        self,
        tenant_id: str,
        department_names: dict[str, str],
    ) -> int:
        """仅更新租户已有 Department 节点的名称，返回实际更新的节点数。

        Args:
            tenant_id: 租户 ID，用于限定只更新该租户拥有的部门节点。
            department_names: 部门 ID 到部门名称的映射；键或值为空、映射为空时直接返回 0。
        """
        # 关键安全关卡：driver 未就绪、租户或映射为空时不执行任何写入，统一返回 0。
        if not self._driver or not tenant_id or not department_names:
            return 0
        cypher, params = build_department_metadata_sync_query(
            tenant_id,
            department_names,
        )
        rows = await self.execute_cypher(cypher, params)
        return int(rows[0].get("updated", 0)) if rows else 0

    async def add_relation_evidence(
        self,
        relation: Relation,
        *,
        context: dict[str, Any],
        chunk_id: str,
        extractor: str = "knowledge_extractor",
        model: str = "",
        version: str = "1",
    ) -> tuple[str, str] | None:
        """Merge one semantic claim while retaining document/chunk evidence."""

        if not self._driver or not context.get("doc_id"):
            return None
        relation_type = normalize_relation_type(relation.relation)
        if not relation_type:
            return None
        tenant_id = str(context.get("tenant_id") or "")
        claim_key = "|".join(
            (tenant_id, relation.head.casefold(), relation_type, relation.tail.casefold())
        )
        claim_id = hashlib.sha256(claim_key.encode()).hexdigest()[:32]
        evidence_key = "|".join(
            (claim_id, str(context.get("doc_id") or ""), chunk_id)
        )
        evidence_id = hashlib.sha256(evidence_key.encode()).hexdigest()[:32]
        cypher, params = build_relation_claim_query(
            claim_id=claim_id,
            evidence_id=evidence_id,
            head=relation.head,
            relation_type=relation_type,
            tail=relation.tail,
            confidence=relation.confidence,
            context=context,
            chunk_id=chunk_id,
            extractor=extractor,
            model=model,
            version=version,
        )
        await self.execute_cypher(cypher, params)
        return claim_id, evidence_id

    async def delete_document_evidence(self, doc_id: str, tenant_id: str) -> int:
        """Delete document evidence and prune empty organization hierarchy."""

        if not self._driver:
            return 0
        steps = build_document_evidence_delete_queries(doc_id, tenant_id)
        claim_rows = await self.execute_cypher(*steps[0])
        claim_ids = [
            str(claim_id)
            for claim_id in (claim_rows[0].get("claim_ids", []) if claim_rows else [])
            if claim_id
        ]
        evidence_rows = await self.execute_cypher(*steps[1])
        deleted_evidence = (
            int(evidence_rows[0].get("deleted_evidence", 0)) if evidence_rows else 0
        )
        await self.execute_cypher(*steps[2])
        orphan_query, orphan_params = steps[3]
        await self.execute_cypher(
            orphan_query,
            {**orphan_params, "claim_ids": claim_ids},
        )
        await self.execute_cypher(*steps[4])
        await self.execute_cypher(*steps[5])
        verification = await self.execute_cypher(*steps[6])
        remaining = verification[0] if verification else {}
        residue_fields = (
            "documents",
            "evidences",
            "orphan_departments",
            "orphan_companies",
        )
        if any(int(remaining.get(field, 0)) for field in residue_fields):
            raise GraphCleanupConsistencyError(
                "document graph cleanup verification failed"
            )
        return deleted_evidence

    async def get_company_overview(self, tenant_id: str) -> dict[str, Any]:
        """查询公司/部门层级及跨部门关系的聚合概览图。

        Args:
            tenant_id: 租户 ID，限定查询范围在该租户内。
        """
        # 关键安全关卡：driver 未就绪时返回空图并标记 disconnected，不抛异常。
        if not self._driver:
            return {"nodes": [], "edges": [], "departments": [], "status": "disconnected"}
        overview_query, overview_params = build_company_overview_query(tenant_id)
        relation_query, relation_params = build_department_relation_summary_query(tenant_id)
        overview_rows = await self.execute_cypher(overview_query, overview_params)
        relation_rows = await self.execute_cypher(relation_query, relation_params)
        if not overview_rows:
            return {"nodes": [], "edges": [], "departments": [], "status": "ok"}
        overview = overview_rows[0]
        company_id = str(overview.get("company_id") or tenant_id)
        company_name = str(overview.get("company_name") or company_id)
        departments = [item for item in overview.get("departments", []) if item]
        nodes = [
            {
                "id": f"company:{company_id}",
                "label": company_name,
                "type": "Company",
                "company_id": company_id,
            }
        ]
        edges: list[dict[str, Any]] = []
        for department in departments:
            department_id = str(department.get("department_id") or "")
            nodes.append(
                {
                    "id": f"department:{department_id}",
                    "label": str(department.get("name") or department_id),
                    "type": "Department",
                    "department_id": department_id,
                    "document_count": int(department.get("document_count") or 0),
                }
            )
            edges.append(
                {
                    "source": f"company:{company_id}",
                    "target": f"department:{department_id}",
                    "label": "HAS_DEPARTMENT",
                    "path_count": int(department.get("document_count") or 0),
                }
            )
        # 跨部门聚合边：附带的 relation_types 与 relation_details 汇总该部门对之间
        # 语义关系断言的类型与头尾实体明细，供前端展示关系构成。
        for relation in relation_rows:
            edges.append(
                {
                    "source": f"department:{relation.get('source_department_id', '')}",
                    "target": f"department:{relation.get('target_department_id', '')}",
                    "label": "CROSS_DEPARTMENT",
                    "path_count": int(relation.get("path_count") or 0),
                    "claim_ids": relation.get("claim_ids") or [],
                    "relation_types": relation.get("relation_types") or [],
                    "relation_details": relation.get("relation_details") or [],
                }
            )
        return {
            "company_id": company_id,
            "company_name": company_name,
            "departments": departments,
            "nodes": nodes,
            "edges": edges,
            "status": "ok",
        }

    async def get_department_graph(
        self,
        department_id: str,
        *,
        tenant_id: str,
        include_cross_department: bool = False,
        limit: int = 200,
    ) -> dict[str, Any]:
        """查询租户隔离的部门图谱投影（节点与带证据计数的边）。

        Args:
            department_id: 目标部门 ID，图以该部门为中心。
            tenant_id: 租户 ID，限定查询范围在该租户内。
            include_cross_department: 是否包含跨部门的关联节点与边。
            limit: 返回节点/行数的上限。
        """
        # 关键安全关卡：driver 未就绪时返回空图并标记 disconnected。
        if not self._driver:
            return {"nodes": [], "edges": [], "status": "disconnected"}
        cypher, params = build_department_graph_query(
            tenant_id,
            department_id,
            include_cross_department=include_cross_department,
            limit=limit,
        )
        rows = await self.execute_cypher(cypher, params)
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            for key, node_type, id_field, label_field in (
                ("department", "Department", "department_id", "name"),
                ("document", "Document", "doc_id", "display_name"),
                ("head", "Entity", "name", "name"),
                ("tail", "Entity", "name", "name"),
                ("claim", "RelationClaim", "claim_id", "relation_type"),
            ):
                value = row.get(key)
                properties = dict(value) if value is not None else {}
                identifier = str(properties.get(id_field) or "")
                if not identifier:
                    continue
                node_id = f"{node_type.lower()}:{identifier}"
                display_type = (
                    str(properties.get("type") or "Entity")
                    if node_type == "Entity"
                    else node_type
                )
                nodes[node_id] = {
                    "id": node_id,
                    "label": str(properties.get(label_field) or identifier),
                    "type": display_type,
                    "department_id": str(properties.get("department_id") or department_id),
                    "description": str(properties.get("description") or ""),
                }
            claim = dict(row.get("claim")) if row.get("claim") is not None else {}
            head = dict(row.get("head")) if row.get("head") is not None else {}
            tail = dict(row.get("tail")) if row.get("tail") is not None else {}
            if claim and head and tail:
                edge_key = (
                    f"entity:{head.get('name')}",
                    f"entity:{tail.get('name')}",
                    str(claim.get("claim_id") or ""),
                )
                edge = edges.setdefault(
                    edge_key,
                    {
                        "source": edge_key[0],
                        "target": edge_key[1],
                        "label": str(claim.get("relation_type") or "RELATED_TO"),
                        "claim_id": edge_key[2],
                        "path_count": 0,
                        # 关系明细：记录该边的断言 ID、关系类型与头尾实体名称/类型，
                        # 前端据此渲染实体间语义关系的具体构成。
                        "relation_details": [
                            {
                                "claim_id": edge_key[2],
                                "relation_type": str(claim.get("relation_type") or "RELATED_TO"),
                                "head_name": str(head.get("name") or ""),
                                "head_type": str(head.get("type") or "Entity"),
                                "tail_name": str(tail.get("name") or ""),
                                "tail_type": str(tail.get("type") or "Entity"),
                            }
                        ],
                    },
                )
                if row.get("evidence") is not None:
                    edge["path_count"] += 1
        return {"nodes": list(nodes.values()), "edges": list(edges.values()), "status": "ok"}

    async def update_entity(
        self,
        *,
        node_id: str,
        tenant_id: str,
        label: str | None = None,
        entity_type: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any] | None:
        """更新租户内一条已抽取的实体节点，返回更新后的节点数据。

        Args:
            node_id: 实体节点 ID（形如 "entity:<名称>"），用于定位待更新实体。
            tenant_id: 租户 ID，限定只能更新该租户的实体。
            label: 新的实体名称；为 None 或空白时保持原值不变。
            entity_type: 新的实体类型；为 None 或空白时保持原值不变。
            description: 新的实体描述；为 None 时保持原值不变。
        """
        # 关键安全关卡：driver 未就绪时不执行更新，直接返回 None 表示未命中。
        if not self._driver:
            return None
        cypher, params = build_entity_update_query(
            tenant_id=tenant_id,
            node_id=node_id,
            label=label,
            entity_type=entity_type,
            description=description,
        )
        rows = await self.execute_cypher(cypher, params)
        return rows[0].get("node") if rows else None

    async def update_relation_claim(
        self,
        *,
        claim_id: str,
        tenant_id: str,
        relation_type: str,
    ) -> dict[str, Any] | None:
        """更新租户内一条语义关系断言的关系类型，返回更新后的断言行。

        Args:
            claim_id: 关系断言 ID，定位待更新的 RelationClaim 节点。
            tenant_id: 租户 ID，限定只能更新该租户的断言。
            relation_type: 新的关系类型；经规范化后非法或为空时不更新并返回 None。
        """
        # 关键安全关卡：先规范化关系类型，driver 未就绪或类型非法时拒绝更新。
        normalized = normalize_relation_type(relation_type)
        if not self._driver or not normalized:
            return None
        cypher, params = build_relation_claim_update_query(
            tenant_id=tenant_id,
            claim_id=claim_id,
            relation_type=normalized,
        )
        rows = await self.execute_cypher(cypher, params)
        return rows[0] if rows else None

    async def get_paths(
        self,
        *,
        from_id: str,
        to_id: str,
        tenant_id: str,
        department_id: str | None = None,
        include_cross_department: bool = False,
        max_hops: int = 4,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return bounded concrete graph paths inside the authorized tenant scope."""

        if not self._driver:
            return {"paths": [], "path_count": 0, "status": "disconnected"}
        cypher, params = build_paths_query(
            tenant_id=tenant_id,
            from_id=from_id,
            to_id=to_id,
            department_id=department_id,
            include_cross_department=include_cross_department,
            max_hops=max_hops,
            limit=limit,
        )
        paths = await self.execute_cypher(cypher, params)
        return {"paths": paths, "path_count": len(paths), "status": "ok"}

    async def search_claim_contexts(
        self,
        keyword: str,
        *,
        tenant_id: str,
        visible_department_ids: tuple[str, ...] | list[str] | None = None,
        limit: int = 8,
        evidence_limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return bounded relation claims with only authorized document evidence."""

        if not self._driver or not tenant_id:
            return []
        cypher, params = build_claim_context_search_query(
            keyword=keyword,
            tenant_id=tenant_id,
            visible_department_ids=visible_department_ids,
            limit=limit,
            evidence_limit=evidence_limit,
        )
        return await self.execute_cypher(cypher, params)

    async def get_claim_evidence(self, claim_id: str, tenant_id: str) -> dict[str, Any]:
        """Return traceable document/chunk evidence for an authorized claim."""

        if not self._driver:
            return {"claim_id": claim_id, "evidence": [], "status": "disconnected"}
        cypher, params = build_claim_evidence_query(claim_id, tenant_id)
        evidence = await self.execute_cypher(cypher, params)
        return {"claim_id": claim_id, "evidence": evidence, "status": "ok"}

    async def execute_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute the cypher."""
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            return await result.data()

    async def get_entity(self, name: str, tenant_id: str | None = None) -> dict | None:
        """Return the entity."""
        cypher, params = build_entity_lookup_query(name, tenant_id)
        records = await self.execute_cypher(cypher, params)
        return records[0] if records else None

    async def list_entity_types(self, tenant_id: str | None = None) -> list[str]:
        """List the entity types."""
        if not self._driver:
            return []
        cypher, params = build_entity_types_query(tenant_id)
        try:
            rows = await self.execute_cypher(cypher, params)
            return [row["type"] for row in rows if row.get("type")]
        except Exception:
            return []

    async def get_subgraph(self, *, keyword: str = "", entity_type: str = "", limit: int = 80, tenant_id: str | None = None) -> dict[str, Any]:
        """Return the subgraph."""
        return await build_subgraph(
            self.execute_cypher,
            driver=self._driver,
            keyword=keyword,
            entity_type=entity_type,
            limit=limit,
            tenant_id=tenant_id,
        )

    async def delete_by_source(self, source: str, tenant_id: str | None = None) -> int:
        """Delete the by source."""
        cypher, params = build_source_delete_query(source, tenant_id)
        records = await self.execute_cypher(cypher, params)
        return records[0].get("deleted", 0) if records else 0

    async def get_stats(self, tenant_id: str | None = None) -> dict:
        """Return graph statistics, optionally scoped to one tenant."""
        disconnected = {
            "total_entities": 0,
            "total_relations": 0,
            "nodes": 0,
            "edges": 0,
            "status": "disconnected",
        }
        if not self._driver:
            return disconnected
        try:
            if tenant_id:
                entity_count = await self.execute_cypher(
                    """
                    MATCH (entity:Entity)
                    WHERE entity.tenant_id = $tenant_id
                    RETURN count(entity) AS cnt
                    """,
                    {"tenant_id": tenant_id},
                )
                relation_count = await self.execute_cypher(
                    """
                    MATCH (source)-[relation]->(target)
                    WHERE source.tenant_id = $tenant_id
                      AND target.tenant_id = $tenant_id
                    RETURN count(relation) AS cnt
                    """,
                    {"tenant_id": tenant_id},
                )
            else:
                entity_count = await self.execute_cypher(
                    "MATCH (entity:Entity) RETURN count(entity) AS cnt"
                )
                relation_count = await self.execute_cypher(
                    "MATCH ()-[relation]->() RETURN count(relation) AS cnt"
                )
            nodes = int(entity_count[0]["cnt"]) if entity_count else 0
            edges = int(relation_count[0]["cnt"]) if relation_count else 0
            return {
                "total_entities": nodes,
                "total_relations": edges,
                "nodes": nodes,
                "edges": edges,
                "status": "ok",
            }
        except Exception:
            return disconnected
