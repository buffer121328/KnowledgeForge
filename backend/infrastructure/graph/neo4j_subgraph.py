"""Visualization-oriented Neo4j subgraph retrieval and result conversion."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

CypherExecutor = Callable[[str, dict | None], Awaitable[list[dict]]]


def empty_subgraph(status: str = "ok") -> dict[str, Any]:
    """Handle empty subgraph for the module."""
    return {"nodes": [], "edges": [], "status": status}


async def get_subgraph(
    execute_cypher: CypherExecutor,
    *,
    driver: Any,
    keyword: str = "",
    entity_type: str = "",
    limit: int = 80,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Return the existing bounded nodes/edges visualization representation."""
    empty = empty_subgraph()
    if not driver:
        return empty_subgraph("disconnected")

    keyword = (keyword or "").strip()
    entity_type = (entity_type or "").strip()
    limit = max(1, min(int(limit), 200))
    params: dict[str, Any] = {"limit": limit, "keyword": keyword, "etype": entity_type}
    tenant_filter = ""
    if tenant_id is not None:
        params["tenant_id"] = tenant_id
        tenant_filter = "AND a.tenant_id = $tenant_id AND b.tenant_id = $tenant_id"

    try:
        if keyword:
            center_params: dict[str, Any] = {"keyword": keyword, "etype": entity_type, "seed_limit": min(20, limit)}
            center_tenant = ""
            if tenant_id is not None:
                center_params["tenant_id"] = tenant_id
                center_tenant = "AND e.tenant_id = $tenant_id"
            type_clause = "AND ($etype = '' OR e.type = $etype)"
            seeds = await execute_cypher(
                f"""
                    MATCH (e:Entity)
                    WHERE (e.name CONTAINS $keyword OR coalesce(e.description, '') CONTAINS $keyword)
                      {center_tenant}
                      {type_clause}
                    RETURN e.name AS name, e.type AS type,
                           coalesce(e.description, '') AS description
                    LIMIT $seed_limit
                    """,
                center_params,
            )
            if not seeds:
                return empty

            nodes_map: dict[str, dict[str, Any]] = {
                seed["name"]: {
                    "id": seed["name"],
                    "label": seed["name"],
                    "type": seed.get("type") or "Concept",
                    "description": seed.get("description") or "",
                }
                for seed in seeds
            }
            edges: list[dict[str, str]] = []
            edge_keys: set[tuple[str, str, str]] = set()
            hop_params: dict[str, Any] = {"names": list(nodes_map), "limit": limit}
            hop_tenant = ""
            if tenant_id is not None:
                hop_params["tenant_id"] = tenant_id
                hop_tenant = "AND a.tenant_id = $tenant_id AND b.tenant_id = $tenant_id"
            rels = await execute_cypher(
                f"""
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE (a.name IN $names OR b.name IN $names)
                      {hop_tenant}
                    RETURN a.name AS source, a.type AS source_type,
                           coalesce(a.description, '') AS source_desc,
                           type(r) AS relation,
                           b.name AS target, b.type AS target_type,
                           coalesce(b.description, '') AS target_desc
                    LIMIT $limit
                    """,
                hop_params,
            )
            for row in rels:
                source, target = row["source"], row["target"]
                for name, kind, description in (
                    (source, row.get("source_type"), row.get("source_desc")),
                    (target, row.get("target_type"), row.get("target_desc")),
                ):
                    if name not in nodes_map:
                        nodes_map[name] = {
                            "id": name,
                            "label": name,
                            "type": kind or "Concept",
                            "description": description or "",
                        }
                key = (source, target, row["relation"])
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges.append({"source": source, "target": target, "label": row["relation"]})
            return {"nodes": list(nodes_map.values()), "edges": edges, "status": "ok"}

        type_filter = ""
        if entity_type:
            type_filter = "AND (a.type = $etype OR b.type = $etype)"
        rels = await execute_cypher(
            f"""
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE true
                  {tenant_filter}
                  {type_filter}
                RETURN a.name AS source, a.type AS source_type,
                       coalesce(a.description, '') AS source_desc,
                       type(r) AS relation,
                       b.name AS target, b.type AS target_type,
                       coalesce(b.description, '') AS target_desc
                LIMIT $limit
                """,
            params,
        )
        nodes_map: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []
        for row in rels:
            source, target = row["source"], row["target"]
            for name, kind, description in (
                (source, row.get("source_type"), row.get("source_desc")),
                (target, row.get("target_type"), row.get("target_desc")),
            ):
                if name not in nodes_map:
                    nodes_map[name] = {"id": name, "label": name, "type": kind or "Concept", "description": description or ""}
            edges.append({"source": source, "target": target, "label": row["relation"]})

        if len(nodes_map) < min(20, limit):
            node_params: dict[str, Any] = {"limit": limit, "etype": entity_type}
            node_tenant = ""
            if tenant_id is not None:
                node_params["tenant_id"] = tenant_id
                node_tenant = "AND e.tenant_id = $tenant_id"
            node_type = "AND ($etype = '' OR e.type = $etype)"
            orphans = await execute_cypher(
                f"""
                    MATCH (e:Entity)
                    WHERE true
                      {node_tenant}
                      {node_type}
                    RETURN e.name AS name, e.type AS type,
                           coalesce(e.description, '') AS description
                    LIMIT $limit
                    """,
                node_params,
            )
            for orphan in orphans:
                name = orphan["name"]
                if name not in nodes_map:
                    nodes_map[name] = {
                        "id": name,
                        "label": name,
                        "type": orphan.get("type") or "Concept",
                        "description": orphan.get("description") or "",
                    }
        return {"nodes": list(nodes_map.values()), "edges": edges, "status": "ok"}
    except Exception:
        return empty_subgraph("error")
