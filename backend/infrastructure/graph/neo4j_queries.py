"""Tenant-safe Cypher query builders used by the graph facade."""

from __future__ import annotations

import re
from typing import Any

from domain.departments import department_display_name

_REL_TYPE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def normalize_relation_type(relation: str) -> str | None:
    """Normalize and validate a dynamic relationship type before interpolation."""
    normalized = relation.upper().replace(" ", "_").replace("-", "_")
    return normalized if _REL_TYPE_PATTERN.match(normalized) else None


def build_entity_lookup_query(name: str, tenant_id: str | None = None) -> tuple[str, dict[str, Any]]:
    """Build the entity lookup query."""
    if tenant_id is not None:
        return (
            "MATCH (e:Entity {name: $name, tenant_id: $tenant_id}) RETURN e",
            {"name": name, "tenant_id": tenant_id},
        )
    return "MATCH (e:Entity {name: $name}) RETURN e", {"name": name}




def build_claim_context_search_query(
    *,
    keyword: str,
    tenant_id: str,
    visible_department_ids: tuple[str, ...] | list[str] | None = None,
    limit: int = 8,
    evidence_limit: int = 5,
) -> tuple[str, dict[str, Any]]:
    """构建生产问答用的、限额且租户隔离的断言/证据检索查询。

    Args:
        keyword: 检索关键词，匹配实体名称或描述。
        tenant_id: 租户 ID，所有 MATCH 均绑定该租户。
        visible_department_ids: 用户可见的部门 ID 集合；None 表示不过滤部门（管理员场景），非 None 时空集合会以哨兵值收紧为不可见任何部门（fail-closed）。
        limit: 返回断言数量的上限，实际被夹紧到 [1, 50]。
        evidence_limit: 每条断言返回证据条数的上限，实际被夹紧到 [1, 20]。
    """
    # ① 夹紧返回上限，防止调用方传入极端值放大查询开销。
    bounded_limit = max(1, min(limit, 50))
    bounded_evidence_limit = max(1, min(evidence_limit, 20))
    # ② 部门过滤关键安全关卡：None 才允许不过滤；空集合换成不可能命中的哨兵值，
    #    保证“无可见部门”的用户查不到任何证据，而不是退化成全量可见。
    if visible_department_ids is None:
        departments: list[str] = []
        department_clause = ""
    else:
        departments = [value for value in visible_department_ids if value] or ["__no_visible_department__"]
        department_clause = "AND evidence.department_id IN $visible_department_ids"
    return (
        f"""
        MATCH (seed:Entity)
        WHERE seed.tenant_id = $tenant_id
          AND (seed.name CONTAINS $keyword OR coalesce(seed.description, '') CONTAINS $keyword)
        CALL {{
            WITH seed
            MATCH (seed)-[:CLAIM_HEAD]->
                  (claim:RelationClaim {{tenant_id: $tenant_id}})-[:CLAIM_TAIL]->
                  (related:Entity {{tenant_id: $tenant_id}})
            RETURN claim, seed AS head, related AS tail
            UNION
            WITH seed
            MATCH (related:Entity {{tenant_id: $tenant_id}})-[:CLAIM_HEAD]->
                  (claim:RelationClaim {{tenant_id: $tenant_id}})-[:CLAIM_TAIL]->(seed)
            RETURN claim, related AS head, seed AS tail
        }}
        MATCH (evidence:RelationEvidence {{tenant_id: $tenant_id}})-[:SUPPORTS]->(claim)
        MATCH (document:Document {{tenant_id: $tenant_id}})-[:PROVIDES_EVIDENCE]->(evidence)
        WHERE document.doc_id = evidence.doc_id
          {department_clause}
        WITH claim, head, tail,
             collect(DISTINCT {{
                 evidence_id: evidence.evidence_id,
                 doc_id: evidence.doc_id,
                 chunk_id: evidence.chunk_id,
                 department_id: evidence.department_id,
                 confidence: evidence.confidence,
                 authority: document.authority,
                 review_status: document.review_status,
                 sensitivity: document.sensitivity,
                 display_name: document.display_name,
                 provenance_source_filename: document.provenance_source_filename,
                 relative_path: document.relative_path,
                 source: CASE
                     WHEN coalesce(document.provenance_source_filename, '') <> ''
                         THEN document.provenance_source_filename
                     WHEN coalesce(document.display_name, '') <> ''
                         THEN document.display_name
                     ELSE evidence.doc_id
                 END
             }}) AS all_evidences
        RETURN claim.claim_id AS claim_id,
               claim.relation_type AS relation_type,
               head.name AS head_name,
               head.type AS head_type,
               tail.name AS tail_name,
               tail.type AS tail_type,
               coalesce(claim.confidence, 0.0) AS claim_confidence,
               size(all_evidences) AS evidence_count,
               all_evidences[..$evidence_limit] AS evidences
        ORDER BY claim_confidence DESC, evidence_count DESC, claim_id
        LIMIT $limit
        """,
        {
            "keyword": keyword,
            "tenant_id": tenant_id,
            "visible_department_ids": departments,
            "limit": bounded_limit,
            "evidence_limit": bounded_evidence_limit,
        },
    )


def build_entity_types_query(tenant_id: str | None = None) -> tuple[str, dict[str, Any]]:
    """Build the entity types query."""
    if tenant_id is not None:
        return (
            """
            MATCH (e:Entity)
            WHERE e.tenant_id = $tenant_id AND e.type IS NOT NULL AND e.type <> ''
            RETURN DISTINCT e.type AS type
            ORDER BY type
            """,
            {"tenant_id": tenant_id},
        )
    return (
        """
        MATCH (e:Entity)
        WHERE e.type IS NOT NULL AND e.type <> ''
        RETURN DISTINCT e.type AS type
        ORDER BY type
        """,
        {},
    )


def build_source_delete_query(source: str, tenant_id: str | None = None) -> tuple[str, dict[str, Any]]:
    """Build the source delete query."""
    if tenant_id is not None:
        return (
            """
            MATCH (e:Entity {source: $source, tenant_id: $tenant_id})
            DETACH DELETE e
            RETURN count(e) AS deleted
            """,
            {"source": source, "tenant_id": tenant_id},
        )
    return (
        """
        MATCH (e:Entity {source: $source})
        DETACH DELETE e
        RETURN count(e) AS deleted
        """,
        {"source": source},
    )


def build_document_context_query(context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """构建幂等的公司/部门/文档归属合并（MERGE）查询。

    Args:
        context: 文档上下文字典，需含 tenant_id、company_id、department_id、doc_id 等键；缺失键按空值兜底。
    """
    params = {
        "tenant_id": str(context.get("tenant_id") or ""),
        "company_id": str(context.get("company_id") or ""),
        "department_id": str(context.get("department_id") or ""),
        # 部门显示名统一走领域规范化：传入名为空时回退为部门 ID 对应的展示名。
        "department_name": department_display_name(
            str(context.get("department_id") or ""),
            str(context.get("department_name") or ""),
        ),
        "doc_id": str(context.get("doc_id") or ""),
        "display_name": str(
            context.get("display_name") or context.get("uploaded_filename") or ""
        ),
        "uploaded_filename": str(context.get("uploaded_filename") or ""),
        "provenance_source_filename": str(
            context.get("provenance_source_filename") or ""
        ),
        "relative_path": str(context.get("relative_path") or ""),
        "folder_path": str(context.get("folder_path") or ""),
        "authority": str(context.get("authority") or ""),
        "review_status": str(context.get("review_status") or ""),
        "sensitivity": str(context.get("sensitivity") or ""),
        "version": int(context.get("version") or 1),
    }
    return (
        """
        MERGE (company:Company {tenant_id: $tenant_id, company_id: $company_id})
        ON CREATE SET company.name = $company_id
        MERGE (department:Department {
            tenant_id: $tenant_id,
            company_id: $company_id,
            department_id: $department_id
        })
        SET department.name = $department_name
        MERGE (company)-[:HAS_DEPARTMENT]->(department)
        MERGE (document:Document {tenant_id: $tenant_id, doc_id: $doc_id})
        SET document.company_id = $company_id,
            document.department_id = $department_id,
            document.display_name = $display_name,
            document.uploaded_filename = $uploaded_filename,
            document.provenance_source_filename = $provenance_source_filename,
            document.relative_path = $relative_path,
            document.folder_path = $folder_path,
            document.authority = $authority,
            document.review_status = $review_status,
            document.sensitivity = $sensitivity,
            document.version = $version
        MERGE (department)-[:OWNS_DOCUMENT]->(document)
        RETURN document.doc_id AS doc_id
        """,
        params,
    )


def build_department_metadata_sync_query(
    tenant_id: str,
    department_names: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    """构建租户限定的 Department 节点名称批量同步查询（只更新已存在节点）。

    Args:
        tenant_id: 租户 ID，限定只更新该租户的部门节点。
        department_names: 部门 ID 到部门名称的映射；键或值为空的条目会被剔除，并按部门 ID 排序保证批量写入顺序稳定。
    """
    departments = [
        {"department_id": department_id, "name": name}
        for department_id, name in sorted(department_names.items())
        if department_id and name
    ]
    return (
        """
        UNWIND $departments AS item
        MATCH (department:Department {
            tenant_id: $tenant_id,
            department_id: item.department_id
        })
        SET department.name = item.name
        RETURN count(department) AS updated
        """,
        {"tenant_id": tenant_id, "departments": departments},
    )


def build_relation_claim_query(
    *,
    claim_id: str,
    evidence_id: str,
    head: str,
    relation_type: str,
    tail: str,
    confidence: float,
    context: dict[str, Any],
    chunk_id: str,
    extractor: str,
    model: str,
    version: str,
) -> tuple[str, dict[str, Any]]:
    """Build a claim merge with one independently idempotent evidence record."""

    params = {
        "tenant_id": str(context.get("tenant_id") or ""),
        "claim_id": claim_id,
        "evidence_id": evidence_id,
        "head": head,
        "relation_type": relation_type,
        "tail": tail,
        "confidence": float(confidence),
        "doc_id": str(context.get("doc_id") or ""),
        "chunk_id": chunk_id,
        "department_id": str(context.get("department_id") or ""),
        "extractor": extractor,
        "model": model,
        "extractor_version": version,
    }
    return (
        """
        MATCH (head:Entity {tenant_id: $tenant_id, name: $head})
        MATCH (tail:Entity {tenant_id: $tenant_id, name: $tail})
        MATCH (document:Document {tenant_id: $tenant_id, doc_id: $doc_id})
        MERGE (claim:RelationClaim {tenant_id: $tenant_id, claim_id: $claim_id})
        ON CREATE SET claim.head_entity_id = $head,
                      claim.relation_type = $relation_type,
                      claim.tail_entity_id = $tail
        SET claim.confidence = CASE
                WHEN coalesce(claim.confidence, 0.0) < $confidence THEN $confidence
                ELSE claim.confidence
            END
        MERGE (head)-[:CLAIM_HEAD]->(claim)
        MERGE (claim)-[:CLAIM_TAIL]->(tail)
        MERGE (evidence:RelationEvidence {
            tenant_id: $tenant_id,
            evidence_id: $evidence_id
        })
        SET evidence.doc_id = $doc_id,
            evidence.chunk_id = $chunk_id,
            evidence.department_id = $department_id,
            evidence.confidence = $confidence,
            evidence.extractor = $extractor,
            evidence.model = $model,
            evidence.extractor_version = $extractor_version
        MERGE (document)-[:PROVIDES_EVIDENCE]->(evidence)
        MERGE (evidence)-[:SUPPORTS]->(claim)
        RETURN claim.claim_id AS claim_id, evidence.evidence_id AS evidence_id
        """,
        params,
    )


def build_document_evidence_delete_queries(
    doc_id: str,
    tenant_id: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Build ordered Neo4j 5-compatible evidence deletion and verification steps."""

    params = {"tenant_id": tenant_id, "doc_id": doc_id}
    return [
        (
            """
            MATCH (document:Document {tenant_id: $tenant_id, doc_id: $doc_id})
            OPTIONAL MATCH (document)-[:PROVIDES_EVIDENCE]->(evidence:RelationEvidence)
            OPTIONAL MATCH (evidence)-[:SUPPORTS]->(claim:RelationClaim)
            RETURN collect(DISTINCT claim.claim_id) AS claim_ids
            """,
            dict(params),
        ),
        (
            """
            MATCH (:Document {tenant_id: $tenant_id, doc_id: $doc_id})
                  -[:PROVIDES_EVIDENCE]->(evidence:RelationEvidence)
            WITH DISTINCT evidence
            DETACH DELETE evidence
            RETURN count(*) AS deleted_evidence
            """,
            dict(params),
        ),
        (
            """
            MATCH (document:Document {tenant_id: $tenant_id, doc_id: $doc_id})
            WITH DISTINCT document
            DETACH DELETE document
            RETURN count(*) AS deleted_documents
            """,
            dict(params),
        ),
        (
            """
            MATCH (claim:RelationClaim {tenant_id: $tenant_id})
            WHERE claim.claim_id IN $claim_ids
              AND NOT EXISTS {
                  MATCH (:RelationEvidence {tenant_id: $tenant_id})-[:SUPPORTS]->(claim)
              }
            WITH DISTINCT claim
            DETACH DELETE claim
            RETURN count(*) AS deleted_claims
            """,
            {**params, "claim_ids": []},
        ),
        (
            """
            MATCH (department:Department {tenant_id: $tenant_id})
            WHERE NOT EXISTS {
                MATCH (department)-[:OWNS_DOCUMENT]->(:Document {tenant_id: $tenant_id})
            }
            WITH DISTINCT department
            DETACH DELETE department
            RETURN count(*) AS deleted_departments
            """,
            dict(params),
        ),
        (
            """
            MATCH (company:Company {tenant_id: $tenant_id})
            WHERE NOT EXISTS {
                MATCH (company)-[:HAS_DEPARTMENT]->(:Department {tenant_id: $tenant_id})
            }
            WITH DISTINCT company
            DETACH DELETE company
            RETURN count(*) AS deleted_companies
            """,
            dict(params),
        ),
        (
            """
            OPTIONAL MATCH (document:Document {tenant_id: $tenant_id, doc_id: $doc_id})
            WITH count(document) AS documents
            OPTIONAL MATCH (evidence:RelationEvidence {tenant_id: $tenant_id, doc_id: $doc_id})
            WITH documents, count(evidence) AS evidences
            OPTIONAL MATCH (department:Department {tenant_id: $tenant_id})
            WHERE NOT EXISTS {
                MATCH (department)-[:OWNS_DOCUMENT]->(:Document {tenant_id: $tenant_id})
            }
            WITH documents, evidences, count(department) AS orphan_departments
            OPTIONAL MATCH (company:Company {tenant_id: $tenant_id})
            WHERE NOT EXISTS {
                MATCH (company)-[:HAS_DEPARTMENT]->(:Department {tenant_id: $tenant_id})
            }
            RETURN documents,
                   evidences,
                   orphan_departments,
                   count(company) AS orphan_companies
            """,
            dict(params),
        ),
    ]


def build_company_overview_query(tenant_id: str) -> tuple[str, dict[str, Any]]:
    """Build a tenant-scoped company overview with deterministic department counts."""

    return (
        """
        MATCH (company:Company {tenant_id: $tenant_id})
        MATCH (company)-[:HAS_DEPARTMENT]->(department:Department)
        MATCH (department)-[:OWNS_DOCUMENT]->(document:Document)
        WITH company, department, count(DISTINCT document) AS document_count
        ORDER BY department.name
        RETURN company.company_id AS company_id,
               company.name AS company_name,
               collect(CASE WHEN department IS NULL THEN NULL ELSE {
                   department_id: department.department_id,
                   name: department.name,
                   document_count: document_count
               } END) AS departments
        """,
        {"tenant_id": tenant_id},
    )


def build_department_graph_query(
    tenant_id: str,
    department_id: str,
    *,
    include_cross_department: bool,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    """Build a department-owned graph projection with optional cross-department evidence."""

    bounded_limit = max(1, min(limit, 500))
    cross_clause = "" if include_cross_department else "AND evidence.department_id = $department_id"
    return (
        f"""
        MATCH (department:Department {{tenant_id: $tenant_id, department_id: $department_id}})
        OPTIONAL MATCH (department)-[:OWNS_DOCUMENT]->(document:Document)
        OPTIONAL MATCH (document)-[:PROVIDES_EVIDENCE]->(evidence:RelationEvidence)-[:SUPPORTS]->(claim:RelationClaim)
        WHERE evidence IS NULL OR (evidence.tenant_id = $tenant_id {cross_clause})
        OPTIONAL MATCH (head:Entity {{tenant_id: $tenant_id}})-[:CLAIM_HEAD]->(claim)-[:CLAIM_TAIL]->(tail:Entity {{tenant_id: $tenant_id}})
        RETURN department, document, evidence, claim, head, tail
        LIMIT $limit
        """,
        {
            "tenant_id": tenant_id,
            "department_id": department_id,
            "limit": bounded_limit,
        },
    )


def build_paths_query(
    *,
    tenant_id: str,
    from_id: str,
    to_id: str,
    department_id: str | None,
    include_cross_department: bool,
    max_hops: int,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    """Build a bounded tenant-safe path query whose results retain evidence nodes."""

    bounded_hops = max(1, min(max_hops, 6))
    bounded_limit = max(1, min(limit, 50))
    department_clause = ""
    if department_id and not include_cross_department:
        department_clause = (
            "AND all(n IN nodes(path) WHERE n.department_id IS NULL "
            "OR n.department_id = $department_id)"
        )
    return (
        f"""
        MATCH (start {{tenant_id: $tenant_id}}), (target {{tenant_id: $tenant_id}})
        WHERE coalesce(start.doc_id, start.department_id, start.claim_id, start.name) = $from_id
          AND coalesce(target.doc_id, target.department_id, target.claim_id, target.name) = $to_id
        MATCH path = (start)-[*1..{bounded_hops}]-(target)
        WHERE all(node IN nodes(path) WHERE node.tenant_id = $tenant_id)
          {department_clause}
        RETURN [node IN nodes(path) | properties(node)] AS nodes,
               [relationship IN relationships(path) | type(relationship)] AS relations
        LIMIT $limit
        """,
        {
            "tenant_id": tenant_id,
            "from_id": from_id,
            "to_id": to_id,
            "department_id": department_id or "",
            "limit": bounded_limit,
        },
    )


def build_claim_evidence_query(
    claim_id: str,
    tenant_id: str,
) -> tuple[str, dict[str, Any]]:
    """Build tenant-scoped evidence detail for one semantic relation claim."""

    return (
        """
        MATCH (claim:RelationClaim {tenant_id: $tenant_id, claim_id: $claim_id})
        MATCH (evidence:RelationEvidence {tenant_id: $tenant_id})-[:SUPPORTS]->(claim)
        MATCH (document:Document {tenant_id: $tenant_id, doc_id: evidence.doc_id})
        RETURN evidence.evidence_id AS evidence_id,
               evidence.doc_id AS doc_id,
               evidence.chunk_id AS chunk_id,
               evidence.department_id AS department_id,
               evidence.confidence AS confidence,
               evidence.extractor AS extractor,
               evidence.model AS model,
               evidence.extractor_version AS extractor_version,
               document.display_name AS display_name,
               document.provenance_source_filename AS provenance_source_filename,
               document.relative_path AS relative_path
        ORDER BY evidence.department_id, document.display_name, evidence.chunk_id
        """,
        {"tenant_id": tenant_id, "claim_id": claim_id},
    )


def build_department_relation_summary_query(
    tenant_id: str,
) -> tuple[str, dict[str, Any]]:
    """构建跨部门聚合边查询：汇总被多部门证据共同支撑的语义关系断言。

    Args:
        tenant_id: 租户 ID，所有 MATCH 均绑定该租户。
    """
    # 返回中的 relation_types/relation_details 聚合每对部门之间的断言类型
    # 与头尾实体明细（最多 8 条），供概览图展示关系构成。

    return (
        """
        MATCH (from_department:Department {tenant_id: $tenant_id})-[:OWNS_DOCUMENT]->
              (from_document:Document {tenant_id: $tenant_id})-[:PROVIDES_EVIDENCE]->
              (from_evidence:RelationEvidence {tenant_id: $tenant_id})-[:SUPPORTS]->
              (claim:RelationClaim {tenant_id: $tenant_id})
        MATCH (to_department:Department {tenant_id: $tenant_id})-[:OWNS_DOCUMENT]->
              (to_document:Document {tenant_id: $tenant_id})-[:PROVIDES_EVIDENCE]->
              (to_evidence:RelationEvidence {tenant_id: $tenant_id})-[:SUPPORTS]->(claim)
        MATCH (head:Entity {tenant_id: $tenant_id})-[:CLAIM_HEAD]->(claim)-[:CLAIM_TAIL]->
              (tail:Entity {tenant_id: $tenant_id})
        WHERE from_department.department_id < to_department.department_id
        RETURN from_department.department_id AS source_department_id,
               to_department.department_id AS target_department_id,
               count(DISTINCT claim) AS path_count,
               collect(DISTINCT claim.claim_id) AS claim_ids,
               collect(DISTINCT claim.relation_type) AS relation_types,
               collect(DISTINCT {
                   claim_id: claim.claim_id,
                   relation_type: claim.relation_type,
                   head_name: head.name,
                   head_type: head.type,
                   tail_name: tail.name,
                   tail_type: tail.type
               })[..8] AS relation_details
        ORDER BY source_department_id, target_department_id
        """,
        {"tenant_id": tenant_id},
    )


def build_entity_update_query(
    *,
    tenant_id: str,
    node_id: str,
    label: str | None,
    entity_type: str | None,
    description: str | None,
) -> tuple[str, dict[str, Any]]:
    """构建租户限定的实体元数据更新查询（空值字段保持原值不变）。

    Args:
        tenant_id: 租户 ID，限定只能更新该租户的实体。
        node_id: 实体节点 ID（形如 "entity:<名称>"），去掉前缀后作为匹配名称。
        label: 新实体名称；为 None 或去空白后为空时保持原名称。
        entity_type: 新实体类型；为 None 或去空白后为空时保持原类型。
        description: 新实体描述；为 None 时保持原描述，空字符串视为有意清空。
    """
    entity_name = node_id.removeprefix("entity:")
    return (
        """
        MATCH (entity:Entity {tenant_id: $tenant_id, name: $entity_name})
        SET entity.name = CASE WHEN $label <> '' THEN $label ELSE entity.name END,
            entity.type = CASE WHEN $entity_type <> '' THEN $entity_type ELSE entity.type END,
            entity.description = CASE WHEN $description IS NOT NULL THEN $description ELSE entity.description END,
            entity.updated_at = timestamp()
        RETURN {
            id: 'entity:' + entity.name,
            label: entity.name,
            type: coalesce(entity.type, 'Entity'),
            description: coalesce(entity.description, '')
        } AS node
        """,
        {
            "tenant_id": tenant_id,
            "entity_name": entity_name,
            "label": (label or "").strip(),
            "entity_type": (entity_type or "").strip(),
            "description": description.strip() if description is not None else None,
        },
    )


def build_relation_claim_update_query(
    *,
    tenant_id: str,
    claim_id: str,
    relation_type: str,
) -> tuple[str, dict[str, Any]]:
    """构建租户限定的语义关系断言类型更新查询。

    Args:
        tenant_id: 租户 ID，限定只能更新该租户的断言。
        claim_id: 待更新的 RelationClaim 断言 ID。
        relation_type: 规范化后的新关系类型，写入前要求调用方先通过 normalize_relation_type 校验。
    """
    return (
        """
        MATCH (claim:RelationClaim {tenant_id: $tenant_id, claim_id: $claim_id})
        MATCH (evidence:RelationEvidence {tenant_id: $tenant_id})-[:SUPPORTS]->(claim)
        WITH DISTINCT claim
        SET claim.relation_type = $relation_type,
            claim.updated_at = timestamp()
        RETURN claim.claim_id AS claim_id, claim.relation_type AS relation_type
        """,
        {"tenant_id": tenant_id, "claim_id": claim_id, "relation_type": relation_type},
    )
