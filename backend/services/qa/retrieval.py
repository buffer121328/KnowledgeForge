"""Vector and graph retrieval collaborators for the QA facade."""

from __future__ import annotations

from typing import Any

import pybreaker

from domain.knowledge import RetrievedContext
from domain.retrieval import RetrievalOutcome
from infrastructure.audit.log import AuditAction, get_audit_service
from shared.utils.circuit_breaker import (
    call_with_fallback,
    is_transient_dependency_error,
    knowledge_graph_breaker,
    vector_store_breaker,
)
from shared.utils.dependency_resilience import execute_with_policy, get_dependency_policy
from shared.utils.logging import get_logger

logger = get_logger(__name__)


async def vector_retrieve(
    vector_store: Any,
    rewritten: dict,
    tenant_id: str | None = None,
    user_id: str = "",
    question_fingerprint: str = "",
    visible_department_ids: tuple[str, ...] | list[str] | None = None,
    top_k: int = 8,
) -> RetrievalOutcome:
    """执行向量检索，按租户与可见部门过滤并封装为检索结果。

    Args:
        vector_store: 向量库服务实例；为空时直接返回空结果。
        rewritten: 改写后的查询结果，取其 queries（最多前 2 条）作为检索输入。
        tenant_id: 租户 ID；为 None 表示不过滤租户（管理员场景）。
        user_id: 发起提问的用户 ID，用于熔断安全审计事件的归属。
        question_fingerprint: 问题指纹，用于审计事件关联。
        visible_department_ids: 用户可见的部门 ID 集合；None 表示不过滤，非 None（含空集合）表示严格按该集合过滤。
        top_k: 每条查询返回的候选块数上限；广义"范围/目录"类问题需要更深的候选池。
    """
    if not vector_store:
        return RetrievalOutcome(contexts=[])

    contexts: list[RetrievedContext] = []
    attempts = 0
    failed_attempts = 0
    breaker_event_recorded = False
    # 多主题改写会产生多条子查询（原句 + 实体串 + 每主题一条）。每条查询
    # 独立 top_k 召回后由 RRF 统一融合去重，扩大次要主题的可见性。
    for query in rewritten.get("queries", [])[:4]:
        attempts += 1
        try:
            results = await execute_with_policy(
                get_dependency_policy("vector"),
                lambda: call_with_fallback(
                    vector_store_breaker,
                    vector_store.search,
                    None,
                    query,
                    top_k,
                    tenant_id,
                    visible_department_ids=visible_department_ids,
                ),
                retry_predicate=is_transient_dependency_error,
            )
            for document, score in results or []:
                metadata = dict(document.get("metadata") or {})
                source_document_id = (
                    document.get("source_document_id")
                    or metadata.get("source_document_id")
                    or metadata.get("doc_id")
                )
                if isinstance(source_document_id, str) and source_document_id.strip():
                    metadata["source_document_id"] = source_document_id
                contexts.append(
                    RetrievedContext(
                        content=document.get("content", ""),
                        source=document.get("source", "vector_store"),
                        score=float(score),
                        retrieval_type="vector",
                        metadata=metadata,
                    )
                )
        except pybreaker.CircuitBreakerError:
            failed_attempts += 1
            if not breaker_event_recorded:
                breaker_event_recorded = True
                try:
                    get_audit_service().log_security_event(
                        user_id=user_id or "system",
                        org_id=tenant_id or "",
                        action=AuditAction.BREAKER_OPEN,
                        reason_code="vector_store_unavailable",
                        metadata={
                            "dependency": "vector_store",
                            "question_fingerprint": question_fingerprint,
                        },
                    )
                except Exception:
                    logger.warning("security_event_audit_failed", security_event="breaker_open")
            logger.warning("vector_store_breaker_open")
        except Exception as error:
            failed_attempts += 1
            logger.warning("vector_retrieve_failed", error_type=type(error).__name__)

    return RetrievalOutcome(contexts=contexts, unavailable=attempts > 0 and failed_attempts == attempts)


async def graph_retrieve(
    knowledge_graph: Any,
    question: str,
    rewritten: dict,
    tenant_id: str | None = None,
    user_id: str = "",
    question_fingerprint: str = "",
    visible_department_ids: tuple[str, ...] | list[str] | None = None,
) -> RetrievalOutcome:
    """Retrieve tenant-safe relation claims backed by visible document evidence."""
    if not knowledge_graph or not tenant_id:
        return RetrievalOutcome(contexts=[])

    contexts: list[RetrievedContext] = []
    seen_claim_ids: set[str] = set()
    keywords = rewritten.get("entities") or rewritten.get("keywords") or []
    if not keywords:
        keywords = [question[:32]]

    attempts = 0
    failed_attempts = 0
    breaker_event_recorded = False
    for keyword in keywords[:3]:
        attempts += 1
        try:
            claims = await execute_with_policy(
                get_dependency_policy("graph"),
                lambda: call_with_fallback(
                    knowledge_graph_breaker,
                    knowledge_graph.search_claim_contexts,
                    None,
                    keyword,
                    tenant_id=tenant_id,
                    visible_department_ids=visible_department_ids,
                    limit=8,
                    evidence_limit=5,
                ),
                retry_predicate=is_transient_dependency_error,
            )
        except pybreaker.CircuitBreakerError:
            failed_attempts += 1
            if not breaker_event_recorded:
                breaker_event_recorded = True
                try:
                    get_audit_service().log_security_event(
                        user_id=user_id or "system",
                        org_id=tenant_id,
                        action=AuditAction.BREAKER_OPEN,
                        reason_code="knowledge_graph_unavailable",
                        metadata={
                            "dependency": "knowledge_graph",
                            "question_fingerprint": question_fingerprint,
                        },
                    )
                except Exception:
                    logger.warning("security_event_audit_failed", security_event="breaker_open")
            logger.warning("knowledge_graph_breaker_open")
            continue
        except Exception as error:
            failed_attempts += 1
            logger.warning("graph_claim_search_failed", error_type=type(error).__name__)
            continue

        for claim in claims or []:
            claim_id = str(claim.get("claim_id") or "")
            evidences = [item for item in (claim.get("evidences") or []) if item]
            if not claim_id or claim_id in seen_claim_ids or not evidences:
                continue
            seen_claim_ids.add(claim_id)

            evidence_count = max(int(claim.get("evidence_count") or 0), len(evidences))
            sources: list[str] = []
            doc_ids: list[str] = []
            chunk_ids: list[str] = []
            department_ids: list[str] = []
            evidence_ids: list[str] = []
            source_labels: list[str] = []
            for evidence in evidences:
                source = str(evidence.get("source") or evidence.get("doc_id") or "").strip()
                doc_id = str(evidence.get("doc_id") or "").strip()
                chunk_id = str(evidence.get("chunk_id") or "").strip()
                department_id = str(evidence.get("department_id") or "").strip()
                evidence_id = str(evidence.get("evidence_id") or "").strip()
                if source and source not in sources:
                    sources.append(source)
                if doc_id and doc_id not in doc_ids:
                    doc_ids.append(doc_id)
                if chunk_id and chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)
                if department_id and department_id not in department_ids:
                    department_ids.append(department_id)
                if evidence_id and evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
                if source:
                    source_labels.append(f"{source}#{chunk_id}" if chunk_id else source)

            confidence = max(0.0, min(float(claim.get("claim_confidence") or 0.0), 1.0))
            evidence_coverage = min(evidence_count / 3.0, 1.0)
            score = min(1.0, 0.3 + (0.5 * confidence) + (0.2 * evidence_coverage))
            head_name = str(claim.get("head_name") or "")
            tail_name = str(claim.get("tail_name") or "")
            relation_type = str(claim.get("relation_type") or "")
            common_metadata = {
                **claim,
                "sources": sources,
                "doc_ids": doc_ids,
                "chunk_ids": chunk_ids,
                "department_ids": department_ids,
                "evidence_ids": evidence_ids,
                "evidence_count": evidence_count,
                "graph_score_components": {
                    "entity_match": 1.0,
                    "claim_confidence": confidence,
                    "evidence_coverage": evidence_coverage,
                },
            }
            for evidence in evidences:
                source = str(evidence.get("source") or evidence.get("doc_id") or "").strip()
                doc_id = str(evidence.get("doc_id") or "").strip()
                chunk_id = str(evidence.get("chunk_id") or "").strip()
                if not source or not doc_id or not chunk_id:
                    continue
                chunk_index = evidence.get("chunk_index")
                if chunk_index is None and "#chunk-" in chunk_id:
                    try:
                        chunk_index = int(chunk_id.rsplit("#chunk-", 1)[1])
                    except ValueError:
                        chunk_index = None
                metadata = {
                    **common_metadata,
                    "source": source,
                    "doc_id": doc_id,
                    "source_document_id": doc_id,
                    "chunk_id": chunk_id,
                    "department_id": str(evidence.get("department_id") or "").strip(),
                    "evidence_id": str(evidence.get("evidence_id") or "").strip(),
                    "tenant_id": tenant_id,
                }
                if isinstance(chunk_index, int) and chunk_index >= 0:
                    metadata["chunk_index"] = chunk_index
                contexts.append(
                    RetrievedContext(
                        content=(
                            f"关系: {head_name} -[{relation_type}]-> {tail_name}\n"
                            f"证据: {evidence_count} 条\n"
                            f"来源: {'；'.join(source_labels)}"
                        ),
                        source=source,
                        score=score,
                        retrieval_type="graph",
                        metadata=metadata,
                    )
                )

    return RetrievalOutcome(contexts=contexts, unavailable=attempts > 0 and failed_attempts == attempts)
