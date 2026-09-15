"""QA 证据缓存治理：读写、资格判定与序列化（自 qa_agent 拆分）。

函数以协作对象（``QAAgent`` 实例）为第一参数；缓存语义与原方法保持一致。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from domain.evidence import (
    QAResponseStatus,
    evidence_assessment_from_dict,
    evidence_assessment_to_dict,
    grounding_result_from_dict,
    grounding_result_to_dict,
    structured_answer_from_dict,
    structured_answer_to_dict,
)
from domain.knowledge import QAResult, QueryIntent, RetrievedContext
from infrastructure.evidence_gate_configuration import get_effective_evidence_gate_configuration
from services.qa.generation import no_context_answer
from shared.config import settings
from shared.utils.logging import get_logger

logger = get_logger(__name__)

_INSUFFICIENT_EVIDENCE = "insufficient_verified_evidence"


async def check_cache(
agent,
    question: str,
    user_id: str,
    retrieval_mode: str,
    tenant_id: str,
    knowledge_revision: int,
) -> dict | None:
    """Check the cache."""
    try:
        if agent.revision_store is None:
            return await agent.qa_cache.get(
                question,
                user_id,
                retrieval_mode,
                tenant_id,
            )
        return await agent.qa_cache.get(
            question,
            user_id,
            retrieval_mode,
            tenant_id,
            knowledge_revision,
        )
    except Exception as error:
        logger.warning("qa_cache_check_failed", error_type=type(error).__name__)
        return None

async def safe_cached_result(
agent,
    cached: dict,
    tenant_id: str | None,
    gate_mode: str | None = None,
) -> QAResult | None:
    """Revalidate legacy cache data before it can bypass current safety checks."""

    try:
        contexts = [
            context
            if isinstance(context, RetrievedContext)
            else RetrievedContext(
                content=context.get("content", ""),
                source=context.get("source", ""),
                score=float(context.get("score", 0)),
                retrieval_type=context.get("retrieval_type", context.get("type", "vector")),
                metadata=context.get("metadata") or {},
            )
            for context in cached.get("contexts") or []
        ]
        context_result = await agent.safety_pipeline.filter_contexts(
            contexts,
            tenant_id=tenant_id,
        )
        answer_result = await agent.safety_pipeline.sanitize_answer(
            str(cached.get("answer") or "")
        )
        if context_result.actions or answer_result.actions:
            return None
        serialized_assessment = cached.get("evidence_assessment")
        if (gate_mode or agent._effective_gate_configuration().mode) == "enforce" and not serialized_assessment:
            return None
        evidence_assessment = (
            evidence_assessment_from_dict(serialized_assessment)
            if serialized_assessment
            else None
        )
        structured_answer = (
            structured_answer_from_dict(cached["structured_answer"])
            if cached.get("structured_answer")
            else None
        )
        grounding_result = (
            grounding_result_from_dict(cached["grounding_result"])
            if cached.get("grounding_result")
            else None
        )
        intent_value = cached.get("intent", QueryIntent.FACTOID)
        return QAResult(
            question=str(cached.get("question") or ""),
            answer=answer_result.answer,
            contexts=contexts,
            intent=intent_value if isinstance(intent_value, QueryIntent) else QueryIntent(intent_value),
            confidence=float(cached.get("confidence", 0)),
            reasoning_steps=list(cached.get("reasoning_steps") or []),
            degradation_code=cached.get("degradation_code"),
            security_actions=list(cached.get("security_actions") or []),
            response_status=(
                QAResponseStatus(cached["response_status"])
                if cached.get("response_status")
                else evidence_assessment.response_status
                if evidence_assessment is not None
                else None
            ),
            evidence_assessment=evidence_assessment,
            structured_answer=structured_answer,
            grounding_result=grounding_result,
        )
    except (TypeError, ValueError, KeyError):
        return None

async def write_cache(
agent,
    question: str,
    user_id: str,
    result: QAResult,
    retrieval_mode: str,
    tenant_id: str,
    knowledge_revision: int,
) -> None:
    """Write the cache."""
    try:
        if agent.revision_store is None:
            await agent.qa_cache.set(
                question,
                user_id,
                agent._serialize_cache_result(result),
                retrieval_mode,
                tenant_id,
            )
            return
        await agent.qa_cache.set(question, user_id, agent._serialize_cache_result(result),
            retrieval_mode, tenant_id, knowledge_revision)
    except Exception as error:
        logger.warning("qa_cache_write_failed", error_type=type(error).__name__)

def serialize_cache_result(result: QAResult) -> dict[str, Any]:
    """Serialize only the safe answer contract, excluding trace internals."""
    return {
            "question": result.question,
            "answer": result.answer,
            "contexts": [
                {
                    "content": context.content,
                    "source": context.source,
                    "score": context.score,
                    "retrieval_type": context.retrieval_type,
                    "metadata": context.metadata,
                }
                for context in result.contexts
            ],
            "intent": result.intent.value,
            "confidence": result.confidence,
            "reasoning_steps": result.reasoning_steps,
            "degradation_code": result.degradation_code,
            "security_actions": result.security_actions,
            "response_status": (
                result.response_status.value if result.response_status else None
            ),
            "evidence_assessment": evidence_assessment_to_dict(
                result.evidence_assessment
            ),
            "structured_answer": structured_answer_to_dict(result.structured_answer),
            "grounding_result": grounding_result_to_dict(result.grounding_result),
            "cached_at": datetime.now(UTC).isoformat(),
        }

def evidence_degradation_code(status: QAResponseStatus) -> str:
    """Map evidence routes to stable bounded degradation codes."""
    return {
        QAResponseStatus.CONFLICTING_EVIDENCE: "conflicting_evidence",
        QAResponseStatus.HUMAN_REVIEW_REQUIRED: "human_review_required",
        QAResponseStatus.NEEDS_CLARIFICATION: "needs_clarification",
        QAResponseStatus.SOURCE_UNAVAILABLE: "retrieval_source_unavailable",
    }.get(status, _INSUFFICIENT_EVIDENCE)

def evidence_controlled_answer(
    status: QAResponseStatus,
    intent: QueryIntent,
) -> tuple[str, list[str]]:
    """Return a safe user-facing response without generation."""
    messages = {
        QAResponseStatus.CONFLICTING_EVIDENCE: "当前资料存在相互冲突的证据，无法自动给出确定结论。",
        QAResponseStatus.HUMAN_REVIEW_REQUIRED: "当前问题需要人工审核后才能回答。",
        QAResponseStatus.NEEDS_CLARIFICATION: "当前问题缺少必要信息，请补充版本、日期或具体对象。",
        QAResponseStatus.SOURCE_UNAVAILABLE: "当前知识检索服务不可用，请稍后重试。",
    }
    if status in messages:
        return messages[status], ["证据资格门禁停止了自动生成。"]
    return no_context_answer(intent)

def effective_gate_configuration(agent) -> Any:
    """Return the composed company-global configuration or a bounded test override."""
    return (
        agent.gate_configuration_provider()
        if agent.gate_configuration_provider is not None
        else get_effective_evidence_gate_configuration()
    )

def is_cache_eligible(agent, result: QAResult, *, user_id: str, gate_mode: str | None = None) -> bool:
    """Apply legacy-compatible or enforced grounded cache eligibility."""
    base_eligible = bool(
        agent.qa_cache
        and user_id
        and result.contexts
        and result.degradation_code is None
        and not result.security_actions
        and result.confidence >= settings.qa_cache_min_confidence
    )
    if not base_eligible:
        return False
    if (gate_mode or effective_gate_configuration(agent).mode) != "enforce":
        return True
    return bool(
        result.response_status
        in {QAResponseStatus.ANSWERED, QAResponseStatus.PARTIALLY_ANSWERED}
        and result.structured_answer is not None
        and result.grounding_result is not None
        and result.grounding_result.passed
    )

def cache_mode_id(strategy: Any, *, gate_configuration: Any | None = None) -> str:
    """Build a stable identity for every policy that can change an answer."""
    retrieval_id = strategy.value
    if strategy.uses_bm25:
        retrieval_id = (
            f"{strategy.value}@{settings.qa_bm25_schema_version}:"
            f"{settings.qa_bm25_tokenizer_version}:"
            f"k1={settings.qa_bm25_k1}:b={settings.qa_bm25_b}:"
            f"rrf={settings.qa_rrf_k}:w={settings.qa_rrf_bm25_weight}"
        )
    configuration = gate_configuration or get_effective_evidence_gate_configuration()
    return (
        f"{retrieval_id}|{configuration.cache_identity}"
        f"|evidence={settings.qa_evidence_policy_version}"
        f"|calibration={configuration.calibration_version}"
        f"|budget={settings.qa_evidence_candidate_budget_version}"
        f"|reranker={settings.qa_cross_encoder_mode}:{settings.qa_cross_encoder_version}"
        f"|answer={settings.qa_structured_answer_schema_version}"
        f"|grounding={settings.qa_grounding_policy_version}"
    )

