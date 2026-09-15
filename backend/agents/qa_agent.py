"""Public QA orchestration facade for local query, retrieval, ranking, and LLM helpers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any

from services.evidence.qualification import (
    EvidenceQualifier,
    context_identity,
)
from services.qa.generation import (
    ANSWER_PROMPT,
    StructuredGenerationError,
    StructuredGenerationResult,
    build_answer_messages,
    call_llm_with_breaker,
    generate_structured_answer,
    no_context_answer,
)
from services.qa.grounding import GroundingVerifier
from services.qa.query import INTENT_RULES as _INTENT_RULES
from services.qa.query import classify_intent_local, rewrite_query_local
from services.qa.ranking import calc_confidence
from services.qa.retrievers import (
    ClaimAwareGraphRetriever,
    HybridRetrievalOrchestrator,
    NativeBM25Retriever,
    NativeDenseRetriever,
)
from services.safety.qa_checks import question_fingerprint
from domain.evidence import (
    EvidenceReasonCode,
    EvidenceState,
    QAResponseStatus,
)
from domain.trace import TraceRecorder
from services.safety.pipeline import QASafetyPipeline
from domain.retrieval import (
    CandidateStageRecorderPort,
    RetrievalOutcome,
    RetrievalRequest,
    RetrievalScope,
    RetrievalStatus,
    resolve_retrieval_strategy,
)
from agents import qa_cache
from domain.knowledge import QAResult, QueryIntent, RetrievedContext
from infrastructure.retrieval.cross_encoder import RerankStatus, SafeCrossEncoderReranker
from infrastructure.cache.semantic import (
    SemanticCacheMode,
    SemanticConfirmationRequired,
)
from shared.config import settings
from shared.utils.logging import get_logger
from shared.utils.metrics import (
    qa_cache_hits_total,
    qa_confidence_score,
    qa_cross_encoder_latency_seconds,
    qa_cross_encoder_results_total,
    qa_evidence_decisions_total,
    qa_grounding_latency_seconds,
    qa_grounding_results_total,
    qa_query_total,
    qa_retrieval_branch_total,
    qa_structured_generation_latency_seconds,
    qa_structured_generation_total,
)

logger = get_logger(__name__)

_INSUFFICIENT_EVIDENCE = "insufficient_verified_evidence"
_GRAPH_UNAVAILABLE = "graph_retrieval_unavailable"
_VECTOR_UNAVAILABLE = "vector_retrieval_unavailable"
_BM25_UNAVAILABLE = "bm25_retrieval_unavailable"
_BM25_CONTRACT_ERROR = "bm25_retrieval_contract_error"
_CONFLICTING_EVIDENCE_NOTICE = "conflicting_evidence"
_STRUCTURED_OUTPUT_INVALID = EvidenceReasonCode.STRUCTURED_OUTPUT_INVALID.value
_GROUNDING_FAILED = "grounding_failed"


@dataclass(frozen=True, slots=True)
class QAAgentDependencies:
    """Explicit runtime collaborators assembled by the workflow composition root."""

    llm: Any
    safety_pipeline: QASafetyPipeline
    cross_encoder_reranker: SafeCrossEncoderReranker
    evidence_qualifier: EvidenceQualifier
    grounding_verifier: GroundingVerifier



class QAAgent:
    """Answer questions while preserving cache, tenant, resilience, and result contracts."""

    def __init__(
        self,
        *,
        dependencies: QAAgentDependencies,
        vector_store: Any = None,
        knowledge_graph: Any = None,
        sparse_index: Any = None,
        qa_cache: Any = None,
        revision_store: Any = None,
        semantic_cache: Any = None,
        confirmation_service: Any = None,
        gate_configuration_provider: Any = None,
    ) -> None:
        """Initialize the QA facade from explicitly assembled collaborators."""
        self.llm = dependencies.llm
        self.safety_pipeline = dependencies.safety_pipeline
        self.cross_encoder_reranker = dependencies.cross_encoder_reranker
        self.evidence_qualifier = dependencies.evidence_qualifier
        self.grounding_verifier = dependencies.grounding_verifier

        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.sparse_index = sparse_index
        self.qa_cache = qa_cache
        self.revision_store = revision_store
        self.semantic_cache = semantic_cache
        self.confirmation_service = confirmation_service
        self.gate_configuration_provider = gate_configuration_provider

    async def answer(
        self,
        question: str,
        tenant_id: str | None = None,
        user_id: str = "",
        retrieval_mode: str = "hybrid",
        visible_department_ids: tuple[str, ...] | list[str] | None = None,
        semantic_bypass_token: str | None = None,
        semantic_confirmation_token: str | None = None,
        evaluation_candidate_recorder: CandidateStageRecorderPort | None = None,
    ) -> QAResult | SemanticConfirmationRequired:
        """生成所给问题的回答，并记录完整执行轨迹。

        Args:
            question: 用户提出的问题文本。
            tenant_id: 租户 ID；为 None 表示不过滤租户（管理员场景）。
            user_id: 发起提问的用户 ID，用于作用域隔离与语义确认绑定。
            retrieval_mode: 检索模式（如 "hybrid"、"vector"），缺省为 hybrid。
            visible_department_ids: 用户可见的部门 ID 集合；None 表示不做部门过滤（管理员），非 None（含空集合）表示严格按该集合过滤。
            semantic_bypass_token: 语义确认绕过令牌；提供且有效时跳过确认流程。
            semantic_confirmation_token: 语义确认令牌；提供且校验通过时直接取回已确认的结果。
        """
        recorder = TraceRecorder()
        try:
            return await self._answer_with_trace(
                question,
                tenant_id=tenant_id,
                user_id=user_id,
                retrieval_mode=retrieval_mode,
                visible_department_ids=visible_department_ids,
                semantic_bypass_token=semantic_bypass_token,
                semantic_confirmation_token=semantic_confirmation_token,
                evaluation_candidate_recorder=evaluation_candidate_recorder,
                recorder=recorder,
            )
        except Exception as error:
            recorder.record(
                "qa.failed",
                status="failed",
                exception_type=type(error).__name__,
            )
            raise

    async def _answer_with_trace(
        self,
        question: str,
        *,
        tenant_id: str | None,
        user_id: str,
        retrieval_mode: str,
        visible_department_ids: tuple[str, ...] | list[str] | None,
        semantic_bypass_token: str | None,
        semantic_confirmation_token: str | None,
        evaluation_candidate_recorder: CandidateStageRecorderPort | None,
        recorder: TraceRecorder,
    ) -> QAResult | SemanticConfirmationRequired:
        """执行问答主流程，同时记录请求级事件轨迹。

        Args:
            question: 用户提出的问题文本。
            tenant_id: 租户 ID；为 None 表示不过滤租户（管理员场景）。
            user_id: 发起提问的用户 ID。
            retrieval_mode: 请求指定的检索模式。
            visible_department_ids: 用户可见的部门 ID 集合；None 表示不过滤，非 None（含空集合）表示严格按该集合过滤。
            semantic_bypass_token: 语义确认绕过令牌，可为 None。
            semantic_confirmation_token: 语义确认令牌，可为 None。
            recorder: 请求级轨迹记录器，负责收集各阶段事件。
        """
        effective_strategy = resolve_retrieval_strategy(
            retrieval_mode, settings.qa_retrieval_strategy
        )
        strategy_id = effective_strategy.value
        gate_configuration = self._effective_gate_configuration()
        cache_mode_id = self._cache_mode_id(effective_strategy, gate_configuration=gate_configuration)

        started = time.time()
        if tenant_id == "":
            tenant_id = None
        effective_tenant = tenant_id or ""
        knowledge_revision = (
            await self.revision_store.get(effective_tenant)
            if self.revision_store is not None and effective_tenant
            else 0
        )
        pre_retrieval_security_actions = await recorder.capture_async(
            "safety.validate_question",
            lambda: self.safety_pipeline.validate_question(question),
        )
        question_id = question_fingerprint(question)

        # 语义缓存作用域与 SemanticQACache / SemanticConfirmationService 的五元组契约一致；
        # 部门可见性随请求由身份服务推导，缓存与令牌按 user_scope 绑定到具体用户。
        confirmation_scope = {
            "question": question,
            "tenant_id": effective_tenant,
            "user_scope": user_id,
            "retrieval_mode": cache_mode_id,
            "knowledge_revision": knowledge_revision,
        }
        if semantic_confirmation_token and self.confirmation_service is not None:
            payload = await self.confirmation_service.consume_confirmation(
                semantic_confirmation_token,
                **confirmation_scope,
            )
            if payload is not None:
                confirmed = await self._safe_cached_result(
                    payload.get("cached_result") or {}, tenant_id=tenant_id,
                    gate_mode=gate_configuration.mode,
                )
                if confirmed is not None:
                    return replace(
                        confirmed,
                        trace=recorder.build(),
                        cache_hit_type="semantic_confirmed",
                        knowledge_revision=knowledge_revision,
                    )
            # Invalid/expired/unsafe confirmations fall through once to full RAG.
        force_full_rag = (
            semantic_confirmation_token is not None
            or evaluation_candidate_recorder is not None
        )
        if semantic_bypass_token and self.confirmation_service is not None:
            force_full_rag = await self.confirmation_service.consume_bypass(
                semantic_bypass_token,
                **confirmation_scope,
            ) or force_full_rag

        if self.qa_cache and user_id and not force_full_rag:
            cached = await recorder.capture_async(
                "cache.lookup",
                lambda: self._check_cache(
                    question,
                    user_id,
                    cache_mode_id,
                    tenant_id=effective_tenant,
                    knowledge_revision=knowledge_revision,
                ),
                metadata={"retrieval_mode": strategy_id},
            )
            if cached is not None:
                cached_result = await self._safe_cached_result(
                    cached,
                    tenant_id=tenant_id,
                    gate_mode=gate_configuration.mode,
                )
                if cached_result is not None:
                    qa_cache_hits_total.inc()
                    recorder.record(
                        "cache.lookup",
                        metadata={"retrieval_mode": strategy_id, "cache_hit": True},
                    )
                    recorder.record(
                        "qa.completed",
                        metadata={"retrieval_mode": strategy_id, "cache_hit": True},
                    )
                    logger.info(
                        "qa_cache_hit",
                        question_id=question_id,
                        user_id=user_id,
                        retrieval_mode=strategy_id,
                    )
                    return replace(
                        cached_result,
                        trace=recorder.build(),
                        cache_hit_type="exact",
                        knowledge_revision=knowledge_revision,
                    )
                pre_retrieval_security_actions.append("unsafe_cache_bypassed")
                logger.warning("qa_unsafe_cache_bypassed", question_id=question_id)

        if self.semantic_cache is not None and user_id and not force_full_rag:
            candidate = await self.semantic_cache.lookup(**confirmation_scope)
            if candidate is not None:
                recorder.record(
                    "cache.semantic_candidate",
                    metadata={"outcome": "matched", "mode": self.semantic_cache.mode.value},
                )
                if (
                    self.semantic_cache.mode is SemanticCacheMode.CONFIRM
                    and self.confirmation_service is not None
                ):
                    token = await self.confirmation_service.issue(candidate, **confirmation_scope)
                    return SemanticConfirmationRequired(
                        similar_question=candidate.question,
                        similarity=candidate.similarity,
                        cached_at=candidate.cached_at,
                        confirmation_token=token,
                    )

        intent = recorder.capture_sync(
            "query.classify",
            lambda: self._classify_intent_local(question),
        )
        rewritten = recorder.capture_sync(
            "query.rewrite",
            lambda: self._rewrite_query_local(question),
        )
        retrieval_calls: list[tuple[str, Any]] = [
            (
                "vector",
                recorder.capture_async(
                    "retrieval.vector",
                    lambda: self._vector_retrieve(
                        rewritten,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        question_fingerprint=question_id,
                        visible_department_ids=visible_department_ids,
                    ),
                    metadata=self._retrieval_trace_metadata,
                ),
            )
        ]
        if effective_strategy.uses_bm25:
            retrieval_calls.append(
                (
                    "bm25",
                    recorder.capture_async(
                        "retrieval.bm25",
                        lambda: self._bm25_retrieve(
                            question,
                            rewritten,
                            tenant_id=tenant_id,
                            user_id=user_id,
                            question_fingerprint=question_id,
                            visible_department_ids=visible_department_ids,
                        ),
                        metadata=self._retrieval_trace_metadata,
                    ),
                )
            )
        if effective_strategy.uses_graph:
            retrieval_calls.append(
                (
                    "graph",
                    recorder.capture_async(
                        "retrieval.graph",
                        lambda: self._graph_retrieve(
                            question,
                            rewritten,
                            tenant_id=tenant_id,
                            user_id=user_id,
                            question_fingerprint=question_id,
                            visible_department_ids=visible_department_ids,
                        ),
                        metadata=self._retrieval_trace_metadata,
                    ),
                )
            )
        retrieval_values = await asyncio.gather(*(call for _, call in retrieval_calls))
        outcomes = {
            name: self._as_retrieval_outcome(value)
            for (name, _), value in zip(retrieval_calls, retrieval_values)
        }
        for branch, outcome in outcomes.items():
            qa_retrieval_branch_total.labels(
                branch=branch, status=outcome.status.value
            ).inc()
        vector_outcome = outcomes["vector"]
        bm25_outcome = outcomes.get("bm25", RetrievalOutcome(contexts=[]))
        graph_outcome = outcomes.get("graph", RetrievalOutcome(contexts=[]))

        if evaluation_candidate_recorder is not None:
            self._record_candidate_outcome(
                evaluation_candidate_recorder, "dense", vector_outcome
            )
            if effective_strategy.uses_bm25:
                self._record_candidate_outcome(
                    evaluation_candidate_recorder, "bm25", bm25_outcome
                )
            else:
                evaluation_candidate_recorder.record(
                    "bm25",
                    [],
                    status="not_executed",
                    reason_code="branch_not_configured",
                )
            if effective_strategy.uses_graph:
                self._record_candidate_outcome(
                    evaluation_candidate_recorder, "graph", graph_outcome
                )
            else:
                evaluation_candidate_recorder.record(
                    "graph",
                    [],
                    status="not_executed",
                    reason_code="branch_not_configured",
                )

        vector_contexts = vector_outcome.contexts
        bm25_contexts = bm25_outcome.contexts
        graph_contexts = graph_outcome.contexts
        degradation_code: str | None = None
        if vector_outcome.unavailable:
            graph_contexts = [
                context for context in graph_contexts if self._has_document_provenance(context)
            ]
            remaining = bm25_contexts + graph_contexts
            degradation_code = _VECTOR_UNAVAILABLE if remaining else _INSUFFICIENT_EVIDENCE
        elif bm25_outcome.status is RetrievalStatus.CONTRACT_ERROR:
            degradation_code = _BM25_CONTRACT_ERROR
        elif bm25_outcome.unavailable and (vector_contexts or graph_contexts):
            degradation_code = _BM25_UNAVAILABLE
        elif graph_outcome.unavailable and (vector_contexts or bm25_contexts):
            degradation_code = _GRAPH_UNAVAILABLE

        safety_result = await recorder.capture_async(
            "safety.filter_contexts",
            lambda: self.safety_pipeline.filter_contexts(
                vector_contexts + bm25_contexts + graph_contexts,
                tenant_id=tenant_id,
            ),
            metadata=lambda value: {
                "contexts_count": len(value.contexts),
                "security_action_count": len(value.actions),
            },
        )
        security_actions = pre_retrieval_security_actions + list(safety_result.actions)
        fused_contexts = recorder.capture_sync(
            "ranking.rerank",
            lambda: self._hybrid_rerank(safety_result.contexts),
            metadata=lambda value: {"contexts_count": len(value)},
        )
        if evaluation_candidate_recorder is not None:
            evaluation_candidate_recorder.record(
                "fused",
                fused_contexts if fused_contexts else [],
                status="executed" if fused_contexts else "executed_empty",
                reason_code=None if fused_contexts else "empty",
            )
        rrf_contexts = fused_contexts[: settings.qa_context_limit]
        rerank_started = time.perf_counter()
        rerank_result = await recorder.capture_async(
            "qa.rerank_completed",
            lambda: self.cross_encoder_reranker.rerank(question, rrf_contexts),
            metadata=lambda value: {
                "mode": settings.qa_cross_encoder_mode,
                "status": value.status.value,
                "contexts_count": len(value.contexts),
                "scored_count": value.scored_count,
                "degradation_code": value.degradation_code,
            },
        )
        qa_cross_encoder_results_total.labels(
            mode=settings.qa_cross_encoder_mode,
            status=rerank_result.status.value,
        ).inc()
        qa_cross_encoder_latency_seconds.labels(
            mode=settings.qa_cross_encoder_mode,
        ).observe(time.perf_counter() - rerank_started)
        top_contexts = await self._expand_scope_contexts(rerank_result.contexts)
        if evaluation_candidate_recorder is not None:
            if rerank_result.status is RerankStatus.APPLIED:
                evaluation_candidate_recorder.record(
                    "reranked",
                    top_contexts,
                    status="executed" if top_contexts else "executed_empty",
                    reason_code=None if top_contexts else "empty",
                )
            elif rerank_result.status is RerankStatus.FALLBACK:
                evaluation_candidate_recorder.record(
                    "reranked",
                    [],
                    status="unavailable",
                    reason_code=rerank_result.degradation_code
                    or "reranker_unavailable",
                )
            else:
                evaluation_candidate_recorder.record(
                    "reranked",
                    [],
                    status="not_executed",
                    reason_code="reranker_disabled",
                )
        if rerank_result.degradation_code and degradation_code is None:
            degradation_code = rerank_result.degradation_code
        evidence_assessment = recorder.capture_sync(
            "qa.evidence_qualified",
            lambda: self.evidence_qualifier.assess(
                question,
                top_contexts,
                intent=intent,
                branch_statuses={
                    "vector": vector_outcome.status,
                    "bm25": bm25_outcome.status,
                    "graph": graph_outcome.status,
                },
                reranker_scores={
                    item.context_id: item.score for item in rerank_result.scores
                },
            ),
            metadata=lambda value: {
                "evidence_state": value.primary_state.value,
                "response_status": value.response_status.value,
                "evaluated_count": len(value.evaluated_context_ids),
                "accepted_count": len(value.supporting_context_ids),
                "policy_version": value.policy_version,
                "calibration_version": value.calibration_version,
                "reason_code": value.reason_codes[0].value,
            },
        )
        qa_evidence_decisions_total.labels(
            state=evidence_assessment.primary_state.value,
            response_status=evidence_assessment.response_status.value,
            policy_version=evidence_assessment.policy_version,
        ).inc()
        gate_mode = gate_configuration.mode
        enforce_evidence = gate_mode == "enforce"
        structured_answer = None
        grounding_result = None
        response_status = evidence_assessment.response_status
        # 冲突路由下的自由生成需要行为提示；该码只进生成提示，不写入降级记录。
        generation_notice_code = degradation_code
        if generation_notice_code is None and (
            evidence_assessment.primary_state is EvidenceState.CONFLICTING_EVIDENCE
        ):
            generation_notice_code = _CONFLICTING_EVIDENCE_NOTICE
        if enforce_evidence and not evidence_assessment.generation_allowed:
            degradation_code = self._evidence_degradation_code(evidence_assessment.response_status)
            answer_text, reasoning = self._evidence_controlled_answer(
                evidence_assessment.response_status,
                intent,
            )
        elif not top_contexts:
            degradation_code = _INSUFFICIENT_EVIDENCE
            answer_text, reasoning = no_context_answer(intent)
        elif gate_mode == "off" or not evidence_assessment.generation_allowed:
            answer_text, reasoning = await recorder.capture_async(
                "generation.answer",
                lambda: self._generate_answer(
                    question,
                    top_contexts,
                    intent,
                    degradation_code=generation_notice_code,
                ),
            )
        else:
            supporting_contexts = self._supporting_contexts(
                top_contexts, evidence_assessment.supporting_context_ids
            )
            structured_started = time.perf_counter()
            try:
                generated = await self._generate_structured_answer(
                    question,
                    supporting_contexts,
                    intent,
                    degradation_code=degradation_code,
                )
            except StructuredGenerationError:
                structured_duration = time.perf_counter() - structured_started
                qa_structured_generation_total.labels(
                    status="invalid",
                    schema_version=settings.qa_structured_answer_schema_version,
                ).inc()
                qa_structured_generation_latency_seconds.labels(
                    status="invalid"
                ).observe(structured_duration)
                recorder.record(
                    "qa.structured_generated",
                    status="failed",
                    duration_ms=structured_duration * 1000,
                    metadata={
                        "status": "invalid",
                        "schema_version": settings.qa_structured_answer_schema_version,
                        "reason_code": _STRUCTURED_OUTPUT_INVALID,
                    },
                    exception_type=StructuredGenerationError.__name__,
                )
                recorder.record(
                    "qa.grounding_verified",
                    status="skipped",
                    metadata={
                        "status": "skipped",
                        "policy_version": settings.qa_grounding_policy_version,
                        "reason_code": _STRUCTURED_OUTPUT_INVALID,
                    },
                )
                if enforce_evidence:
                    degradation_code = _STRUCTURED_OUTPUT_INVALID
                    response_status = QAResponseStatus.INSUFFICIENT_EVIDENCE
                    answer_text, reasoning = no_context_answer(intent)
                else:
                    answer_text, reasoning = await recorder.capture_async(
                        "generation.answer",
                        lambda: self._generate_answer(
                            question,
                            top_contexts,
                            intent,
                            degradation_code=generation_notice_code,
                        ),
                    )
            else:
                structured_duration = time.perf_counter() - structured_started
                qa_structured_generation_total.labels(
                    status="succeeded",
                    schema_version=generated.answer.schema_version,
                ).inc()
                qa_structured_generation_latency_seconds.labels(
                    status="succeeded"
                ).observe(structured_duration)
                recorder.record(
                    "qa.structured_generated",
                    duration_ms=structured_duration * 1000,
                    metadata={
                        "status": "succeeded",
                        "result_count": len(generated.answer.claims),
                        "accepted_count": len(generated.answer.citations),
                        "schema_version": generated.answer.schema_version,
                    },
                )
                grounding_started = time.perf_counter()
                grounding_result = self.grounding_verifier.verify(
                    generated.answer, generated.authorized_contexts
                )
                structured_answer = self.grounding_verifier.supported_subset(
                    generated.answer, grounding_result
                )
                grounding_duration = time.perf_counter() - grounding_started
                grounded_status = (
                    structured_answer.status
                    if structured_answer is not None
                    else QAResponseStatus.INSUFFICIENT_EVIDENCE
                )
                qa_grounding_results_total.labels(
                    passed=str(grounding_result.passed).lower(),
                    response_status=grounded_status.value,
                    policy_version=grounding_result.policy_version,
                ).inc()
                qa_grounding_latency_seconds.labels(
                    policy_version=grounding_result.policy_version
                ).observe(grounding_duration)
                recorder.record(
                    "qa.grounding_verified",
                    duration_ms=grounding_duration * 1000,
                    metadata={
                        "grounding_passed": grounding_result.passed,
                        "accepted_count": len(grounding_result.accepted_claim_ids),
                        "rejected_count": len(grounding_result.rejected_claim_ids),
                        "response_status": grounded_status.value,
                        "policy_version": grounding_result.policy_version,
                        "reason_code": (
                            grounding_result.reason_codes[0].value
                            if grounding_result.reason_codes
                            else None
                        ),
                    },
                )
                if gate_mode == "shadow":
                    answer_text, reasoning = await recorder.capture_async(
                        "generation.answer",
                        lambda: self._generate_answer(
                            question,
                            top_contexts,
                            intent,
                            degradation_code=generation_notice_code,
                        ),
                    )
                elif structured_answer is None:
                    degradation_code = _GROUNDING_FAILED
                    response_status = QAResponseStatus.INSUFFICIENT_EVIDENCE
                    answer_text, reasoning = no_context_answer(intent)
                else:
                    response_status = structured_answer.status
                    answer_text = structured_answer.answer
                    reasoning = list(generated.reasoning_steps)

        sanitized_answer = await recorder.capture_async(
            "safety.sanitize_answer",
            lambda: self.safety_pipeline.sanitize_answer(answer_text),
            metadata=lambda value: {"security_action_count": len(value.actions)},
        )
        security_actions.extend(
            action for action in sanitized_answer.actions if action not in security_actions
        )
        result = QAResult(
            question=question,
            answer=sanitized_answer.answer,
            contexts=top_contexts,
            intent=intent,
            confidence=self._calc_confidence(top_contexts),
            reasoning_steps=reasoning,
            degradation_code=degradation_code,
            security_actions=security_actions,
            knowledge_revision=knowledge_revision,
            response_status=response_status,
            evidence_assessment=evidence_assessment,
            structured_answer=(
                None if sanitized_answer.actions else structured_answer
            ),
            grounding_result=grounding_result,
        )
        recorder.record(
            "qa.completed",
            metadata={
                "retrieval_mode": strategy_id,
                "contexts_count": len(top_contexts),
                "vector_count": len(vector_contexts),
                "bm25_count": len(bm25_contexts),
                "graph_count": len(graph_contexts),
                "degradation_code": degradation_code,
                "security_action_count": len(security_actions),
            },
        )
        result = replace(result, trace=recorder.build())
        qa_query_total.labels(intent=intent.value).inc()
        qa_confidence_score.observe(result.confidence)
        logger.info(
            "qa_completed",
            intent=intent.value,
            confidence=result.confidence,
            latency_ms=(time.time() - started) * 1000,
            contexts_count=len(top_contexts),
            vector_count=len(vector_contexts),
            bm25_count=len(bm25_contexts),
            graph_count=len(graph_contexts),
            retrieval_mode=strategy_id,
            degradation_code=degradation_code,
            security_actions=security_actions,
        )
        cache_eligible = self._is_cache_eligible(result, user_id=user_id, gate_mode=gate_configuration.mode)
        if cache_eligible:
            await self._write_cache(
                question,
                user_id,
                result,
                cache_mode_id,
                tenant_id=effective_tenant,
                knowledge_revision=knowledge_revision,
            )
            if self.semantic_cache is not None:
                await self.semantic_cache.record(
                    **confirmation_scope,
                    result=self._serialize_cache_result(result),
                )
        return result


    # ── 缓存治理委托（实现见 agents/qa_cache.py）──────────────

    async def _check_cache(self, *args, **kwargs):
        return await qa_cache.check_cache(self, *args, **kwargs)

    async def _safe_cached_result(self, *args, **kwargs):
        return await qa_cache.safe_cached_result(self, *args, **kwargs)

    async def _write_cache(self, *args, **kwargs):
        return await qa_cache.write_cache(self, *args, **kwargs)

    @staticmethod
    def _serialize_cache_result(result: QAResult) -> dict[str, Any]:
        return qa_cache.serialize_cache_result(result)

    @staticmethod
    def _evidence_degradation_code(status: QAResponseStatus) -> str:
        return qa_cache.evidence_degradation_code(status)

    @staticmethod
    def _evidence_controlled_answer(status: QAResponseStatus, intent: QueryIntent) -> tuple[str, list[str]]:
        return qa_cache.evidence_controlled_answer(status, intent)

    def _effective_gate_configuration(self) -> Any:
        return qa_cache.effective_gate_configuration(self)

    def _is_cache_eligible(self, result: QAResult, *, user_id: str, gate_mode: str | None = None) -> bool:
        return qa_cache.is_cache_eligible(self, result, user_id=user_id, gate_mode=gate_mode)

    @staticmethod
    def _cache_mode_id(strategy: Any, *, gate_configuration: Any | None = None) -> str:
        return qa_cache.cache_mode_id(strategy, gate_configuration=gate_configuration)


    def _retrieval_trace_metadata(
        cls, value: RetrievalOutcome | list[RetrievedContext]
    ) -> dict[str, Any]:
        """Return bounded branch diagnostics without source text."""
        outcome = cls._as_retrieval_outcome(value)
        return {
            "contexts_count": len(outcome.contexts),
            "status": outcome.status.value,
            "unavailable": outcome.unavailable,
        }

    @staticmethod
    def _as_retrieval_outcome(value: RetrievalOutcome | list[RetrievedContext]) -> RetrievalOutcome:
        """Keep focused collaborator tests/callers compatible with list stubs."""

        if isinstance(value, RetrievalOutcome):
            return value
        return RetrievalOutcome(contexts=value)

    @staticmethod
    def _record_candidate_outcome(
        recorder: CandidateStageRecorderPort,
        stage: str,
        outcome: RetrievalOutcome,
    ) -> None:
        """Map retrieval outcomes to the evaluation stage status contract."""

        if outcome.status is RetrievalStatus.SUCCESS:
            recorder.record(stage, outcome.contexts, status="executed")
        elif outcome.status is RetrievalStatus.EMPTY:
            recorder.record(
                stage, [], status="executed_empty", reason_code="empty"
            )
        else:
            recorder.record(
                stage,
                [],
                status="unavailable",
                reason_code=outcome.status.value,
            )

    @staticmethod
    def _has_document_provenance(context: RetrievedContext) -> bool:
        """Report whether the question-answering agent has the document provenance."""
        return context.retrieval_type == "graph" and bool(str(context.metadata.get("source") or "").strip())

    @staticmethod
    def _classify_intent_local(question: str) -> QueryIntent:
        """Classify query intent with the local fallback."""
        return classify_intent_local(question)

    @staticmethod
    def _rewrite_query_local(question: str) -> dict:
        """Rewrite the query with the local fallback."""
        return rewrite_query_local(question)

    async def _vector_retrieve(
        self,
        rewritten: dict,
        tenant_id: str | None = None,
        user_id: str = "",
        question_fingerprint: str = "",
        visible_department_ids: tuple[str, ...] | list[str] | None = None,
    ) -> RetrievalOutcome:
        """从向量库检索上下文。

        Args:
            rewritten: 改写后的查询结果，取其 queries 首条作为检索输入。
            tenant_id: 租户 ID；为 None 表示不过滤租户（管理员场景）。
            user_id: 发起提问的用户 ID，用于检索作用域。
            question_fingerprint: 问题指纹，用于缓存与追踪关联。
            visible_department_ids: 用户可见的部门 ID 集合；None 表示不过滤，非 None（含空集合）表示严格按该集合过滤。
        """
        request = RetrievalRequest(
            question=str((rewritten.get("queries") or [""])[0]),
            rewritten=rewritten,
            scope=RetrievalScope(tenant_id=tenant_id or "", visible_department_ids=tuple(visible_department_ids) if visible_department_ids is not None else None),
            user_id=user_id,
            question_fingerprint=question_fingerprint,
        )
        return await NativeDenseRetriever(
            self.vector_store, top_k=settings.qa_vector_top_k
        ).retrieve(request)

    async def _bm25_retrieve(
        self,
        question: str,
        rewritten: dict,
        tenant_id: str | None = None,
        user_id: str = "",
        question_fingerprint: str = "",
        visible_department_ids: tuple[str, ...] | list[str] | None = None,
    ) -> RetrievalOutcome:
        """从原生稀疏（BM25）索引检索上下文。

        Args:
            question: 原始问题文本。
            rewritten: 改写后的查询结果。
            tenant_id: 租户 ID；为 None 表示不过滤租户（管理员场景）。
            user_id: 发起提问的用户 ID，用于检索作用域。
            question_fingerprint: 问题指纹，用于缓存与追踪关联。
            visible_department_ids: 用户可见的部门 ID 集合；None 表示不过滤，非 None（含空集合）表示严格按该集合过滤。
        """
        request = RetrievalRequest(
            question=question,
            rewritten=rewritten,
            scope=RetrievalScope(tenant_id=tenant_id or "", visible_department_ids=tuple(visible_department_ids) if visible_department_ids is not None else None),
            user_id=user_id,
            question_fingerprint=question_fingerprint,
        )
        return await NativeBM25Retriever(
            self.sparse_index, top_k=settings.qa_bm25_top_k
        ).retrieve(request)

    async def _graph_retrieve(
        self,
        question: str,
        rewritten: dict,
        tenant_id: str | None = None,
        user_id: str = "",
        question_fingerprint: str = "",
        visible_department_ids: tuple[str, ...] | list[str] | None = None,
    ) -> RetrievalOutcome:
        """从知识图谱检索上下文。

        Args:
            question: 原始问题文本。
            rewritten: 改写后的查询结果。
            tenant_id: 租户 ID；为 None 表示不过滤租户（管理员场景）。
            user_id: 发起提问的用户 ID，用于检索作用域。
            question_fingerprint: 问题指纹，用于缓存与追踪关联。
            visible_department_ids: 用户可见的部门 ID 集合；None 表示不过滤，非 None（含空集合）表示严格按该集合过滤。
        """
        request = RetrievalRequest(
            question=question,
            rewritten=rewritten,
            scope=RetrievalScope(tenant_id=tenant_id or "", visible_department_ids=tuple(visible_department_ids) if visible_department_ids is not None else None),
            user_id=user_id,
            question_fingerprint=question_fingerprint,
        )
        return await ClaimAwareGraphRetriever(self.knowledge_graph).retrieve(request)

    async def _call_llm_with_breaker(self, messages: list) -> Any:
        """Call the LLM with breaker."""
        return await call_llm_with_breaker(self.llm, messages)

    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Fuse native adapter contexts with configured deterministic RRF."""
        return HybridRetrievalOrchestrator(
            rrf_k=settings.qa_rrf_k,
            source_weights={
                "vector": settings.qa_rrf_vector_weight,
                "bm25": settings.qa_rrf_bm25_weight,
                "graph": settings.qa_rrf_graph_weight,
            },
        ).fuse(RetrievalOutcome(contexts=contexts))

    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
        *,
        degradation_code: str | None = None,
    ) -> tuple[str, list[str]]:
        """Generate the answer."""
        if not contexts:
            return no_context_answer(intent)
        messages, reasoning_steps = build_answer_messages(
            question,
            contexts,
            intent,
            degradation_code=degradation_code,
        )
        response = await self._call_llm_with_breaker(messages)
        reasoning_steps.append("答案生成完成")
        return response.content, reasoning_steps

    async def _expand_scope_contexts(
        self,
        contexts: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        """Append neighbor chunks for scope questions and top-ranked evidence.

        广义"管理范围/目录"类问题的答案分布在整个目录区，而长目录会被分块
        切碎：当排序第一的分块本身含目录时，顺序补拉同文档的后续分块。同时
        为前三个入选分块补拉同文档相邻分块，覆盖恰好落在被选分块隔壁的规则
        正文。扩展块取自已授权上下文的同一文档，不扩大授权范围。
        """
        budget = settings.qa_scope_expansion_chunks
        if not contexts or budget <= 0 or self.vector_store is None:
            return contexts
        existing = {
            str((context.metadata or {}).get("chunk_id") or "").strip()
            for context in contexts
        }
        wanted: list[str] = []

        head = contexts[0]
        head_metadata = head.metadata or {}
        head_doc = str(head_metadata.get("source_document_id") or "").strip()
        head_index = head_metadata.get("chunk_index")
        content = head.content or ""
        toc_head = (
            head_doc
            and isinstance(head_index, int)
            and not isinstance(head_index, bool)
            and ("目 录" in content or "目录" in content)
        )
        if toc_head:
            wanted.extend(
                f"{head_doc}#chunk-{head_index + offset}"
                for offset in range(1, budget + 1)
            )

        for context in contexts[:3]:
            metadata = context.metadata or {}
            doc_id = str(metadata.get("source_document_id") or "").strip()
            chunk_index = metadata.get("chunk_index")
            if not doc_id or not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
                continue
            for neighbor in (chunk_index - 1, chunk_index + 1):
                if neighbor >= 0:
                    wanted.append(f"{doc_id}#chunk-{neighbor}")

        wanted = [chunk_id for chunk_id in dict.fromkeys(wanted) if chunk_id not in existing]
        if not wanted:
            return contexts
        try:
            fetched = await self.vector_store.fetch_chunks(wanted)
        except Exception as error:
            logger.warning(
                "scope_context_expansion_failed", error_type=type(error).__name__
            )
            return contexts
        expanded = list(contexts)
        for document, score in fetched or []:
            chunk_metadata = dict(document.get("metadata") or {})
            chunk_id = str(chunk_metadata.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in existing:
                continue
            existing.add(chunk_id)
            expanded.append(
                RetrievedContext(
                    content=str(document.get("content") or ""),
                    source=str(document.get("source") or "vector_store"),
                    score=float(score) * 0.5,
                    retrieval_type="vector",
                    metadata={
                        **chunk_metadata,
                        "source_document_id": str(chunk_metadata.get("doc_id") or doc_id),
                        "scope_expansion": True,
                    },
                )
            )
        return expanded

    async def _generate_structured_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
        *,
        degradation_code: str | None = None,
    ) -> StructuredGenerationResult:
        """Generate a bounded structured answer for deterministic grounding."""
        return await generate_structured_answer(
            self.llm,
            question,
            contexts,
            intent,
            schema_version=settings.qa_structured_answer_schema_version,
            degradation_code=degradation_code,
        )

    @staticmethod
    def _supporting_contexts(
        contexts: list[RetrievedContext],
        supporting_context_ids: tuple[str, ...],
    ) -> list[RetrievedContext]:
        """Select only current-run contexts authorized by evidence qualification."""
        allowed = set(supporting_context_ids)
        return [
            context
            for index, context in enumerate(contexts)
            if context_identity(context, index) in allowed
        ]

    @staticmethod
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        """Calculate the confidence."""
        return calc_confidence(contexts)


__all__ = [
    "ANSWER_PROMPT",
    "QAAgent",
    "_INTENT_RULES",
]
