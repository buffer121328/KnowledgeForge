"""Question-answering HTTP routes."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from services.qa.generation import QAGenerationUnavailableError
from services.safety.qa_checks import QASafetyRefusalError, question_fingerprint
from services.safety.pipeline import QASafetyUnavailableError
from api.contracts import CursorToken, PageLimit, ResourceId
from api.dependencies import QARouteDependencies, get_qa_route_dependencies, require_permission
from api.dependencies.qa import (
    AuditAction,
    AuditResult,
    WebhookEvent,
    get_audit_service,
    get_webhook_service,
)
from api.schemas import (
    QAConversationCreateRequest,
    QAConversationDetailResponse,
    QAConversationItem,
    QAConversationPageResponse,
    QAConversationDeletedResponse,
    QAFeedbackRequest,
    QAFeedbackResponse,
    QuestionRequest,
    QuestionResponse,
    SemanticCacheDecisionRequest,
    SemanticCacheRejectResponse,
    SemanticConfirmationResponse,
)
from domain.identity import Permission, UserContext, UserRole
from domain.qa_history import QAFeedbackRating
from shared.utils.logging import bind_context, get_logger
from shared.utils.ratelimit import RATE_LIMITS, authenticated_composite_key, limiter


qa_router = APIRouter(prefix="/qa", tags=["智能问答"])
logger = get_logger(__name__)


def _controlled_error(status_code: int, code: str, message: str) -> HTTPException:
    """Return the controlled error."""
    request_id = uuid4().hex
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


def _resolved_qa_dependencies(
    request: Request,
    candidate: QARouteDependencies | object,
) -> QARouteDependencies:
    """Use FastAPI injection, with a bounded direct-handler test fallback."""

    if isinstance(candidate, QARouteDependencies):
        return candidate
    return replace(
        get_qa_route_dependencies(request),
        audit_service_factory=get_audit_service,
        webhook_service_factory=get_webhook_service,
    )


def _history(dependencies: QARouteDependencies):
    """Return the configured QA history repository or a controlled unavailable error."""
    history = dependencies.history
    if history is None:
        raise _controlled_error(503, "qa_history_unavailable", "问答历史服务暂不可用。")
    return history


def _conversation_item(conversation) -> QAConversationItem:
    """Convert a QA conversation domain record to the HTTP schema."""
    return QAConversationItem(
        id=conversation.id,
        title=conversation.title,
        status=conversation.status,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


@qa_router.get("/conversations", response_model=QAConversationPageResponse)
async def list_conversations(
    request: Request,
    cursor: CursorToken | None = None,
    limit: PageLimit = 20,
    user: UserContext = Depends(require_permission(Permission.QA_HISTORY)),
    qa_dependencies: QARouteDependencies = Depends(get_qa_route_dependencies),
):
    """List the authenticated user's durable conversations."""
    qa_dependencies = _resolved_qa_dependencies(request, qa_dependencies)
    try:
        page = _history(qa_dependencies).list_conversations(
            tenant_id=user.org_id,
            user_id=user.user_id,
            cursor=cursor,
            limit=limit,
        )
    except ValueError as error:
        raise _controlled_error(400, "invalid_conversation_cursor", "会话游标无效。") from error
    return QAConversationPageResponse(
        items=[_conversation_item(item) for item in page.items],
        next_cursor=page.next_cursor,
    )


@qa_router.post("/conversations", response_model=QAConversationItem)
async def create_conversation(
    request: Request,
    body: QAConversationCreateRequest,
    user: UserContext = Depends(require_permission(Permission.QA_HISTORY)),
    qa_dependencies: QARouteDependencies = Depends(get_qa_route_dependencies),
):
    """Create an empty durable conversation owned by the caller."""
    qa_dependencies = _resolved_qa_dependencies(request, qa_dependencies)
    conversation = _history(qa_dependencies).create_conversation(
        tenant_id=user.org_id,
        user_id=user.user_id,
        title=body.title,
    )
    return _conversation_item(conversation)


@qa_router.get("/conversations/{conversation_id}", response_model=QAConversationDetailResponse)
async def get_conversation(
    conversation_id: ResourceId,
    request: Request,
    user: UserContext = Depends(require_permission(Permission.QA_HISTORY)),
    qa_dependencies: QARouteDependencies = Depends(get_qa_route_dependencies),
):
    """Return one owned conversation or the common not-found outcome."""
    qa_dependencies = _resolved_qa_dependencies(request, qa_dependencies)
    detail = _history(qa_dependencies).get_conversation(
        conversation_id=conversation_id,
        tenant_id=user.org_id,
        user_id=user.user_id,
    )
    if detail is None:
        raise _controlled_error(404, "qa_conversation_not_found", "会话不存在。")
    return QAConversationDetailResponse(
        conversation=_conversation_item(detail["conversation"]),
        messages=[
            {
                "id": item.id,
                "sequence": item.sequence,
                "role": item.role,
                "content": item.content,
                "created_at": item.created_at.isoformat(),
            }
            for item in detail["messages"]
        ],
        runs=detail["runs"],
    )


@qa_router.delete("/conversations/{conversation_id}", response_model=QAConversationDeletedResponse)
async def delete_conversation(
    conversation_id: ResourceId,
    request: Request,
    user: UserContext = Depends(require_permission(Permission.QA_HISTORY)),
    qa_dependencies: QARouteDependencies = Depends(get_qa_route_dependencies),
):
    """Delete one owned conversation before best-effort cache invalidation."""
    qa_dependencies = _resolved_qa_dependencies(request, qa_dependencies)
    run_ids = _history(qa_dependencies).delete_conversation(
        conversation_id=conversation_id,
        tenant_id=user.org_id,
        user_id=user.user_id,
    )
    if run_ids is None:
        raise _controlled_error(404, "qa_conversation_not_found", "会话不存在。")
    cache = qa_dependencies.cache
    if cache is not None:
        for run_id in run_ids:
            try:
                await cache.invalidate_run(run_id)
            except Exception as error:
                logger.warning("qa_cache_invalidation_failed", error_type=type(error).__name__)
    return {"status": "deleted", "conversation_id": conversation_id}


@qa_router.post("/runs/{qa_run_id}/feedback", response_model=QAFeedbackResponse)
async def submit_feedback(
    qa_run_id: ResourceId,
    body: QAFeedbackRequest,
    request: Request,
    user: UserContext = Depends(require_permission(Permission.QA_FEEDBACK)),
    qa_dependencies: QARouteDependencies = Depends(get_qa_route_dependencies),
):
    """Create or replace the caller's bounded feedback for an owned run."""
    qa_dependencies = _resolved_qa_dependencies(request, qa_dependencies)
    feedback = _history(qa_dependencies).upsert_feedback(
        qa_run_id=qa_run_id,
        tenant_id=user.org_id,
        user_id=user.user_id,
        rating=QAFeedbackRating(body.rating),
        note=body.note,
    )
    if feedback is None:
        raise _controlled_error(404, "qa_run_not_found", "问答记录不存在。")
    return feedback


@qa_router.post("/ask", response_model=QuestionResponse | SemanticConfirmationResponse)
@limiter.limit(RATE_LIMITS["qa_ask"], key_func=authenticated_composite_key)
async def ask_question(
    request: Request,
    response: Response,
    req: QuestionRequest,
    user: UserContext = Depends(require_permission(Permission.QA_QUERY)),
    qa_dependencies: QARouteDependencies = Depends(get_qa_route_dependencies),
):
    """Answer a question with safe retrieval degradation and error semantics."""

    qa_dependencies = _resolved_qa_dependencies(request, qa_dependencies)
    bind_context(user_id=user.user_id, action="qa.query")
    # The authenticated identity is authoritative for cache and workflow scope;
    # middleware state is transport context and must never weaken tenant isolation.
    tenant_id = user.org_id
    question_resource = f"qa/{question_fingerprint(req.question)}"
    qa_wf = qa_dependencies.workflow
    if not qa_wf:
        raise _controlled_error(503, "qa_service_unavailable", "问答服务暂不可用，请稍后重试。")

    try:
        result = await qa_wf.ainvoke({
            "question": req.question,
            "tenant_id": tenant_id,
            "user_id": user.user_id,
            "retrieval_mode": req.retrieval_mode,
            "visible_department_ids": (
                None
                if user.role == UserRole.ORGANIZATION_ADMIN
                else ((user.department_id,) if user.department_id else ())
            ),
            "semantic_bypass_token": req.semantic_bypass_token,
            "semantic_confirmation_token": req.semantic_confirmation_token,
        })
    except QASafetyRefusalError as error:
        controlled = _controlled_error(400, "qa_safety_refusal", "请求因安全策略被拒绝。")
        try:
            qa_dependencies.audit_service().log_security_event(
                user_id=user.user_id,
                org_id=user.org_id,
                action=AuditAction.QA_SAFETY_REFUSAL,
                reason_code=error.code,
                request_id=controlled.detail["request_id"],
                metadata={
                    "question_fingerprint": question_fingerprint(req.question),
                    "security_action": error.code,
                },
                result=AuditResult.DENIED,
                required=True,
            )
        except Exception as audit_error:
            # Refusal is already the safe boundary; never disclose audit internals.
            logger.warning(
                "security_event_audit_failed",
                security_event="qa_safety_refusal",
                error_type=type(audit_error).__name__,
            )
        logger.warning(
            "qa_safety_refusal",
            security_action=error.code,
            request_id=controlled.detail["request_id"],
        )
        raise controlled from error
    except QAGenerationUnavailableError as error:
        controlled = _controlled_error(503, "qa_generation_unavailable", "问答生成服务暂不可用，请稍后重试。")
        try:
            qa_dependencies.audit_service().log_security_event(
                user_id=user.user_id,
                org_id=user.org_id,
                action=AuditAction.BREAKER_OPEN,
                reason_code="llm_unavailable",
                request_id=controlled.detail["request_id"],
                metadata={
                    "dependency": "llm",
                    "question_fingerprint": question_fingerprint(req.question),
                    "failure_class": "generation_unavailable",
                },
                result=AuditResult.FAILURE,
            )
        except Exception as audit_error:
            logger.warning(
                "security_event_audit_failed",
                security_event="breaker_open",
                error_type=type(audit_error).__name__,
            )
        logger.warning(
            "qa_generation_unavailable",
            error_type=type(error).__name__,
            request_id=controlled.detail["request_id"],
        )
        raise controlled from error
    except QASafetyUnavailableError as error:
        controlled = _controlled_error(
            503,
            "qa_safety_unavailable",
            "问答安全校验暂不可用，请稍后重试。",
        )
        qa_dependencies.audit_service().log(
            user_id=user.user_id,
            action=AuditAction.QA_QUERY,
            resource=question_resource,
            result=AuditResult.FAILURE,
            username=user.username,
            ip=request.client.host if request.client else "",
            user_agent=request.headers.get("user-agent", ""),
            org_id=user.org_id,
            metadata={
                "security_action": "external_validator_unavailable",
                "stage": error.stage,
            },
        )
        logger.warning(
            "qa_safety_unavailable",
            stage=error.stage,
            request_id=controlled.detail["request_id"],
        )
        raise controlled from error
    except Exception as error:
        controlled = _controlled_error(500, "qa_internal_error", "问答服务暂时无法完成请求。")
        logger.error(
            "qa_ask_failed",
            error_type=type(error).__name__,
            request_id=controlled.detail["request_id"],
        )
        raise controlled from error

    confirmation = result.get("confirmation")
    if confirmation is not None:
        return SemanticConfirmationResponse(
            similar_question=confirmation.similar_question,
            similarity=confirmation.similarity,
            cached_at=confirmation.cached_at,
            confirmation_token=confirmation.confirmation_token,
        )

    qa_result = result.get("result")
    if not qa_result:
        qa_dependencies.audit_service().log(
            user_id=user.user_id,
            action=AuditAction.QA_QUERY,
            resource=question_resource,
            result=AuditResult.FAILURE,
            username=user.username,
            ip=request.client.host if request.client else "",
            user_agent=request.headers.get("user-agent", ""),
            org_id=user.org_id,
        )
        raise _controlled_error(500, "qa_internal_error", "问答服务暂时无法完成请求。")

    qa_dependencies.audit_service().log(
        user_id=user.user_id,
        action=AuditAction.QA_QUERY,
        resource=question_resource,
        result=AuditResult.SUCCESS,
        username=user.username,
        ip=request.client.host if request.client else "",
        user_agent=request.headers.get("user-agent", ""),
        org_id=user.org_id,
        metadata={
            "intent": qa_result.intent.value,
            "confidence": qa_result.confidence,
            "degradation_code": qa_result.degradation_code,
            "security_actions": qa_result.security_actions,
        },
    )

    webhook_service = qa_dependencies.webhook_service()
    await webhook_service.trigger(
        WebhookEvent.QA_COMPLETED,
        {
            "schema_version": "v1",
            "intent": qa_result.intent.value,
            "confidence": qa_result.confidence,
            "has_degraded_retrieval": qa_result.degradation_code is not None,
        },
        org_id=user.org_id,
    )

    logger.info(
        "qa_completed",
        confidence=qa_result.confidence,
        intent=qa_result.intent.value,
        tenant_id=tenant_id,
        degradation_code=qa_result.degradation_code,
    )
    conversation_id = req.conversation_id
    qa_run_id = None
    history_saved = False
    warning_code = None
    history = qa_dependencies.history
    if history is not None:
        try:
            conversation_id, qa_run_id = history.record_answer(
                tenant_id=user.org_id,
                user_id=user.user_id,
                question=req.question,
                result=qa_result,
                conversation_id=req.conversation_id,
                retrieval_mode=req.retrieval_mode,
                knowledge_revision=int(result.get("knowledge_revision") or 0),
                cache_hit_type=str(result.get("cache_hit_type") or "none"),
                duration_ms=int(result.get("duration_ms") or 0),
                semantic_decision=result.get("semantic_decision"),
            )
            history_saved = True
        except Exception as error:
            warning_code = "qa_history_not_saved"
            logger.warning("qa_history_write_failed", error_type=type(error).__name__)
    else:
        warning_code = "qa_history_not_saved"
    qa_cache = qa_dependencies.cache
    if history_saved and qa_run_id and qa_cache is not None:
        try:
            await qa_cache.associate_answer(
                qa_run_id,
                req.question,
                user.user_id,
                req.retrieval_mode,
                user.org_id,
                int(result.get("knowledge_revision") or 0),
            )
        except Exception as error:
            logger.warning("qa_cache_run_association_failed", error_type=type(error).__name__)
    return QuestionResponse(
        question=qa_result.question,
        answer=qa_result.answer,
        confidence=qa_result.confidence,
        intent=qa_result.intent.value,
        sources=[
            {
                "content": c.content[:200],
                "source": c.source,
                "score": c.score,
                "type": c.retrieval_type,
                "document_id": str(c.metadata.get("source_document_id") or c.metadata.get("doc_id") or ""),
                "chunk_id": str(c.metadata.get("source_chunk_id") or c.metadata.get("chunk_id") or ""),
                "chunk_index": c.metadata.get("chunk_index"),
            }
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
        degradation_code=qa_result.degradation_code,
        conversation_id=conversation_id,
        qa_run_id=qa_run_id,
        cache_hit_type=str(result.get("cache_hit_type") or "none"),
        history_saved=history_saved,
        warning_code=warning_code,
        response_status=(
            qa_result.response_status.value if qa_result.response_status else None
        ),
        evidence_state=(
            qa_result.evidence_assessment.primary_state.value
            if qa_result.evidence_assessment
            else None
        ),
        evidence_reason_codes=(
            [reason.value for reason in qa_result.evidence_assessment.reason_codes]
            if qa_result.evidence_assessment
            else []
        ),
        claims=(
            [
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "citation_ids": list(claim.citation_ids),
                    "material": claim.material,
                }
                for claim in qa_result.structured_answer.claims
            ]
            if qa_result.structured_answer
            else []
        ),
        citations=(
            [
                {
                    "citation_id": citation.citation_id,
                    "source": citation.source,
                    "content": citation.content[:1000],
                    "document_id": citation.document_id,
                    "chunk_id": citation.chunk_id,
                    "chunk_index": citation.chunk_index,
                    "highlight": citation.highlight[:4000],
                }
                for citation in qa_result.structured_answer.citations
            ]
            if qa_result.structured_answer
            else []
        ),
        missing_information=[
            {"field": item.field, "description": item.description}
            for item in (
                qa_result.structured_answer.missing_information
                if qa_result.structured_answer
                else qa_result.evidence_assessment.missing_information
                if qa_result.evidence_assessment
                else ()
            )
        ],
        grounding_result=(
            {
                "passed": qa_result.grounding_result.passed,
                "accepted_claim_ids": list(
                    qa_result.grounding_result.accepted_claim_ids
                ),
                "rejected_claim_ids": list(
                    qa_result.grounding_result.rejected_claim_ids
                ),
                "reason_codes": [
                    reason.value for reason in qa_result.grounding_result.reason_codes
                ],
                "policy_version": qa_result.grounding_result.policy_version,
            }
            if qa_result.grounding_result
            else None
        ),
        grounding_passed=(
            qa_result.grounding_result.passed if qa_result.grounding_result else None
        ),
        policy_version=(
            qa_result.evidence_assessment.policy_version
            if qa_result.evidence_assessment
            else None
        ),
    )


@qa_router.post(
    "/cache/confirm",
    response_model=QuestionResponse | SemanticConfirmationResponse,
)
async def confirm_semantic_cache(
    request: Request,
    response: Response,
    body: SemanticCacheDecisionRequest,
    user: UserContext = Depends(require_permission(Permission.QA_QUERY)),
    qa_dependencies: QARouteDependencies = Depends(get_qa_route_dependencies),
):
    """Confirm once; invalid tokens safely execute the original question via full RAG."""
    qa_dependencies = _resolved_qa_dependencies(request, qa_dependencies)
    return await ask_question(
        request=request,
        response=response,
        req=QuestionRequest(
            question=body.question,
            conversation_id=body.conversation_id,
            retrieval_mode=body.retrieval_mode,
            semantic_confirmation_token=body.confirmation_token,
        ),
        user=user,
        qa_dependencies=qa_dependencies,
    )


@qa_router.post("/cache/reject", response_model=SemanticCacheRejectResponse)
async def reject_semantic_cache(
    request: Request,
    body: SemanticCacheDecisionRequest,
    user: UserContext = Depends(require_permission(Permission.QA_QUERY)),
    qa_dependencies: QARouteDependencies = Depends(get_qa_route_dependencies),
):
    """Reject or close a candidate and issue a scoped, one-use full-RAG bypass."""
    qa_dependencies = _resolved_qa_dependencies(request, qa_dependencies)
    revision = await qa_dependencies.knowledge_revision.get(user.org_id)
    token = await qa_dependencies.semantic_confirmation.reject(
        body.confirmation_token,
        question=body.question,
        tenant_id=user.org_id,
        user_scope=user.user_id,
        retrieval_mode=body.retrieval_mode,
        knowledge_revision=revision,
    )
    return SemanticCacheRejectResponse(semantic_bypass_token=token)
