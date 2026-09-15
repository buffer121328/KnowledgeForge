"""ATDD coverage for QA collaborators, workflow assembly, and Celery task boundaries."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from agents.qa_agent import QAAgent
from services.qa.generation import generate_answer
from services.qa.query import classify_intent_local, rewrite_query_local
from services.qa.ranking import calc_confidence, hybrid_rerank
from domain.evidence import (
    AnswerCitation,
    AnswerClaim,
    EvidenceAssessment,
    EvidenceReasonCode,
    EvidenceState,
    GroundingResult,
    QAResponseStatus,
    StructuredAnswer,
)
from domain.knowledge import QAResult, QueryIntent, RetrievedContext
from infrastructure.celery_app import celery_app
from infrastructure.celery_tasks import batch_ingest_task, cleanup_task, ingest_document_task, knowledge_update_task, ping
from workflows.ask_question import build_ask_question_workflow
from workflows.workflow_factory import build_workflow_registry


def test_qa_facade_uses_focused_query_and_ranking_collaborators():
    contexts = [
        RetrievedContext(content="same context", source="vector", score=0.5, retrieval_type="vector"),
        RetrievedContext(content="same context", source="graph", score=0.5, retrieval_type="graph"),
    ]

    assert QAAgent._classify_intent_local("两个方案有什么区别") == classify_intent_local("两个方案有什么区别")
    assert QAAgent._rewrite_query_local("如何配置知识库") == rewrite_query_local("如何配置知识库")
    ranked = QAAgent._hybrid_rerank(contexts)
    assert ranked == hybrid_rerank(contexts)
    assert QAAgent._calc_confidence(ranked) == calc_confidence(ranked)
    assert len(ranked) == 1


@pytest.mark.asyncio
async def test_qa_generation_preserves_no_context_semantics_without_llm_call():
    answer, reasoning = await generate_answer(
        llm=None,
        question="没有上下文怎么办？",
        contexts=[],
        intent=QueryIntent.FACTOID,
    )

    assert answer == "未检索到可验证的证据，无法生成可靠答案。"
    assert reasoning[-1] == "跳过答案生成"


def test_workflow_factory_returns_compatible_registry_keys(monkeypatch):
    from types import SimpleNamespace
    from workflows import workflow_factory

    workflow = SimpleNamespace(ainvoke=object())
    monkeypatch.setattr(workflow_factory, "DocParserAgent", lambda: object())
    monkeypatch.setattr(workflow_factory, "KnowledgeExtractAgent", lambda: object())
    dependencies = object()
    received_qa_kwargs = {}

    def build_qa_agent(**kwargs):
        received_qa_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(workflow_factory, "QAAgent", build_qa_agent)
    monkeypatch.setattr(workflow_factory, "build_qa_agent_dependencies", lambda: dependencies)
    monkeypatch.setattr(workflow_factory, "KnowledgeUpdateAgent", lambda **_: object())
    monkeypatch.setattr(workflow_factory, "build_ingest_document_workflow", lambda *args: workflow)
    monkeypatch.setattr(workflow_factory, "build_ask_question_workflow", lambda *args: workflow)
    monkeypatch.setattr(workflow_factory, "build_update_knowledge_workflow", lambda *args: workflow)

    registry = build_workflow_registry(vector_store=None, knowledge_graph=None)

    assert set(registry) == {"ingest", "qa", "update"}
    assert all(hasattr(item, "ainvoke") for item in registry.values())
    assert received_qa_kwargs["dependencies"] is dependencies


def test_celery_compatibility_facade_preserves_include_task_names_and_exports():
    assert "infrastructure.celery_tasks" in celery_app.conf.include
    assert ingest_document_task.name == "infrastructure.celery_tasks.ingest_document_task"
    assert batch_ingest_task.name == "infrastructure.celery_tasks.batch_ingest_task"
    assert knowledge_update_task.name == "infrastructure.celery_tasks.knowledge_update_task"
    assert cleanup_task.name == "infrastructure.celery_tasks.cleanup_task"
    assert ping.name == "infrastructure.celery_tasks.ping"


@pytest.mark.asyncio
async def test_ask_question_workflow_preserves_structured_grounding_result() -> None:
    context = RetrievedContext(
        content="培训预算为100万元。",
        source="policy.md",
        score=0.9,
        retrieval_type="vector",
        metadata={"context_id": "ctx_1"},
    )
    structured = StructuredAnswer(
        status=QAResponseStatus.ANSWERED,
        answer="培训预算为100万元。",
        claims=(
            AnswerClaim(
                claim_id="claim_1",
                text="培训预算为100万元。",
                citation_ids=("cite_current",),
            ),
        ),
        citations=(
            AnswerCitation(citation_id="cite_current", context_id="ctx_1"),
        ),
        schema_version="structured-answer-v1",
    )
    assessment = EvidenceAssessment(
        states=(EvidenceState.DIRECT_EVIDENCE,),
        response_status=QAResponseStatus.ANSWERED,
        reason_codes=(EvidenceReasonCode.DIRECT_SUPPORT,),
        evaluated_context_ids=("ctx_1",),
        supporting_context_ids=("ctx_1",),
    )
    grounding = GroundingResult(
        passed=True,
        policy_version="grounding-v1",
        accepted_claim_ids=("claim_1",),
    )
    expected = QAResult(
        question="培训预算是多少？",
        answer=structured.answer,
        contexts=[context],
        intent=QueryIntent.FACTOID,
        confidence=0.9,
        response_status=QAResponseStatus.ANSWERED,
        evidence_assessment=assessment,
        structured_answer=structured,
        grounding_result=grounding,
    )
    qa_agent = type("StubQAAgent", (), {"answer": AsyncMock(return_value=expected)})()
    workflow = build_ask_question_workflow(qa_agent)

    state = await workflow.ainvoke(
        {
            "question": expected.question,
            "tenant_id": "org-001",
            "user_id": "user-1",
            "retrieval_mode": "hybrid",
        }
    )

    assert state["result"] is expected
    assert state["result"].structured_answer == structured
    assert state["result"].grounding_result == grounding
