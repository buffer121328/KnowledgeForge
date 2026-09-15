"""ATDD for durable, owner-scoped QA history and degradation semantics."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Response
from sqlalchemy import create_engine
from starlette.requests import Request

from api.routers.qa import ask_question
from api.schemas import QuestionRequest
from infrastructure.postgres.identity_store import PostgreSQLIdentityStore
from auth.jwt_service import AuthService
from domain.identity import UserContext, UserRole
from domain.knowledge import QAResult, QueryIntent, RetrievedContext
from domain.qa_history import QAFeedbackRating
from infrastructure.postgres.database import DatabaseService
from infrastructure.postgresql_migrations import upgrade
from infrastructure.documents.qa_history import PostgreSQLQAHistoryRepository


@pytest.fixture
def history(tmp_path) -> PostgreSQLQAHistoryRepository:
    """Create a migrated history repository with two tenant users."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'qa-history.sqlite3'}")
    upgrade(engine)
    database = DatabaseService.from_engine(engine)
    identity = PostgreSQLIdentityStore(database)
    auth = AuthService()
    for username, tenant in (("alice", "org-a"), ("bob", "org-b")):
        assert identity.create_user(
            {
                "user_id": f"user-{username}",
                "username": username,
                "email": f"{username}@example.test",
                "display_name": username,
                "password_hash": auth.hash_password("legacy-password"),
                "role": UserRole.VIEWER,
                "org_id": tenant,
                "is_active": True,
                "token_version": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return PostgreSQLQAHistoryRepository(database)


def _result(answer: str = "answer") -> QAResult:
    """Build one source-backed QA result."""
    return QAResult(
        question="question",
        answer=answer,
        contexts=[
            RetrievedContext(
                content="controlled source summary",
                source="doc-1",
                score=0.91,
                retrieval_type="vector",
                metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
            )
        ],
        intent=QueryIntent.FACTOID,
        confidence=0.9,
    )


def test_history_records_messages_sources_and_cursor_pages(history: PostgreSQLQAHistoryRepository) -> None:
    """Completed answers are recoverable in stable order and bounded pages."""
    first_conversation, first_run = history.record_answer(
        tenant_id="org-a",
        user_id="user-alice",
        question="first question",
        result=_result("first answer"),
    )
    history.record_answer(
        tenant_id="org-a",
        user_id="user-alice",
        question="second question",
        result=_result("second answer"),
    )

    page = history.list_conversations(tenant_id="org-a", user_id="user-alice", limit=1)
    assert len(page.items) == 1
    assert page.next_cursor
    assert history.list_conversations(
        tenant_id="org-a", user_id="user-alice", limit=1, cursor=page.next_cursor
    ).items[0].id == first_conversation

    detail = history.get_conversation(
        conversation_id=first_conversation,
        tenant_id="org-a",
        user_id="user-alice",
    )
    assert [message.role for message in detail["messages"]] == ["user", "assistant"]
    assert [message.content for message in detail["messages"]] == ["first question", "first answer"]
    assert detail["runs"][0]["id"] == first_run
    assert detail["runs"][0]["sources"][0]["chunk_id"] == "chunk-1"


def test_cross_owner_is_hidden_and_feedback_is_idempotent(history: PostgreSQLQAHistoryRepository) -> None:
    """Conversation and feedback access require both tenant and owner."""
    conversation_id, run_id = history.record_answer(
        tenant_id="org-a",
        user_id="user-alice",
        question="question",
        result=_result(),
    )
    assert history.get_conversation(
        conversation_id=conversation_id,
        tenant_id="org-b",
        user_id="user-bob",
    ) is None
    assert history.upsert_feedback(
        qa_run_id=run_id,
        tenant_id="org-b",
        user_id="user-bob",
        rating=QAFeedbackRating.UP,
    ) is None

    first = history.upsert_feedback(
        qa_run_id=run_id,
        tenant_id="org-a",
        user_id="user-alice",
        rating=QAFeedbackRating.UP,
    )
    second = history.upsert_feedback(
        qa_run_id=run_id,
        tenant_id="org-a",
        user_id="user-alice",
        rating=QAFeedbackRating.DOWN,
        note="incorrect",
    )
    assert first["feedback_id"] == second["feedback_id"]
    assert second["rating"] == "down"


def test_deletion_hides_conversation_and_returns_run_ids(history: PostgreSQLQAHistoryRepository) -> None:
    """Durable deletion precedes cache invalidation and cleans visible content."""
    conversation_id, run_id = history.record_answer(
        tenant_id="org-a",
        user_id="user-alice",
        question="delete me",
        result=_result(),
    )

    assert history.delete_conversation(
        conversation_id=conversation_id,
        tenant_id="org-a",
        user_id="user-alice",
    ) == [run_id]
    assert history.get_conversation(
        conversation_id=conversation_id,
        tenant_id="org-a",
        user_id="user-alice",
    ) is None


@pytest.mark.asyncio
async def test_answer_survives_history_write_failure_without_database_details() -> None:
    """History-only failure returns the answer with an explicit unsaved warning."""
    workflow = AsyncMock()
    workflow.ainvoke.return_value = {"result": _result("available answer")}
    history = MagicMock()
    history.record_answer.side_effect = RuntimeError("postgres://user:secret@internal")
    app = FastAPI()
    app.state.workflows = {"qa": workflow}
    app.state.qa_history = history
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/qa/ask",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "app": app,
        }
    )
    user = UserContext("user-alice", "alice", UserRole.VIEWER, "org-a")
    audit = MagicMock()
    webhook = MagicMock(trigger=AsyncMock())

    with patch("api.routers.qa.get_audit_service", return_value=audit), patch(
        "api.routers.qa.get_webhook_service", return_value=webhook
    ):
        response = await ask_question(request, Response(), QuestionRequest(question="question"), user)

    assert response.answer == "available answer"
    assert response.history_saved is False
    assert response.warning_code == "qa_history_not_saved"
    assert "secret" not in repr(response)


@pytest.mark.asyncio
async def test_saved_answer_associates_cache_with_persisted_run() -> None:
    """History run IDs are linked to exact cache keys for bounded deletion."""
    workflow = AsyncMock()
    workflow.ainvoke.return_value = {
        "result": _result("cached answer"),
        "knowledge_revision": 7,
    }
    history = MagicMock()
    history.record_answer.return_value = ("conversation-1", "run-1")
    qa_cache = MagicMock(associate_answer=AsyncMock())
    app = FastAPI()
    app.state.workflows = {"qa": workflow}
    app.state.qa_history = history
    app.state.qa_cache = qa_cache
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/qa/ask",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "app": app,
        }
    )
    user = UserContext("user-alice", "alice", UserRole.VIEWER, "org-a")

    with patch("api.routers.qa.get_audit_service", return_value=MagicMock()), patch(
        "api.routers.qa.get_webhook_service", return_value=MagicMock(trigger=AsyncMock())
    ):
        await ask_question(request, Response(), QuestionRequest(question="question"), user)

    assert workflow.ainvoke.await_args.args[0]["tenant_id"] == "org-a"
    qa_cache.associate_answer.assert_awaited_once_with(
        "run-1", "question", "user-alice", "hybrid", "org-a", 7
    )


def test_controlled_non_answer_is_persisted_in_authorized_history(
    history: PostgreSQLQAHistoryRepository,
) -> None:
    from domain.evidence import (
        EvidenceAssessment,
        EvidenceReasonCode,
        EvidenceState,
        QAResponseStatus,
    )

    assessment = EvidenceAssessment(
        states=(EvidenceState.INSUFFICIENT_EVIDENCE,),
        response_status=QAResponseStatus.INSUFFICIENT_EVIDENCE,
        reason_codes=(EvidenceReasonCode.ZERO_RESULTS,),
        policy_version="evidence-v1",
        calibration_version="calibration-v1",
    )
    controlled = QAResult(
        question="未知记录是什么？",
        answer="未检索到可验证的证据，无法生成可靠答案。",
        contexts=[],
        intent=QueryIntent.FACTOID,
        confidence=0.0,
        degradation_code="insufficient_verified_evidence",
        response_status=QAResponseStatus.INSUFFICIENT_EVIDENCE,
        evidence_assessment=assessment,
    )

    conversation_id, run_id = history.record_answer(
        tenant_id="org-a",
        user_id="user-alice",
        question=controlled.question,
        result=controlled,
    )

    detail = history.get_conversation(
        conversation_id=conversation_id,
        tenant_id="org-a",
        user_id="user-alice",
    )
    assert detail is not None
    assert detail["messages"][-1].content == controlled.answer
    assert detail["runs"][0]["id"] == run_id
    assert detail["runs"][0]["degradation_code"] == "insufficient_verified_evidence"
