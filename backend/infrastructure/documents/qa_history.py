"""PostgreSQL repository and service for owner-scoped QA history."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, insert, or_, select, update

from domain.knowledge import QAResult
from domain.qa_history import QAConversation, QAConversationPage, QAFeedbackRating, QAMessage
from infrastructure.postgres.database import DatabaseService, get_database_service
from infrastructure.postgres.models import (
    qa_conversations,
    qa_feedback,
    qa_messages,
    qa_runs,
    qa_sources,
)


class QAHistoryUnavailableError(RuntimeError):
    """Report durable history unavailability without backend details."""


def _cursor(updated_at: datetime, conversation_id: str) -> str:
    """Encode a stable opaque pagination cursor."""
    raw = json.dumps([updated_at.isoformat(), conversation_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(value: str) -> tuple[datetime, str]:
    """Decode and validate a conversation cursor."""
    try:
        padding = "=" * (-len(value) % 4)
        timestamp, conversation_id = json.loads(base64.urlsafe_b64decode(value + padding))
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc), str(conversation_id)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid conversation cursor") from error


class PostgreSQLQAHistoryRepository:
    """Persist QA conversations, messages, runs, sources and feedback."""

    def __init__(self, database: DatabaseService | None = None) -> None:
        """Initialize the QA history repository."""
        self.database = database or get_database_service()

    @staticmethod
    def _conversation(row: Any) -> QAConversation:
        """Hydrate a conversation domain record."""
        data = row._mapping
        return QAConversation(
            id=data["id"],
            tenant_id=data["tenant_id"],
            user_id=data["user_id"],
            title=data["title"],
            status=data["status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    def create_conversation(self, *, tenant_id: str, user_id: str, title: str = "") -> QAConversation:
        """Create one user-owned conversation."""
        now = datetime.now(timezone.utc)
        conversation_id = f"conv_{uuid4().hex}"
        with self.database.session() as session:
            session.execute(
                insert(qa_conversations).values(
                    id=conversation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    title=title[:255],
                    status="active",
                    created_at=now,
                    updated_at=now,
                )
            )
        return QAConversation(conversation_id, tenant_id, user_id, title[:255], "active", now, now)

    def list_conversations(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
        cursor: str | None = None,
    ) -> QAConversationPage:
        """List one owner's conversations using stable reverse cursor pagination."""
        bounded_limit = min(max(limit, 1), 100)
        query = select(qa_conversations).where(
            qa_conversations.c.tenant_id == tenant_id,
            qa_conversations.c.user_id == user_id,
            qa_conversations.c.deleted_at.is_(None),
        )
        if cursor:
            updated_at, conversation_id = _decode_cursor(cursor)
            query = query.where(
                or_(
                    qa_conversations.c.updated_at < updated_at,
                    (
                        (qa_conversations.c.updated_at == updated_at)
                        & (qa_conversations.c.id < conversation_id)
                    ),
                )
            )
        query = query.order_by(qa_conversations.c.updated_at.desc(), qa_conversations.c.id.desc()).limit(bounded_limit + 1)
        with self.database.session() as session:
            rows = session.execute(query).all()
        has_more = len(rows) > bounded_limit
        selected = rows[:bounded_limit]
        items = [self._conversation(row) for row in selected]
        next_cursor = _cursor(items[-1].updated_at, items[-1].id) if has_more and items else None
        return QAConversationPage(items=items, next_cursor=next_cursor)

    def get_conversation(self, *, conversation_id: str, tenant_id: str, user_id: str) -> dict[str, Any] | None:
        """Return an owned conversation with ordered messages and run metadata."""
        with self.database.session() as session:
            conversation_row = session.execute(
                select(qa_conversations).where(
                    qa_conversations.c.id == conversation_id,
                    qa_conversations.c.tenant_id == tenant_id,
                    qa_conversations.c.user_id == user_id,
                    qa_conversations.c.deleted_at.is_(None),
                )
            ).first()
            if conversation_row is None:
                return None
            message_rows = session.execute(
                select(qa_messages).where(
                    qa_messages.c.tenant_id == tenant_id,
                    qa_messages.c.conversation_id == conversation_id,
                ).order_by(qa_messages.c.sequence)
            ).all()
            run_rows = session.execute(
                select(qa_runs).where(
                    qa_runs.c.tenant_id == tenant_id,
                    qa_runs.c.conversation_id == conversation_id,
                    qa_runs.c.user_id == user_id,
                ).order_by(qa_runs.c.created_at)
            ).all()
            run_payload: list[dict[str, Any]] = []
            for run_row in run_rows:
                run = dict(run_row._mapping)
                source_rows = session.execute(
                    select(qa_sources).where(
                        qa_sources.c.tenant_id == tenant_id,
                        qa_sources.c.qa_run_id == run["id"],
                    ).order_by(qa_sources.c.position)
                ).all()
                run["sources"] = [dict(source._mapping) for source in source_rows]
                run_payload.append(run)
        return {
            "conversation": self._conversation(conversation_row),
            "messages": [
                QAMessage(
                    id=row._mapping["id"],
                    sequence=row._mapping["sequence"],
                    role=row._mapping["role"],
                    content=row._mapping["content"],
                    created_at=row._mapping["created_at"],
                )
                for row in message_rows
            ],
            "runs": run_payload,
        }

    def record_answer(
        self,
        *,
        tenant_id: str,
        user_id: str,
        question: str,
        result: QAResult,
        conversation_id: str | None = None,
        retrieval_mode: str = "hybrid",
        knowledge_revision: int = 0,
        cache_hit_type: str = "none",
        duration_ms: int = 0,
        semantic_decision: str | None = None,
        model_version: str = "",
        rule_version: str = "",
    ) -> tuple[str, str]:
        """Persist messages, run and controlled sources in one owner-scoped transaction."""
        now = datetime.now(timezone.utc)
        run_id = f"run_{uuid4().hex}"
        with self.database.session() as session:
            if conversation_id:
                owned = session.execute(
                    select(qa_conversations.c.id).where(
                        qa_conversations.c.id == conversation_id,
                        qa_conversations.c.tenant_id == tenant_id,
                        qa_conversations.c.user_id == user_id,
                        qa_conversations.c.deleted_at.is_(None),
                    )
                ).scalar_one_or_none()
                if owned is None:
                    raise LookupError("conversation not found")
            else:
                conversation_id = f"conv_{uuid4().hex}"
                session.execute(
                    insert(qa_conversations).values(
                        id=conversation_id,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        title=question.strip()[:120],
                        status="active",
                        created_at=now,
                        updated_at=now,
                    )
                )
            last_sequence = session.execute(
                select(func.max(qa_messages.c.sequence)).where(
                    qa_messages.c.tenant_id == tenant_id,
                    qa_messages.c.conversation_id == conversation_id,
                )
            ).scalar_one_or_none()
            sequence = int(last_sequence or 0)
            session.execute(
                insert(qa_messages),
                [
                    {
                        "id": f"msg_{uuid4().hex}",
                        "tenant_id": tenant_id,
                        "conversation_id": conversation_id,
                        "sequence": sequence + 1,
                        "role": "user",
                        "content": question,
                        "created_at": now,
                    },
                    {
                        "id": f"msg_{uuid4().hex}",
                        "tenant_id": tenant_id,
                        "conversation_id": conversation_id,
                        "sequence": sequence + 2,
                        "role": "assistant",
                        "content": result.answer,
                        "created_at": now,
                    },
                ],
            )
            session.execute(
                insert(qa_runs).values(
                    id=run_id,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    retrieval_mode=retrieval_mode,
                    confidence=result.confidence,
                    intent=result.intent.value,
                    knowledge_revision=knowledge_revision,
                    model_version=model_version[:128],
                    rule_version=rule_version[:128],
                    degradation_code=result.degradation_code,
                    cache_hit_type=cache_hit_type,
                    duration_ms=max(duration_ms, 0),
                    semantic_decision=semantic_decision,
                    created_at=now,
                )
            )
            source_rows = []
            for position, context in enumerate(result.contexts[:20]):
                metadata = context.metadata or {}
                source_rows.append(
                    {
                        "id": f"source_{uuid4().hex}",
                        "tenant_id": tenant_id,
                        "qa_run_id": run_id,
                        "position": position,
                        "doc_id": str(metadata.get("doc_id") or context.source)[:128],
                        "chunk_id": str(metadata.get("chunk_id") or "")[:255],
                        "retrieval_type": context.retrieval_type[:32],
                        "score": context.score,
                        "display_summary": context.content[:500],
                    }
                )
            if source_rows:
                session.execute(insert(qa_sources), source_rows)
            session.execute(
                update(qa_conversations)
                .where(
                    qa_conversations.c.id == conversation_id,
                    qa_conversations.c.tenant_id == tenant_id,
                )
                .values(updated_at=now)
            )
        return conversation_id, run_id

    def delete_conversation(self, *, conversation_id: str, tenant_id: str, user_id: str) -> list[str] | None:
        """Soft-delete one owned conversation and clean user-visible content."""
        now = datetime.now(timezone.utc)
        with self.database.session() as session:
            owned = session.execute(
                select(qa_conversations.c.id).where(
                    qa_conversations.c.id == conversation_id,
                    qa_conversations.c.tenant_id == tenant_id,
                    qa_conversations.c.user_id == user_id,
                    qa_conversations.c.deleted_at.is_(None),
                )
            ).scalar_one_or_none()
            if owned is None:
                return None
            run_ids = list(
                session.execute(
                    select(qa_runs.c.id).where(
                        qa_runs.c.tenant_id == tenant_id,
                        qa_runs.c.conversation_id == conversation_id,
                    )
                ).scalars()
            )
            session.execute(
                update(qa_conversations)
                .where(qa_conversations.c.id == conversation_id, qa_conversations.c.tenant_id == tenant_id)
                .values(status="deleted", deleted_at=now, updated_at=now)
            )
            session.execute(
                update(qa_messages)
                .where(qa_messages.c.tenant_id == tenant_id, qa_messages.c.conversation_id == conversation_id)
                .values(content="")
            )
            if run_ids:
                session.execute(
                    update(qa_sources)
                    .where(qa_sources.c.tenant_id == tenant_id, qa_sources.c.qa_run_id.in_(run_ids))
                    .values(display_summary="")
                )
        return run_ids

    def upsert_feedback(
        self,
        *,
        qa_run_id: str,
        tenant_id: str,
        user_id: str,
        rating: QAFeedbackRating,
        note: str = "",
    ) -> dict[str, Any] | None:
        """Create or update one bounded owner feedback record."""
        now = datetime.now(timezone.utc)
        with self.database.session() as session:
            owned = session.execute(
                select(qa_runs.c.id).where(
                    qa_runs.c.id == qa_run_id,
                    qa_runs.c.tenant_id == tenant_id,
                    qa_runs.c.user_id == user_id,
                )
            ).scalar_one_or_none()
            if owned is None:
                return None
            existing = session.execute(
                select(qa_feedback.c.id).where(
                    qa_feedback.c.tenant_id == tenant_id,
                    qa_feedback.c.qa_run_id == qa_run_id,
                    qa_feedback.c.user_id == user_id,
                )
            ).scalar_one_or_none()
            values = {"rating": rating.value, "note": note[:1000], "updated_at": now}
            if existing:
                session.execute(update(qa_feedback).where(qa_feedback.c.id == existing).values(**values))
                feedback_id = existing
            else:
                feedback_id = f"feedback_{uuid4().hex}"
                session.execute(
                    insert(qa_feedback).values(
                        id=feedback_id,
                        tenant_id=tenant_id,
                        qa_run_id=qa_run_id,
                        user_id=user_id,
                        created_at=now,
                        **values,
                    )
                )
        return {"feedback_id": feedback_id, "qa_run_id": qa_run_id, "rating": rating.value, "note": note[:1000]}

    def cleanup_expired(self, *, tenant_id: str, retention_days: int) -> int:
        """Soft-delete and content-clean expired conversations for one tenant."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        with self.database.session() as session:
            ids = list(
                session.execute(
                    select(qa_conversations.c.id).where(
                        qa_conversations.c.tenant_id == tenant_id,
                        qa_conversations.c.updated_at < cutoff,
                        qa_conversations.c.deleted_at.is_(None),
                    )
                ).scalars()
            )
        count = 0
        for conversation_id in ids:
            with self.database.session() as session:
                owner = session.execute(
                    select(qa_conversations.c.user_id).where(
                        qa_conversations.c.id == conversation_id,
                        qa_conversations.c.tenant_id == tenant_id,
                    )
                ).scalar_one()
            if self.delete_conversation(conversation_id=conversation_id, tenant_id=tenant_id, user_id=owner) is not None:
                count += 1
        return count


__all__ = ["PostgreSQLQAHistoryRepository", "QAHistoryUnavailableError"]
