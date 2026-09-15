"""Durable task ownership registry used before any Celery control access."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import insert, select, update

from infrastructure.postgres.database import (
    DatabaseConflictError,
    DatabaseService,
    DatabaseUnavailableError,
)
from infrastructure.postgres.models import task_runs
from infrastructure.security.security_state import (
    CorruptSecurityStateError,
    SecurityStateBackend,
    SecurityStateUnavailableError,
)
from shared.config import settings
from shared.utils.metrics import task_registry_operations_total

TaskState = Literal[
    "reserved",
    "queued",
    "started",
    "retrying",
    "succeeded",
    "failed",
    "revoked",
    "publish_failed",
]
_STATES = {
    "reserved",
    "queued",
    "started",
    "retrying",
    "succeeded",
    "failed",
    "revoked",
    "publish_failed",
}


class TaskRegistryUnavailableError(RuntimeError):
    """Signal a task registry unavailable failure."""
    def __init__(self) -> None:
        """Initialize the task registry unavailable error."""
        super().__init__("task ownership registry is unavailable")


@dataclass(frozen=True)
class TaskRecord:
    """Represent task record."""
    task_id: str
    org_id: str
    actor_id: str
    kind: str
    state: TaskState
    file_reference: str
    created_at: str
    updated_at: str

    @property
    def description(self) -> str:
        """Return a readable operation description without exposing internal IDs."""
        labels = {
            "ingest": "文档入库",
            "batch_ingest": "批量文档入库",
            "knowledge_update": "知识更新",
            "cleanup": "任务清理",
            "evidence_release": "发布评测",
        }
        label = labels.get(self.kind, self.kind or "后台任务")
        return f"{label}：{self.file_reference}" if self.file_reference else label


class TaskRegistry:
    """Represent task registry."""
    _PREFIX = "security:v1:task:"

    def __init__(
        self,
        backend: SecurityStateBackend,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Initialize the task registry."""
        self.backend = backend
        self.ttl_seconds = ttl_seconds or settings.task_registry_ttl_seconds

    @classmethod
    def _key(cls, task_id: str) -> str:
        """Build the storage key for a task record."""
        return f"{cls._PREFIX}{task_id}"

    @staticmethod
    def _org_index(org_id: str) -> str:
        """Build the task index for an organization."""
        return f"security:v1:task-org:{org_id}"

    @staticmethod
    def _now() -> str:
        """Return the current UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _envelope(record: TaskRecord) -> dict:
        """Wrap a task record in its versioned storage envelope."""
        return {"version": 1, "kind": "task", "data": asdict(record)}

    @staticmethod
    def _load(value: dict) -> TaskRecord:
        """Load the task registry."""
        if value.get("version") != 1 or value.get("kind") != "task":
            raise CorruptSecurityStateError()
        data = value.get("data")
        if not isinstance(data, dict) or data.get("state") not in _STATES:
            raise CorruptSecurityStateError()
        try:
            return TaskRecord(**data)
        except (TypeError, ValueError) as error:
            raise CorruptSecurityStateError() from error

    def _run_named(self, operation: str, callback):
        """Run the named."""
        try:
            value = callback()
            task_registry_operations_total.labels(operation, "success").inc()
            return value
        except TaskRegistryUnavailableError:
            task_registry_operations_total.labels(operation, "unavailable").inc()
            raise
        except (SecurityStateUnavailableError, CorruptSecurityStateError) as error:
            task_registry_operations_total.labels(operation, "unavailable").inc()
            raise TaskRegistryUnavailableError() from error

    def reserve(
        self,
        task_id: str,
        *,
        org_id: str,
        actor_id: str,
        kind: str,
        file_reference: str = "",
    ) -> TaskRecord:
        """Handle reserve for the task registry."""
        if not task_id or not org_id or not actor_id:
            raise ValueError("task reservation requires host identity")

        def operation() -> TaskRecord:
            """Handle operation for the task registry."""
            with self.backend.locked(f"task-reserve:{task_id}"):
                if self.backend.read(self._key(task_id)) is not None:
                    raise ValueError("task ID is already reserved")
                now = self._now()
                record = TaskRecord(
                    task_id=task_id,
                    org_id=org_id,
                    actor_id=actor_id,
                    kind=kind,
                    state="reserved",
                    file_reference=file_reference,
                    created_at=now,
                    updated_at=now,
                )
                self.backend.write(
                    self._key(task_id),
                    self._envelope(record),
                    add_indexes=(self._org_index(org_id),),
                    ttl_seconds=self.ttl_seconds,
                )
                return record

        return self._run_named("reserve", operation)

    def update_state(self, task_id: str, state: TaskState) -> TaskRecord | None:
        """Update the state."""
        if state not in _STATES:
            raise ValueError("unsupported task state")

        def operation() -> TaskRecord | None:
            """Handle operation for the task registry."""
            value = self.backend.read(self._key(task_id))
            if value is None:
                return None
            current = self._load(value)
            updated = TaskRecord(
                **{
                    **asdict(current),
                    "state": state,
                    "updated_at": self._now(),
                }
            )
            self.backend.write(
                self._key(task_id),
                self._envelope(updated),
                add_indexes=(self._org_index(updated.org_id),),
                ttl_seconds=self.ttl_seconds,
            )
            return updated

        return self._run_named("update", operation)

    def release(self, task_id: str) -> bool:
        """Handle release for the task registry."""
        def operation() -> bool:
            """Handle operation for the task registry."""
            value = self.backend.read(self._key(task_id))
            if value is None:
                return False
            current = self._load(value)
            return self.backend.delete(
                self._key(task_id),
                remove_indexes=(self._org_index(current.org_id),),
            )

        return bool(self._run_named("release", operation))

    def get_owned(self, task_id: str, org_id: str) -> TaskRecord | None:
        """Return the owned."""
        def operation() -> TaskRecord | None:
            """Handle operation for the task registry."""
            value = self.backend.read(self._key(task_id))
            if value is None:
                return None
            record = self._load(value)
            return record if record.org_id == org_id else None

        return self._run_named("get", operation)

    def list_owned(self, org_id: str, limit: int = 50) -> list[TaskRecord]:
        """List the owned."""
        def operation() -> list[TaskRecord]:
            """Handle operation for the task registry."""
            records: list[TaskRecord] = []
            for key in self.backend.members(self._org_index(org_id)):
                value = self.backend.read(key)
                if value is None:
                    continue
                record = self._load(value)
                if record.org_id == org_id:
                    records.append(record)
            records.sort(key=lambda item: item.created_at, reverse=True)
            return records[: max(0, min(limit, 200))]

        return self._run_named("list", operation)


class PostgreSQLTaskRunRepository:
    """Retain task ownership and terminal history beyond Redis expiry."""

    def __init__(self, database: DatabaseService) -> None:
        self.database = database

    @staticmethod
    def _from_row(row) -> TaskRecord:
        values = dict(row)
        return TaskRecord(
            task_id=values["id"],
            org_id=values["tenant_id"],
            actor_id=values["user_id"],
            kind=values["task_type"],
            state=values["status"],
            file_reference=values["safe_file_reference"],
            created_at=values["created_at"].isoformat(),
            updated_at=values["updated_at"].isoformat(),
        )

    def reserve(
        self,
        task_id: str,
        *,
        org_id: str,
        actor_id: str,
        kind: str,
        file_reference: str = "",
    ) -> TaskRecord:
        now = datetime.now(timezone.utc)
        try:
            with self.database.session() as session:
                session.execute(
                    insert(task_runs).values(
                        id=task_id,
                        tenant_id=org_id,
                        user_id=actor_id,
                        task_type=kind,
                        status="reserved",
                        safe_file_reference=file_reference,
                        created_at=now,
                        updated_at=now,
                    )
                )
        except DatabaseConflictError as error:
            raise ValueError("task ID is already reserved") from error
        return TaskRecord(
            task_id=task_id,
            org_id=org_id,
            actor_id=actor_id,
            kind=kind,
            state="reserved",
            file_reference=file_reference,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )

    def update_state(self, task_id: str, state: TaskState) -> TaskRecord | None:
        now = datetime.now(timezone.utc)
        values = {
            "status": state,
            "updated_at": now,
            "started_at": now if state == "started" else None,
            "finished_at": now if state in {"succeeded", "failed", "revoked", "publish_failed"} else None,
            "failure_category": (
                "publish_failed" if state == "publish_failed" else "execution_failed" if state == "failed" else None
            ),
        }
        # Do not erase a previously recorded start time on later transitions.
        if state != "started":
            values.pop("started_at")
        with self.database.session() as session:
            result = session.execute(
                update(task_runs).where(task_runs.c.id == task_id).values(**values)
            )
            if result.rowcount != 1:
                return None
            row = session.execute(
                select(task_runs).where(task_runs.c.id == task_id)
            ).mappings().one()
        return self._from_row(row)

    def get_owned(self, task_id: str, org_id: str) -> TaskRecord | None:
        with self.database.session() as session:
            row = session.execute(
                select(task_runs).where(
                    task_runs.c.id == task_id,
                    task_runs.c.tenant_id == org_id,
                )
            ).mappings().one_or_none()
        return self._from_row(row) if row else None

    def list_owned(self, org_id: str, limit: int = 50) -> list[TaskRecord]:
        with self.database.session() as session:
            rows = session.execute(
                select(task_runs)
                .where(task_runs.c.tenant_id == org_id)
                .order_by(task_runs.c.created_at.desc())
                .limit(max(0, min(limit, 200)))
            ).mappings().all()
        return [self._from_row(row) for row in rows]


class PersistentTaskRegistry:
    """Use PostgreSQL as durable fact source and Redis for active coordination."""

    def __init__(self, active: TaskRegistry, history: PostgreSQLTaskRunRepository) -> None:
        self.active = active
        self.history = history

    @staticmethod
    def _durable(callback):
        try:
            return callback()
        except DatabaseUnavailableError as error:
            raise TaskRegistryUnavailableError() from error

    def reserve(self, task_id: str, **kwargs) -> TaskRecord:
        record = self._durable(lambda: self.history.reserve(task_id, **kwargs))
        try:
            self.active.reserve(task_id, **kwargs)
        except Exception:
            self._durable(lambda: self.history.update_state(task_id, "publish_failed"))
            raise
        return record

    def update_state(self, task_id: str, state: TaskState) -> TaskRecord | None:
        durable = self._durable(lambda: self.history.update_state(task_id, state))
        try:
            self.active.update_state(task_id, state)
        except TaskRegistryUnavailableError:
            pass
        return durable

    def release(self, task_id: str) -> bool:
        durable = self._durable(
            lambda: self.history.update_state(task_id, "publish_failed")
        ) is not None
        try:
            self.active.release(task_id)
        except TaskRegistryUnavailableError:
            pass
        return durable

    def get_owned(self, task_id: str, org_id: str) -> TaskRecord | None:
        try:
            active = self.active.get_owned(task_id, org_id)
        except TaskRegistryUnavailableError:
            active = None
        return active or self._durable(lambda: self.history.get_owned(task_id, org_id))

    def list_owned(self, org_id: str, limit: int = 50) -> list[TaskRecord]:
        return self._durable(lambda: self.history.list_owned(org_id, limit))


def update_task_state_safely(task_id: str | None, state: TaskState) -> None:
    """Best-effort worker lifecycle writeback with no unsafe payloads."""
    if not task_id:
        return
    try:
        get_task_registry().update_state(task_id, state)
    except (TaskRegistryUnavailableError, ValueError):
        task_registry_operations_total.labels("lifecycle", "unavailable").inc()


_registry: TaskRegistry | PersistentTaskRegistry | None = None
def get_task_registry() -> TaskRegistry | PersistentTaskRegistry:
    """Return PostgreSQL durable history with Redis active coordination."""
    global _registry
    if _registry is None:
        from infrastructure.postgres.database import get_database_service
        from infrastructure.security.security_state import RedisSecurityStateBackend

        active_backend = RedisSecurityStateBackend(
            settings.redis_url,
            max_record_bytes=settings.security_state_max_record_bytes,
            socket_timeout_seconds=settings.security_state_timeout_seconds,
        )
        _registry = PersistentTaskRegistry(
            TaskRegistry(active_backend),
            PostgreSQLTaskRunRepository(get_database_service()),
        )
    return _registry


def reset_task_registry() -> None:
    """Reset the task registry."""
    global _registry
    _registry = None


__all__ = [
    "TaskRecord",
    "TaskRegistry",
    "PersistentTaskRegistry",
    "PostgreSQLTaskRunRepository",
    "TaskRegistryUnavailableError",
    "TaskState",
    "get_task_registry",
    "reset_task_registry",
    "update_task_state_safely",
]
