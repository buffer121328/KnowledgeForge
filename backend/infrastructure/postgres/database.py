"""Bounded SQLAlchemy database service for relational business repositories."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from infrastructure.postgres.models import schema_migrations
from infrastructure.security.security_state import SecurityStateUnavailableError


class DatabaseConfigurationError(RuntimeError):
    """Report an invalid database configuration without echoing secret values."""


class DatabaseUnavailableError(SecurityStateUnavailableError):
    """Report an unavailable or incompatible relational fact source."""


class DatabaseConflictError(RuntimeError):
    """Report a stable relational uniqueness or integrity conflict."""


class DatabaseService:
    """Own a SQLAlchemy engine, short transaction sessions and safe probes."""

    def __init__(
        self,
        database_url: str,
        *,
        pool_size: int = 5,
        max_overflow: int = 10,
        connect_timeout_seconds: float = 3.0,
        statement_timeout_ms: int = 5000,
        required_schema_version: int = 1,
    ) -> None:
        """Create a bounded engine without creating or altering database schema."""
        if not database_url.strip():
            raise DatabaseConfigurationError("database URL is required")
        connect_args: dict[str, Any] = {}
        if database_url.startswith("postgresql"):
            connect_args = {
                "connect_timeout": max(1, int(connect_timeout_seconds)),
                "options": f"-c statement_timeout={statement_timeout_ms}",
            }
        kwargs: dict[str, Any] = {
            "pool_pre_ping": True,
            "connect_args": connect_args,
        }
        if not database_url.startswith("sqlite"):
            kwargs.update(pool_size=pool_size, max_overflow=max_overflow)
        self.engine = create_engine(database_url, **kwargs)
        self.required_schema_version = required_schema_version
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)
        self._closed = False

    @classmethod
    def from_engine(
        cls,
        engine: Engine,
        *,
        required_schema_version: int = 1,
    ) -> "DatabaseService":
        """Build a service around a caller-owned engine for tests or embedding."""
        service = cls.__new__(cls)
        service.engine = engine
        service.required_schema_version = required_schema_version
        service._session_factory = sessionmaker(engine, expire_on_commit=False)
        service._closed = False
        return service

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Commit one short transaction or roll it back with safe error types."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except IntegrityError as error:
            session.rollback()
            raise DatabaseConflictError("relational constraint conflict") from error
        except SQLAlchemyError as error:
            session.rollback()
            raise DatabaseUnavailableError() from error
        finally:
            session.close()

    def schema_version(self) -> int:
        """Return the highest applied schema version or zero for missing schema."""
        try:
            with self.engine.connect() as connection:
                value = connection.execute(select(schema_migrations.c.version).order_by(schema_migrations.c.version.desc()).limit(1)).scalar_one_or_none()
        except SQLAlchemyError:
            return 0
        return int(value or 0)

    def ping(self) -> bool:
        """Return whether connectivity and the required explicit schema are ready."""
        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            return self.schema_version() >= self.required_schema_version
        except SQLAlchemyError:
            return False

    def close(self) -> None:
        """Dispose the engine pool exactly once."""
        if self._closed:
            return
        self.engine.dispose()
        self._closed = True


_database_service: DatabaseService | None = None


def get_database_service() -> DatabaseService:
    """Return the configured process database service without changing schema."""
    global _database_service
    from shared.config import settings

    if _database_service is None or _database_service._closed:
        _database_service = DatabaseService(
            settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            connect_timeout_seconds=settings.database_connect_timeout_seconds,
            statement_timeout_ms=settings.database_statement_timeout_ms,
            required_schema_version=settings.database_required_schema_version,
        )
    return _database_service


def set_database_service(service: DatabaseService | None) -> None:
    """Override the process database service for application assembly or tests."""
    global _database_service
    _database_service = service


__all__ = [
    "DatabaseConfigurationError",
    "DatabaseConflictError",
    "DatabaseService",
    "DatabaseUnavailableError",
    "get_database_service",
    "set_database_service",
]
