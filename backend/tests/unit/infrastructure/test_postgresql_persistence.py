"""ATDD coverage for explicit PostgreSQL persistence and schema readiness."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine

from infrastructure.postgres.database import (
    DatabaseConfigurationError,
    DatabaseService,
)
from infrastructure.postgres.models import metadata
from infrastructure.postgresql_migrations import REQUIRED_SCHEMA_VERSION, upgrade
from shared.config.settings import Settings


def _settings(**overrides: object) -> Settings:
    """Build isolated settings without reading a developer environment file."""
    return Settings(_env_file=None, **overrides)


@pytest.mark.parametrize("environment", ["development", "test", "production", "staging", ""])
def test_every_runtime_requires_database_url(environment: str) -> None:
    """Every assembled runtime requires its sole relational fact source."""
    candidate = _settings(
        app_environment=environment,
        database_url="",
    )

    with pytest.raises(DatabaseConfigurationError):
        candidate.validate_database_for_runtime(environment)


def test_removed_backend_selectors_cannot_change_runtime_repository_family() -> None:
    """Obsolete environment values are ignored instead of selecting a fallback."""
    candidate = _settings(
        database_url="postgresql://user:secret@db/internal",
        webhook_secret_encryption_key="protected-key",
        webhook_secret_key_reference="key-v1",
        security_state_backend="memory",
        document_catalog_backend="sqlite",
        qa_history_backend="disabled",
        webhook_fact_backend="legacy",
        task_history_backend="legacy",
    )

    candidate.validate_database_for_runtime("test")
    for name in (
        "security_state_backend",
        "document_catalog_backend",
        "qa_history_backend",
        "webhook_fact_backend",
        "task_history_backend",
    ):
        assert not hasattr(candidate, name)


def test_explicit_migration_advances_schema_version_idempotently(tmp_path: Path) -> None:
    """Migration command records one required version and can be repeated."""
    database_path = tmp_path / "migration.sqlite3"
    engine = create_engine(f"sqlite+pysqlite:///{database_path}")

    assert upgrade(engine) == REQUIRED_SCHEMA_VERSION
    assert upgrade(engine) == REQUIRED_SCHEMA_VERSION

    service = DatabaseService.from_engine(engine)
    assert service.schema_version() == REQUIRED_SCHEMA_VERSION
    assert service.ping() is True


def test_application_probe_does_not_create_missing_schema() -> None:
    """A readiness probe reports missing schema without running hidden DDL."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    service = DatabaseService.from_engine(engine)

    assert service.ping() is False
    assert "schema_migrations" not in metadata.tables or not engine.dialect.has_table(
        engine.connect(), "schema_migrations"
    )


def test_database_service_disposes_pool_once() -> None:
    """Lifespan cleanup releases the configured engine pool."""
    engine = MagicMock()
    service = DatabaseService.from_engine(engine)

    service.close()
    service.close()

    engine.dispose.assert_called_once_with()
