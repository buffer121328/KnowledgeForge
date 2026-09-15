"""ATDD for explicit PostgreSQL-backed initial administrator bootstrap."""

from __future__ import annotations

import pytest

from auth.bootstrap import AdminBootstrapError, bootstrap_initial_admin
from infrastructure.postgres.identity_store import PostgreSQLIdentityStore
from auth.jwt_service import AuthService
from domain.identity import UserRole
from infrastructure.postgres.database import DatabaseService
from infrastructure.postgresql_migrations import upgrade
from sqlalchemy import create_engine


def _database(tmp_path, name: str) -> DatabaseService:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / name}")
    upgrade(engine)
    return DatabaseService.from_engine(engine)


def test_empty_durable_store_creates_one_active_admin(tmp_path) -> None:
    database = _database(tmp_path, "empty.sqlite3")
    result = bootstrap_initial_admin(
        username="root-admin",
        email="root@example.test",
        org_id="org-root",
        password="correct-horse-battery",
        database=database,
    )

    users = PostgreSQLIdentityStore(database).list_users()
    assert len(users) == 1
    assert users[0]["user_id"] == result.user_id
    assert users[0]["role"] == UserRole.ORGANIZATION_ADMIN
    assert users[0]["is_active"] is True
    assert AuthService().verify_password(
        "correct-horse-battery", users[0]["password_hash"]
    )
    assert "correct-horse-battery" not in repr(result)
    assert "correct-horse-battery" not in repr(users)


def test_existing_identity_refuses_bootstrap_without_mutation(tmp_path) -> None:
    database = _database(tmp_path, "existing.sqlite3")
    kwargs = dict(
        username="root-admin",
        email="root@example.test",
        org_id="org-root",
        password="correct-horse-battery",
        database=database,
    )
    bootstrap_initial_admin(**kwargs)

    with pytest.raises(AdminBootstrapError, match="not empty"):
        bootstrap_initial_admin(**{**kwargs, "username": "second-admin"})

    assert len(PostgreSQLIdentityStore(database).list_users()) == 1


def test_weak_password_is_rejected_before_postgresql_mutation(tmp_path) -> None:
    database = _database(tmp_path, "weak.sqlite3")
    with pytest.raises(AdminBootstrapError, match="12 to 128"):
        bootstrap_initial_admin(
            username="admin",
            email="admin@example.test",
            org_id="org-root",
            password="short",
            database=database,
        )
    assert PostgreSQLIdentityStore(database).list_users() == []


def test_postgresql_bootstrap_creates_one_admin(tmp_path) -> None:
    """Initial administrator bootstrap also supports the relational fact source."""
    database = _database(tmp_path, "bootstrap.sqlite3")

    result = bootstrap_initial_admin(
        username="root-postgres",
        email="root-postgres@example.test",
        org_id="org-root",
        password="correct-horse-battery",
        database=database,
    )

    users = PostgreSQLIdentityStore(database).list_users()
    assert [user["user_id"] for user in users] == [result.user_id]
    assert users[0]["role"] == UserRole.ORGANIZATION_ADMIN
