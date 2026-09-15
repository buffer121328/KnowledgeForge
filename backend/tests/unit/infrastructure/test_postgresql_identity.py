"""ATDD for relational identity, invitation and API Key fact stores."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select

from auth.apikey_service import APIKeyService
from auth.apikey_store import PostgreSQLKeyStore
from auth.jwt_service import AuthService
from auth.user_service import IdentityServiceError, UserService
from domain.identity import Permission, UserRole, builtin_role_records
from infrastructure.postgres.database import DatabaseService
from infrastructure.postgres.identity_store import PostgreSQLIdentityStore
from infrastructure.postgresql_migrations import upgrade
from infrastructure.postgres.models import api_keys, users
from shared.config import settings


@pytest.fixture
def database(tmp_path) -> DatabaseService:
    """Create one migrated isolated relational database."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'identity.sqlite3'}")
    upgrade(engine)
    return DatabaseService.from_engine(engine)


def _record(auth: AuthService, *, username: str = "alice", email: str = "alice@example.test", tenant: str = "org-a") -> dict:
    """Build a compatible migrated user record."""
    return {
        "user_id": f"user-{username}",
        "username": username,
        "email": email,
        "display_name": username.title(),
        "password_hash": auth.hash_password("legacy-password"),
        "role": UserRole.VIEWER,
        "org_id": tenant,
        "department_id": "finance",
        "is_department_manager": True,
        "is_active": True,
        "token_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_login_at": None,
    }


def test_migrated_password_and_token_version_are_preserved(database: DatabaseService) -> None:
    """Relational users authenticate with their existing digest and version."""
    auth = AuthService()
    store = PostgreSQLIdentityStore(database)
    record = _record(auth)

    assert store.create_user(record)
    loaded = UserService(auth_service=auth, store=store).authenticate("alice", "legacy-password")

    assert loaded["user_id"] == record["user_id"]
    assert loaded["token_version"] == 3
    assert loaded["department_id"] == "finance"
    assert loaded["is_department_manager"] is True


def test_duplicate_username_or_email_is_rejected_by_database(database: DatabaseService) -> None:
    """Concurrent-safe database constraints reject identity duplicates."""
    auth = AuthService()
    store = PostgreSQLIdentityStore(database)

    assert store.create_user(_record(auth))
    assert not store.create_user(_record(auth, username="alice", email="other@example.test", tenant="org-b"))
    assert not store.create_user(_record(auth, username="bob", email="alice@example.test", tenant="org-b"))


def test_soft_delete_invalidates_user_without_cross_tenant_lookup(database: DatabaseService) -> None:
    """Deletion hides the user and persists credential invalidation metadata."""
    auth = AuthService()
    store = PostgreSQLIdentityStore(database)
    record = _record(auth)
    assert store.create_user(record)

    assert UserService(store=store).find_by_id(record["user_id"], org_id="org-b", raise_on_missing=False) is None
    assert store.delete_user(record["user_id"])
    assert store.find_user_by_id(record["user_id"]) is None

    with database.engine.connect() as connection:
        row = connection.execute(select(users).where(users.c.id == record["user_id"])).first()
    assert row._mapping["is_active"] is False
    assert row._mapping["token_version"] == 4
    assert row._mapping["deleted_at"] is not None


def test_postgresql_store_exposes_canonical_builtin_roles(
    database: DatabaseService,
) -> None:
    """Relational identity uses domain-owned immutable role definitions."""

    roles = PostgreSQLIdentityStore(database).list_roles()

    assert {role["name"]: role for role in roles} == builtin_role_records()


def test_production_invitation_is_one_time_and_binds_tenant_role(
    database: DatabaseService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invitation consumption and user creation share one transaction."""
    store = PostgreSQLIdentityStore(database)
    raw_token = "invite-once"
    store.create_invitation(
        invitation_id="invite-1",
        token_hash=hashlib.sha256(raw_token.encode()).hexdigest(),
        tenant_id="org-invited",
        role=UserRole.EDITOR,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    monkeypatch.setattr(settings, "app_environment", "production")
    monkeypatch.setattr(settings, "allow_demo_default_org_registration", False)
    service = UserService(store=store)

    created = service.register_public_user_with_invitation(
        "invited",
        "a-long-enough-password",
        invitation_token=raw_token,
    )
    assert created["org_id"] == "org-invited"
    assert created["role"] == UserRole.EDITOR

    with pytest.raises(IdentityServiceError) as reused:
        service.register_public_user_with_invitation(
            "invited-again",
            "a-long-enough-password",
            invitation_token=raw_token,
        )
    assert reused.value.detail == "registration_invitation_invalid"


def test_postgresql_api_key_store_persists_digest_only(database: DatabaseService) -> None:
    """The relational Key store never receives or stores raw API Key material."""
    auth = AuthService()
    identity = PostgreSQLIdentityStore(database)
    assert identity.create_user(_record(auth))
    service = APIKeyService(store=PostgreSQLKeyStore(database))

    key, raw = service.create_key(
        name="automation",
        user_id="user-alice",
        org_id="org-a",
        role=UserRole.VIEWER,
        scopes=[Permission.QA_QUERY],
        actor_permissions=[Permission.QA_QUERY],
    )

    assert service.verify(raw, record_use=False).id == key.id
    with database.engine.connect() as connection:
        row = connection.execute(select(api_keys).where(api_keys.c.id == key.id)).first()
    assert row._mapping["key_hash"] == service.hash_key(raw)
    assert raw not in repr(dict(row._mapping))
