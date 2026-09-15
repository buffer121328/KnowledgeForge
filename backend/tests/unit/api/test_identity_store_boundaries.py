"""ATDD for explicit identity storage contracts and adapter ownership."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[3]


def _imports(path: Path) -> set[str]:
    """Return direct import targets declared by one Python source file."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_identity_modules_enforce_contract_and_adapter_dependency_direction() -> None:
    """Contracts/test doubles are neutral and adapters avoid auth policy state."""

    contracts = _imports(BACKEND_ROOT / "auth" / "identity_contracts.py")
    memory = _imports(BACKEND_ROOT / "auth" / "memory_identity_store.py")
    redis = _imports(BACKEND_ROOT / "infrastructure" / "cache" / "identity_store.py")
    postgresql = _imports(
        BACKEND_ROOT / "infrastructure" / "postgres" / "identity_store.py"
    )

    assert not any(name.startswith("infrastructure") for name in contracts | memory)
    for adapter_imports in (redis, postgresql):
        assert "auth.jwt_service" not in adapter_imports
        assert "auth.memory_accounts" not in adapter_imports
        assert "auth.user_service" not in adapter_imports
        assert "auth.role_service" not in adapter_imports
    assert not (BACKEND_ROOT / "auth" / "identity_store.py").exists()


def test_default_identity_provider_selects_only_postgresql(monkeypatch) -> None:
    """Runtime composition lazily constructs one PostgreSQL identity adapter."""

    from auth import identity_provider
    from infrastructure.postgres import identity_store as postgresql_identity_store

    created: list[object] = []

    class PostgreSQLDouble:
        def __init__(self) -> None:
            created.append(self)

    monkeypatch.setattr(
        postgresql_identity_store,
        "PostgreSQLIdentityStore",
        PostgreSQLDouble,
    )
    identity_provider.reset_identity_store()

    first = identity_provider.get_identity_store()
    second = identity_provider.get_identity_store()

    assert first is second
    assert created == [first]


def test_invitation_registration_capability_is_protocol_based() -> None:
    """Invitation admission does not depend on a concrete PostgreSQL class."""

    from auth.identity_contracts import InvitationIdentityStore
    from auth.memory_identity_store import MemoryIdentityStore

    memory = MemoryIdentityStore({}, {})
    assert not isinstance(memory, InvitationIdentityStore)

    class InvitationCapableStore(MemoryIdentityStore):
        def create_invitation(self, **_kwargs) -> None:
            return None

        def consume_invitation_and_create_user(self, **kwargs):
            return kwargs["record"]

    assert isinstance(InvitationCapableStore({}, {}), InvitationIdentityStore)


def test_invitation_registration_fails_stably_for_incapable_store() -> None:
    """Business policy checks protocol capability rather than concrete type."""

    from auth.memory_identity_store import MemoryIdentityStore
    from auth.user_service import IdentityServiceError, UserService

    service = UserService(store=MemoryIdentityStore({}, {}))

    with pytest.raises(IdentityServiceError) as captured:
        service.register_public_user_with_invitation(
            "invitee",
            "a-long-enough-password",
            invitation_token="opaque-invitation",
        )

    assert captured.value.detail == "registration_invitation_unavailable"


def test_builtin_roles_are_canonical_across_memory_and_redis_adapters() -> None:
    """Every adapter exposes independent copies of canonical built-in roles."""

    from auth.memory_identity_store import MemoryIdentityStore
    from domain.identity import builtin_role_records
    from infrastructure.cache.identity_store import RedisIdentityStore
    from infrastructure.security.security_state import MemorySecurityStateBackend

    expected = builtin_role_records()
    memory = MemoryIdentityStore({}, builtin_role_records())
    redis = RedisIdentityStore(MemorySecurityStateBackend())

    assert {role["name"]: role for role in memory.list_roles()} == expected
    assert {role["name"]: role for role in redis.list_roles()} == expected
    memory.roles["viewer"]["description"] = "mutated"
    assert builtin_role_records()["viewer"]["description"] != "mutated"


def test_redis_identity_adapter_reads_existing_versioned_envelope() -> None:
    """The focused Redis adapter preserves existing keys and envelope shape."""

    from domain.identity import UserRole
    from infrastructure.cache.identity_store import RedisIdentityStore
    from infrastructure.security.security_state import MemorySecurityStateBackend

    backend = MemorySecurityStateBackend()
    backend.write(
        "security:v1:user:user-existing",
        {
            "version": 1,
            "kind": "user",
            "data": {
                "user_id": "user-existing",
                "username": "existing",
                "role": "viewer",
                "org_id": "org-existing",
                "token_version": 4,
            },
        },
        add_indexes=("security:v1:users",),
    )

    loaded = RedisIdentityStore(backend).find_user_by_id("user-existing")

    assert loaded is not None
    assert loaded["role"] is UserRole.VIEWER
    assert loaded["token_version"] == 4


def test_concrete_adapter_import_does_not_load_auth_service_or_demo_store() -> None:
    """Focused adapter imports do not execute authentication policy modules."""

    for name in (
        "auth.jwt_service",
        "auth.memory_accounts",
        "infrastructure.cache.identity_store",
        "infrastructure.postgres.identity_store",
    ):
        sys.modules.pop(name, None)

    importlib.import_module("infrastructure.cache.identity_store")
    importlib.import_module("infrastructure.postgres.identity_store")

    assert "auth.jwt_service" not in sys.modules
    assert "auth.memory_accounts" not in sys.modules
