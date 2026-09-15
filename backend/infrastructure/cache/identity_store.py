"""Legacy Redis identity adapter over bounded shared security state."""

from __future__ import annotations

import copy
from contextlib import AbstractContextManager
from typing import Any, cast

from domain.identity import UserRole, builtin_role_records

from infrastructure.security.security_state import (
    CorruptSecurityStateError,
    SecurityStateBackend,
)

_USER_PREFIX = "security:v1:user:"
_ROLE_PREFIX = "security:v1:role:"
_USERS_INDEX = "security:v1:users"
_ROLES_INDEX = "security:v1:roles"

def _user_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return the user record."""
    value = copy.deepcopy(record)
    role = value.get("role")
    value["role"] = role.value if isinstance(role, UserRole) else str(role)
    return {"version": 1, "kind": "user", "data": value}


def _load_user(envelope: dict[str, Any]) -> dict[str, Any]:
    """Load the user."""
    if envelope.get("version") != 1 or envelope.get("kind") != "user":
        raise CorruptSecurityStateError()
    value = copy.deepcopy(envelope.get("data"))
    if not isinstance(value, dict):
        raise CorruptSecurityStateError()
    try:
        value["role"] = UserRole(value["role"])
    except (KeyError, ValueError, TypeError) as error:
        raise CorruptSecurityStateError() from error
    return value


def _role_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return the role record."""
    return {
        "version": 1,
        "kind": "role",
        "data": copy.deepcopy(record),
    }


def _load_role(envelope: dict[str, Any]) -> dict[str, Any]:
    """Load the role."""
    if envelope.get("version") != 1 or envelope.get("kind") != "role":
        raise CorruptSecurityStateError()
    value = copy.deepcopy(envelope.get("data"))
    if not isinstance(value, dict):
        raise CorruptSecurityStateError()
    return value

class RedisIdentityStore:
    """Shared identity state using the repository security-state backend."""

    def __init__(self, backend: SecurityStateBackend) -> None:
        """Initialize the Redis identity store."""
        self.backend = backend

    def _locked(self, name: str) -> AbstractContextManager[None]:
        """Narrow the legacy backend lock annotation to its runtime contract."""
        return cast(AbstractContextManager[None], self.backend.locked(name))

    @staticmethod
    def _user_key(user_id: str) -> str:
        """Build the Redis key for a user record."""
        return f"{_USER_PREFIX}{user_id}"

    @staticmethod
    def _role_key(role_name: str, org_id: str | None) -> str:
        """Build the Redis key for a role record."""
        owner = org_id or "builtin"
        return f"{_ROLE_PREFIX}{owner}:{role_name}"

    def find_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Find the user by username."""
        return next(
            (u for u in self.list_users() if u.get("username") == username),
            None,
        )

    def find_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Find the user by ID."""
        value = self.backend.read(self._user_key(user_id))
        return None if value is None else _load_user(value)

    def list_users(self) -> list[dict[str, Any]]:
        """List the users."""
        users: list[dict[str, Any]] = []
        for key in self.backend.members(_USERS_INDEX):
            value = self.backend.read(key)
            if value is not None:
                users.append(_load_user(value))
        return users

    def create_user(self, record: dict[str, Any]) -> bool:
        """Create the user."""
        with self._locked("identity-users"):
            if self.find_user_by_username(record["username"]):
                return False
            email = str(record.get("email") or "")
            if email and any(u.get("email") == email for u in self.list_users()):
                return False
            self.save_user(record)
            return True

    def save_user(self, record: dict[str, Any]) -> None:
        """Persist the user."""
        self.backend.write(
            self._user_key(record["user_id"]),
            _user_record(record),
            add_indexes=(_USERS_INDEX,),
        )

    def create_initial_user(self, record: dict[str, Any]) -> bool:
        """Create the first durable identity only while the shared store is empty."""
        with self._locked("identity-users"):
            if self.list_users():
                return False
            self.save_user(record)
            return True

    def delete_user(self, user_id: str) -> bool:
        """Delete the user."""
        return self.backend.delete(
            self._user_key(user_id),
            remove_indexes=(_USERS_INDEX,),
        )

    def find_role(
        self,
        role_name: str,
        org_id: str | None,
    ) -> dict[str, Any] | None:
        """Find the role."""
        builtin = builtin_role_records().get(role_name)
        if builtin and builtin.get("is_builtin"):
            return copy.deepcopy(builtin)
        value = self.backend.read(self._role_key(role_name, org_id))
        return None if value is None else _load_role(value)

    def list_roles(self) -> list[dict[str, Any]]:
        """List the roles."""
        roles = [
            copy.deepcopy(role)
            for role in builtin_role_records().values()
            if role.get("is_builtin")
        ]
        for key in self.backend.members(_ROLES_INDEX):
            value = self.backend.read(key)
            if value is not None:
                roles.append(_load_role(value))
        return roles

    def create_role(self, record: dict[str, Any]) -> bool:
        """Create the role."""
        with self._locked("identity-roles"):
            if self.find_role(record["name"], record.get("org_id") or None):
                return False
            self.save_role(record)
            return True

    def save_role(self, record: dict[str, Any]) -> None:
        """Persist the role."""
        self.backend.write(
            self._role_key(record["name"], record.get("org_id") or None),
            _role_record(record),
            add_indexes=(_ROLES_INDEX,),
        )

    def delete_role(self, role_name: str, org_id: str | None) -> bool:
        """Delete the role."""
        return self.backend.delete(
            self._role_key(role_name, org_id),
            remove_indexes=(_ROLES_INDEX,),
        )


__all__ = ["RedisIdentityStore"]
