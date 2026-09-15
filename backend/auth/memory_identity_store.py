"""Explicit in-memory identity adapter for tests and local service construction."""

from __future__ import annotations

from typing import Any


class MemoryIdentityStore:
    """Persist and retrieve memory identity data."""
    def __init__(
        self,
        users: dict[str, dict[str, Any]],
        roles: dict[str, dict[str, Any]],
    ) -> None:
        """Initialize the memory identity store."""
        self.users = users
        self.roles = roles

    @staticmethod
    def _role_key(role_name: str, org_id: str | None) -> str:
        """Build the storage key for a tenant-scoped role."""
        return f"{org_id}::{role_name}" if org_id else role_name

    def find_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Find the user by username."""
        return self.users.get(username)

    def find_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Find the user by ID."""
        return next(
            (user for user in self.users.values() if user["user_id"] == user_id),
            None,
        )

    def list_users(self) -> list[dict[str, Any]]:
        """List the users."""
        return list(self.users.values())

    def create_user(self, record: dict[str, Any]) -> bool:
        """Create the user."""
        if record["username"] in self.users:
            return False
        self.users[record["username"]] = record
        return True

    def save_user(self, record: dict[str, Any]) -> None:
        """Persist the user."""
        self.users[record["username"]] = record

    def delete_user(self, user_id: str) -> bool:
        """Delete the user."""
        user = self.find_user_by_id(user_id)
        if not user:
            return False
        return self.users.pop(user["username"], None) is not None

    def find_role(
        self,
        role_name: str,
        org_id: str | None,
    ) -> dict[str, Any] | None:
        """Find the role."""
        builtin = self.roles.get(role_name)
        if builtin and builtin.get("is_builtin"):
            return builtin
        return self.roles.get(self._role_key(role_name, org_id))

    def list_roles(self) -> list[dict[str, Any]]:
        """List the roles."""
        return list(self.roles.values())

    def create_role(self, record: dict[str, Any]) -> bool:
        """Create the role."""
        key = self._role_key(record["name"], record.get("org_id") or None)
        if key in self.roles or (
            record["name"] in self.roles
            and self.roles[record["name"]].get("is_builtin")
        ):
            return False
        self.roles[key] = record
        return True

    def save_role(self, record: dict[str, Any]) -> None:
        """Persist the role."""
        key = self._role_key(record["name"], record.get("org_id") or None)
        self.roles[key] = record

    def delete_role(self, role_name: str, org_id: str | None) -> bool:
        """Delete the role."""
        return self.roles.pop(self._role_key(role_name, org_id), None) is not None


__all__ = ["MemoryIdentityStore"]
