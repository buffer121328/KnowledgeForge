"""Stable dependency-light protocols for identity persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from domain.identity import UserRole


class IdentityStore(Protocol):
    """Define user and role persistence required by identity services."""

    def find_user_by_username(self, username: str) -> dict[str, Any] | None: ...
    def find_user_by_id(self, user_id: str) -> dict[str, Any] | None: ...
    def list_users(self) -> list[dict[str, Any]]: ...
    def create_user(self, record: dict[str, Any]) -> bool: ...
    def save_user(self, record: dict[str, Any]) -> None: ...
    def delete_user(self, user_id: str) -> bool: ...
    def find_role(
        self,
        role_name: str,
        org_id: str | None,
    ) -> dict[str, Any] | None: ...
    def list_roles(self) -> list[dict[str, Any]]: ...
    def create_role(self, record: dict[str, Any]) -> bool: ...
    def save_role(self, record: dict[str, Any]) -> None: ...
    def delete_role(self, role_name: str, org_id: str | None) -> bool: ...


@runtime_checkable
class InvitationIdentityStore(IdentityStore, Protocol):
    """Identity store capable of atomic invitation-backed registration."""

    def create_invitation(
        self,
        *,
        invitation_id: str,
        token_hash: str,
        tenant_id: str,
        role: UserRole,
        expires_at: datetime,
        email: str | None = None,
    ) -> None: ...

    def consume_invitation_and_create_user(
        self,
        *,
        token_hash: str,
        record: dict[str, Any],
    ) -> dict[str, Any] | None: ...


__all__ = ["IdentityStore", "InvitationIdentityStore"]
