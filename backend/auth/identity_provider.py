"""Composition provider for the authoritative runtime identity store."""

from __future__ import annotations

from auth.identity_contracts import IdentityStore

_identity_store: IdentityStore | None = None


def get_identity_store() -> IdentityStore:
    """Return the process-wide PostgreSQL identity fact store."""

    global _identity_store
    if _identity_store is None:
        from infrastructure.postgres.identity_store import (
            PostgreSQLIdentityStore,
        )

        _identity_store = PostgreSQLIdentityStore()
    return _identity_store


def reset_identity_store() -> None:
    """Reset the process-wide identity store for isolated application tests."""

    global _identity_store
    _identity_store = None


__all__ = ["get_identity_store", "reset_identity_store"]
