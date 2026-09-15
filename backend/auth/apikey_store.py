"""API Key persistence protocol and in-memory implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import delete, insert, select, update

from domain.identity import APIKey, Permission, UserRole
from infrastructure.security.security_state import CorruptSecurityStateError, SecurityStateBackend
from infrastructure.postgres.database import DatabaseService, get_database_service
from infrastructure.postgres.models import api_keys


class KeyStore(Protocol):
    """Persistence boundary used by :class:`auth.apikey.APIKeyService`."""

    def save(self, api_key: APIKey) -> None:
        """Persist a record through the key store."""
        ...

    def get_by_id(self, key_id: str) -> APIKey | None:
        """Return the by ID."""
        ...

    def get_by_hash(self, key_hash: str) -> APIKey | None:
        """Return the by hash."""
        ...

    def list_by_user(self, user_id: str) -> list[APIKey]:
        """List the by user."""
        ...

    def delete(self, key_id: str) -> bool:
        """Delete a record through the key store."""
        ...

    def update_last_used(self, key_id: str) -> None:
        """Update the last used."""
        ...


class MemoryKeyStore:
    """In-memory API Key persistence for local runtime and tests."""

    def __init__(self) -> None:
        """Initialize the memory key store."""
        self._store: dict[str, APIKey] = {}

    def save(self, api_key: APIKey) -> None:
        """Persist a record through the memory key store."""
        self._store[api_key.id] = api_key

    def get_by_id(self, key_id: str) -> APIKey | None:
        """Return the by ID."""
        return self._store.get(key_id)

    def get_by_hash(self, key_hash: str) -> APIKey | None:
        """Return the by hash."""
        for api_key in self._store.values():
            if api_key.key_hash == key_hash and api_key.is_active:
                return api_key
        return None

    def list_by_user(self, user_id: str) -> list[APIKey]:
        """List the by user."""
        return [api_key for api_key in self._store.values() if api_key.user_id == user_id]

    def delete(self, key_id: str) -> bool:
        """Delete a record through the memory key store."""
        return self._store.pop(key_id, None) is not None

    def update_last_used(self, key_id: str) -> None:
        """Update the last used."""
        api_key = self._store.get(key_id)
        if api_key:
            api_key.last_used_at = datetime.now(timezone.utc).isoformat()

    def clear(self) -> None:
        """Clear local state for isolated tests without changing the protocol."""
        self._store.clear()


def _serialize(api_key: APIKey) -> dict:
    """Return the serialize."""
    return {
        "version": 1,
        "kind": "apikey",
        "data": {
            "id": api_key.id,
            "key_prefix": api_key.key_prefix,
            "key_hash": api_key.key_hash,
            "name": api_key.name,
            "user_id": api_key.user_id,
            "org_id": api_key.org_id,
            "role": api_key.role.value,
            "scopes": [
                scope.value if isinstance(scope, Permission) else str(scope)
                for scope in api_key.scopes
            ],
            "expires_at": api_key.expires_at,
            "created_at": api_key.created_at,
            "last_used_at": api_key.last_used_at,
            "is_active": api_key.is_active,
            "disabled_at": api_key.disabled_at,
            "rotated_at": api_key.rotated_at,
            "rotation_count": api_key.rotation_count,
            "revoked_at": api_key.revoked_at,
            "revoked_by": api_key.revoked_by,
        },
    }


def _deserialize(envelope: dict) -> APIKey:
    """Return the deserialize."""
    if envelope.get("version") != 1 or envelope.get("kind") != "apikey":
        raise CorruptSecurityStateError()
    data = envelope.get("data")
    if not isinstance(data, dict):
        raise CorruptSecurityStateError()
    try:
        return APIKey(
            id=str(data["id"]),
            key_prefix=str(data["key_prefix"]),
            key_hash=str(data["key_hash"]),
            name=str(data["name"]),
            user_id=str(data["user_id"]),
            org_id=str(data["org_id"]),
            role=UserRole(data["role"]),
            scopes=[Permission(scope) for scope in data.get("scopes") or []],
            expires_at=data.get("expires_at"),
            created_at=str(data.get("created_at") or ""),
            last_used_at=data.get("last_used_at"),
            is_active=bool(data.get("is_active", True)),
            disabled_at=data.get("disabled_at"),
            rotated_at=data.get("rotated_at"),
            rotation_count=int(data.get("rotation_count") or 0),
            revoked_at=data.get("revoked_at"),
            revoked_by=data.get("revoked_by"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CorruptSecurityStateError() from error


class RedisKeyStore:
    """Multi-replica KeyStore over the shared security-state backend."""

    _PREFIX = "security:v1:apikey:"

    def __init__(self, backend: SecurityStateBackend) -> None:
        """Initialize the Redis key store."""
        self.backend = backend

    @classmethod
    def _key(cls, key_id: str) -> str:
        """Build the Redis key for an API key record."""
        return f"{cls._PREFIX}{key_id}"

    @staticmethod
    def _user_index(user_id: str) -> str:
        """Build the Redis index key for a user's API keys."""
        return f"security:v1:apikey-user:{user_id}"

    @staticmethod
    def _hash_index(key_hash: str) -> str:
        """Hash the index."""
        return f"security:v1:apikey-hash:{key_hash}"

    def save(self, api_key: APIKey) -> None:
        """Persist a record through the Redis key store."""
        key = self._key(api_key.id)
        existing = self.backend.read(key)
        remove_indexes: tuple[str, ...] = ()
        if existing is not None:
            previous = _deserialize(existing)
            remove_indexes = (
                self._user_index(previous.user_id),
                self._hash_index(previous.key_hash),
            )
        self.backend.write(
            key,
            _serialize(api_key),
            add_indexes=(
                self._user_index(api_key.user_id),
                self._hash_index(api_key.key_hash),
            ),
            remove_indexes=remove_indexes,
        )

    def get_by_id(self, key_id: str) -> APIKey | None:
        """Return the by ID."""
        value = self.backend.read(self._key(key_id))
        return None if value is None else _deserialize(value)

    def get_by_hash(self, key_hash: str) -> APIKey | None:
        """Return the by hash."""
        for key in self.backend.members(self._hash_index(key_hash)):
            value = self.backend.read(key)
            if value is None:
                continue
            api_key = _deserialize(value)
            if api_key.key_hash == key_hash and api_key.is_active:
                return api_key
        return None

    def list_by_user(self, user_id: str) -> list[APIKey]:
        """List the by user."""
        records: list[APIKey] = []
        for key in self.backend.members(self._user_index(user_id)):
            value = self.backend.read(key)
            if value is not None:
                records.append(_deserialize(value))
        return records

    def delete(self, key_id: str) -> bool:
        """Delete a record through the Redis key store."""
        api_key = self.get_by_id(key_id)
        if api_key is None:
            return False
        return self.backend.delete(
            self._key(key_id),
            remove_indexes=(
                self._user_index(api_key.user_id),
                self._hash_index(api_key.key_hash),
            ),
        )

    def update_last_used(self, key_id: str) -> None:
        """Update the last used."""
        api_key = self.get_by_id(key_id)
        if api_key:
            api_key.last_used_at = datetime.now(timezone.utc).isoformat()
            self.save(api_key)


def _parse_time(value: str | None) -> datetime | None:
    """Parse a compatible ISO timestamp for relational persistence."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    """Serialize a relational timestamp for the domain API Key contract."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


class PostgreSQLKeyStore:
    """Persist API Key digests and lifecycle metadata in PostgreSQL."""

    def __init__(self, database: DatabaseService | None = None) -> None:
        """Initialize the relational API Key store."""
        self.database = database or get_database_service()

    @staticmethod
    def _from_row(row) -> APIKey:
        """Hydrate one domain API Key without exposing its digest through HTTP."""
        data = row._mapping
        return APIKey(
            id=data["id"],
            key_prefix=data["key_prefix"],
            key_hash=data["key_hash"],
            name=data["name"],
            user_id=data["user_id"],
            org_id=data["tenant_id"],
            role=UserRole(data["role"]),
            scopes=[Permission(value) for value in (data["scopes"] or [])],
            expires_at=_iso(data["expires_at"]),
            created_at=_iso(data["created_at"]) or "",
            last_used_at=_iso(data["last_used_at"]),
            is_active=bool(data["is_active"]),
            disabled_at=_iso(data["disabled_at"]),
            rotated_at=_iso(data["rotated_at"]),
            rotation_count=int(data["rotation_count"]),
            revoked_at=_iso(data["revoked_at"]),
            revoked_by=data["revoked_by"],
        )

    @staticmethod
    def _values(api_key: APIKey) -> dict:
        """Return the allowlisted digest-only relational representation."""
        return {
            "id": api_key.id,
            "tenant_id": api_key.org_id,
            "user_id": api_key.user_id,
            "key_prefix": api_key.key_prefix,
            "key_hash": api_key.key_hash,
            "name": api_key.name,
            "role": api_key.role.value,
            "scopes": [scope.value if isinstance(scope, Permission) else str(scope) for scope in api_key.scopes],
            "expires_at": _parse_time(api_key.expires_at),
            "created_at": _parse_time(api_key.created_at) or datetime.now(timezone.utc),
            "last_used_at": _parse_time(api_key.last_used_at),
            "is_active": api_key.is_active,
            "disabled_at": _parse_time(api_key.disabled_at),
            "rotated_at": _parse_time(api_key.rotated_at),
            "rotation_count": api_key.rotation_count,
            "revoked_at": _parse_time(api_key.revoked_at),
            "revoked_by": api_key.revoked_by,
        }

    def save(self, api_key: APIKey) -> None:
        """Insert or atomically replace one API Key lifecycle record."""
        values = self._values(api_key)
        with self.database.session() as session:
            exists = session.execute(select(api_keys.c.id).where(api_keys.c.id == api_key.id)).scalar_one_or_none()
            if exists is None:
                session.execute(insert(api_keys).values(**values))
            else:
                session.execute(update(api_keys).where(api_keys.c.id == api_key.id).values(**values))

    def get_by_id(self, key_id: str) -> APIKey | None:
        """Return an API Key record by stable ID."""
        with self.database.session() as session:
            row = session.execute(select(api_keys).where(api_keys.c.id == key_id)).first()
        return self._from_row(row) if row else None

    def get_by_hash(self, key_hash: str) -> APIKey | None:
        """Resolve only an active Key by its unique digest."""
        with self.database.session() as session:
            row = session.execute(
                select(api_keys).where(api_keys.c.key_hash == key_hash, api_keys.c.is_active.is_(True))
            ).first()
        return self._from_row(row) if row else None

    def list_by_user(self, user_id: str) -> list[APIKey]:
        """List all lifecycle records for one owner."""
        with self.database.session() as session:
            rows = session.execute(
                select(api_keys).where(api_keys.c.user_id == user_id).order_by(api_keys.c.created_at.desc())
            ).all()
        return [self._from_row(row) for row in rows]

    def delete(self, key_id: str) -> bool:
        """Hard-delete a staged record only for existing lifecycle compensation."""
        with self.database.session() as session:
            result = session.execute(delete(api_keys).where(api_keys.c.id == key_id))
        return result.rowcount == 1

    def update_last_used(self, key_id: str) -> None:
        """Record successful use only after service owner checks complete."""
        with self.database.session() as session:
            session.execute(
                update(api_keys)
                .where(api_keys.c.id == key_id, api_keys.c.is_active.is_(True))
                .values(last_used_at=datetime.now(timezone.utc))
            )


__all__ = ["KeyStore", "MemoryKeyStore", "PostgreSQLKeyStore", "RedisKeyStore"]
