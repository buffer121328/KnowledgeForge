"""PostgreSQL identity adapter and atomic identity relationships."""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from typing import Any

from domain.identity import Permission, UserRole, builtin_role_records
from sqlalchemy import delete, insert, select, update

from infrastructure.postgres.database import (
    DatabaseConflictError,
    DatabaseService,
    get_database_service,
)
from infrastructure.postgres.models import (
    organizations,
    permissions,
    registration_invitations,
    role_permissions,
    roles,
    user_roles,
)
from infrastructure.postgres.models import (
    users as pg_users,
)


def _as_datetime(value: Any, *, default: datetime | None = None) -> datetime | None:
    """Convert an ISO value to an aware UTC datetime for relational storage."""
    if value is None or value == "":
        return default
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return default
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_iso(value: Any) -> str | None:
    """Convert a relational timestamp to the compatible ISO representation."""
    if value is None:
        return None
    parsed = _as_datetime(value)
    return parsed.isoformat() if parsed is not None else None


class PostgreSQLIdentityStore:
    """Persist users and custom roles in the configured relational fact source."""

    def __init__(self, database: DatabaseService | None = None) -> None:
        """Initialize the PostgreSQL identity store."""
        self.database = database or get_database_service()

    @staticmethod
    def _role_id(tenant_id: str, role_name: str) -> str:
        """Return a bounded stable per-tenant role identifier."""
        digest = hashlib.sha256(f"{tenant_id}\0{role_name}".encode()).hexdigest()[:24]
        return f"role_{digest}"

    def _ensure_tenant(self, session, tenant_id: str) -> None:
        """Ensure organization, permission dictionary and built-in roles exist."""
        now = datetime.now(timezone.utc)
        if session.execute(select(organizations.c.id).where(organizations.c.id == tenant_id)).scalar_one_or_none() is None:
            session.execute(
                insert(organizations).values(
                    id=tenant_id,
                    name=tenant_id,
                    status="active",
                    created_at=now,
                    updated_at=now,
                )
            )
        for permission in Permission:
            if session.execute(select(permissions.c.code).where(permissions.c.code == permission.value)).scalar_one_or_none() is None:
                session.execute(insert(permissions).values(code=permission.value, description=""))
        for role_name, role in builtin_role_records().items():
            role_id = self._role_id(tenant_id, role_name)
            if session.execute(select(roles.c.id).where(roles.c.id == role_id)).scalar_one_or_none() is None:
                session.execute(
                    insert(roles).values(
                        id=role_id,
                        tenant_id=tenant_id,
                        name=role_name,
                        description=str(role.get("description") or ""),
                        is_builtin=True,
                        created_at=now,
                        updated_at=now,
                    )
                )
                permission_rows = [
                    {
                        "tenant_id": tenant_id,
                        "role_id": role_id,
                        "permission_code": code,
                    }
                    for code in role.get("permissions") or []
                ]
                if permission_rows:
                    session.execute(insert(role_permissions), permission_rows)

    @staticmethod
    def _user_from_row(row: Any) -> dict[str, Any]:
        """Hydrate one compatible user dictionary from a relational row."""
        data = dict(row._mapping)
        return {
            "user_id": data["id"],
            "username": data["username"],
            "email": data.get("email") or "",
            "display_name": data.get("display_name") or data["username"],
            "password_hash": data["password_hash"],
            "role": UserRole(data["role"]),
            "org_id": data["tenant_id"],
            "department_id": data.get("department_id"),
            "is_department_manager": bool(data.get("is_department_manager", False)),
            "is_active": bool(data["is_active"]),
            "token_version": int(data["token_version"]),
            "created_at": _as_iso(data.get("created_at")),
            "last_login_at": _as_iso(data.get("last_login_at")),
            "deleted_at": _as_iso(data.get("deleted_at")),
        }

    def find_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Return an active identity record by globally unique normalized username."""
        normalized = username.strip().lower()
        with self.database.session() as session:
            row = session.execute(
                select(pg_users).where(
                    pg_users.c.username == normalized,
                    pg_users.c.deleted_at.is_(None),
                )
            ).first()
        return self._user_from_row(row) if row else None

    def find_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Return a non-deleted identity record by stable ID."""
        with self.database.session() as session:
            row = session.execute(
                select(pg_users).where(
                    pg_users.c.id == user_id,
                    pg_users.c.deleted_at.is_(None),
                )
            ).first()
        return self._user_from_row(row) if row else None

    def list_users(self) -> list[dict[str, Any]]:
        """List non-deleted users for existing service-level filtering."""
        with self.database.session() as session:
            rows = session.execute(
                select(pg_users).where(pg_users.c.deleted_at.is_(None)).order_by(pg_users.c.username)
            ).all()
        return [self._user_from_row(row) for row in rows]

    def _insert_user(self, session, record: dict[str, Any]) -> None:
        """Insert one compatible user record and role relationship."""
        now = datetime.now(timezone.utc)
        tenant_id = str(record["org_id"])
        role_name = record["role"].value if isinstance(record["role"], UserRole) else str(record["role"])
        self._ensure_tenant(session, tenant_id)
        session.execute(
            insert(pg_users).values(
                id=record["user_id"],
                tenant_id=tenant_id,
                username=str(record["username"]).strip().lower(),
                email=(str(record.get("email") or "").strip().lower() or None),
                display_name=str(record.get("display_name") or record["username"]),
                password_hash=str(record["password_hash"]),
                role=role_name,
                department_id=(str(record.get("department_id") or "").strip() or None),
                is_department_manager=bool(record.get("is_department_manager", False)),
                is_active=bool(record.get("is_active", True)),
                token_version=int(record.get("token_version", 0)),
                created_at=_as_datetime(record.get("created_at"), default=now),
                updated_at=now,
                last_login_at=_as_datetime(record.get("last_login_at")),
                deleted_at=None,
            )
        )
        session.execute(
            insert(user_roles).values(
                tenant_id=tenant_id,
                user_id=record["user_id"],
                role_id=self._role_id(tenant_id, role_name),
            )
        )

    def create_user(self, record: dict[str, Any]) -> bool:
        """Atomically create a unique user and its tenant role binding."""
        try:
            with self.database.session() as session:
                self._insert_user(session, record)
        except DatabaseConflictError:
            return False
        return True

    def create_initial_user(self, record: dict[str, Any]) -> bool:
        """Create the first relational user only while no active user exists."""
        try:
            with self.database.session() as session:
                if session.execute(select(pg_users.c.id).limit(1)).first():
                    return False
                self._insert_user(session, record)
        except DatabaseConflictError:
            return False
        return True

    def save_user(self, record: dict[str, Any]) -> None:
        """Replace mutable identity fields and its single role binding atomically."""
        now = datetime.now(timezone.utc)
        tenant_id = str(record["org_id"])
        role_name = record["role"].value if isinstance(record["role"], UserRole) else str(record["role"])
        with self.database.session() as session:
            self._ensure_tenant(session, tenant_id)
            result = session.execute(
                update(pg_users)
                .where(pg_users.c.id == record["user_id"])
                .values(
                    tenant_id=tenant_id,
                    username=str(record["username"]).strip().lower(),
                    email=(str(record.get("email") or "").strip().lower() or None),
                    display_name=str(record.get("display_name") or record["username"]),
                    password_hash=str(record["password_hash"]),
                    role=role_name,
                    department_id=(str(record.get("department_id") or "").strip() or None),
                    is_department_manager=bool(record.get("is_department_manager", False)),
                    is_active=bool(record.get("is_active", True)),
                    token_version=int(record.get("token_version", 0)),
                    updated_at=now,
                    last_login_at=_as_datetime(record.get("last_login_at")),
                    deleted_at=_as_datetime(record.get("deleted_at")),
                )
            )
            if result.rowcount != 1:
                raise LookupError("identity record not found")
            session.execute(delete(user_roles).where(user_roles.c.user_id == record["user_id"]))
            session.execute(
                insert(user_roles).values(
                    tenant_id=tenant_id,
                    user_id=record["user_id"],
                    role_id=self._role_id(tenant_id, role_name),
                )
            )

    def delete_user(self, user_id: str) -> bool:
        """Soft-delete a user and invalidate previously issued credentials."""
        now = datetime.now(timezone.utc)
        with self.database.session() as session:
            result = session.execute(
                update(pg_users)
                .where(pg_users.c.id == user_id, pg_users.c.deleted_at.is_(None))
                .values(
                    is_active=False,
                    token_version=pg_users.c.token_version + 1,
                    deleted_at=now,
                    updated_at=now,
                )
            )
        return result.rowcount == 1

    @staticmethod
    def _role_from_row(row: Any, permission_values: list[str]) -> dict[str, Any]:
        """Hydrate one compatible custom role dictionary."""
        data = dict(row._mapping)
        return {
            "role_id": data["id"],
            "name": data["name"],
            "display_name": data["name"],
            "description": data.get("description") or "",
            "permissions": permission_values,
            "is_builtin": bool(data["is_builtin"]),
            "user_count": 0,
            "org_id": data["tenant_id"],
        }

    def find_role(self, role_name: str, org_id: str | None) -> dict[str, Any] | None:
        """Return a built-in role or tenant-owned custom role."""
        builtin = builtin_role_records().get(role_name)
        if builtin and builtin.get("is_builtin"):
            return copy.deepcopy(builtin)
        if not org_id:
            return None
        with self.database.session() as session:
            row = session.execute(
                select(roles).where(roles.c.tenant_id == org_id, roles.c.name == role_name)
            ).first()
            if row is None:
                return None
            permission_values = list(
                session.execute(
                    select(role_permissions.c.permission_code).where(
                        role_permissions.c.tenant_id == org_id,
                        role_permissions.c.role_id == row._mapping["id"],
                    )
                ).scalars()
            )
        return self._role_from_row(row, permission_values)

    def list_roles(self) -> list[dict[str, Any]]:
        """List built-in definitions and all custom relational roles."""
        result = [
            copy.deepcopy(role)
            for role in builtin_role_records().values()
            if role.get("is_builtin")
        ]
        with self.database.session() as session:
            rows = session.execute(select(roles).where(roles.c.is_builtin.is_(False))).all()
            for row in rows:
                data = row._mapping
                values = list(
                    session.execute(
                        select(role_permissions.c.permission_code).where(
                            role_permissions.c.tenant_id == data["tenant_id"],
                            role_permissions.c.role_id == data["id"],
                        )
                    ).scalars()
                )
                result.append(self._role_from_row(row, values))
        return result

    def create_role(self, record: dict[str, Any]) -> bool:
        """Create a tenant-owned custom role with permissions atomically."""
        if record.get("is_builtin") or not record.get("org_id"):
            return False
        try:
            self.save_role(record, create_only=True)
        except DatabaseConflictError:
            return False
        return True

    def save_role(self, record: dict[str, Any], *, create_only: bool = False) -> None:
        """Upsert a custom role and replace its validated permissions."""
        tenant_id = str(record.get("org_id") or "")
        if not tenant_id or record.get("is_builtin"):
            raise DatabaseConflictError("built-in roles are immutable")
        now = datetime.now(timezone.utc)
        role_id = str(record.get("role_id") or self._role_id(tenant_id, str(record["name"])))
        with self.database.session() as session:
            self._ensure_tenant(session, tenant_id)
            existing = session.execute(select(roles.c.id).where(roles.c.id == role_id)).scalar_one_or_none()
            values = {
                "tenant_id": tenant_id,
                "name": str(record["name"]),
                "description": str(record.get("description") or ""),
                "is_builtin": False,
                "updated_at": now,
            }
            if existing is None:
                session.execute(insert(roles).values(id=role_id, created_at=now, **values))
            elif create_only:
                raise DatabaseConflictError("role already exists")
            else:
                session.execute(update(roles).where(roles.c.id == role_id).values(**values))
            session.execute(delete(role_permissions).where(role_permissions.c.role_id == role_id))
            permission_values = [str(item) for item in record.get("permissions") or []]
            if permission_values:
                session.execute(
                    insert(role_permissions),
                    [
                        {"tenant_id": tenant_id, "role_id": role_id, "permission_code": item}
                        for item in permission_values
                    ],
                )

    def delete_role(self, role_name: str, org_id: str | None) -> bool:
        """Delete only a tenant-owned custom role."""
        if not org_id:
            return False
        with self.database.session() as session:
            row = session.execute(
                select(roles.c.id).where(
                    roles.c.tenant_id == org_id,
                    roles.c.name == role_name,
                    roles.c.is_builtin.is_(False),
                )
            ).first()
            if row is None:
                return False
            role_id = row._mapping["id"]
            session.execute(delete(role_permissions).where(role_permissions.c.role_id == role_id))
            result = session.execute(delete(roles).where(roles.c.id == role_id))
        return result.rowcount == 1

    def create_invitation(
        self,
        *,
        invitation_id: str,
        token_hash: str,
        tenant_id: str,
        role: UserRole,
        expires_at: datetime,
        email: str | None = None,
    ) -> None:
        """Persist a digest-only one-time registration invitation."""
        with self.database.session() as session:
            self._ensure_tenant(session, tenant_id)
            session.execute(
                insert(registration_invitations).values(
                    id=invitation_id,
                    tenant_id=tenant_id,
                    token_hash=token_hash,
                    role=role.value,
                    email=email.lower() if email else None,
                    expires_at=expires_at,
                    created_at=datetime.now(timezone.utc),
                )
            )

    def consume_invitation_and_create_user(
        self,
        *,
        token_hash: str,
        record: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Consume one valid invitation and create its user in one transaction."""
        now = datetime.now(timezone.utc)
        try:
            with self.database.session() as session:
                invitation = session.execute(
                    select(registration_invitations).where(
                        registration_invitations.c.token_hash == token_hash,
                        registration_invitations.c.consumed_at.is_(None),
                        registration_invitations.c.expires_at > now,
                    ).with_for_update()
                ).first()
                if invitation is None:
                    return None
                data = invitation._mapping
                record["org_id"] = data["tenant_id"]
                record["role"] = UserRole(data["role"])
                self._insert_user(session, record)
                session.execute(
                    update(registration_invitations)
                    .where(registration_invitations.c.id == data["id"])
                    .values(consumed_at=now, consumed_by=record["user_id"])
                )
        except DatabaseConflictError:
            return None
        return record


__all__ = ["PostgreSQLIdentityStore"]
