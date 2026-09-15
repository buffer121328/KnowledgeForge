"""Identity user business rules independent of FastAPI request/response models."""

from __future__ import annotations

import secrets
import hashlib
from datetime import datetime, timezone
from typing import Any

from auth.identity_contracts import (
    IdentityStore,
    InvitationIdentityStore,
)
from auth.identity_provider import get_identity_store
from auth.memory_identity_store import MemoryIdentityStore
from auth.jwt_service import AuthService
from auth.memory_accounts import ROLE_DB
from domain.identity import UserRole


class IdentityServiceError(Exception):
    """Stable domain-neutral error translated by the HTTP routers."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Initialize the identity service error."""
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class UserService:
    """User-record operations over an explicit identity store."""

    def __init__(
        self,
        users: dict[str, dict[str, Any]] | None = None,
        auth_service: AuthService | None = None,
        *,
        store: IdentityStore | None = None,
    ) -> None:
        """Initialize the user service."""
        if store is not None:
            self.store = store
        elif users is not None:
            self.store = MemoryIdentityStore(users, ROLE_DB)
        else:
            self.store = get_identity_store()
        self.users = getattr(self.store, "users", None)
        self.auth_service = auth_service or AuthService()

    def find_by_id(
        self,
        user_id: str,
        *,
        org_id: str | None = None,
        raise_on_missing: bool = True,
    ) -> dict[str, Any] | None:
        """Find the by ID."""
        user = self.store.find_user_by_id(user_id)
        if user and (org_id is None or user.get("org_id") == org_id):
            return user
        if raise_on_missing:
            raise IdentityServiceError(404, "用户不存在")
        return None

    def find_by_username(self, username: str) -> dict[str, Any] | None:
        """Find the by username."""
        return self.store.find_user_by_username(username)

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        """Authenticate the user service."""
        user = self.find_by_username(username)
        if not user or not self.auth_service.verify_password(password, user["password_hash"]):
            raise IdentityServiceError(401, "用户名或密码错误")
        if not user.get("is_active", True):
            raise IdentityServiceError(403, "账号已被禁用，请联系管理员")
        return user

    def register_public_user(self, username: str, password: str) -> dict[str, Any]:
        """Register through a trusted invitation or explicit local demo policy."""
        return self.register_public_user_with_invitation(username, password)

    def register_public_user_with_invitation(
        self,
        username: str,
        password: str,
        *,
        invitation_token: str | None = None,
    ) -> dict[str, Any]:
        """Consume an invitation atomically or use the bounded local demo path."""
        from shared.config import settings

        now = datetime.now(timezone.utc).isoformat()
        record = {
            "user_id": f"user_{secrets.token_hex(16)}",
            "username": username,
            "password_hash": self.auth_service.hash_password(password),
            "display_name": username,
            "email": "",
            "role": UserRole.VIEWER,
            "org_id": "org_default",
            "department_id": None,
            "is_department_manager": False,
            "is_active": True,
            "token_version": 0,
            "created_at": now,
            "last_login_at": None,
        }
        if invitation_token:
            if not isinstance(self.store, InvitationIdentityStore):
                raise IdentityServiceError(503, "registration_invitation_unavailable")
            consumed = self.store.consume_invitation_and_create_user(
                token_hash=hashlib.sha256(invitation_token.encode()).hexdigest(),
                record=record,
            )
            if consumed is None:
                raise IdentityServiceError(400, "registration_invitation_invalid")
            return consumed
        environment = settings.app_environment.strip().lower()
        if environment not in {"development", "test"} or not settings.allow_demo_default_org_registration:
            raise IdentityServiceError(403, "registration_invitation_required")
        if not self.store.create_user(record):
            raise IdentityServiceError(409, "用户名已存在")
        return record

    def list_users(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        role: UserRole | None = None,
        org_id: str | None = None,
        search: str | None = None,
        is_active: bool | None = None,
    ) -> list[dict[str, Any]]:
        """List the users."""
        users = self.store.list_users()
        if role:
            users = [user for user in users if user["role"] == role]
        if org_id:
            users = [user for user in users if user["org_id"] == org_id]
        if is_active is not None:
            users = [user for user in users if user["is_active"] == is_active]
        if search:
            query = search.lower()
            users = [
                user
                for user in users
                if query in user["username"].lower()
                or query in user.get("display_name", "").lower()
                or query in user.get("email", "").lower()
            ]
        start = (page - 1) * page_size
        return users[start : start + page_size]

    def create_user(
        self,
        *,
        username: str,
        password: str,
        display_name: str,
        email: str,
        role: UserRole,
        org_id: str,
        department_id: str | None = None,
        is_department_manager: bool = False,
    ) -> dict[str, Any]:
        """Create the user."""
        if role == UserRole.ADMIN and not (department_id or "").strip():
            raise IdentityServiceError(400, "部门负责人必须绑定部门")
        if self.find_by_username(username):
            raise IdentityServiceError(409, "用户名已存在")
        if email and any(
            user.get("email") == email for user in self.store.list_users()
        ):
            raise IdentityServiceError(409, "邮箱已被使用")
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "user_id": f"user_{secrets.token_hex(16)}",
            "username": username,
            "password_hash": self.auth_service.hash_password(password),
            "display_name": display_name or username,
            "email": email,
            "role": role,
            "org_id": org_id,
            "department_id": None if role == UserRole.ORGANIZATION_ADMIN else department_id,
            "is_department_manager": role == UserRole.ADMIN,
            "is_active": True,
            "token_version": 0,
            "created_at": now,
            "last_login_at": None,
        }
        if not self.store.create_user(record):
            raise IdentityServiceError(409, "用户名或邮箱已存在")
        return record

    def update_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        email: str | None = None,
        role: UserRole | None = None,
        org_id: str | None = None,
        is_active: bool | None = None,
        department_id: str | None = None,
        is_department_manager: bool | None = None,
        actor_user_id: str | None = None,
        scope_org_id: str | None = None,
    ) -> dict[str, Any]:
        """Update the user."""
        target = self.find_by_id(user_id, org_id=scope_org_id)
        next_role = role if role is not None else target["role"]
        next_active = is_active if is_active is not None else target.get("is_active", True)
        next_department = (
            None
            if next_role == UserRole.ORGANIZATION_ADMIN
            else (
                department_id.strip() or None
                if department_id is not None
                else target.get("department_id")
            )
        )
        if target["role"] == UserRole.ORGANIZATION_ADMIN and (
            next_role != UserRole.ORGANIZATION_ADMIN or not next_active
        ):
            if actor_user_id == user_id:
                raise IdentityServiceError(400, "不能降级或禁用自己的组织管理员账号")
            if self._active_organization_admin_count(target["org_id"]) <= 1:
                raise IdentityServiceError(400, "不能移除最后一个组织管理员")
        if next_role == UserRole.ADMIN and not next_department:
            raise IdentityServiceError(400, "部门负责人必须绑定部门")
        if display_name is not None:
            target["display_name"] = display_name
        if email is not None:
            if any(
                user["user_id"] != user_id and user.get("email") == email
                for user in self.store.list_users()
            ):
                raise IdentityServiceError(409, "邮箱已被使用")
            target["email"] = email
        if role is not None:
            old_role = target["role"]
            target["role"] = role
            if old_role != role:
                target["token_version"] = target.get("token_version", 0) + 1
        if org_id is not None:
            old_org_id = target.get("org_id")
            target["org_id"] = org_id
            if old_org_id != org_id:
                target["token_version"] = target.get("token_version", 0) + 1
        if next_role == UserRole.ORGANIZATION_ADMIN or department_id is not None:
            if target.get("department_id") != next_department:
                target["department_id"] = next_department
                target["token_version"] = target.get("token_version", 0) + 1
        normalized_manager = target["role"] == UserRole.ADMIN
        if bool(target.get("is_department_manager", False)) != normalized_manager:
            target["is_department_manager"] = normalized_manager
            target["token_version"] = target.get("token_version", 0) + 1
        if is_active is not None:
            target["is_active"] = is_active
            if not is_active:
                target["token_version"] = target.get("token_version", 0) + 1
        self.store.save_user(target)
        return target

    def _active_organization_admin_count(self, org_id: str) -> int:
        """Count active organization administrators inside one tenant."""
        return sum(
            1
            for user in self.store.list_users()
            if user.get("org_id") == org_id
            and user.get("role") == UserRole.ORGANIZATION_ADMIN
            and user.get("is_active", True)
        )

    def trusted_promote_organization_admin(
        self,
        user_id: str,
        *,
        org_id: str,
    ) -> dict[str, Any]:
        """Promote one explicitly identified tenant user and invalidate old tokens."""
        target = self.find_by_id(user_id, org_id=org_id)
        changed = False
        if target["role"] != UserRole.ORGANIZATION_ADMIN:
            target["role"] = UserRole.ORGANIZATION_ADMIN
            changed = True
        if target.get("department_id") is not None:
            target["department_id"] = None
            changed = True
        if bool(target.get("is_department_manager", False)):
            target["is_department_manager"] = False
            changed = True
        if changed:
            target["token_version"] = target.get("token_version", 0) + 1
            self.store.save_user(target)
        return target

    def delete_user(
        self,
        user_id: str,
        *,
        actor_user_id: str,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete the user."""
        if actor_user_id == user_id:
            raise IdentityServiceError(400, "不能删除自己的账号")
        target = self.find_by_id(user_id, org_id=org_id)
        if (
            target["role"] == UserRole.ORGANIZATION_ADMIN
            and target.get("is_active", True)
            and self._active_organization_admin_count(target["org_id"]) <= 1
        ):
            raise IdentityServiceError(400, "不能移除最后一个组织管理员")
        self.store.delete_user(target["user_id"])
        return target

    def batch_change_role(
        self,
        user_ids: list[str],
        role: UserRole,
        *,
        org_id: str | None = None,
        actor_user_id: str | None = None,
    ) -> int:
        """Process the change role."""
        updated = 0
        for user_id in user_ids:
            target = self.find_by_id(
                user_id,
                org_id=org_id,
                raise_on_missing=False,
            )
            if target:
                if role == UserRole.ADMIN and not target.get("department_id"):
                    raise IdentityServiceError(400, "部门负责人必须绑定部门")
                if target["role"] == UserRole.ORGANIZATION_ADMIN and role != UserRole.ORGANIZATION_ADMIN:
                    if actor_user_id == user_id:
                        raise IdentityServiceError(400, "不能降级自己的组织管理员账号")
                    if self._active_organization_admin_count(target["org_id"]) <= 1:
                        raise IdentityServiceError(400, "不能移除最后一个组织管理员")
                old_role = target["role"]
                target["role"] = role
                if role == UserRole.ORGANIZATION_ADMIN:
                    target["department_id"] = None
                target["is_department_manager"] = role == UserRole.ADMIN
                if old_role != role:
                    target["token_version"] = target.get("token_version", 0) + 1
                self.store.save_user(target)
                updated += 1
        return updated

    def change_own_password(self, user_id: str, *, old_password: str, new_password: str) -> None:
        """Change the own password."""
        target = self.find_by_id(user_id)
        if not self.auth_service.verify_password(old_password, target["password_hash"]):
            raise IdentityServiceError(400, "原密码错误")
        target["password_hash"] = self.auth_service.hash_password(new_password)
        target["token_version"] = target.get("token_version", 0) + 1
        self.store.save_user(target)

    def reset_password(
        self,
        user_id: str,
        *,
        new_password: str,
        org_id: str | None = None,
    ) -> None:
        """Reset the password."""
        target = self.find_by_id(user_id, org_id=org_id)
        target["password_hash"] = self.auth_service.hash_password(new_password)
        target["token_version"] = target.get("token_version", 0) + 1
        self.store.save_user(target)

    def toggle_active(
        self,
        user_id: str,
        *,
        actor_user_id: str,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Toggle the active."""
        if actor_user_id == user_id:
            raise IdentityServiceError(400, "不能禁用自己的账号")
        target = self.find_by_id(user_id, org_id=org_id)
        if (
            target["role"] == UserRole.ORGANIZATION_ADMIN
            and target.get("is_active", True)
            and self._active_organization_admin_count(target["org_id"]) <= 1
        ):
            raise IdentityServiceError(400, "不能移除最后一个组织管理员")
        target["is_active"] = not target.get("is_active", True)
        if not target["is_active"]:
            target["token_version"] = target.get("token_version", 0) + 1
        self.store.save_user(target)
        return target
