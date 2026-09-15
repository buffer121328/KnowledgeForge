"""Explicit initial-administrator bootstrap for durable identity state."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from auth.jwt_service import AuthService
from auth.user_service import UserService
from domain.identity import UserRole
from infrastructure.audit.log import AuditAction, AuditResult, AuditService, get_audit_service
from infrastructure.postgres.database import DatabaseService, get_database_service
from infrastructure.postgres.identity_store import PostgreSQLIdentityStore


class AdminBootstrapError(RuntimeError):
    """Stable, secret-free administrator bootstrap failure."""


@dataclass(frozen=True)
class AdminBootstrapResult:
    """Secret-free identity summary returned after successful bootstrap."""

    user_id: str
    username: str
    org_id: str


@dataclass(frozen=True)
class OrganizationAdminPromotionResult:
    """Bounded promotion result that excludes credentials and personal content."""

    user_id: str
    org_id: str
    token_version: int


def bootstrap_initial_admin(
    *,
    username: str,
    email: str,
    org_id: str,
    password: str,
    database: DatabaseService | None = None,
) -> AdminBootstrapResult:
    """Create the first administrator in the PostgreSQL identity store."""
    normalized_username = username.strip()
    normalized_email = email.strip()
    normalized_org = org_id.strip()
    if not normalized_username or not normalized_email or not normalized_org:
        raise AdminBootstrapError("username, email and organization are required")
    if not 12 <= len(password) <= 128:
        raise AdminBootstrapError("password must contain 12 to 128 characters")

    database_service = database or get_database_service()
    if not database_service.ping():
        raise AdminBootstrapError("durable security state is unavailable")
    store = PostgreSQLIdentityStore(database_service)
    auth = AuthService()
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "user_id": f"user_{secrets.token_hex(16)}",
        "username": normalized_username,
        "password_hash": auth.hash_password(password),
        "display_name": normalized_username,
        "email": normalized_email,
        "role": UserRole.ORGANIZATION_ADMIN,
        "org_id": normalized_org,
        "is_active": True,
        "token_version": 0,
        "created_at": now,
        "last_login_at": None,
    }
    if not store.create_initial_user(record):
        raise AdminBootstrapError("identity store is not empty")
    return AdminBootstrapResult(
        user_id=record["user_id"],
        username=normalized_username,
        org_id=normalized_org,
    )


def promote_organization_admin(
    *,
    org_id: str,
    user_id: str,
    user_service: UserService | None = None,
    audit_service: AuditService | None = None,
) -> OrganizationAdminPromotionResult:
    """Promote one explicit tenant user without guessing a bootstrap identity."""
    normalized_org = org_id.strip()
    normalized_user = user_id.strip()
    if not normalized_org or not normalized_user:
        raise AdminBootstrapError("organization and user IDs are required")
    service = user_service or UserService()
    target = service.trusted_promote_organization_admin(
        normalized_user,
        org_id=normalized_org,
    )
    (audit_service or get_audit_service()).log(
        user_id="trusted-migration",
        username="trusted-migration",
        org_id=normalized_org,
        action=AuditAction.USER_ROLE_CHANGE,
        resource=f"user:{normalized_user}",
        result=AuditResult.SUCCESS,
        metadata={
            "target_type": "user",
            "target_id": normalized_user,
            "role": UserRole.ORGANIZATION_ADMIN.value,
            "reason_code": "trusted_organization_admin_promotion",
        },
    )
    return OrganizationAdminPromotionResult(
        user_id=normalized_user,
        org_id=normalized_org,
        token_version=int(target.get("token_version", 0)),
    )
