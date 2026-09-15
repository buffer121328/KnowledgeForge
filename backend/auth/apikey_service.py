"""Authoritative, least-privilege API Key lifecycle service."""

from __future__ import annotations

import copy
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from auth.apikey_store import KeyStore, MemoryKeyStore, PostgreSQLKeyStore
from auth.config import AuthSettings, auth_settings
from domain.identity import APIKey, Permission, ROLE_PERMISSIONS, UserRole

@dataclass(frozen=True)
class APIKeyPolicy:
    """Validated expiry and warning policy for API Keys."""

    min_expiry_days: int = 1
    default_expiry_days: int = 30
    max_expiry_days: int = 90
    stale_warning_days: int = 30
    expiry_warning_days: int = 7

    def __post_init__(self) -> None:
        """Validate API key policy values after initialization."""
        values = (
            self.min_expiry_days,
            self.default_expiry_days,
            self.max_expiry_days,
            self.stale_warning_days,
            self.expiry_warning_days,
        )
        if any(value <= 0 for value in values):
            raise ValueError("API Key policy values must be positive")
        if not self.min_expiry_days <= self.default_expiry_days <= self.max_expiry_days:
            raise ValueError("API Key expiry policy is inconsistent")
        if (
            self.stale_warning_days > self.max_expiry_days
            or self.expiry_warning_days > self.max_expiry_days
        ):
            raise ValueError("API Key warning threshold exceeds maximum lifetime")

    @classmethod
    def from_settings(cls, settings: AuthSettings) -> APIKeyPolicy:
        """Create the API key policy from settings."""
        return cls(
            min_expiry_days=settings.api_key_min_expiry_days,
            default_expiry_days=settings.api_key_default_expiry_days,
            max_expiry_days=settings.api_key_max_expiry_days,
            stale_warning_days=settings.api_key_stale_warning_days,
            expiry_warning_days=settings.api_key_expiry_warning_days,
        )


@dataclass(frozen=True)
class APIKeyLifecycleEvent:
    """Secret-free event passed to the required lifecycle audit boundary."""

    action: str
    actor_user_id: str
    actor_username: str
    org_id: str
    key_id: str
    scopes: tuple[str, ...]
    expires_at: str | None
    state: str
    rotation_count: int


class APIKeyLifecycleError(RuntimeError):
    """Stable lifecycle error safe to translate at the HTTP boundary."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        """Initialize the API key lifecycle error."""
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


EventSink = Callable[[APIKeyLifecycleEvent], None]
Clock = Callable[[], datetime]


class APIKeyService:
    """API Key domain service backed by an injected persistence boundary."""

    KEY_PREFIX = "ak_"

    def __init__(
        self,
        store: KeyStore | None = None,
        *,
        policy: APIKeyPolicy | None = None,
        event_sink: EventSink | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialize the API key service."""
        self.store = store or MemoryKeyStore()
        self.policy = policy or APIKeyPolicy.from_settings(auth_settings)
        self.event_sink = event_sink
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def generate_key(self) -> tuple[str, str, str]:
        """Return the raw key, display prefix, and SHA-256 digest."""
        raw = secrets.token_urlsafe(32)
        full_key = f"{self.KEY_PREFIX}{raw}"
        prefix = f"{full_key[:7]}...{full_key[-4:]}"
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        return full_key, prefix, key_hash

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash the key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def create_key(
        self,
        name: str,
        user_id: str,
        org_id: str,
        role: UserRole,
        scopes: list[Permission] | list[str] | None = None,
        expires_days: int | None = None,
        *,
        actor_permissions: list[Permission] | None = None,
        actor_username: str = "",
    ) -> tuple[APIKey, str]:
        """Create the key."""
        normalized_scopes = self._authorize_scopes(
            scopes or [],
            actor_permissions
            if actor_permissions is not None
            else ROLE_PERMISSIONS.get(role, []),
        )
        lifetime = self._expiry_days(expires_days)
        now = self._now()
        full_key, prefix, key_hash = self.generate_key()
        api_key = APIKey(
            id=f"key_{secrets.token_hex(4)}",
            key_prefix=prefix,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            org_id=org_id,
            # A Key's authority is its explicit scope list, never its creator role.
            role=UserRole.API_USER,
            scopes=normalized_scopes,
            expires_at=(now + timedelta(days=lifetime)).isoformat(),
            created_at=now.isoformat(),
        )
        self.store.save(api_key)
        try:
            self._emit("auth.apikey_create", api_key, user_id, actor_username)
        except APIKeyLifecycleError:
            self.store.delete(api_key.id)
            raise
        return api_key, full_key

    def verify(self, raw_key: str, *, record_use: bool = True) -> APIKey | None:
        """Return an active Key with valid expiry and optionally record its use."""
        if not raw_key.startswith(self.KEY_PREFIX):
            return None
        api_key = self.store.get_by_hash(self.hash_key(raw_key))
        if not api_key or self.status(api_key) != "active":
            return None
        if record_use:
            self.mark_used(api_key.id)
        return api_key

    def mark_used(self, key_id: str, *, raw_key: str | None = None) -> bool:
        """Reconfirm current secret/state and record use after owner checks."""
        api_key = self.store.get_by_id(key_id)
        if not api_key or self.status(api_key) != "active":
            return False
        if raw_key is not None and not secrets.compare_digest(
            api_key.key_hash,
            self.hash_key(raw_key),
        ):
            return False
        api_key.last_used_at = self._now().isoformat()
        self.store.save(api_key)
        return True

    def status(self, api_key: APIKey) -> str:
        """Calculate the current API key lifecycle status."""
        if api_key.revoked_at:
            return "revoked"
        expiry = self._parse_time(api_key.expires_at)
        if expiry is None:
            return "invalid_expiry"
        if expiry <= self._now():
            return "expired"
        if not api_key.is_active:
            return "disabled"
        return "active"

    def warnings(self, api_key: APIKey) -> list[str]:
        """Handle warnings for the API key service."""
        if self.status(api_key) != "active":
            return []
        warnings: list[str] = []
        now = self._now()
        created_at = self._parse_time(api_key.created_at)
        last_used_at = self._parse_time(api_key.last_used_at)
        expiry = self._parse_time(api_key.expires_at)
        stale_after = timedelta(days=self.policy.stale_warning_days)
        if last_used_at is None:
            if created_at is not None and now - created_at >= stale_after:
                warnings.append("never_used")
        elif now - last_used_at >= stale_after:
            warnings.append("stale")
        if expiry is not None and expiry - now <= timedelta(
            days=self.policy.expiry_warning_days
        ):
            warnings.append("expires_soon")
        return warnings

    def list_keys(self, user_id: str, org_id: str | None = None) -> list[APIKey]:
        """List the keys."""
        keys = self.store.list_by_user(user_id)
        if org_id is not None:
            keys = [api_key for api_key in keys if api_key.org_id == org_id]
        return keys

    def warning_projection(self, user_id: str, org_id: str | None = None) -> list[dict[str, object]]:
        """Return scheduler-friendly warning records without secret material."""
        projection: list[dict[str, object]] = []
        for api_key in self.list_keys(user_id, org_id):
            if self.status(api_key) != "active":
                continue
            warnings = self.warnings(api_key)
            if not warnings:
                continue
            projection.append(
                {
                    "key_id": api_key.id,
                    "user_id": api_key.user_id,
                    "org_id": api_key.org_id,
                    "warnings": warnings,
                    "expires_at": api_key.expires_at,
                    "created_at": api_key.created_at,
                    "last_used_at": api_key.last_used_at,
                    "rotation_count": api_key.rotation_count,
                }
            )
        return sorted(projection, key=lambda item: str(item["key_id"]))

    def get_owned_key(
        self,
        key_id: str,
        user_id: str,
        org_id: str | None = None,
    ) -> APIKey | None:
        """Return the owned key."""
        api_key = self.store.get_by_id(key_id)
        if not api_key or api_key.user_id != user_id:
            return None
        if org_id is not None and api_key.org_id != org_id:
            return None
        return api_key

    def revoke(
        self,
        key_id: str,
        user_id: str,
        org_id: str | None = None,
        *,
        actor_username: str = "",
    ) -> APIKey | None:
        """Handle revoke for the API key service."""
        api_key = self.get_owned_key(key_id, user_id, org_id)
        if not api_key:
            return None
        if api_key.revoked_at:
            return api_key
        previous = copy.deepcopy(api_key)
        updated = copy.deepcopy(api_key)
        updated.is_active = False
        updated.revoked_at = self._now().isoformat()
        updated.revoked_by = user_id
        self._save_with_audit(
            previous,
            updated,
            "auth.apikey_revoke",
            user_id,
            actor_username,
        )
        return updated

    def toggle(
        self,
        key_id: str,
        user_id: str,
        org_id: str | None = None,
        *,
        actor_username: str = "",
    ) -> APIKey | None:
        """Toggle the api key service."""
        api_key = self.get_owned_key(key_id, user_id, org_id)
        if not api_key:
            return None
        if api_key.revoked_at:
            raise APIKeyLifecycleError(
                "api_key_revoked",
                "已撤销的 API Key 不能重新启用",
                409,
            )
        previous = copy.deepcopy(api_key)
        updated = copy.deepcopy(api_key)
        updated.is_active = not updated.is_active
        updated.disabled_at = None if updated.is_active else self._now().isoformat()
        self._save_with_audit(
            previous,
            updated,
            "auth.apikey_toggle",
            user_id,
            actor_username,
        )
        return updated

    def update_scopes(
        self,
        key_id: str,
        user_id: str,
        org_id: str,
        scopes: list[Permission] | list[str],
        *,
        actor_permissions: list[Permission],
        actor_username: str = "",
    ) -> APIKey | None:
        """Update the scopes."""
        api_key = self.get_owned_key(key_id, user_id, org_id)
        if not api_key:
            return None
        if api_key.revoked_at:
            raise APIKeyLifecycleError(
                "api_key_revoked",
                "已撤销的 API Key 不能修改 scope",
                409,
            )
        normalized = self._authorize_scopes(scopes, actor_permissions)
        if not set(normalized).issubset(set(api_key.scopes)):
            raise APIKeyLifecycleError(
                "api_key_scope_expansion_forbidden",
                "API Key scope 只能缩减；扩权请创建新 Key",
                409,
            )
        previous = copy.deepcopy(api_key)
        updated = copy.deepcopy(api_key)
        updated.scopes = normalized
        self._save_with_audit(
            previous,
            updated,
            "auth.apikey_scope_update",
            user_id,
            actor_username,
        )
        return updated

    def rotate(
        self,
        key_id: str,
        user_id: str,
        org_id: str,
        *,
        expires_days: int | None = None,
        actor_username: str = "",
    ) -> tuple[APIKey, str] | None:
        """Handle rotate for the API key service."""
        api_key = self.get_owned_key(key_id, user_id, org_id)
        if not api_key:
            return None
        if api_key.revoked_at:
            raise APIKeyLifecycleError(
                "api_key_revoked",
                "已撤销的 API Key 不能轮换",
                409,
            )
        lifetime = self._expiry_days(expires_days)
        previous = copy.deepcopy(api_key)
        updated = copy.deepcopy(api_key)
        full_key, prefix, key_hash = self.generate_key()
        now = self._now()
        updated.key_prefix = prefix
        updated.key_hash = key_hash
        updated.expires_at = (now + timedelta(days=lifetime)).isoformat()
        updated.last_used_at = None
        updated.is_active = True
        updated.disabled_at = None
        updated.rotated_at = now.isoformat()
        updated.rotation_count += 1
        self._save_with_audit(
            previous,
            updated,
            "auth.apikey_rotate",
            user_id,
            actor_username,
        )
        return updated, full_key

    def _save_with_audit(
        self,
        previous: APIKey,
        updated: APIKey,
        action: str,
        actor_user_id: str,
        actor_username: str,
    ) -> None:
        """Persist the with audit."""
        self.store.save(updated)
        try:
            self._emit(action, updated, actor_user_id, actor_username)
        except APIKeyLifecycleError:
            self.store.save(previous)
            raise

    def _emit(
        self,
        action: str,
        api_key: APIKey,
        actor_user_id: str,
        actor_username: str,
    ) -> None:
        """Emit the api key service."""
        if self.event_sink is None:
            return
        event = APIKeyLifecycleEvent(
            action=action,
            actor_user_id=actor_user_id,
            actor_username=actor_username,
            org_id=api_key.org_id,
            key_id=api_key.id,
            scopes=tuple(scope.value for scope in api_key.scopes),
            expires_at=api_key.expires_at,
            state=self.status(api_key),
            rotation_count=api_key.rotation_count,
        )
        try:
            self.event_sink(event)
        except Exception as error:
            raise APIKeyLifecycleError(
                "api_key_audit_unavailable",
                "API Key 生命周期审计暂不可用，请稍后重试",
                503,
            ) from error

    def _expiry_days(self, requested: int | None) -> int:
        """Resolve and validate the requested API key lifetime."""
        days = self.policy.default_expiry_days if requested is None else requested
        if not self.policy.min_expiry_days <= days <= self.policy.max_expiry_days:
            raise APIKeyLifecycleError(
                "invalid_api_key_expiry",
                "API Key 有效期超出允许范围",
            )
        return days

    @staticmethod
    def _authorize_scopes(
        scopes: list[Permission] | list[str],
        actor_permissions: list[Permission],
    ) -> list[Permission]:
        """Validate requested API key scopes against the owner permissions."""
        normalized: list[Permission] = []
        for scope in scopes:
            if isinstance(scope, Permission):
                permission = scope
            else:
                try:
                    permission = Permission(scope)
                except ValueError as error:
                    raise APIKeyLifecycleError(
                        "invalid_api_key_scope",
                        "API Key 包含未知 scope",
                    ) from error
            if permission not in normalized:
                normalized.append(permission)
        if not set(normalized).issubset(set(actor_permissions)):
            raise APIKeyLifecycleError(
                "api_key_scope_not_allowed",
                "API Key scope 超出当前用户权限",
            )
        return normalized

    def _now(self) -> datetime:
        """Return the current UTC timestamp."""
        value = self.clock()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _parse_time(value: str | None) -> datetime | None:
        """Parse the time."""
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)


def _persist_lifecycle_event(event: APIKeyLifecycleEvent) -> None:
    """Persist a required, secret-free lifecycle record."""
    from infrastructure.audit.log import AuditAction, AuditResult, get_audit_service

    get_audit_service().log(
        user_id=event.actor_user_id,
        username=event.actor_username,
        org_id=event.org_id,
        action=AuditAction(event.action),
        resource=f"apikey/{event.key_id}",
        result=AuditResult.SUCCESS,
        metadata={
            "scopes": list(event.scopes),
            "expires_at": event.expires_at,
            "state": event.state,
            "rotation_count": event.rotation_count,
        },
        required=True,
    )


_default_service: APIKeyService | None = None
def get_default_service() -> APIKeyService:
    """Return the process-wide API Key authority used by HTTP boundaries."""
    global _default_service
    if _default_service is None:
        _default_service = APIKeyService(
            store=PostgreSQLKeyStore(),
            event_sink=_persist_lifecycle_event,
        )
    return _default_service


def reset_default_service() -> None:
    """Reset the process client cache without deleting persistent records."""
    global _default_service
    _default_service = None
