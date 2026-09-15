"""Authentication core services and configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from auth.config import AuthSettings
    from auth.jwt_service import AuthService
    from domain.identity import Permission, UserContext, UserRole

__all__ = [
    "AuthSettings",
    "AuthService",
    "Permission",
    "UserContext",
    "UserRole",
]


def __getattr__(name: str) -> Any:
    """Lazily preserve the small public auth facade without eager policy imports."""

    if name == "AuthSettings":
        from auth.config import AuthSettings

        return AuthSettings
    if name == "AuthService":
        from auth.jwt_service import AuthService

        return AuthService
    if name in {"Permission", "UserContext", "UserRole"}:
        from domain import identity

        return getattr(identity, name)
    raise AttributeError(name)
