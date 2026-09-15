"""Authentication configuration and fail-closed runtime validation."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class AuthConfigurationError(RuntimeError):
    """Raised before application assembly when authentication settings are unsafe."""


_LOCAL_ENVIRONMENTS = frozenset({"development", "test"})
_SUPPORTED_JWT_ALGORITHMS = frozenset({"HS256", "HS384", "HS512"})
_PLACEHOLDER_SECRET_VALUES = frozenset(
    {
        "change-me-in-production-use-openssl-rand-hex-32",
        "change-me",
        "changeme",
        "replace-me",
        "your-secret-key",
        "secret",
    }
)
_PLACEHOLDER_SECRET_MARKERS = ("replace", "placeholder", "example", "change-me")
_MIN_SECRET_BYTES = 32
_MIN_SECRET_UNIQUE_CHARACTERS = 8
_MAX_ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60
_MAX_REFRESH_TOKEN_EXPIRE_DAYS = 31
_MIN_PRODUCTION_BCRYPT_ROUNDS = 12


class AuthSettings(BaseSettings):
    # JWT signing material is intentionally unset by default. Production startup
    # rejects it unless the deployment injects a secure value.
    """Define and validate auth configuration."""
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # This is only a deliberate developer/test convenience. It is rejected for
    # production and unknown environments by validate_for_runtime().
    allow_insecure_local_development: bool = False

    # Redis (token blacklist / rate-limit counters)
    redis_url: str = "redis://localhost:6379/1"

    # Password policy
    bcrypt_rounds: int = 12

    # API Keys are always time bounded. These settings also drive lifecycle
    # status warnings returned to the owning user.
    api_key_min_expiry_days: int = Field(default=1, gt=0, le=3650)
    api_key_default_expiry_days: int = Field(default=30, gt=0, le=3650)
    api_key_max_expiry_days: int = Field(default=90, gt=0, le=3650)
    api_key_stale_warning_days: int = Field(default=30, gt=0, le=3650)
    api_key_expiry_warning_days: int = Field(default=7, gt=0, le=3650)

    model_config = {"env_prefix": "AUTH_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def validate_for_runtime(self, app_environment: str) -> None:
        """Reject unsafe JWT configuration before a process starts serving traffic."""
        environment = app_environment.strip().lower()
        is_local = environment in _LOCAL_ENVIRONMENTS
        secret_issue = _secret_issue(self.secret_key)

        _validate_algorithm(self.algorithm)
        _validate_token_lifetimes(
            access_token_expire_minutes=self.access_token_expire_minutes,
            refresh_token_expire_days=self.refresh_token_expire_days,
        )
        _validate_api_key_policy(
            minimum=self.api_key_min_expiry_days,
            default=self.api_key_default_expiry_days,
            maximum=self.api_key_max_expiry_days,
            stale_warning=self.api_key_stale_warning_days,
            expiry_warning=self.api_key_expiry_warning_days,
        )

        if is_local:
            if secret_issue is not None and not self.allow_insecure_local_development:
                raise AuthConfigurationError(
                    "Invalid authentication configuration: insecure JWT secret requires "
                    "AUTH_ALLOW_INSECURE_LOCAL_DEVELOPMENT=true in development or test"
                )
            return

        if self.allow_insecure_local_development:
            raise AuthConfigurationError(
                "Invalid authentication configuration: local insecure-auth override is forbidden outside development or test"
            )
        if secret_issue is not None:
            raise AuthConfigurationError(f"Invalid authentication configuration: jwt_secret_{secret_issue}")
        if self.bcrypt_rounds < _MIN_PRODUCTION_BCRYPT_ROUNDS:
            raise AuthConfigurationError(
                "Invalid authentication configuration: bcrypt rounds below the production minimum"
            )


def _secret_issue(secret_key: str) -> str | None:
    """Return the secret issue."""
    normalized = secret_key.strip()
    folded = normalized.casefold()
    if not normalized:
        return "missing"
    if folded in _PLACEHOLDER_SECRET_VALUES or any(marker in folded for marker in _PLACEHOLDER_SECRET_MARKERS):
        return "placeholder"
    if len(normalized.encode("utf-8")) < _MIN_SECRET_BYTES:
        return "too_short"
    if len(set(normalized)) < _MIN_SECRET_UNIQUE_CHARACTERS:
        return "low_diversity"
    return None


def _validate_algorithm(algorithm: str) -> None:
    """Validate the algorithm."""
    if algorithm.strip().upper() not in _SUPPORTED_JWT_ALGORITHMS:
        raise AuthConfigurationError("Invalid authentication configuration: unsupported JWT algorithm")


def _validate_token_lifetimes(*, access_token_expire_minutes: int, refresh_token_expire_days: int) -> None:
    """Validate the token lifetimes."""
    if not 0 < access_token_expire_minutes <= _MAX_ACCESS_TOKEN_EXPIRE_MINUTES:
        raise AuthConfigurationError("Invalid authentication configuration: access token lifetime is outside allowed bounds")
    if not 0 < refresh_token_expire_days <= _MAX_REFRESH_TOKEN_EXPIRE_DAYS:
        raise AuthConfigurationError("Invalid authentication configuration: refresh token lifetime is outside allowed bounds")


def _validate_api_key_policy(
    *,
    minimum: int,
    default: int,
    maximum: int,
    stale_warning: int,
    expiry_warning: int,
) -> None:
    """Validate the API key policy."""
    if not minimum <= default <= maximum:
        raise AuthConfigurationError(
            "Invalid authentication configuration: API Key expiry policy is inconsistent"
        )
    if stale_warning > maximum or expiry_warning > maximum:
        raise AuthConfigurationError(
            "Invalid authentication configuration: API Key warning threshold exceeds maximum lifetime"
        )


auth_settings = AuthSettings()
