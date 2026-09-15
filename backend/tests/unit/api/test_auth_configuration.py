"""Acceptance coverage for fail-closed runtime authentication settings."""
from __future__ import annotations

from pathlib import Path

import pytest

from auth.config import AuthConfigurationError, AuthSettings


SECURE_SECRET = "D6o4j3qVn1Wz8KeR5sX2Yp7Lm9Ac4HtB0rF6uN3iC8vQ"
DEFAULT_SECRET = "change-me-in-production-use-openssl-rand-hex-32"


def settings(**overrides: object) -> AuthSettings:
    values: dict[str, object] = {"secret_key": SECURE_SECRET, "bcrypt_rounds": 12}
    values.update(overrides)
    return AuthSettings(**values)


@pytest.mark.parametrize(
    "secret",
    ["", DEFAULT_SECRET, "replace-me-before-deploy", "short-key", "a" * 64],
)
def test_production_rejects_missing_placeholder_or_weak_secret_without_echoing_it(secret: str) -> None:
    candidate = AuthSettings(secret_key=secret)

    with pytest.raises(AuthConfigurationError) as captured:
        candidate.validate_for_runtime("production")

    if secret:
        assert secret not in str(captured.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"algorithm": "none"},
        {"access_token_expire_minutes": 0},
        {"access_token_expire_minutes": 1441},
        {"refresh_token_expire_days": 0},
        {"refresh_token_expire_days": 32},
        {"bcrypt_rounds": 11},
        {"allow_insecure_local_development": True},
        {"api_key_min_expiry_days": 40},
        {"api_key_expiry_warning_days": 91},
    ],
)
def test_production_rejects_unsafe_auth_settings(overrides: dict[str, object]) -> None:
    with pytest.raises(AuthConfigurationError):
        settings(**overrides).validate_for_runtime("production")


def test_secure_production_settings_pass_runtime_validation() -> None:
    settings().validate_for_runtime("production")


@pytest.mark.parametrize("environment", ["development", "test"])
def test_explicit_local_opt_in_allows_missing_or_placeholder_secret(environment: str) -> None:
    AuthSettings(secret_key="", allow_insecure_local_development=True).validate_for_runtime(environment)
    AuthSettings(
        secret_key=DEFAULT_SECRET,
        allow_insecure_local_development=True,
    ).validate_for_runtime(environment)


def test_local_placeholder_without_explicit_opt_in_is_rejected() -> None:
    with pytest.raises(AuthConfigurationError):
        AuthSettings(secret_key=DEFAULT_SECRET).validate_for_runtime("development")


@pytest.mark.parametrize("environment", ["production", "staging", ""])
def test_local_exception_never_relaxes_production_or_unknown_environments(environment: str) -> None:
    candidate = AuthSettings(
        secret_key=DEFAULT_SECRET,
        allow_insecure_local_development=True,
    )

    with pytest.raises(AuthConfigurationError):
        candidate.validate_for_runtime(environment)


def test_create_app_rejects_invalid_production_auth_before_assembly(monkeypatch) -> None:
    from shared.config import settings as runtime_settings

    monkeypatch.setattr(runtime_settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(runtime_settings, "dashscope_api_key", "test-key")
    from api import app as app_module

    monkeypatch.setattr(app_module, "auth_settings", AuthSettings(secret_key=DEFAULT_SECRET))
    monkeypatch.setattr(app_module.settings, "app_environment", "production")

    with pytest.raises(AuthConfigurationError):
        app_module.create_app()


def test_configuration_templates_declare_production_fail_closed_intent() -> None:
    backend_root = Path(__file__).resolve().parents[3]
    repository_root = backend_root.parent

    environment_template = (repository_root / "config" / ".env.example").read_text(encoding="utf-8")
    kubernetes_config = (repository_root / "deploy" / "k8s" / "config.yaml").read_text(encoding="utf-8")

    assert "AUTH_SECRET_KEY=REPLACE_VIA_SECRET_MANAGER" in environment_template
    assert "AUTH_ALLOW_INSECURE_LOCAL_DEVELOPMENT=false" in environment_template
    assert "AUTH_API_KEY_MIN_EXPIRY_DAYS=1" in environment_template
    assert "AUTH_API_KEY_DEFAULT_EXPIRY_DAYS=30" in environment_template
    assert "AUTH_API_KEY_MAX_EXPIRY_DAYS=90" in environment_template
    assert 'APP_ENVIRONMENT: "production"' in kubernetes_config
    assert 'AUTH_ALLOW_INSECURE_LOCAL_DEVELOPMENT: "false"' in kubernetes_config
    assert 'AUTH_API_KEY_DEFAULT_EXPIRY_DAYS: "30"' in kubernetes_config
