"""Lazy Guardrails AI adapter; imports the optional package only on demand."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

from domain.content_safety import ContentValidationOutcome

GuardrailCallback = Callable[[str, dict], tuple[bool, str | None]]


class GuardrailsAIAdapter:
    """Adapt guardrails ai to the application contract."""
    name = "guardrails_ai"

    def __init__(self, guards: dict[str, Any]) -> None:
        """Initialize the guardrails ai adapter."""
        self.guards = dict(guards)

    async def validate(
        self,
        stage: str,
        value: str,
        metadata: dict,
    ) -> ContentValidationOutcome:
        """Validate a value with the guardrails ai adapter."""
        guard = self.guards.get(stage)
        if guard is None:
            raise RuntimeError("guard is not configured")
        outcome = await asyncio.to_thread(
            guard.validate,
            value,
            metadata=metadata,
        )
        passed = getattr(outcome, "validation_passed", None)
        validated = getattr(outcome, "validated_output", value)
        if passed is True:
            if isinstance(validated, str) and validated != value:
                return ContentValidationOutcome.sanitize(validated)
            return ContentValidationOutcome.pass_(value)
        if passed is False:
            if isinstance(validated, str) and validated != value:
                return ContentValidationOutcome.sanitize(validated)
            return ContentValidationOutcome.deny()
        raise TypeError("unsupported Guardrails validation outcome")


def _guardrails_components():
    """Return the guardrails components."""
    from guardrails import Guard, OnFailAction, register_validator
    from guardrails.validators import FailResult, PassResult, Validator

    return (
        Guard,
        OnFailAction,
        register_validator,
        Validator,
        PassResult,
        FailResult,
    )


def _build_callback_guard(stage: str, callback):
    """Build the callback guard."""
    (
        Guard,
        OnFailAction,
        register_validator,
        Validator,
        PassResult,
        FailResult,
    ) = _guardrails_components()

    @register_validator(
        name=f"knowledgeforge/local-{stage}",
        data_type="string",
    )
    class _LocalValidator(Validator):  # type: ignore[valid-type,misc]
        """Represent local validator."""
        def __init__(self) -> None:
            """Initialize the local validator."""
            super().__init__(
                on_fail=(
                    OnFailAction.FIX
                    if stage == "output"
                    else OnFailAction.NOOP
                )
            )

        def _validate(self, value: str, metadata: dict):
            """Apply the local validator callback to the supplied value."""
            accepted, fixed = callback(value, metadata)
            if accepted:
                return PassResult()
            return FailResult(
                error_message="repository safety policy rejected content",
                fix_value=fixed,
            )

    guard = Guard()
    guard.configure(allow_metrics_collection=False)
    return guard.use(_LocalValidator())


def build_callback_guardrails(
    callbacks: Mapping[str, GuardrailCallback],
) -> dict[str, Any]:
    """Build local/no-network Guards from explicit stage callbacks."""
    required_stages = {"input", "context", "output"}
    if set(callbacks) != required_stages:
        raise ValueError("callbacks must configure input, context, and output")
    return {
        stage: _build_callback_guard(stage, callbacks[stage])
        for stage in ("input", "context", "output")
    }


def build_local_guardrails_poc() -> dict[str, Any]:
    """Construct deterministic POC guards with no Hub/model/network dependency."""

    def deny_marker(value: str, _metadata: dict) -> tuple[bool, str | None]:
        """Deny the marker."""
        return ("POC_BLOCK" not in value), None

    def redact_marker(value: str, _metadata: dict) -> tuple[bool, str | None]:
        """Redact the marker."""
        if "POC_SECRET" not in value:
            return True, None
        return False, value.replace("POC_SECRET", "[REDACTED]")

    return {
        **build_callback_guardrails(
            {
                "input": deny_marker,
                "context": deny_marker,
                "output": redact_marker,
            }
        )
    }


__all__ = [
    "GuardrailCallback",
    "GuardrailsAIAdapter",
    "build_callback_guardrails",
    "build_local_guardrails_poc",
]
