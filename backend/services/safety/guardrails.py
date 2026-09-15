"""Compose repository QA policy with the optional Guardrails adapter."""

from __future__ import annotations

from domain.content_safety import ContentValidator

from services.safety.qa_checks import (
    QASafetyRefusalError,
    sanitize_answer,
    validate_question,
)


def _input_callback(value: str, _metadata: dict) -> tuple[bool, str | None]:
    """Apply deterministic question validation through a Guard callback."""
    try:
        validate_question(value)
    except QASafetyRefusalError:
        return False, None
    return True, None


def _context_callback(_value: str, _metadata: dict) -> tuple[bool, str | None]:
    """Leave deterministic context ownership checks to the safety pipeline."""
    return True, None


def _output_callback(value: str, _metadata: dict) -> tuple[bool, str | None]:
    """Apply deterministic answer sanitization through a Guard callback."""
    result = sanitize_answer(value)
    return result.answer == value, (
        result.answer if result.answer != value else None
    )


def build_repository_validator() -> ContentValidator:
    """Build the optional validator with explicit repository callbacks."""
    from infrastructure.security.guardrails_adapter import (
        GuardrailsAIAdapter,
        build_callback_guardrails,
    )

    callbacks = {
        "input": _input_callback,
        "context": _context_callback,
        "output": _output_callback,
    }
    return GuardrailsAIAdapter(build_callback_guardrails(callbacks))


__all__ = ["build_repository_validator"]
