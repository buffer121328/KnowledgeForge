"""Composable QA content-safety pipeline with deterministic hard controls."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace
from typing import Literal

from services.safety.qa_checks import (
    ContextSafetyResult,
    QASafetyRefusalError,
    SanitizedAnswer,
    filter_contexts as deterministic_filter_contexts,
    sanitize_answer as deterministic_sanitize_answer,
    validate_question as deterministic_validate_question,
)
from domain.content_safety import ContentValidationOutcome, ContentValidator
from domain.knowledge import RetrievedContext
from shared.utils.metrics import (
    guardrail_decisions_total,
    guardrail_validation_errors_total,
    guardrail_validation_latency_seconds,
    security_safe_refusals_total,
)

GuardrailMode = Literal["builtin", "monitor", "enforce"]
_SAFE_LABEL = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class QASafetyUnavailableError(RuntimeError):
    """Controlled enforce-mode signal for an unavailable optional validator."""

    def __init__(self, stage: str) -> None:
        """Initialize the question-answering safety unavailable error."""
        super().__init__("QA safety validation is temporarily unavailable")
        self.stage = stage


def _safe_label(value: str, fallback: str) -> str:
    """Return the safe label."""
    normalized = value.strip().lower()
    return normalized if _SAFE_LABEL.fullmatch(normalized) else fallback


class QASafetyPipeline:
    """Run mandatory repository controls plus an optional bounded validator."""

    def __init__(
        self,
        *,
        mode: GuardrailMode = "builtin",
        validator: ContentValidator | None = None,
        timeout_seconds: float = 1.0,
        rule_version: str = "builtin-v1",
    ) -> None:
        """Initialize the question-answering safety pipeline."""
        if mode not in {"builtin", "monitor", "enforce"}:
            raise ValueError("mode must be builtin, monitor, or enforce")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.mode = mode
        self.validator = validator
        self.timeout_seconds = timeout_seconds
        self.rule_version = _safe_label(rule_version, "invalid-version")
        self.validator_name = _safe_label(
            getattr(validator, "name", "unconfigured"),
            "unknown",
        )

    async def validate_question(self, question: str) -> list[str]:
        """Validate the question."""
        try:
            deterministic_validate_question(question)
        except QASafetyRefusalError as error:
            self._record_decision("input", "deny")
            security_safe_refusals_total.labels(reason=error.code).inc()
            raise
        outcome, action = await self._optional("input", question, {})
        if action:
            return [action]
        if outcome and outcome.decision in {"deny", "sanitize"}:
            self._record_decision("input", "deny")
            security_safe_refusals_total.labels(
                reason="external_input_denied"
            ).inc()
            raise QASafetyRefusalError("external_input_denied")
        self._record_decision("input", "pass")
        return []

    async def filter_contexts(
        self,
        contexts: list[RetrievedContext],
        tenant_id: str | None,
    ) -> ContextSafetyResult:
        """Filter the contexts."""
        deterministic = deterministic_filter_contexts(contexts, tenant_id)
        if self.mode == "builtin":
            return deterministic

        accepted: list[RetrievedContext] = []
        actions = list(deterministic.actions)
        for context in deterministic.contexts:
            outcome, action = await self._optional(
                "context",
                context.content,
                {
                    "source_type": context.retrieval_type,
                    "has_tenant": bool(tenant_id),
                },
            )
            if action:
                if action not in actions:
                    actions.append(action)
                accepted.append(context)
                continue
            if outcome is None or outcome.decision == "pass":
                self._record_decision("context", "pass")
                accepted.append(context)
                continue
            if outcome.decision == "sanitize" and outcome.value is not None:
                self._record_decision("context", "sanitize")
                accepted.append(replace(context, content=outcome.value))
                if "external_context_redacted" not in actions:
                    actions.append("external_context_redacted")
                continue
            self._record_decision("context", "deny")
            if "external_context_denied" not in actions:
                actions.append("external_context_denied")
        return ContextSafetyResult(contexts=accepted, actions=actions)

    async def sanitize_answer(self, answer: str) -> SanitizedAnswer:
        """Sanitize the answer."""
        deterministic = deterministic_sanitize_answer(answer)
        outcome, action = await self._optional(
            "output",
            deterministic.answer,
            {},
        )
        actions = list(deterministic.actions)
        if action:
            if action not in actions:
                actions.append(action)
            return SanitizedAnswer(deterministic.answer, actions)
        if outcome is None or outcome.decision == "pass":
            self._record_decision("output", "pass")
            return deterministic
        if outcome.decision == "sanitize" and outcome.value is not None:
            self._record_decision("output", "sanitize")
            if "external_output_redacted" not in actions:
                actions.append("external_output_redacted")
            return SanitizedAnswer(outcome.value, actions)
        self._record_decision("output", "deny")
        security_safe_refusals_total.labels(reason="external_output_denied").inc()
        raise QASafetyRefusalError("external_output_denied")

    async def _optional(
        self,
        stage: str,
        value: str,
        metadata: dict,
    ) -> tuple[ContentValidationOutcome | None, str | None]:
        """Run the optional external validator for one safety stage."""
        if self.mode == "builtin":
            return None, None
        if self.validator is None:
            guardrail_validation_errors_total.labels(
                stage=stage,
                validator=self.validator_name,
            ).inc()
            return self._handle_validator_error(stage)

        started = time.perf_counter()
        try:
            outcome = await asyncio.wait_for(
                self.validator.validate(stage, value, metadata),
                timeout=self.timeout_seconds,
            )
            if not isinstance(outcome, ContentValidationOutcome):
                raise TypeError("invalid content validator outcome")
        except Exception:
            guardrail_validation_errors_total.labels(
                stage=stage,
                validator=self.validator_name,
            ).inc()
            return self._handle_validator_error(stage)
        finally:
            guardrail_validation_latency_seconds.labels(
                stage=stage,
                validator=self.validator_name,
            ).observe(time.perf_counter() - started)

        if self.mode == "monitor" and outcome.decision != "pass":
            self._record_decision(stage, f"monitor_{outcome.decision}")
            suffix = (
                "denied"
                if outcome.decision == "deny"
                else "redacted"
            )
            return None, f"external_{stage}_monitor_{suffix}"
        return outcome, None

    def _handle_validator_error(
        self,
        stage: str,
    ) -> tuple[None, str | None]:
        """Handle the validator error."""
        self._record_decision(stage, "error")
        if self.mode == "enforce":
            raise QASafetyUnavailableError(stage)
        return None, f"external_{stage}_validator_error"

    def _record_decision(self, stage: str, decision: str) -> None:
        """Record the decision."""
        guardrail_decisions_total.labels(
            stage=stage,
            decision=decision,
            rule_version=self.rule_version,
        ).inc()


def build_runtime_safety_pipeline(
    *,
    mode: GuardrailMode,
    timeout_seconds: float,
    rule_version: str,
) -> QASafetyPipeline:
    """Lazily assemble the optional adapter only when configured."""
    validator: ContentValidator | None = None
    if mode != "builtin":
        try:
            from services.safety.guardrails import build_repository_validator

            validator = build_repository_validator()
        except Exception:
            validator = None
    return QASafetyPipeline(
        mode=mode,
        validator=validator,
        timeout_seconds=timeout_seconds,
        rule_version=rule_version,
    )


__all__ = [
    "ContentValidationOutcome",
    "ContentValidator",
    "GuardrailMode",
    "QASafetyPipeline",
    "QASafetyUnavailableError",
    "build_runtime_safety_pipeline",
]
