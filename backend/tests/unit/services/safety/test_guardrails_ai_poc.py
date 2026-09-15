"""Optional no-network Guardrails AI adapter POC."""

from __future__ import annotations

import pytest

pytest.importorskip("guardrails")

from infrastructure.security.guardrails_adapter import (
    GuardrailsAIAdapter,
    build_local_guardrails_poc,
)


@pytest.mark.asyncio
async def test_local_guardrails_poc_passes_denies_and_fixes_without_network():
    guards = build_local_guardrails_poc()

    class _NetworkTelemetryMustNotStart:
        def start_span(self, *_args, **_kwargs):
            raise AssertionError("Guardrails telemetry must remain disabled")

    for guard in guards.values():
        assert guard._allow_metrics_collection is False
        assert guard._hub_telemetry._enabled is False
        guard._hub_telemetry._tracer = _NetworkTelemetryMustNotStart()

    adapter = GuardrailsAIAdapter(guards)

    passed = await adapter.validate("input", "normal business question", {})
    denied = await adapter.validate("input", "POC_BLOCK this request", {})
    fixed = await adapter.validate(
        "output",
        "answer contains POC_SECRET",
        {},
    )

    assert passed.decision == "pass"
    assert denied.decision == "deny"
    assert fixed.decision == "sanitize"
    assert fixed.value == "answer contains [REDACTED]"
