from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pybreaker
import pytest

from shared.utils.circuit_breaker import AsyncCircuitBreaker, call_with_fallback
from shared.utils.metrics import (
    circuit_breaker_consecutive_failures,
    circuit_breaker_fallback_calls_total,
    circuit_breaker_rejected_calls_total,
    circuit_breaker_state_changed_timestamp_seconds,
)


@dataclass
class HttpStatusError(RuntimeError):
    status_code: int
    message: str = "dependency request failed"

    def __post_init__(self) -> None:
        super().__init__(self.message)


async def open_breaker(breaker: AsyncCircuitBreaker) -> None:
    failing_dependency = AsyncMock(side_effect=ConnectionError("dependency unavailable"))
    with pytest.raises(pybreaker.CircuitBreakerError):
        await breaker.call(failing_dependency)


class TestAsyncCircuitBreakerLifecycle:
    @pytest.mark.asyncio
    async def test_counted_failures_open_breaker_and_reject_without_calling_dependency(self) -> None:
        breaker = AsyncCircuitBreaker(name="threshold-test", fail_max=3, reset_timeout=60)
        failing_dependency = AsyncMock(side_effect=ConnectionError("dependency unavailable"))

        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_dependency)

        with pytest.raises(pybreaker.CircuitBreakerError, match="temporarily unavailable"):
            await breaker.call(failing_dependency)

        with pytest.raises(pybreaker.CircuitBreakerError, match="temporarily unavailable"):
            await breaker.call(failing_dependency)

        assert failing_dependency.await_count == 3
        snapshot = breaker.snapshot()
        assert snapshot.state == "open"
        assert snapshot.consecutive_failures == 3
        assert snapshot.rejected_calls == 1

    @pytest.mark.asyncio
    async def test_successful_half_open_probe_closes_breaker_and_resets_failures(self) -> None:
        breaker = AsyncCircuitBreaker(name="successful-probe", fail_max=1, reset_timeout=0)
        await open_breaker(breaker)
        successful_dependency = AsyncMock(return_value="recovered")

        assert await breaker.call(successful_dependency) == "recovered"

        snapshot = breaker.snapshot()
        assert snapshot.state == "closed"
        assert snapshot.consecutive_failures == 0
        assert snapshot.last_state_change_timestamp > 0

    @pytest.mark.asyncio
    async def test_failed_half_open_probe_reopens_without_leaking_dependency_error(self) -> None:
        breaker = AsyncCircuitBreaker(name="failed-probe", fail_max=1, reset_timeout=0)
        await open_breaker(breaker)
        failing_probe = AsyncMock(side_effect=ConnectionError("upstream-secret.example.internal"))

        with pytest.raises(pybreaker.CircuitBreakerError) as captured:
            await breaker.call(failing_probe)

        assert "upstream-secret.example.internal" not in str(captured.value)
        assert breaker.snapshot().state == "open"

    @pytest.mark.asyncio
    async def test_half_open_probe_limit_rejects_concurrent_call_without_invoking_dependency(self) -> None:
        breaker = AsyncCircuitBreaker(
            name="bounded-probe",
            fail_max=1,
            reset_timeout=0,
            half_open_max_calls=1,
        )
        await open_breaker(breaker)

        admitted = asyncio.Event()
        release = asyncio.Event()

        async def blocking_probe() -> str:
            admitted.set()
            await release.wait()
            return "recovered"

        probe = asyncio.create_task(breaker.call(blocking_probe))
        await admitted.wait()
        rejected_dependency = AsyncMock(return_value="must not run")

        with pytest.raises(pybreaker.CircuitBreakerError, match="temporarily unavailable"):
            await breaker.call(rejected_dependency)

        assert rejected_dependency.await_count == 0
        release.set()
        assert await probe == "recovered"
        assert breaker.snapshot().state == "closed"


class TestAsyncCircuitBreakerClassificationAndObservability:
    @pytest.mark.asyncio
    async def test_validation_error_does_not_change_existing_failure_count(self) -> None:
        breaker = AsyncCircuitBreaker(name="validation-error", fail_max=3, reset_timeout=60)

        with pytest.raises(ConnectionError):
            await breaker.call(AsyncMock(side_effect=ConnectionError("temporary")))
        before = breaker.snapshot()

        with pytest.raises(ValueError, match="invalid caller input"):
            await breaker.call(AsyncMock(side_effect=ValueError("invalid caller input")))

        after = breaker.snapshot()
        assert after.state == "closed"
        assert after.consecutive_failures == before.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_cancelled_half_open_probe_releases_the_probe_slot(self) -> None:
        breaker = AsyncCircuitBreaker(name="cancelled-probe", fail_max=1, reset_timeout=0)
        await open_breaker(breaker)
        admitted = asyncio.Event()

        async def blocked_probe() -> None:
            admitted.set()
            await asyncio.Event().wait()

        probe = asyncio.create_task(breaker.call(blocked_probe))
        await admitted.wait()
        probe.cancel()

        with pytest.raises(asyncio.CancelledError):
            await probe

        assert breaker.snapshot().state == "half-open"
        assert await breaker.call(AsyncMock(return_value="recovered")) == "recovered"
        assert breaker.snapshot().state == "closed"

    @pytest.mark.asyncio
    async def test_non_counted_half_open_error_does_not_reopen_the_breaker(self) -> None:
        breaker = AsyncCircuitBreaker(name="half-open-validation", fail_max=1, reset_timeout=0)
        await open_breaker(breaker)

        with pytest.raises(ValueError, match="invalid caller input"):
            await breaker.call(AsyncMock(side_effect=ValueError("invalid caller input")))

        snapshot = breaker.snapshot()
        assert snapshot.state == "half-open"
        assert snapshot.consecutive_failures == 1

        assert await breaker.call(AsyncMock(return_value="recovered")) == "recovered"
        assert breaker.snapshot().state == "closed"
        assert breaker.snapshot().consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_http_rate_limit_counts_toward_breaker_health(self) -> None:
        breaker = AsyncCircuitBreaker(name="http-rate-limit", fail_max=2, reset_timeout=60)
        rate_limited = AsyncMock(side_effect=HttpStatusError(429))

        with pytest.raises(HttpStatusError):
            await breaker.call(rate_limited)

        assert breaker.snapshot().consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_rejected_call_fallback_and_metrics_are_recorded(self) -> None:
        breaker = AsyncCircuitBreaker(name="fallback-observability", fail_max=1, reset_timeout=60)
        failing_dependency = AsyncMock(side_effect=ConnectionError("temporary"))
        fallback = AsyncMock(return_value="safe fallback")

        assert await call_with_fallback(breaker, failing_dependency, fallback) == "safe fallback"
        assert await call_with_fallback(breaker, failing_dependency, fallback) == "safe fallback"

        snapshot = breaker.snapshot()
        assert failing_dependency.await_count == 1
        assert fallback.await_count == 2
        assert snapshot.state == "open"
        assert snapshot.consecutive_failures == 1
        assert snapshot.rejected_calls == 1
        assert snapshot.fallback_calls == 2
        assert snapshot.last_state_change_timestamp > 0
        assert circuit_breaker_consecutive_failures.labels(breaker="fallback-observability")._value.get() == 1
        assert circuit_breaker_state_changed_timestamp_seconds.labels(breaker="fallback-observability")._value.get() > 0
        assert circuit_breaker_rejected_calls_total.labels(breaker="fallback-observability")._value.get() == 1
        assert circuit_breaker_fallback_calls_total.labels(breaker="fallback-observability")._value.get() == 2
