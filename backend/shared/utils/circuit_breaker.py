"""Asynchronous circuit breakers for unstable external dependencies.

The adapter in this module owns asynchronous state transitions instead of
manually mutating PyBreaker's private counters. It preserves
``pybreaker.CircuitBreakerError`` as the controlled, caller-visible signal for
an unavailable dependency so existing QA fallbacks continue to work.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
import pybreaker

from shared.utils.metrics import (
    circuit_breaker_consecutive_failures,
    circuit_breaker_fallback_calls_total,
    circuit_breaker_rejected_calls_total,
    circuit_breaker_state,
    circuit_breaker_state_changed_timestamp_seconds,
)

T = TypeVar("T")
FailureClassifier = Callable[[Exception], bool]

_CLOSED = "closed"
_OPEN = "open"
_HALF_OPEN = "half-open"
_STATE_METRIC_VALUES = {_CLOSED: 0, _OPEN: 1, _HALF_OPEN: 2}
_UNAVAILABLE_MESSAGE = "External dependency is temporarily unavailable"


@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    """Safe, read-only state exported for tests and operational adapters."""

    name: str
    state: str
    consecutive_failures: int
    last_state_change_timestamp: float
    rejected_calls: int
    fallback_calls: int


def is_transient_dependency_error(error: Exception) -> bool:
    """Return whether an exception represents a dependency failure worth counting.

    Validation, authentication, and content-validation errors are intentionally
    not counted unless a dependency client supplies a more precise classifier.
    """

    if isinstance(
        error,
        (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            httpx.NetworkError,
            httpx.TimeoutException,
        ),
    ):
        return True

    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)

    if isinstance(status_code, int):
        return status_code == 429 or 500 <= status_code <= 599
    return False


class AsyncCircuitBreaker:
    """A coroutine-aware circuit breaker with bounded half-open probes.

    State updates are serialized, but dependency calls run outside the lock.
    State generations prevent an in-flight success admitted before an outage
    from closing a breaker that a concurrent failure has since opened.
    """

    def __init__(
        self,
        *,
        name: str,
        fail_max: int,
        reset_timeout: float,
        half_open_max_calls: int = 1,
        failure_classifier: FailureClassifier = is_transient_dependency_error,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the async circuit breaker."""
        if fail_max < 1:
            raise ValueError("fail_max must be at least 1")
        if reset_timeout < 0:
            raise ValueError("reset_timeout must be non-negative")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")

        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self._failure_classifier = failure_classifier
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._lock = asyncio.Lock()
        self._state = _CLOSED
        self._consecutive_failures = 0
        self._last_state_change_timestamp = self._wall_clock()
        self._opened_at: float | None = None
        self._half_open_calls = 0
        self._rejected_calls = 0
        self._fallback_calls = 0
        self._state_generation = 0
        self._publish_metrics()

    @property
    def current_state(self) -> str:
        """Compatibility-friendly current state accessor."""

        return self._state

    @property
    def fail_counter(self) -> int:
        """Return the current consecutive counted-failure count."""

        return self._consecutive_failures

    def snapshot(self) -> CircuitBreakerSnapshot:
        """Return safe operational state without exposing failure details."""

        return CircuitBreakerSnapshot(
            name=self.name,
            state=self._state,
            consecutive_failures=self._consecutive_failures,
            last_state_change_timestamp=self._last_state_change_timestamp,
            rejected_calls=self._rejected_calls,
            fallback_calls=self._fallback_calls,
        )

    async def call(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Invoke an async dependency with circuit-breaker protection."""

        is_half_open_probe, state_generation = await self._admit_call()
        try:
            result = await fn(*args, **kwargs)
        except asyncio.CancelledError:
            await self._finish_non_counted_call(is_half_open_probe, state_generation)
            raise
        except Exception as error:
            opened = False
            if self._failure_classifier(error):
                opened = await self._record_counted_failure(
                    is_half_open_probe=is_half_open_probe,
                    state_generation=state_generation,
                )
            else:
                await self._finish_non_counted_call(is_half_open_probe, state_generation)

            if opened:
                raise pybreaker.CircuitBreakerError(_UNAVAILABLE_MESSAGE) from None
            raise
        else:
            await self._record_success(is_half_open_probe, state_generation)
            return result

    async def _admit_call(self) -> tuple[bool, int]:
        """Admit a call when the circuit state permits execution."""
        async with self._lock:
            if self._state == _OPEN:
                opened_at = self._opened_at
                if opened_at is None or self._monotonic_clock() - opened_at < self.reset_timeout:
                    self._record_rejection_locked()
                    raise pybreaker.CircuitBreakerError(_UNAVAILABLE_MESSAGE)
                self._transition_locked(_HALF_OPEN)

            if self._state == _HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._record_rejection_locked()
                    raise pybreaker.CircuitBreakerError(_UNAVAILABLE_MESSAGE)
                self._half_open_calls += 1
                return True, self._state_generation

            return False, self._state_generation

    async def _record_counted_failure(self, *, is_half_open_probe: bool, state_generation: int) -> bool:
        """Record the counted failure."""
        async with self._lock:
            if is_half_open_probe:
                self._half_open_calls = max(0, self._half_open_calls - 1)
                if self._state == _HALF_OPEN and state_generation == self._state_generation:
                    self._transition_locked(_OPEN)
                    return True
                return self._state == _OPEN

            if self._state != _CLOSED or state_generation != self._state_generation:
                return self._state == _OPEN

            self._consecutive_failures += 1
            self._publish_metrics()
            if self._consecutive_failures >= self.fail_max:
                self._transition_locked(_OPEN)
                return True
            return False

    async def _finish_non_counted_call(self, is_half_open_probe: bool, state_generation: int) -> None:
        """Release a probe without changing dependency-health state.

        A validation or authorization failure does not prove that the external
        dependency is unhealthy. In half-open state, leave the breaker open to
        another valid probe rather than reopening it or declaring recovery.
        """

        if not is_half_open_probe:
            return
        async with self._lock:
            if self._state == _HALF_OPEN and state_generation == self._state_generation:
                self._half_open_calls = max(0, self._half_open_calls - 1)
                self._publish_metrics()

    async def _record_success(self, is_half_open_probe: bool, state_generation: int) -> None:
        """Record the success."""
        async with self._lock:
            if is_half_open_probe:
                self._half_open_calls = max(0, self._half_open_calls - 1)
                if self._state == _HALF_OPEN and state_generation == self._state_generation:
                    self._consecutive_failures = 0
                    self._transition_locked(_CLOSED)
                return

            if self._state == _CLOSED and state_generation == self._state_generation:
                self._consecutive_failures = 0
                self._publish_metrics()

    def record_fallback(self) -> None:
        """Record an invoked fallback without storing failure payloads."""

        self._fallback_calls += 1
        circuit_breaker_fallback_calls_total.labels(breaker=self.name).inc()

    def _record_rejection_locked(self) -> None:
        """Record the rejection locked."""
        self._rejected_calls += 1
        circuit_breaker_rejected_calls_total.labels(breaker=self.name).inc()
        self._publish_metrics()

    def _transition_locked(self, new_state: str) -> None:
        """Transition circuit state while holding the internal lock."""
        self._state = new_state
        self._state_generation += 1
        self._last_state_change_timestamp = self._wall_clock()
        self._opened_at = self._monotonic_clock() if new_state == _OPEN else None
        if new_state != _HALF_OPEN:
            self._half_open_calls = 0
        self._publish_metrics()

    def _publish_metrics(self) -> None:
        """Publish the metrics."""
        circuit_breaker_state.labels(breaker=self.name).set(_STATE_METRIC_VALUES[self._state])
        circuit_breaker_consecutive_failures.labels(breaker=self.name).set(self._consecutive_failures)
        circuit_breaker_state_changed_timestamp_seconds.labels(breaker=self.name).set(self._last_state_change_timestamp)


# Dependency-specific breakers. A stricter classifier can be injected where an
# SDK exposes a richer, stable transient-error hierarchy.
llm_breaker = AsyncCircuitBreaker(name="llm_api", fail_max=5, reset_timeout=30)
vector_store_breaker = AsyncCircuitBreaker(name="vector_store", fail_max=3, reset_timeout=15)
knowledge_graph_breaker = AsyncCircuitBreaker(name="knowledge_graph", fail_max=3, reset_timeout=15)
redis_breaker = AsyncCircuitBreaker(name="redis", fail_max=10, reset_timeout=10)


async def call_async(breaker: AsyncCircuitBreaker, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    """Call an asynchronous dependency through the supplied breaker."""

    return await breaker.call(fn, *args, **kwargs)


async def call_with_fallback(
    breaker: AsyncCircuitBreaker,
    primary: Callable[..., Awaitable[T]],
    fallback: Callable[..., Awaitable[T]] | None = None,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Call a protected dependency and use the optional fallback on failure."""

    try:
        return await call_async(breaker, primary, *args, **kwargs)
    except Exception:
        if fallback is None:
            raise
        breaker.record_fallback()
        return await fallback(*args, **kwargs)
