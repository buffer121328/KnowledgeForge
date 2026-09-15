"""Bounded, observable execution for safe external dependency calls."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from shared.config import settings
from shared.config.settings import Settings
from shared.utils.circuit_breaker import is_transient_dependency_error
from shared.utils.metrics import (
    dependency_final_failures_total,
    dependency_retries_total,
    dependency_timeouts_total,
)

T = TypeVar("T")
RetryPredicate = Callable[[Exception], bool]
Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]
RandomValue = Callable[[], float]

_SUPPORTED_DEPENDENCIES = ("vector", "graph", "llm", "webhook")


class DependencyRequestTimeoutError(TimeoutError):
    """Safe timeout signal for an exhausted dependency request budget."""

    def __init__(self, dependency: str) -> None:
        """Initialize the dependency request timeout error."""
        super().__init__(f"{dependency} dependency request timed out")
        self.dependency = dependency


@dataclass(frozen=True)
class DependencyExecutionPolicy:
    """Validated budget and retry limits for one external dependency class."""

    dependency: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    total_timeout_seconds: float
    max_retries: int
    base_backoff_seconds: float = 0.1
    max_backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        """Validate dependency execution policy values after initialization."""
        if not self.dependency:
            raise ValueError("dependency must be non-empty")
        for value_name in (
            "connect_timeout_seconds",
            "read_timeout_seconds",
            "total_timeout_seconds",
            "base_backoff_seconds",
            "max_backoff_seconds",
        ):
            if getattr(self, value_name) <= 0:
                raise ValueError(f"{value_name} must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.max_backoff_seconds < self.base_backoff_seconds:
            raise ValueError("max_backoff_seconds must be at least base_backoff_seconds")


def build_dependency_policies(config: Settings) -> dict[str, DependencyExecutionPolicy]:
    """Build separate, validated policies from the application configuration."""

    return {
        dependency: DependencyExecutionPolicy(
            dependency=dependency,
            connect_timeout_seconds=getattr(config, f"{dependency}_connect_timeout_seconds"),
            read_timeout_seconds=getattr(config, f"{dependency}_read_timeout_seconds"),
            total_timeout_seconds=getattr(config, f"{dependency}_total_timeout_seconds"),
            max_retries=getattr(config, f"{dependency}_max_retries"),
        )
        for dependency in _SUPPORTED_DEPENDENCIES
    }


def get_dependency_policy(dependency: str, config: Settings | None = None) -> DependencyExecutionPolicy:
    """Resolve one policy without exposing mutable global policy state."""

    try:
        return build_dependency_policies(config or settings)[dependency]
    except KeyError as error:
        raise ValueError(f"unsupported dependency policy: {dependency}") from error


def _jittered_backoff(policy: DependencyExecutionPolicy, retry_index: int, random_value: RandomValue) -> float:
    """Return bounded exponential backoff with a positive 0.5x–1.5x jitter."""

    exponential = min(policy.max_backoff_seconds, policy.base_backoff_seconds * (2**retry_index))
    bounded_random = min(1.0, max(0.0, random_value()))
    return min(policy.max_backoff_seconds, exponential * (0.5 + bounded_random))


async def execute_with_policy(
    policy: DependencyExecutionPolicy,
    operation: Callable[[], Awaitable[T]],
    *,
    retry_predicate: RetryPredicate = is_transient_dependency_error,
    sleep: Sleep = asyncio.sleep,
    clock: Clock = time.monotonic,
    random_value: RandomValue = random.random,
) -> T:
    """Execute a safe external operation within one total deadline.

    ``max_retries`` counts attempts after the first. The executor deliberately
    leaves state-changing workflows out of scope; callers opt in only for
    idempotent reads or explicitly retry-safe Webhook delivery attempts.
    """

    deadline = clock() + policy.total_timeout_seconds
    retries_performed = 0
    timeout_recorded = False

    def record_timeout() -> None:
        """Record the timeout."""
        nonlocal timeout_recorded
        if not timeout_recorded:
            dependency_timeouts_total.labels(dependency=policy.dependency).inc()
            timeout_recorded = True

    def record_final_failure() -> None:
        """Record the final failure."""
        dependency_final_failures_total.labels(dependency=policy.dependency).inc()

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            record_timeout()
            record_final_failure()
            raise DependencyRequestTimeoutError(policy.dependency)

        try:
            async with asyncio.timeout(remaining):
                return await operation()
        except asyncio.CancelledError:
            raise
        except TimeoutError as error:
            record_timeout()
            failure: Exception = error
        except Exception as error:
            failure = error

        if not retry_predicate(failure) or retries_performed >= policy.max_retries:
            record_final_failure()
            raise failure

        delay = _jittered_backoff(policy, retries_performed, random_value)
        # A retry that cannot begin before the deadline merely creates a retry
        # storm. Fail at the boundary instead of sleeping pointlessly.
        if clock() + delay >= deadline:
            record_timeout()
            record_final_failure()
            raise DependencyRequestTimeoutError(policy.dependency) from failure

        dependency_retries_total.labels(dependency=policy.dependency).inc()
        retries_performed += 1
        await sleep(delay)
