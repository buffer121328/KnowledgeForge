from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from shared.config.settings import Settings
from shared.utils.dependency_resilience import (
    DependencyExecutionPolicy,
    DependencyRequestTimeoutError,
    build_dependency_policies,
    execute_with_policy,
)
from shared.utils.metrics import (
    dependency_final_failures_total,
    dependency_retries_total,
    dependency_timeouts_total,
)


class TestDependencyExecutionPolicies:
    def test_settings_resolve_independent_policies(self) -> None:
        configured = Settings(
            vector_connect_timeout_seconds=1.5,
            vector_read_timeout_seconds=2.5,
            vector_total_timeout_seconds=3.5,
            vector_max_retries=1,
            llm_connect_timeout_seconds=4.5,
            llm_read_timeout_seconds=5.5,
            llm_total_timeout_seconds=10.5,
            llm_max_retries=2,
        )

        policies = build_dependency_policies(configured)

        assert policies["vector"] == DependencyExecutionPolicy(
            dependency="vector",
            connect_timeout_seconds=1.5,
            read_timeout_seconds=2.5,
            total_timeout_seconds=3.5,
            max_retries=1,
        )
        assert policies["llm"] == DependencyExecutionPolicy(
            dependency="llm",
            connect_timeout_seconds=4.5,
            read_timeout_seconds=5.5,
            total_timeout_seconds=10.5,
            max_retries=2,
        )
        assert policies["graph"].total_timeout_seconds != policies["vector"].total_timeout_seconds

    def test_settings_reject_non_positive_timeout_values(self) -> None:
        with pytest.raises(ValidationError):
            Settings(vector_total_timeout_seconds=0)

        with pytest.raises(ValidationError):
            Settings(webhook_max_retries=-1)


class TestBoundedDependencyExecution:
    @pytest.mark.asyncio
    async def test_transient_failure_retries_with_jittered_backoff_then_recovers(self) -> None:
        policy = DependencyExecutionPolicy(
            dependency="unit-retry",
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            total_timeout_seconds=5,
            max_retries=1,
            base_backoff_seconds=0.2,
            max_backoff_seconds=1,
        )
        attempts = 0
        delays: list[float] = []

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("temporary")
            return "recovered"

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        result = await execute_with_policy(
            policy,
            operation,
            sleep=record_sleep,
            random_value=lambda: 0.75,
        )

        assert result == "recovered"
        assert attempts == 2
        assert delays == [pytest.approx(0.25)]

    @pytest.mark.asyncio
    async def test_non_transient_failure_preserves_original_error_without_retry(self) -> None:
        policy = DependencyExecutionPolicy(
            dependency="unit-non-transient",
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            total_timeout_seconds=5,
            max_retries=2,
        )
        operation = AsyncMock(side_effect=ValueError("invalid query"))
        sleep = AsyncMock()

        with pytest.raises(ValueError, match="invalid query"):
            await execute_with_policy(policy, operation, sleep=sleep)

        assert operation.await_count == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_total_budget_cancels_slow_operation_and_records_timeout_failure(self) -> None:
        policy = DependencyExecutionPolicy(
            dependency="unit-timeout",
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            total_timeout_seconds=0.01,
            max_retries=2,
        )
        cancelled = asyncio.Event()

        async def slow_operation() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        before_timeout = dependency_timeouts_total.labels(dependency=policy.dependency)._value.get()
        before_final_failure = dependency_final_failures_total.labels(dependency=policy.dependency)._value.get()

        with pytest.raises(DependencyRequestTimeoutError):
            await execute_with_policy(policy, slow_operation)

        assert cancelled.is_set()
        assert dependency_timeouts_total.labels(dependency=policy.dependency)._value.get() == before_timeout + 1
        assert (
            dependency_final_failures_total.labels(dependency=policy.dependency)._value.get()
            == before_final_failure + 1
        )

    @pytest.mark.asyncio
    async def test_budget_prevents_a_retry_that_cannot_start_before_deadline(self) -> None:
        policy = DependencyExecutionPolicy(
            dependency="unit-budget",
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            total_timeout_seconds=0.01,
            max_retries=3,
            base_backoff_seconds=0.2,
            max_backoff_seconds=1,
        )
        operation = AsyncMock(side_effect=ConnectionError("temporary"))
        sleep = AsyncMock()

        with pytest.raises(DependencyRequestTimeoutError):
            await execute_with_policy(policy, operation, sleep=sleep, random_value=lambda: 0.5)

        assert operation.await_count == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_metrics_have_fixed_payload_free_dependency_labels(self) -> None:
        policy = DependencyExecutionPolicy(
            dependency="unit-metrics",
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            total_timeout_seconds=5,
            max_retries=1,
        )
        attempts = 0

        async def operation() -> None:
            nonlocal attempts
            attempts += 1
            raise ConnectionError("https://secret.internal/?token=not-a-label")

        before_retry = dependency_retries_total.labels(dependency=policy.dependency)._value.get()
        before_final_failure = dependency_final_failures_total.labels(dependency=policy.dependency)._value.get()

        with pytest.raises(ConnectionError):
            await execute_with_policy(policy, operation, sleep=AsyncMock())

        assert attempts == 2
        assert dependency_retries_total.labels(dependency=policy.dependency)._value.get() == before_retry + 1
        assert (
            dependency_final_failures_total.labels(dependency=policy.dependency)._value.get()
            == before_final_failure + 1
        )
        for metric in (
            dependency_timeouts_total,
            dependency_retries_total,
            dependency_final_failures_total,
        ):
            assert metric._labelnames == ("dependency",)
