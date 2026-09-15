"""Helpers for constructing model-provider HTTP transport settings."""

from __future__ import annotations

from typing import Protocol

import httpx


class ModelProviderTimeoutConfig(Protocol):
    """Minimal configuration required to build a provider timeout."""

    llm_connect_timeout_seconds: float
    llm_read_timeout_seconds: float
    llm_total_timeout_seconds: float


def build_model_provider_timeout(config: ModelProviderTimeoutConfig) -> httpx.Timeout:
    """Build one explicit timeout shared by SDK clients and operational probes."""

    return httpx.Timeout(
        timeout=float(config.llm_total_timeout_seconds),
        connect=float(config.llm_connect_timeout_seconds),
        read=float(config.llm_read_timeout_seconds),
        write=float(config.llm_total_timeout_seconds),
        pool=float(config.llm_connect_timeout_seconds),
    )
