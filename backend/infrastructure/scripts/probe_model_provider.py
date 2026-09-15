"""Secret-safe runtime probe for the configured OpenAI-compatible text model."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from shared.config import settings
from shared.utils.model_provider import build_model_provider_timeout


class ProbeConfig(Protocol):
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    llm_connect_timeout_seconds: float
    llm_read_timeout_seconds: float
    llm_total_timeout_seconds: float


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _emit(output: Callable[[str], Any], **payload: Any) -> None:
    output(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def run_probe(
    *,
    config: ProbeConfig = settings,
    client_factory: Callable[..., Any] = httpx.Client,
    output: Callable[[str], Any] = print,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Probe model discovery and one minimal chat request without echoing secrets."""

    if not config.deepseek_api_key:
        _emit(output, probe="configuration", ok=False, error_type="MissingCredentialError")
        return 1

    headers = {
        "Authorization": f"Bearer {config.deepseek_api_key}",
        "Content-Type": "application/json",
    }
    timeout = build_model_provider_timeout(config)
    probes = (
        ("models", "GET", _endpoint(config.deepseek_base_url, "models"), None),
        (
            "chat",
            "POST",
            _endpoint(config.deepseek_base_url, "chat/completions"),
            {
                "model": config.deepseek_model,
                "messages": [{"role": "user", "content": "Reply with OK only."}],
                "temperature": 0,
                "max_tokens": 8,
            },
        ),
    )

    try:
        with client_factory(timeout=timeout, follow_redirects=False) as client:
            for name, method, url, payload in probes:
                started = clock()
                response = client.request(method, url, headers=headers, json=payload)
                elapsed = max(0.0, clock() - started)
                _emit(
                    output,
                    probe=name,
                    status=response.status_code,
                    elapsed_seconds=round(elapsed, 3),
                    ok=200 <= response.status_code < 300,
                )
                if not 200 <= response.status_code < 300:
                    return 1
    except Exception as error:  # noqa: BLE001 - probe output must sanitize arbitrary client errors.
        _emit(output, probe="request", ok=False, error_type=type(error).__name__)
        return 1
    return 0


def main() -> int:
    return run_probe()


if __name__ == "__main__":
    raise SystemExit(main())
