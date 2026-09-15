"""ATDD for shallow liveness and dependency-aware readiness."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware.auth import _is_public_path
from api.routers.admin import health_router
from infrastructure.readiness import ReadinessService


def _client(report: dict) -> TestClient:
    app = FastAPI()
    app.state.readiness = MagicMock(check=AsyncMock(return_value=report))
    app.include_router(health_router)
    return TestClient(app)


def test_liveness_remains_shallow_when_dependencies_are_unavailable() -> None:
    client = _client({"status": "not_ready", "components": {"knowledge_graph": "unavailable"}})

    response = client.get("/api/health/live")
    compatibility = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"
    assert compatibility.json()["status"] == "ok"


def test_readiness_returns_200_only_when_all_components_are_ready() -> None:
    components = {
        "vector_store": "ready",
        "knowledge_graph": "ready",
        "security_state": "ready",
        "task_broker": "ready",
    }
    response = _client({"status": "ready", "components": components}).get(
        "/api/health/ready"
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "components": components}


def test_readiness_failure_is_503_and_secret_free() -> None:
    components = {
        "vector_store": "ready",
        "knowledge_graph": "unavailable",
        "security_state": "ready",
        "task_broker": "ready",
    }
    response = _client({"status": "not_ready", "components": components}).get(
        "/api/health/ready"
    )

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "components": components}
    assert "bolt://" not in response.text


@pytest.mark.asyncio
async def test_readiness_coordinator_marks_failed_probe_without_exception_details() -> None:
    vector = MagicMock(health_check=AsyncMock(return_value=True))
    graph = MagicMock(
        health_check=AsyncMock(side_effect=RuntimeError("bolt://neo4j:secret@internal"))
    )
    security = MagicMock(ping=MagicMock(return_value=True))
    service = ReadinessService(
        vector_store=vector,
        knowledge_graph=graph,
        security_state=security,
        timeout_seconds=0.5,
        broker_probe=lambda: True,
    )

    report = await service.check()

    assert report["status"] == "not_ready"
    assert report["components"]["knowledge_graph"] == "unavailable"
    assert "secret" not in repr(report)


@pytest.mark.asyncio
async def test_readiness_includes_required_database_schema_probe() -> None:
    """Configured PostgreSQL facts participate in aggregate readiness."""
    vector = MagicMock(health_check=AsyncMock(return_value=True))
    graph = MagicMock(health_check=AsyncMock(return_value=True))
    security = MagicMock(ping=MagicMock(return_value=True))
    database = MagicMock(ping=MagicMock(return_value=False))
    service = ReadinessService(
        vector_store=vector,
        knowledge_graph=graph,
        security_state=security,
        database=database,
        timeout_seconds=0.5,
        broker_probe=lambda: True,
    )

    report = await service.check()

    assert report["status"] == "not_ready"
    assert report["components"]["postgresql"] == "unavailable"


def test_health_contract_paths_are_public() -> None:
    assert _is_public_path("/api/health/live")
    assert _is_public_path("/api/health/ready")
