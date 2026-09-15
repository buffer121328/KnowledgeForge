"""Acceptance tests for the read-only evaluation results API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_current_user
from api.routers.evaluation_pkg import evaluation_router
from domain.identity import Permission, UserRole, UserContext


def _user(*, role: UserRole = UserRole.ORGANIZATION_ADMIN) -> UserContext:
    return UserContext(
        user_id="user-a",
        username="alice",
        role=role,
        org_id="tenant-a",
        permissions=(
        list(Permission)
        if role == UserRole.ORGANIZATION_ADMIN
        else [Permission.DOC_READ]
    ),
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """A routed app whose results root points at a temporary directory."""
    app = FastAPI()
    app.include_router(evaluation_router)
    app.dependency_overrides[get_current_user] = lambda: _user()
    monkeypatch.setattr("api.routers.evaluation_pkg._common.RESULTS_ROOT", tmp_path)
    return TestClient(app)


def _write_run(root: Path, run_id: str, responses: list[dict]) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "started_at": "2026-08-01T12:00:00+00:00",
                "retrieval_modes": ["vector", "hybrid"],
                "benchmark_source": "/private/tmp/benchmark.jsonl",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "responses.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in responses) + "\n",
        encoding="utf-8",
    )


def _record(benchmark_id: str) -> dict:
    return {
        "run_id": "ragas-20260801T120000Z-aaaa1111",
        "benchmark_id": benchmark_id,
        "retrieval_mode": "vector",
        "category": "single_document_fact",
        "question": "公司人力资源管理制度的主要管理范围是什么？",
        "response": "制度共十八章……" + "长" * 300,
        "contexts": [
            {
                "rank": 1,
                "content": "完整检索文本",
                "source": "056公司人力资源管理制度",
                "score": 0.9,
                "retrieval_type": "vector",
                "source_document_id": "5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9",
                "metadata": {"source_document_id": "5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9"},
            }
        ],
        "retrieved_context_ids": ["5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9"],
        "reference_context_ids": ["5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9"],
        "status": "succeeded",
        "latency_ms": 12.3,
        "exception": None,
    }


def test_empty_results_root_returns_empty_list(client: TestClient) -> None:
    response = client.get("/evaluation/runs")

    assert response.status_code == 200
    assert response.json() == {"runs": []}


def test_non_admin_is_forbidden(client: TestClient) -> None:
    client.app.dependency_overrides[get_current_user] = lambda: _user(role=UserRole.EDITOR)

    response = client.get("/evaluation/runs")

    assert response.status_code == 403


def test_lists_runs_with_bounded_summaries(
    client: TestClient, tmp_path: Path,
) -> None:
    _write_run(tmp_path, "ragas-20260801T120000Z-aaaa1111", [_record("company-demo-01")])

    response = client.get("/evaluation/runs")

    assert response.status_code == 200
    body = response.json()
    assert len(body["runs"]) == 1
    run = body["runs"][0]
    assert run["run_id"] == "ragas-20260801T120000Z-aaaa1111"
    assert run["retrieval_modes"] == ["vector", "hybrid"]
    assert run["completed"] == 1
    assert "tmp" not in response.text


def test_run_detail_returns_metadata_and_404_for_invalid_id(
    client: TestClient, tmp_path: Path,
) -> None:
    _write_run(tmp_path, "ragas-20260801T120000Z-aaaa1111", [])

    ok = client.get("/evaluation/runs/ragas-20260801T120000Z-aaaa1111")
    assert ok.status_code == 200
    assert ok.json()["metadata"]["benchmark_source"] == "benchmark.jsonl"
    assert ok.json()["incomplete"] is False

    for bad_id in ("../secret", "ragas-20260801T120000Z-aaaa1111/x", "not-a-run"):
        response = client.get(f"/evaluation/runs/{bad_id}")
        assert response.status_code == 404, bad_id


def test_records_paginate_and_truncate_response(
    client: TestClient, tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "ragas-20260801T120000Z-aaaa1111",
        [_record(f"company-demo-{i:02d}") for i in range(1, 26)],
    )

    page = client.get(
        "/evaluation/runs/ragas-20260801T120000Z-aaaa1111/records",
        params={"page": 1, "page_size": 10},
    )
    assert page.status_code == 200
    body = page.json()
    assert body["total"] == 25
    assert len(body["records"]) == 10
    first = body["records"][0]
    assert first["response"].endswith("…")
    assert "content" not in first["contexts"][0]

    missing = client.get("/evaluation/runs/ragas-20260801T120000Z-zzzz9999/records")
    assert missing.status_code == 404


def test_records_pagination_validation(client: TestClient) -> None:
    response = client.get(
        "/evaluation/runs/ragas-20260801T120000Z-aaaa1111/records",
        params={"page": 0},
    )
    assert response.status_code == 422
