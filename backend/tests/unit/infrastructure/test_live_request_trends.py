from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from api.middleware.request_trend import RequestTrendMiddleware
from infrastructure.audit import trends as request_trends


NOW = datetime(2026, 8, 9, 3, 27, tzinfo=timezone.utc)


def _use_file(monkeypatch, tmp_path):
    path = tmp_path / "request-trend.json"
    monkeypatch.setattr(request_trends, "_DEFAULT_FILE", path)
    return path


def test_minute_trends_are_utc_and_organization_isolated(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    request_trends.record_request(
        "/api/v1/docs", org_id="org-a", status_code=200, duration_ms=40, recorded_at=NOW
    )
    request_trends.record_request(
        "/api/v1/qa/ask", org_id="org-a", is_qa=True, status_code=422, duration_ms=60, recorded_at=NOW
    )
    request_trends.record_request(
        "/api/v1/users", org_id="org-b", status_code=200, duration_ms=10, recorded_at=NOW
    )

    org_a = request_trends.get_trend("org-a", "60m", now=NOW)[-1]
    org_b = request_trends.get_trend("org-b", "60m", now=NOW)[-1]

    assert org_a["timestamp"] == "2026-08-09T03:27:00Z"
    assert org_a["requests"] == 2
    assert org_a["qa"] == 1
    assert org_a["errors"] == 1
    assert org_a["average_latency_ms"] == 50
    assert org_b["requests"] == 1
    assert "/api/v1/users" not in str(org_a["details"])


def test_24_hour_window_rolls_minutes_into_utc_hours(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    request_trends.record_request("/api/v1/docs", org_id="org-a", recorded_at=NOW - timedelta(minutes=26))
    request_trends.record_request("/api/v1/docs", org_id="org-a", recorded_at=NOW)

    points = request_trends.get_trend("org-a", "24h", now=NOW)

    assert len(points) == 24
    assert points[-1]["timestamp"] == "2026-08-09T03:00:00Z"
    assert points[-1]["requests"] == 2


def test_retention_and_atomic_v2_file(monkeypatch, tmp_path):
    path = _use_file(monkeypatch, tmp_path)
    monkeypatch.setattr(request_trends, "_RETENTION_MINUTES", 2)
    data = request_trends._empty_store()
    for minute in range(3):
        key = f"2026-08-09T03:0{minute}Z"
        data["minutes"][key] = {"organizations": {}}

    request_trends._save_unlocked(path, data)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["schema_version"] == 2
    assert list(saved["minutes"]) == ["2026-08-09T03:01Z", "2026-08-09T03:02Z"]
    assert not list(tmp_path.glob(".request-trend.json.*"))


def test_self_and_health_endpoints_are_excluded_from_collection():
    assert "/api/v1/admin/request-trends" in RequestTrendMiddleware.SKIP_EXACT
    assert "/api/health/ready" in RequestTrendMiddleware.SKIP_EXACT


def test_error_aggregate_is_included_in_summary(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    request_trends.record_request(
        "/api/v1/docs", org_id="org-a", status_code=500, duration_ms=75, recorded_at=NOW
    )
    points = request_trends.get_trend("org-a", "60m", now=NOW)

    assert request_trends.summarize_trend(points) == {
        "requests": 1,
        "qa": 0,
        "errors": 1,
        "average_latency_ms": 75.0,
        "details": points[-1]["details"],
    }
