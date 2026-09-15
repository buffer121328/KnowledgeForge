"""Organization-isolated UTC request trend aggregation for the admin dashboard."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from shared.config import settings

TrendWindow = Literal["60m", "24h"]

_LOCK = threading.Lock()
_SCHEMA_VERSION = 2
_RETENTION_MINUTES = 48 * 60
_DEFAULT_FILE = Path(settings.upload_dir).resolve().parent / "logs" / "request_trend.json"


def _trend_file() -> Path:
    path = _DEFAULT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _minute_start(value: datetime | None = None) -> datetime:
    candidate = value or datetime.now(timezone.utc)
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc).replace(second=0, microsecond=0)


def _minute_key(value: datetime | None = None) -> str:
    return _minute_start(value).strftime("%Y-%m-%dT%H:%MZ")


def _empty_store() -> dict[str, Any]:
    return {"schema_version": _SCHEMA_VERSION, "minutes": {}}


def _load_unlocked(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_store()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return _empty_store()
    if data.get("schema_version") != _SCHEMA_VERSION or not isinstance(data.get("minutes"), dict):
        return _empty_store()
    return data


def _trim(data: dict[str, Any]) -> None:
    minutes = data.setdefault("minutes", {})
    keys = sorted(minutes)
    for key in keys[:-_RETENTION_MINUTES]:
        minutes.pop(key, None)


def _save_unlocked(path: Path, data: dict[str, Any]) -> None:
    _trim(data)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _file_lock(path: Path):
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_stream = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
    return lock_stream


def _new_aggregate() -> dict[str, Any]:
    return {
        "requests": 0,
        "qa": 0,
        "errors": 0,
        "total_latency_ms": 0.0,
        "details": {},
    }


def _merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for field in ("requests", "qa", "errors"):
        target[field] = int(target.get(field, 0)) + int(source.get(field, 0))
    target["total_latency_ms"] = round(
        float(target.get("total_latency_ms", 0)) + float(source.get("total_latency_ms", 0)), 2
    )
    for request_class, detail in (source.get("details") or {}).items():
        detail_target = target.setdefault("details", {}).setdefault(
            request_class,
            {"count": 0, "error_count": 0, "total_latency_ms": 0.0, "methods": {}, "routes": {}},
        )
        for field in ("count", "error_count"):
            detail_target[field] = int(detail_target.get(field, 0)) + int(detail.get(field, 0))
        detail_target["total_latency_ms"] = round(
            float(detail_target.get("total_latency_ms", 0)) + float(detail.get("total_latency_ms", 0)), 2
        )
        for method, count in (detail.get("methods") or {}).items():
            methods = detail_target.setdefault("methods", {})
            methods[method] = int(methods.get(method, 0)) + int(count)
        for route, values in (detail.get("routes") or {}).items():
            routes = detail_target.setdefault("routes", {})
            route_target = routes.setdefault(route, {"count": 0, "error_count": 0, "last_status": 0})
            route_target["count"] += int(values.get("count", 0))
            route_target["error_count"] += int(values.get("error_count", 0))
            route_target["last_status"] = int(values.get("last_status", 0))


def record_request(
    path: str,
    *,
    org_id: str,
    is_qa: bool = False,
    method: str = "GET",
    status_code: int = 200,
    duration_ms: float = 0.0,
    recorded_at: datetime | None = None,
) -> None:
    """Atomically record one authenticated organization-scoped API request."""
    normalized_org = org_id.strip()[:128]
    if not normalized_org:
        return
    key = _minute_key(recorded_at)
    request_class = "ai" if is_qa else "system_api"
    normalized_path = path[:256]
    normalized_method = method.upper()[:16]
    status_value = max(0, min(int(status_code), 999))
    duration_value = round(max(0.0, min(float(duration_ms), 3_600_000.0)), 2)
    trend_path = _trend_file()
    with _LOCK:
        lock_stream = _file_lock(trend_path)
        try:
            data = _load_unlocked(trend_path)
            minute = data["minutes"].setdefault(key, {"organizations": {}})
            bucket = minute.setdefault("organizations", {}).setdefault(normalized_org, _new_aggregate())
            bucket["requests"] = int(bucket.get("requests", 0)) + 1
            bucket["qa"] = int(bucket.get("qa", 0)) + int(is_qa)
            bucket["errors"] = int(bucket.get("errors", 0)) + int(status_value >= 400)
            bucket["total_latency_ms"] = round(float(bucket.get("total_latency_ms", 0)) + duration_value, 2)
            detail = bucket.setdefault("details", {}).setdefault(
                request_class,
                {"count": 0, "error_count": 0, "total_latency_ms": 0.0, "methods": {}, "routes": {}},
            )
            detail["count"] += 1
            detail["error_count"] += int(status_value >= 400)
            detail["total_latency_ms"] = round(float(detail["total_latency_ms"]) + duration_value, 2)
            detail["methods"][normalized_method] = int(detail["methods"].get(normalized_method, 0)) + 1
            route = detail["routes"].setdefault(normalized_path, {"count": 0, "error_count": 0, "last_status": status_value})
            route["count"] += 1
            route["error_count"] += int(status_value >= 400)
            route["last_status"] = status_value
            _save_unlocked(trend_path, data)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()


def get_trend(org_id: str, window: TrendWindow = "60m", *, now: datetime | None = None) -> list[dict[str, Any]]:
    """Return UTC minute or hour points for one organization only."""
    current_minute = _minute_start(now)
    trend_path = _trend_file()
    with _LOCK:
        lock_stream = _file_lock(trend_path)
        try:
            data = _load_unlocked(trend_path)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()
    minutes = data.get("minutes", {})
    if window == "60m":
        slots = [current_minute - timedelta(minutes=index) for index in range(59, -1, -1)]
        groups = [(slot, [slot]) for slot in slots]
    elif window == "24h":
        current_hour = current_minute.replace(minute=0)
        hours = [current_hour - timedelta(hours=index) for index in range(23, -1, -1)]
        groups = [(hour, [hour + timedelta(minutes=minute) for minute in range(60)]) for hour in hours]
    else:
        raise ValueError("unsupported trend window")

    points: list[dict[str, Any]] = []
    for slot, members in groups:
        aggregate = _new_aggregate()
        for member in members:
            minute = minutes.get(_minute_key(member), {})
            bucket = (minute.get("organizations") or {}).get(org_id, {})
            _merge(aggregate, bucket)
        average_latency = round(aggregate["total_latency_ms"] / aggregate["requests"], 2) if aggregate["requests"] else 0.0
        points.append({
            "timestamp": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "requests": aggregate["requests"],
            "qa": aggregate["qa"],
            "errors": aggregate["errors"],
            "average_latency_ms": average_latency,
            "details": aggregate["details"],
        })
    return points


def summarize_trend(points: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _new_aggregate()
    for point in points:
        point_copy = dict(point)
        point_copy["total_latency_ms"] = float(point.get("average_latency_ms", 0)) * int(point.get("requests", 0))
        _merge(aggregate, point_copy)
    return {
        "requests": aggregate["requests"],
        "qa": aggregate["qa"],
        "errors": aggregate["errors"],
        "average_latency_ms": round(aggregate["total_latency_ms"] / aggregate["requests"], 2) if aggregate["requests"] else 0.0,
        "details": aggregate["details"],
    }
