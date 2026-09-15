"""结构化日志配置（structlog + JSON 输出）

提供:
  - setup_logging():       进程启动时调用一次，配置 structlog
  - get_logger(name):      获取绑定上下文的 logger
  - bind_context(**kv):    绑定 trace_id / user_id 等上下文字段
  - clear_context():       请求结束时清理上下文

日志规范:
  - 输出 JSON 格式（便于 Loki / ELK 采集）
  - 必含字段: timestamp, level, event, service
  - 可选字段: trace_id, span_id, user_id, org_id, latency_ms 等
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import structlog

# 服务名（部署时通过环境变量覆盖）
_SERVICE_NAME = "agenthub"

# 已配置标记，避免重复初始化
_CONFIGURED = False

REDACTED = "[REDACTED]"
TRUNCATED = "[TRUNCATED]"
_DEFAULT_MAX_DEPTH = 5
_DEFAULT_MAX_COLLECTION_ITEMS = 50
_DEFAULT_MAX_STRING_LENGTH = 512

_SAFE_FIELDS = frozenset(
    {
        "action",
        "api_key_id",
        "benchmark_id",
        "change_type",
        "code",
        "dependency",
        "doc_id",
        "document_id",
        "error_type",
        "event",
        "file_deleted",
        "file_reference",
        "file_size",
        "key_fingerprint",
        "level",
        "method",
        "operation",
        "org_id",
        "question_id",
        "question_fingerprint",
        "request_id",
        "resource_id",
        "result",
        "route",
        "service",
        "span_id",
        "source_document_id",
        "status",
        "size_bytes",
        "target_id",
        "target_type",
        "task_id",
        "timestamp",
        "trace_id",
        "user_id",
        "webhook_event",
        "webhook_id",
    }
)
_SENSITIVE_FIELD_PARTS = frozenset(
    {
        "answer",
        "apikey",
        "authorization",
        "body",
        "connection",
        "content",
        "context",
        "cookie",
        "credential",
        "dsn",
        "endpoint",
        "error",
        "exception",
        "file",
        "filename",
        "jwt",
        "message",
        "password",
        "path",
        "payload",
        "question",
        "raw",
        "request",
        "response",
        "secret",
        "source",
        "stack",
        "token",
        "traceback",
        "uri",
        "url",
    }
)
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|password|passwd|secret|token)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s]+"),
    re.compile(r"(?<![\w:])/(?:[^/\s]+/)+[^/\s]*"),
)


def safe_fingerprint(value: str, *, namespace: str = "value") -> str:
    """Return a deterministic, versioned, non-reversible correlation value."""
    digest = hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()
    return f"v1:{digest}"


def safe_file_reference(file_path: str) -> str:
    """Return a correlation-safe reference for a filesystem path."""
    return f"file:{safe_fingerprint(file_path, namespace='file-path')}"


def _field_is_sensitive(field_name: str | None) -> bool:
    """Return the field is sensitive."""
    if not field_name:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", field_name.lower()).strip("_")
    if normalized in _SAFE_FIELDS or normalized.endswith(("_count", "_total", "_ms")):
        return False
    parts = set(normalized.split("_"))
    return bool(parts & _SENSITIVE_FIELD_PARTS)


def _sanitize_string(value: str, max_string_length: int) -> str:
    """Sanitize the string."""
    sanitized = value
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        sanitized = pattern.sub(REDACTED, sanitized)
    if len(sanitized) > max_string_length:
        return f"{sanitized[:max_string_length]}{TRUNCATED}"
    return sanitized


def sanitize_for_observability(
    value: Any,
    *,
    field_name: str | None = None,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_collection_items: int = _DEFAULT_MAX_COLLECTION_ITEMS,
    max_string_length: int = _DEFAULT_MAX_STRING_LENGTH,
    _depth: int = 0,
) -> Any:
    """Recursively sanitize untrusted values before logs or audit persistence."""
    if _field_is_sensitive(field_name):
        return REDACTED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _sanitize_string(value, max_string_length)
    if isinstance(value, BaseException):
        return type(value).__name__
    if _depth >= max_depth:
        return TRUNCATED
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:max_collection_items]:
            key_string = str(key)
            sanitized[key_string] = sanitize_for_observability(
                item,
                field_name=key_string,
                max_depth=max_depth,
                max_collection_items=max_collection_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
            )
        if len(items) > max_collection_items:
            sanitized["_truncated"] = TRUNCATED
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        sanitized_items = [
            sanitize_for_observability(
                item,
                max_depth=max_depth,
                max_collection_items=max_collection_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
            )
            for item in items[:max_collection_items]
        ]
        if len(items) > max_collection_items:
            sanitized_items.append(TRUNCATED)
        return sanitized_items
    return f"<{type(value).__name__}>"


def _sanitize_event(_logger, _method, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Sanitize the event."""
    return sanitize_for_observability(event_dict)


class SensitiveDataFilter(logging.Filter):
    """Apply the minimum safe boundary to standard-library log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter the sensitive data filter."""
        if record.exc_info:
            error = record.exc_info[1]
            record.error_type = type(error).__name__ if error else "Exception"
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        record.msg = sanitize_for_observability(record.msg)
        if isinstance(record.args, Mapping):
            record.args = sanitize_for_observability(record.args)
        elif isinstance(record.args, tuple):
            record.args = tuple(sanitize_for_observability(item) for item in record.args)
        elif record.args:
            record.args = sanitize_for_observability(record.args)
        return True


def setup_logging(
    log_level: str = "INFO",
    service_name: str = _SERVICE_NAME,
    json_output: bool = True,
    force: bool = False,
) -> None:
    """配置 structlog，应在进程启动时调用一次。

    Args:
        log_level: 日志级别字符串（DEBUG/INFO/WARN/ERROR）
        service_name: 服务名，写入每条日志的 service 字段
        json_output: True 输出 JSON，False 输出人类可读的控制台格式
        force: 强制重新配置（测试用）
    """
    global _CONFIGURED, _SERVICE_NAME
    if _CONFIGURED and not force:
        _SERVICE_NAME = service_name
        return
    _SERVICE_NAME = service_name
    _CONFIGURED = True

    # force 模式下重置缓存，避免 logger 持有旧 stdout 引用（测试中 capsys 会替换 sys.stdout）
    if force:
        structlog.reset_defaults()

    # 同步标准 logging 级别（structlog 内部会用到）
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        stream=sys.stdout,
        format="%(message)s",
        force=force,
    )
    for handler in logging.getLogger().handlers:
        if not any(isinstance(filter_, SensitiveDataFilter) for filter_ in handler.filters):
            handler.addFilter(SensitiveDataFilter())

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _add_service_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _sanitize_event,
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def _add_service_name(_logger, _method, event_dict: dict) -> dict:
    """注入 service 字段"""
    event_dict["service"] = _SERVICE_NAME
    return event_dict


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """获取 logger，可绑定模块名"""
    if not _CONFIGURED:
        setup_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()


def bind_context(**kwargs: Any) -> None:
    """绑定请求级上下文字段（trace_id, user_id 等）

    典型用法: 中间件中调用 bind_context(trace_id=..., user_id=...)
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """清理上下文，应在请求结束时调用"""
    structlog.contextvars.clear_contextvars()
