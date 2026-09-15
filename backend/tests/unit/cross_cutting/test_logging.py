"""structlog 配置的单元测试"""
from __future__ import annotations

import json


from shared.utils.logging import bind_context, clear_context, get_logger, setup_logging


def test_setup_logging_configures_structlog():
    """测试 setup_logging 配置 structlog"""
    setup_logging(log_level="DEBUG", service_name="test-svc", json_output=True, force=True)
    logger = get_logger("test")
    assert logger is not None


def test_json_output_format(capsys):
    """测试 JSON 输出格式"""
    setup_logging(log_level="DEBUG", service_name="test-svc", json_output=True, force=True)
    logger = get_logger()
    logger.info("test_event", key="value")
    captured = capsys.readouterr()
    parsed = json.loads(captured.out.strip())
    assert parsed["event"] == "test_event"
    assert parsed["key"] == "value"
    assert parsed["service"] == "test-svc"
    assert parsed["level"] == "info"
    assert "timestamp" in parsed


def test_context_binding(capsys):
    """测试上下文绑定"""
    setup_logging(log_level="DEBUG", json_output=True, force=True)
    bind_context(trace_id="trace-123", user_id="user-001")
    logger = get_logger()
    logger.info("with_context")
    captured = capsys.readouterr()
    parsed = json.loads(captured.out.strip())
    assert parsed["trace_id"] == "trace-123"
    assert parsed["user_id"] == "user-001"

    clear_context()
    logger.info("without_context")
    captured = capsys.readouterr()
    parsed = json.loads(captured.out.strip())
    assert "trace_id" not in parsed


def test_log_level_filtering(capsys):
    """测试日志级别过滤"""
    setup_logging(log_level="WARNING", json_output=True, force=True)
    logger = get_logger()
    logger.info("info_should_not_appear")
    logger.warning("warn_should_appear")
    captured = capsys.readouterr()
    assert "info_should_not_appear" not in captured.out
    assert "warn_should_appear" in captured.out
