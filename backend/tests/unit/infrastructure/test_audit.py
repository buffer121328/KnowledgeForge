"""审计日志服务的单元测试"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from infrastructure.audit.log import (
    AuditAction,
    AuditResult,
    AuditService,
    FileAuditStore,
    MemoryAuditStore,
)


@pytest.fixture
def memory_service() -> AuditService:
    return AuditService(store=MemoryAuditStore())


@pytest.fixture
def file_service() -> tuple[AuditService, str]:
    """文件存储服务"""
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        path = f.name
    os.unlink(path)  # 让服务自己创建
    yield AuditService(store=FileAuditStore(path)), path
    if os.path.exists(path):
        os.unlink(path)


class TestAuditAction:
    def test_audit_action_values(self):
        """测试 AuditAction 枚举值"""
        assert AuditAction.LOGIN.value == "auth.login"
        assert AuditAction.DOC_UPLOAD.value == "doc.upload"
        assert AuditAction.QA_QUERY.value == "qa.query"

    def test_audit_result_values(self):
        """测试 AuditResult 枚举值"""
        assert AuditResult.SUCCESS.value == "success"
        assert AuditResult.FAILURE.value == "failure"
        assert AuditResult.DENIED.value == "denied"


class TestMemoryAuditStore:
    def test_log_basic(self, memory_service: AuditService):
        """测试基础日志记录"""
        log = memory_service.log(
            user_id="user_001",
            action=AuditAction.LOGIN,
            ip="192.168.1.1",
            username="admin",
        )
        assert log.user_id == "user_001"
        assert log.action == "auth.login"
        assert log.result == "success"
        assert log.audit_id.startswith("aud_")
        assert log.timestamp != ""

    def test_log_with_metadata(self, memory_service: AuditService):
        """测试带 metadata 的日志"""
        log = memory_service.log(
            user_id="user_001",
            action=AuditAction.DOC_UPLOAD,
            resource="doc/report.pdf",
            metadata={"file_size": 1024, "chunks": 10},
        )
        assert log.metadata["file_size"] == 1024
        assert log.metadata["chunks"] == 10
        assert log.resource == "doc/report.pdf"

    def test_log_failure_result(self, memory_service: AuditService):
        """测试失败结果"""
        log = memory_service.log(
            user_id="user_001",
            action=AuditAction.LOGIN,
            result=AuditResult.FAILURE,
        )
        assert log.result == "failure"

    def test_query_by_user(self, memory_service: AuditService):
        """测试按用户查询"""
        memory_service.log(user_id="user_001", action=AuditAction.LOGIN)
        memory_service.log(user_id="user_002", action=AuditAction.LOGIN)
        memory_service.log(user_id="user_001", action=AuditAction.LOGOUT)

        results = memory_service.query(user_id="user_001")
        assert len(results) == 2
        assert all(r.user_id == "user_001" for r in results)

    def test_query_by_action(self, memory_service: AuditService):
        """测试按操作类型查询"""
        memory_service.log(user_id="user_001", action=AuditAction.LOGIN)
        memory_service.log(user_id="user_001", action=AuditAction.DOC_UPLOAD)

        results = memory_service.query(action="auth.login")
        assert len(results) == 1
        assert results[0].action == "auth.login"

    def test_query_limit(self, memory_service: AuditService):
        """测试查询限制"""
        for i in range(10):
            memory_service.log(user_id="user_001", action=AuditAction.LOGIN)
        results = memory_service.query(limit=5)
        assert len(results) == 5


class TestFileAuditStore:
    def test_file_log_write_and_read(self, file_service):
        """测试文件存储写入和读取"""
        service, path = file_service
        service.log(
            user_id="user_001",
            action=AuditAction.LOGIN,
            username="admin",
        )
        results = service.query(user_id="user_001")
        assert len(results) == 1
        assert results[0].user_id == "user_001"

    def test_file_log_hash_chain(self, file_service):
        """测试哈希链: 前后日志通过 hash 关联"""
        service, path = file_service
        service.log(user_id="user_001", action=AuditAction.LOGIN)
        service.log(user_id="user_001", action=AuditAction.DOC_UPLOAD)

        results = service.query(limit=10)
        assert len(results) == 2
        # 后一条的 prev_hash 应等于前一条的 hash
        # query 返回顺序为逆序，所以 results[1] 是较早的
        assert results[0].prev_hash == results[1].hash
        assert results[1].prev_hash == ""

    def test_reconstructs_existing_record_without_service_import(
        self,
        tmp_path: Path,
    ) -> None:
        """A pre-extraction JSON record uses the neutral type and tenant filter."""

        from domain.audit import AuditLog as ContractAuditLog

        payload = {
            "audit_id": "aud_existing",
            "timestamp": "2026-08-12T00:00:00+00:00",
            "user_id": "user_existing",
            "action": "auth.login",
            "username": "existing-user",
            "resource": "",
            "result": "success",
            "ip": "",
            "user_agent": "",
            "org_id": "org-existing",
            "metadata": {"safe_code": "existing"},
            "prev_hash": "",
            "hash": "",
        }
        canonical = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
        payload["hash"] = hashlib.sha256(canonical).hexdigest()
        path = tmp_path / "existing-audit.log"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        store = FileAuditStore(str(path))
        records = store.query(org_id="org-existing")

        assert len(records) == 1
        assert type(records[0]) is ContractAuditLog
        assert records[0].audit_id == "aud_existing"
        assert store.query(org_id="org-other") == []


def test_store_roundtrip_does_not_load_audit_service_module() -> None:
    """Importing and using a store does not pull in audit service policy."""

    backend_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(backend_root)
    program = """
import sys
from domain.audit import AuditLog
from infrastructure.audit.stores import MemoryAuditStore

store = MemoryAuditStore()
record = AuditLog(
    audit_id="aud_isolated",
    timestamp="2026-08-12T00:00:00+00:00",
    user_id="user",
)
store.append(record)
assert store.query() == [record]
assert "infrastructure.audit_log" not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=backend_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
