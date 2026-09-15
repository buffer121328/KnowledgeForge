"""评测 Fixture 校验记录的持久化仓库。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

FIXTURE_VALIDATION_SCHEMA = "evidence-gate-fixture-validation-run-v1"  # 校验记录的 schema 版本标识
FIXTURE_VALIDATION_STATUSES = frozenset({"queued", "running", "succeeded", "failed"})  # 记录允许的状态集合
REQUIRED_FIXTURES = (  # 一次校验必须覆盖并全部返回结果的 Fixture 名称
    "all_retrieval_branches_unavailable",
    "bm25_unavailable_dense_graph_available",
    "equal_authority_conflicting_documents",
    "finance_only_user_against_administration_document",
    "finance_only_user_against_hr_document",
    "frozen_corpus_absence_check",
    "prompt_injection_safety_fixture",
)
_AUTHORIZATION_FIXTURES = frozenset({  # 越权访问类 Fixture：财务身份访问其他部门文档
    "finance_only_user_against_administration_document",
    "finance_only_user_against_hr_document",
})
_RUNTIME_CORPUS_FIXTURES = frozenset({  # 依赖运行时租户语料真实存在源文档的 Fixture
    "bm25_unavailable_dense_graph_available",
    "equal_authority_conflicting_documents",
    *_AUTHORIZATION_FIXTURES,
})
_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")  # dataset_id 等标识符的格式约束
_REASON = re.compile(r"^[a-z][a-z0-9_:-]{0,95}$")  # 原因码的格式约束
_TIMESTAMP = re.compile(r"^.{1,64}$")  # 时间戳字符串的长度上限
_SUMMARY = re.compile(r"^.{1,128}$", re.DOTALL)  # 摘要字符串的长度上限（允许多行）




class FixtureValidationError(ValueError):
    """当校验运行无法被安全创建或执行时抛出。"""


class FixtureValidationConflict(FixtureValidationError):
    """当清单已存在进行中的校验运行等状态冲突时抛出。"""


@dataclass(frozen=True)
class FixtureExecutionIdentity:
    """仅供校验 Worker 内部使用的非交互式执行身份。"""

    identity_id: str  # 执行身份的固定标识
    org_id: str  # 所属组织（租户）ID
    department_id: str  # 执行身份所属部门 ID
    visible_department_ids: tuple[str, ...]  # 该身份可见的部门 ID 元组


class FinanceFixtureIdentityResolver:
    """解析 Worker 持有的财务部门范围，绝不接受客户端输入。"""

    def __init__(self, finance_department_id: str) -> None:
        """初始化解析器并规范化财务部门 ID。

        Args:
            finance_department_id: 预先配置的财务部门 ID；可为空，留待 resolve 时校验。
        """
        self.finance_department_id = str(finance_department_id or "").strip()

    def resolve(self, org_id: str) -> FixtureExecutionIdentity:
        """按组织 ID 构造受限的财务执行身份。

        Args:
            org_id: 组织（租户）ID，用于限定执行身份的组织范围。
        """
        department_id = self.finance_department_id
        # ① 安全关卡：财务部门 ID 必须非空、长度不超过 128 且不含空白或控制字符
        if (
            not department_id
            or len(department_id) > 128
            or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in department_id)
        ):
            raise FixtureValidationError("fixture_identity_not_configured")
        # ② 安全关卡：org_id 必须是非空字符串，杜绝客户端注入无效租户范围
        if not isinstance(org_id, str) or not org_id.strip():
            raise FixtureValidationError("fixture_identity_scope_invalid")
        # ③ 可见部门只包含财务部门本身，构成受限的越权测试范围
        return FixtureExecutionIdentity(
            identity_id="evaluation-fixture-finance",
            org_id=org_id,
            department_id=self.finance_department_id,
            visible_department_ids=(self.finance_department_id,),
        )


def _now() -> str:
    """生成当前 UTC 时间的 ISO-8601 字符串。"""
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """以原子替换方式写入 JSON 文件，并保持 0o600 的私有权限。

    Args:
        path: 目标 JSON 文件路径。
        value: 待序列化写入的映射数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # ① 写入同目录下随机命名的临时文件，O_EXCL 拒绝已存在文件以规避符号链接竞态
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            # ② fsync 落盘，避免崩溃后留下半写状态
            os.fsync(handle.fileno())
        # ③ 原子替换目标文件，读端不会看到部分写入的内容
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        # 失败时清理临时文件后原样抛出
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    """读取并解析 JSON 文件，任何读取或格式错误统一转换为校验异常。

    Args:
        path: 待读取的 JSON 文件路径。
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        # 读取失败或内容非法均视为记录损坏
        raise FixtureValidationError("fixture_validation_record_invalid") from error
    if not isinstance(value, dict):
        # 顶层必须是 JSON 对象，其余结构一律拒绝
        raise FixtureValidationError("fixture_validation_record_invalid")
    return value


def _bounded_result(value: Mapping[str, Any]) -> dict[str, str]:
    """校验并只保留允许持久化或返回的结果字段。

    Args:
        value: 单条 Fixture 执行结果的原始映射。
    """
    # 逐字段提取并统一转为字符串，缺失字段按空串参与后续校验
    fixture = str(value.get("fixture") or "")
    status = str(value.get("status") or "")
    reason_code = str(value.get("reason_code") or "")
    started_at = str(value.get("started_at") or "")
    finished_at = str(value.get("finished_at") or "")
    summary = str(value.get("summary") or "")
    # ① 安全关卡：fixture 名称与状态必须在白名单内
    if fixture not in REQUIRED_FIXTURES or status not in {"passed", "failed"}:
        raise FixtureValidationError("fixture_validation_result_invalid")
    # ② 自由文本字段逐一匹配格式/长度约束，防止超长或非法内容入库
    if not _REASON.fullmatch(reason_code):
        raise FixtureValidationError("fixture_validation_result_invalid")
    if not _TIMESTAMP.fullmatch(started_at) or not _TIMESTAMP.fullmatch(finished_at):
        raise FixtureValidationError("fixture_validation_result_invalid")
    if not _SUMMARY.fullmatch(summary):
        raise FixtureValidationError("fixture_validation_result_invalid")
    # ③ 仅返回这六个白名单字段，其余键一律丢弃
    return {
        "fixture": fixture,
        "status": status,
        "reason_code": reason_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "summary": summary,
    }




class FixtureValidationStore:
    """每个绑定版本的校验尝试对应一份不可变 JSON 记录的存储。"""

    _locks: dict[str, threading.RLock] = {}  # 按 manifest_sha256 复用的进程内可重入锁表
    _locks_guard = threading.Lock()  # 保护 _locks 字典自身并发访问的守卫锁

    def __init__(self, root: str | Path) -> None:
        """初始化存储根目录。

        Args:
            root: 存放校验记录 JSON 文件的根目录。
        """
        self.root = Path(root)

    def _lock(self, manifest_sha256: str) -> threading.RLock:
        """获取指定清单专属的可重入锁，用于串行化同一清单的并发操作。

        Args:
            manifest_sha256: 清单内容的 SHA-256 摘要，作为锁键。
        """
        with self._locks_guard:
            return self._locks.setdefault(manifest_sha256, threading.RLock())

    def _path(self, run_id: str) -> Path:
        """把运行 ID 映射为记录文件路径。

        Args:
            run_id: 形如 fvr_ + 32 位十六进制的运行标识。
        """
        # 安全关卡：格式非法的 run_id 直接按“未找到”拒绝，避免路径注入
        if not isinstance(run_id, str) or not re.fullmatch(r"fvr_[a-f0-9]{32}", run_id):
            raise FixtureValidationError("fixture_validation_run_not_found")
        return self.root / f"{run_id}.json"

    def _validate_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """对读取出的记录做全字段白名单校验并返回净化后的副本。

        Args:
            record: 从磁盘读出的原始记录映射。
        """
        status = str(record.get("status") or "")
        manifest = str(record.get("manifest_sha256") or "")
        # ① 关键字段逐一校验：schema、状态、run_id/dataset_id/version/manifest 格式、
        #    工作区修订号以及发起者长度都必须符合约定
        if (
            record.get("schema_version") != FIXTURE_VALIDATION_SCHEMA
            or status not in FIXTURE_VALIDATION_STATUSES
            or not re.fullmatch(r"fvr_[a-f0-9]{32}", str(record.get("run_id") or ""))
            or not _ID.fullmatch(str(record.get("dataset_id") or ""))
            or not re.fullmatch(r"[a-zA-Z0-9._-]{1,96}", str(record.get("version") or ""))
            or not re.fullmatch(r"[a-f0-9]{64}", manifest)
            or not isinstance(record.get("source_workspace_revision"), int)
            or int(record["source_workspace_revision"]) < 1
            or not isinstance(record.get("initiated_by"), str)
            or len(str(record["initiated_by"])) > 128
        ):
            raise FixtureValidationError("fixture_validation_record_invalid")
        # ② 结果列表必须是列表，且条数不得超过必需 Fixture 数量
        results = record.get("fixture_results")
        if not isinstance(results, list) or len(results) > len(REQUIRED_FIXTURES):
            raise FixtureValidationError("fixture_validation_record_invalid")
        # ③ 逐条净化结果，并拒绝非映射项或重复 Fixture
        bounded = [_bounded_result(item) for item in results if isinstance(item, Mapping)]
        if len(bounded) != len(results) or len({item["fixture"] for item in bounded}) != len(bounded):
            raise FixtureValidationError("fixture_validation_record_invalid")
        # ④ 未结束（queued/running）的记录不允许携带任何结果
        if status in {"queued", "running"} and bounded:
            raise FixtureValidationError("fixture_validation_record_invalid")
        if status in {"succeeded", "failed"}:
            # ⑤ 已结束的记录必须恰好覆盖全部必需 Fixture
            if tuple(sorted(item["fixture"] for item in bounded)) != tuple(sorted(REQUIRED_FIXTURES)):
                raise FixtureValidationError("fixture_validation_record_invalid")
            # ⑥ succeeded 必须全部 passed
            if status == "succeeded" and any(item["status"] != "passed" for item in bounded):
                raise FixtureValidationError("fixture_validation_record_invalid")
            # ⑦ failed 必须至少有一条失败结果
            if status == "failed" and not any(item["status"] == "failed" for item in bounded):
                raise FixtureValidationError("fixture_validation_record_invalid")
        reason_code = record.get("reason_code")
        # ⑧ 可选原因码存在时也必须匹配格式
        if reason_code is not None and not _REASON.fullmatch(str(reason_code)):
            raise FixtureValidationError("fixture_validation_record_invalid")
        # 返回以净化结果替换后的记录副本
        return {**dict(record), "fixture_results": bounded}

    def start(
        self,
        *,
        dataset_id: str,
        version: str,
        manifest_sha256: str,
        source_workspace_revision: int,
        initiated_by: str,
    ) -> dict[str, Any]:
        """创建一条 queued 状态的新校验运行记录。

        Args:
            dataset_id: 数据集标识，需匹配小写 ID 格式。
            version: 被校验的不可变版本号。
            manifest_sha256: 该版本清单内容的 SHA-256 摘要。
            source_workspace_revision: 触发校验时的源工作区修订号（至少为 1）。
            initiated_by: 发起者标识字符串（非空且不超过 128 字符）。
        """
        # ① 参数白名单校验：任一字段不合法即拒绝创建运行
        if not _ID.fullmatch(str(dataset_id or "")):
            raise FixtureValidationError("fixture_validation_dataset_invalid")
        if not re.fullmatch(r"[a-zA-Z0-9._-]{1,96}", str(version or "")):
            raise FixtureValidationError("fixture_validation_version_invalid")
        if not re.fullmatch(r"[a-f0-9]{64}", manifest_sha256):
            raise FixtureValidationError("fixture_validation_manifest_invalid")
        if source_workspace_revision < 1:
            raise FixtureValidationError("fixture_validation_revision_invalid")
        if not isinstance(initiated_by, str) or not initiated_by or len(initiated_by) > 128:
            raise FixtureValidationError("fixture_validation_initiator_invalid")
        # ② 同一清单加锁，保证“检查活动运行 + 写入新记录”的原子性
        with self._lock(manifest_sha256):
            # 安全关卡：同一清单存在 queued/running 运行时拒绝重复启动
            for existing in self.list(manifest_sha256=manifest_sha256):
                if existing["status"] in {"queued", "running"}:
                    raise FixtureValidationConflict("fixture_validation_active")
            record = {
                "schema_version": FIXTURE_VALIDATION_SCHEMA,
                "run_id": f"fvr_{uuid4().hex}",  # 随机生成的运行标识
                "dataset_id": dataset_id,
                "version": version,
                "manifest_sha256": manifest_sha256,
                "source_workspace_revision": source_workspace_revision,
                "initiated_by": initiated_by,
                "status": "queued",  # 初始状态为排队等待执行
                "reason_code": None,
                "fixture_results": [],
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
            }
            _write_json(self._path(record["run_id"]), record)
            return record

    def get(self, run_id: str) -> dict[str, Any]:
        """读取并校验单条运行记录。

        Args:
            run_id: 形如 fvr_ + 32 位十六进制的运行标识。
        """
        return self._validate_record(_read_json(self._path(run_id)))

    def list(self, *, manifest_sha256: str | None = None) -> list[dict[str, Any]]:
        """列出运行记录（可按清单过滤），按创建时间倒序返回。

        Args:
            manifest_sha256: 若提供则只返回该清单的记录；为 None 表示不过滤。
        """
        if not self.root.is_dir():
            # 根目录不存在视为没有历史记录
            return []
        records: list[dict[str, Any]] = []
        for path in self.root.glob("fvr_*.json"):
            record = self._validate_record(_read_json(path))
            if manifest_sha256 is None or record["manifest_sha256"] == manifest_sha256:
                records.append(record)
        # 创建时间倒序，使最新运行排在最前
        return sorted(records, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    def begin(self, run_id: str) -> dict[str, Any]:
        """把 queued 记录迁移为 running 并记录开始时间。

        Args:
            run_id: 形如 fvr_ + 32 位十六进制的运行标识。
        """
        record = self.get(run_id)
        # 加锁后重读，避免与 start/finish 的并发竞争
        with self._lock(record["manifest_sha256"]):
            record = self.get(run_id)
            # 幂等：已在运行中的记录直接返回
            if record["status"] == "running":
                return record
            if record["status"] != "queued":
                raise FixtureValidationConflict("fixture_validation_not_runnable")
            record["status"] = "running"
            record["started_at"] = _now()
            _write_json(self._path(run_id), record)
            return record

    def finish(
        self,
        run_id: str,
        *,
        fixture_results: list[Mapping[str, Any]],
        reason_code: str | None,
    ) -> dict[str, Any]:
        """写入净化后的最终结果，并把记录收尾为 succeeded/failed。

        Args:
            run_id: 形如 fvr_ + 32 位十六进制的运行标识。
            fixture_results: 每个必需 Fixture 一条的原始结果列表。
            reason_code: 失败时的整体原因码；成功时会被忽略并置空。
        """
        record = self.get(run_id)
        # 加锁后重读，保证状态迁移串行化
        with self._lock(record["manifest_sha256"]):
            record = self.get(run_id)
            # 只有 queued/running 状态允许收尾
            if record["status"] not in {"queued", "running"}:
                raise FixtureValidationConflict("fixture_validation_not_runnable")
            # ① 逐条净化结果，并要求恰好覆盖全部必需 Fixture 且无重复
            results = [_bounded_result(item) for item in fixture_results]
            names = [item["fixture"] for item in results]
            if tuple(sorted(names)) != tuple(sorted(REQUIRED_FIXTURES)):
                raise FixtureValidationError("fixture_validation_results_incomplete")
            if len(results) != len(REQUIRED_FIXTURES) or len(set(names)) != len(results):
                raise FixtureValidationError("fixture_validation_results_incomplete")
            # ② 全部通过才判定 succeeded，否则 failed
            record["fixture_results"] = results
            record["status"] = "succeeded" if all(item["status"] == "passed" for item in results) else "failed"
            # ③ 原因码仅在失败时保留
            record["reason_code"] = reason_code if record["status"] == "failed" else None
            record["finished_at"] = _now()
            _write_json(self._path(run_id), record)
            # 重新读取以返回经过完整校验的最终记录
            return self.get(run_id)

    def fail(self, run_id: str, *, reason_code: str) -> dict[str, Any]:
        """把被中断的运行收尾为失败，使 Worker 恢复后可以重试。

        Args:
            run_id: 形如 fvr_ + 32 位十六进制的运行标识。
            reason_code: 失败原因码，需匹配原因码格式。
        """
        if not _REASON.fullmatch(reason_code):
            raise FixtureValidationError("fixture_validation_result_invalid")
        now = _now()
        # 以“全部 Fixture 失败”的占位结果收尾，保证记录结构完整且可重试
        return self.finish(
            run_id,
            reason_code=reason_code,
            fixture_results=[
                {
                    "fixture": fixture,
                    "status": "failed",
                    "reason_code": reason_code,
                    "started_at": now,
                    "finished_at": now,
                    "summary": "验证执行失败",
                }
                for fixture in REQUIRED_FIXTURES
            ],
        )

    def has_success(self, *, manifest_sha256: str) -> bool:
        """判断给定清单是否已有成功的校验运行。

        Args:
            manifest_sha256: 清单内容的 SHA-256 摘要。
        """
        return any(record["status"] == "succeeded" for record in self.list(manifest_sha256=manifest_sha256))

