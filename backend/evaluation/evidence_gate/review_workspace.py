"""证据数据集的受控编写、maker-checker 双人复核与冻结（门面模块）。

实现按领域拆分为三个 mixin：
- ``CurrentCorpusBindingMixin``：现行语料装订与代表性模板导入；
- ``CaseLifecycleMixin``：案例校验、CRUD、评审流转与批量操作；
- ``VersionFreezeMixin``：版本读取、数据集摘要与冻结产物。
常量、异常与 IO 原语在 ``workspace_io``。公开类与方法面保持不变。
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.evidence_gate.case_lifecycle import CaseLifecycleMixin
from evaluation.evidence_gate.current_corpus_binding import CurrentCorpusBindingMixin
from evaluation.evidence_gate.version_freeze import VersionFreezeMixin
from evaluation.evidence_gate.workspace_io import (  # noqa: F401
    _AUTHORIZATION_FIXTURES,
    _AUTHORING_FIELDS,
    _CASE_ID,
    _CASE_STATUSES,
    _CURRENT_CORPUS_TEMPLATE_IMPORT_LIMIT,
    _CURRENT_CORPUS_TEMPLATE_STANDARD_CATEGORIES,
    _DECISIONS,
    _ID,
    _ORG_NAMESPACE,
    _REFERENCE_ANSWER_CONTRACT_FIELDS,
    _REFERENCE_ANSWER_CANDIDATE_EDITOR,
    _REFERENCE_ANSWER_CANDIDATE_SCHEMA,
    _REFERENCE_ANSWER_CATEGORIES,
    _REQUIRED_FIXTURES,
    _SOURCE_EVALUATION_DATASETS,
    _SOURCE_TITLE_PREFIX,
    _TENANT_ID,
    EvidenceReviewConflict,
    EvidenceReviewError,
    EvidenceReviewNotFound,
    EvidenceReviewPermissionError,
    EvidenceReviewValidationError,
    _read_json,
    _read_jsonl,
    _string_list,
    _utc_now,
    _write_json,
    _write_jsonl,
    sha256_file,
)

__all__ = [
    "EvidenceReviewWorkspace",
    "EvidenceReviewError",
    "EvidenceReviewNotFound",
    "EvidenceReviewConflict",
    "EvidenceReviewPermissionError",
    "EvidenceReviewValidationError",
]


class EvidenceReviewWorkspace(
    CurrentCorpusBindingMixin,
    CaseLifecycleMixin,
    VersionFreezeMixin,
):
    """把单个组织隔离的编写工作区以本地原子文件形式持久化。"""

    # 对外暴露的必需 Fixture 名称集合（与模块级常量一致）
    required_fixtures = _REQUIRED_FIXTURES

    def __init__(self, root: str | Path, source_bundle: str | Path) -> None:
        """初始化工作区根目录与旧版源数据包路径。

        Args:
            root: 存放所有组织工作区的根目录。
            source_bundle: 旧版演示数据包目录（manifest/cases/context/fixture 所在处）。
        """
        self.root = Path(root)
        self.source_bundle = Path(source_bundle)
        self.source_bundles = {
            "evidence-gates-v1": self.source_bundle,
            "evidence-gates-v1-routine": self.source_bundle / "evidence-gates-v1-routine",
        }
        # 该目录只存放由冻结文档整理出的候选答案，不是已批准的评测版本。
        # 在自定义/旧版测试数据包中它可以不存在，届时迁移操作会安全地不写入。
        self.reference_answer_candidates_path = (
            self.source_bundle.parent / "company-demo" / "reference-answer-candidates.json"
        )
        self.company_demo_manifest_path = (
            self.source_bundle.parent / "company-demo" / "corpus_manifest.json"
        )
        # 按 "org:dataset" 粒度复用的可重入锁表；_locks_guard 保护锁表自身的并发创建
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _source_bundle(self, dataset_id: str) -> Path:
        """Return the checked-in source bundle for a supported evaluation suite."""
        return self.source_bundles.get(dataset_id, self.source_bundle)

    def _lock(self, org_id: str, dataset_id: str) -> threading.RLock:
        """获取指定组织与数据集的可重入锁，不存在则创建。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
        """
        key = f"{org_id}:{dataset_id}"
        with self._locks_guard:
            return self._locks.setdefault(key, threading.RLock())

    @staticmethod
    def _validate_dataset_id(dataset_id: str) -> str:
        """校验数据集 ID 格式，非法时统一按“未找到”拒绝（避免探测）。

        Args:
            dataset_id: 待校验的数据集 ID。
        """
        if not isinstance(dataset_id, str) or not _ID.fullmatch(dataset_id):
            raise EvidenceReviewNotFound("dataset_not_found")
        return dataset_id

    @staticmethod
    def _org_namespace(org_id: str) -> str:
        """把组织 ID 哈希成 24 位十六进制目录名，避免磁盘暴露原始 ID。

        Args:
            org_id: 组织 ID，必须是非空字符串。
        """
        if not isinstance(org_id, str) or not org_id.strip():
            raise EvidenceReviewNotFound("organization_not_found")
        return hashlib.sha256(org_id.encode("utf-8")).hexdigest()[:24]

    def workspace_path(self, org_id: str, dataset_id: str) -> Path:
        """返回组织命名空间下的工作区目录。

        Args:
            org_id: 组织 ID。
            dataset_id: 已通过格式校验的数据集 ID。
        """
        dataset = self._validate_dataset_id(dataset_id)
        return self.root / self._org_namespace(org_id) / dataset

    @staticmethod
    def _workspace_paths(root: Path) -> dict[str, Path]:
        """给出工作区内各类文件的固定相对布局。

        Args:
            root: 工作区根目录。
        """
        # 固定布局：元数据、用例、上下文目录、Fixture 档案、评审事件与冻结版本目录
        return {
            "root": root,
            "workspace": root / "workspace.json",
            "cases": root / "cases.jsonl",
            "contexts": root / "context-catalog.jsonl",
            "fixtures": root / "fixture-profile.json",
            "events": root / "review-events.jsonl",
            "versions": root / "versions",
        }

    def _paths(self, org_id: str, dataset_id: str) -> dict[str, Path]:
        """解析组织与数据集对应的工作区文件路径表。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
        """
        return self._workspace_paths(self.workspace_path(org_id, dataset_id))

    @staticmethod
    def _system_import_cases(
        source_cases: list[dict[str, Any]],
        *,
        imported_at: str,
    ) -> list[dict[str, Any]]:
        """把源数据包用例重置为“系统导入”的初始草稿状态。

        Args:
            source_cases: 源数据包中的原始用例列表。
            imported_at: 导入时间戳（ISO 字符串）。
        """
        cases: list[dict[str, Any]] = []
        for source in source_cases:
            case = dict(source)
            # 安全关卡：源用例必须自带部门归属，否则拒绝导入
            if not str(case.get("department_id") or "").strip():
                raise EvidenceReviewValidationError("case_department_required")
            case.update(
                {
                    "review_status": "draft",
                    "case_revision": 1,
                    "last_editor_id": "system-import",
                    "last_edited_at": imported_at,
                    "submitted_at": None,
                    "reviewer_id": None,
                    "reviewed_at": None,
                    "review_reason": "",
                    "rejection_reason": "",
                }
            )
            cases.append(case)
        return cases

    @staticmethod
    def _is_pristine_import(
        workspace: Mapping[str, Any],
        cases: list[dict[str, Any]],
        fixtures: Mapping[str, Any],
        events: list[dict[str, Any]],
    ) -> bool:
        """判断工作区是否仍是未被人工改动过的原始系统导入状态。

        Args:
            workspace: 工作区元数据。
            cases: 当前用例列表。
            fixtures: Fixture 档案数据。
            events: 评审事件列表。
        """
        fixture_values = fixtures.get("fixtures")
        if not isinstance(fixture_values, dict):
            return False
        # 判定标准：authoring、从未冻结、无事件、Fixture 待配置且未经人工核验/租户配置，
        # 并且全部用例仍是 revision 为 1 的 system-import 草稿
        return bool(cases) and all(
            (
                workspace.get("status") == "authoring",
                not workspace.get("last_frozen_version"),
                not events,
                fixtures.get("status") == "pending_operator_configuration",
                all(
                    isinstance(item, dict)
                    and not item.get("verified")
                    and not str(item.get("tenant_id") or "").strip()
                    for item in fixture_values.values()
                ),
                all(
                    case.get("review_status") == "draft"
                    and case.get("case_revision") == 1
                    and case.get("last_editor_id") == "system-import"
                    for case in cases
                ),
            )
        )

    def _refresh_pristine_import(
        self,
        paths: dict[str, Path],
        workspace: dict[str, Any],
        *,
        source_manifest_sha256: str,
    ) -> None:
        """仅当工作区仍处于原始导入状态时，用最新源数据包整体刷新。

        Args:
            paths: 工作区文件路径表。
            workspace: 工作区元数据（就地更新）。
            source_manifest_sha256: 最新源数据包 manifest 的 SHA-256。
        """
        # ① manifest 未变化则无需刷新
        if workspace.get("source_manifest_sha256") == source_manifest_sha256:
            return
        # ② 四类工作区文件必须齐全，否则保持现状
        if not all(
            paths[name].is_file()
            for name in ("cases", "contexts", "fixtures", "events")
        ):
            return
        # ③ 仅未被人工改动过的原始导入才允许整体覆盖刷新
        cases = _read_jsonl(paths["cases"])
        fixtures = _read_json(paths["fixtures"])
        events = _read_jsonl(paths["events"])
        if not self._is_pristine_import(workspace, cases, fixtures, events):
            return

        # ④ 从源数据包重建导入内容并递增工作区版本
        source_bundle = self._source_bundle(paths["root"].name)
        source_cases = _read_jsonl(source_bundle / "cases.jsonl")
        source_contexts = _read_jsonl(source_bundle / "context-catalog.jsonl")
        source_fixtures = _read_json(
            source_bundle / "fixture-profile.example.json"
        )
        now = _utc_now()
        imported_cases = self._system_import_cases(source_cases, imported_at=now)
        workspace["revision"] = int(workspace.get("revision") or 0) + 1
        workspace["updated_at"] = now
        workspace["source_manifest_sha256"] = source_manifest_sha256

        # 最后写入工作区标记：刷新中断时旧源哈希仍在，可安全重试。
        _write_jsonl(paths["contexts"], source_contexts)
        _write_json(paths["fixtures"], source_fixtures)
        _write_jsonl(paths["cases"], imported_cases)
        _write_json(paths["workspace"], workspace)

    def _merge_governed_source_additions(
        self,
        paths: dict[str, Path],
        workspace: dict[str, Any],
        *,
        source_manifest_sha256: str,
    ) -> None:
        """在保留既有复核治理状态的前提下，向工作区追加源数据包新增用例。

        Args:
            paths: 工作区文件路径表。
            workspace: 工作区元数据（就地更新）。
            source_manifest_sha256: 最新源数据包 manifest 的 SHA-256。
        """
        # ① 已同步到最新哈希或已冻结的工作区不做增量合并
        if (
            workspace.get("source_manifest_sha256") == source_manifest_sha256
            or workspace.get("last_frozen_version")
        ):
            return
        # ② 四类工作区文件必须齐全，否则保持现状
        if not all(
            paths[name].is_file()
            for name in ("cases", "contexts", "fixtures", "events")
        ):
            return

        # ③ 安全前提：既有用例必须是源用例的子集（不允许本地删除后被源回填）
        cases = _read_jsonl(paths["cases"])
        source_bundle = self._source_bundle(paths["root"].name)
        source_cases = _read_jsonl(source_bundle / "cases.jsonl")
        existing_ids = {str(case.get("id") or "") for case in cases}
        source_ids = {str(case.get("id") or "") for case in source_cases}
        if not existing_ids or not existing_ids.issubset(source_ids):
            return

        # ④ 被治理用例引用的全部上下文都必须存在于源目录中
        source_contexts = _read_jsonl(source_bundle / "context-catalog.jsonl")
        source_context_ids = {
            str(context.get("context_id") or "") for context in source_contexts
        }
        governed_context_ids = {
            str(context_id)
            for case in cases
            for field in (
                "expected_evidence_context_ids",
                "expected_citation_context_ids",
            )
            for context_id in case.get(field) or []
        }
        if not governed_context_ids.issubset(source_context_ids):
            return

        # ⑤ 新增用例按“系统导入草稿”追加，并只为缺失的 Fixture 补上源侧默认绑定
        additions = [
            case for case in source_cases if str(case.get("id") or "") not in existing_ids
        ]
        now = _utc_now()
        imported_additions = self._system_import_cases(additions, imported_at=now)
        source_fixtures = _read_json(
            source_bundle / "fixture-profile.example.json"
        )
        fixtures = _read_json(paths["fixtures"])
        source_fixture_values = source_fixtures.get("fixtures")
        fixture_values = fixtures.get("fixtures")
        if isinstance(source_fixture_values, dict) and isinstance(fixture_values, dict):
            for fixture_id, fixture in source_fixture_values.items():
                fixture_values.setdefault(fixture_id, fixture)

        workspace["revision"] = int(workspace.get("revision") or 0) + 1
        workspace["updated_at"] = now
        workspace["source_manifest_sha256"] = source_manifest_sha256

        # 保留受治理的用例/事件状态：先发布源方拥有的支撑数据、最后写源标记，
        # 使中断的合并仍可重试。
        _write_jsonl(paths["contexts"], source_contexts)
        _write_json(paths["fixtures"], fixtures)
        _write_jsonl(paths["cases"], [*cases, *imported_additions])
        _write_json(paths["workspace"], workspace)

    def _refresh_source_import(
        self,
        paths: dict[str, Path],
        workspace: dict[str, Any],
        *,
        source_manifest_sha256: str,
    ) -> None:
        """按“原始导入刷新 → 治理态增量合并”两级策略同步源数据包。

        Args:
            paths: 工作区文件路径表。
            workspace: 工作区元数据（就地更新）。
            source_manifest_sha256: 最新源数据包 manifest 的 SHA-256。
        """
        self._refresh_pristine_import(
            paths,
            workspace,
            source_manifest_sha256=source_manifest_sha256,
        )
        if workspace.get("source_manifest_sha256") != source_manifest_sha256:
            self._merge_governed_source_additions(
                paths,
                workspace,
                source_manifest_sha256=source_manifest_sha256,
            )

    def _backfill_case_departments(self, paths: dict[str, Path]) -> int:
        """仅回填缺失的源方部门归属，不覆盖既有治理字段。

        Args:
            paths: 工作区文件路径表。
        """
        if not paths["workspace"].is_file() or not paths["cases"].is_file():
            return 0
        source_departments = {
            str(case.get("id")): str(case.get("department_id") or "")
            for case in _read_jsonl(
                self._source_bundle(paths["root"].name) / "cases.jsonl"
            )
        }
        cases = _read_jsonl(paths["cases"])
        changed = 0
        for case in cases:
            # 已有部门或源包中找不到对应部门的用例一律不动
            if str(case.get("department_id") or "").strip():
                continue
            department_id = source_departments.get(str(case.get("id") or ""), "")
            if not department_id:
                continue
            case["department_id"] = department_id
            # 回填后清除复核部门，避免遗留旧部门的复核归属
            case.pop("reviewer_department_id", None)
            changed += 1
        if changed:
            workspace = _read_json(paths["workspace"])
            self._bump(workspace)
            _write_jsonl(paths["cases"], cases)
            _write_json(paths["workspace"], workspace)
        return changed

    def refresh_pristine_workspaces(self) -> int:
        """刷新全部原始导入工作区并向治理态工作区追加源新增，返回刷新个数。"""
        if not self.root.is_dir():
            return 0
        # ① 计算最新源 manifest 哈希，作为所有工作区的同步基准
        source_manifest_sha256 = sha256_file(self.source_bundle / "manifest.json")
        refreshed = 0
        # ② 只遍历组织命名空间目录下的 evidence-gates-v1 工作区，跳过非法目录名
        for workspace_file in self.root.glob("*/evidence-gates-v1/workspace.json"):
            namespace = workspace_file.parents[1].name
            if not _ORG_NAMESPACE.fullmatch(namespace):
                continue
            paths = self._workspace_paths(workspace_file.parent)
            with self._lock(namespace, "evidence-gates-v1"):
                workspace = _read_json(workspace_file)
                previous_hash = workspace.get("source_manifest_sha256")
                # ③ 同步源导入并回填部门；哈希从旧变新才计入一次刷新
                self._refresh_source_import(
                    paths,
                    workspace,
                    source_manifest_sha256=source_manifest_sha256,
                )
                self._backfill_case_departments(paths)
                if (
                    previous_hash != source_manifest_sha256
                    and _read_json(workspace_file).get("source_manifest_sha256")
                    == source_manifest_sha256
                ):
                    refreshed += 1
        return refreshed

    def _ensure(self, org_id: str, dataset_id: str) -> dict[str, Path]:
        """解析已存在的工作区；仅对旧版数据集惰性导入源数据包。

        当前语料工作区只能由 create_current_corpus_workspace 创建；
        未知数据集 ID 绝不触发演示导入，也不会隐式创建数据集。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
        """

        paths = self._paths(org_id, dataset_id)
        # ① 已存在：校验组织归属，防止跨组织探测其它组织的数据集
        if paths["workspace"].is_file():
            workspace = _read_json(paths["workspace"])
            if workspace.get("org_id_hash") != self._org_namespace(org_id):
                raise EvidenceReviewNotFound("dataset_not_found")
            # ② 当前语料工作区：来源必须正确且四类文件齐全，否则视为不存在
            if dataset_id not in _SOURCE_EVALUATION_DATASETS:
                if workspace.get("source_type") != "current_corpus":
                    raise EvidenceReviewNotFound("dataset_not_found")
                if not all(paths[name].is_file() for name in ("cases", "contexts", "fixtures", "events")):
                    raise EvidenceReviewNotFound("dataset_not_found")
                return paths

            source_manifest_sha256 = sha256_file(
                self._source_bundle(dataset_id) / "manifest.json"
            )
            # ③ 旧版数据集：manifest 未变化时只做部门回填
            if workspace.get("source_manifest_sha256") == source_manifest_sha256:
                self._backfill_case_departments(paths)
                return paths
            # ④ manifest 已更新：加锁执行两级源同步后再回填部门
            with self._lock(org_id, dataset_id):
                self._refresh_source_import(
                    paths,
                    _read_json(paths["workspace"]),
                    source_manifest_sha256=source_manifest_sha256,
                )
                self._backfill_case_departments(paths)
                return paths

        # ⑤ 首次访问：只有 checked-in evaluation suites 允许惰性导入。
        if dataset_id not in _SOURCE_EVALUATION_DATASETS:
            raise EvidenceReviewNotFound("dataset_not_found")
        source_manifest_sha256 = sha256_file(
            self._source_bundle(dataset_id) / "manifest.json"
        )
        with self._lock(org_id, dataset_id):
            # 双重检查：并发下可能已被其它线程创建
            if paths["workspace"].is_file():
                return self._ensure(org_id, dataset_id)
            source_bundle = self._source_bundle(dataset_id)
            source_cases = _read_jsonl(source_bundle / "cases.jsonl")
            contexts = _read_jsonl(source_bundle / "context-catalog.jsonl")
            fixture_path = source_bundle / "fixture-profile.example.json"
            fixtures = _read_json(fixture_path)
            now = _utc_now()
            cases = self._system_import_cases(source_cases, imported_at=now)
            workspace = {
                "schema_version": "evidence-review-workspace-v1",
                "dataset_id": dataset_id,
                "title": (
                    "证据门日常基准 50 条"
                    if dataset_id == "evidence-gates-v1-routine"
                    else "证据门候选集 v1"
                ),
                "org_id_hash": self._org_namespace(org_id),
                "revision": 1,
                "status": "authoring",
                "created_at": now,
                "updated_at": now,
                "last_frozen_version": None,
                "last_frozen_manifest_sha256": None,
                "last_frozen_at": None,
                "last_frozen_by": None,
                "source_frozen_version": None,
                "source_manifest_sha256": source_manifest_sha256,
            }
            _write_jsonl(paths["cases"], cases)
            _write_jsonl(paths["contexts"], contexts)
            _write_json(paths["fixtures"], fixtures)
            _write_json(paths["workspace"], workspace)
            _write_jsonl(paths["events"], [])
        return paths
