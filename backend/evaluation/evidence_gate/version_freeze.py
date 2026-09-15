"""版本读取、数据集摘要与冻结产物生成（mixin）。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import unicodedata
from datetime import UTC, datetime
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from evaluation.evidence_gate.benchmark import (
    EVIDENCE_GATE_MANIFEST_SCHEMA,
    EVIDENCE_GATE_REVIEW_SCHEMA,
    REFERENCE_ANSWER_SCHEMA_VERSION,
    REQUIRED_EVIDENCE_GATE_CATEGORIES,
    SUPPORTED_RESPONSE_STATUSES,
    load_evidence_gate_benchmark,
)
from evaluation.evidence_gate.workspace_io import (  # noqa: F401
    _CASE_ID,
    _CASE_STATUSES,
    _CURRENT_CORPUS_TEMPLATE_IMPORT_LIMIT,
    _CURRENT_CORPUS_TEMPLATE_STANDARD_CATEGORIES,
    _DECISIONS,
    _ID,
    _ORG_NAMESPACE,
    _REFERENCE_ANSWER_CONTRACT_FIELDS,
    _REFERENCE_ANSWER_CANDIDATE_SCHEMA,
    _REFERENCE_ANSWER_CANDIDATE_EDITOR,
    _REFERENCE_ANSWER_CATEGORIES,
    _REQUIRED_FIXTURES,
    _SOURCE_EVALUATION_DATASETS,
    _SOURCE_TITLE_PREFIX,
    _AUTHORIZATION_FIXTURES,
    _TENANT_ID,
    _AUTHORING_FIELDS,
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


class VersionFreezeMixin:
    def _load(self, org_id: str, dataset_id: str) -> tuple[dict[str, Path], dict[str, Any], list[dict[str, Any]]]:
        """加载工作区路径表、元数据与用例列表。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
        """
        paths = self._ensure(org_id, dataset_id)
        return paths, _read_json(paths["workspace"]), _read_jsonl(paths["cases"])

    @staticmethod
    def _check_revision(workspace: dict[str, Any], expected_revision: int) -> None:
        """乐观并发控制：期望版本与当前版本不一致时抛出冲突。

        Args:
            workspace: 工作区元数据。
            expected_revision: 调用方持有的期望 revision。
        """
        if expected_revision != workspace.get("revision"):
            # 错误信息携带服务端当前 revision，便于调用方感知并重试
            raise EvidenceReviewConflict(
                f"revision_conflict:{workspace.get('revision')}"
            )

    @staticmethod
    def _require_authoring(workspace: Mapping[str, Any]) -> None:
        """拒绝工作区冻结后的一切编写类变更。

        Args:
            workspace: 工作区元数据。
        """
        if workspace.get("status") != "authoring":
            raise EvidenceReviewConflict("dataset_frozen")

    @staticmethod
    def _safe_version_name(version: str) -> str:
        """校验版本名可安全用作单段目录名（防路径穿越）。

        Args:
            version: 冻结版本名。
        """
        if (
            not isinstance(version, str)
            or not version
            or len(version) > 128
            or Path(version).name != version
            or version in {".", ".."}
        ):
            raise EvidenceReviewNotFound("version_not_found")
        return version

    def _version_root(self, paths: Mapping[str, Path], version: str) -> Path:
        """返回冻结版本目录，并确保其位于 versions 根之下且含 manifest。

        Args:
            paths: 工作区文件路径表。
            version: 冻结版本名。
        """
        safe_version = self._safe_version_name(version)
        versions_root = paths["versions"].resolve()
        root = (versions_root / safe_version).resolve()
        # resolve 后必须仍是 versions 根的直接子目录，防止符号链接逃逸
        if root.parent != versions_root or not (root / "manifest.json").is_file():
            raise EvidenceReviewNotFound("version_not_found")
        return root

    def _version_summary(self, paths: Mapping[str, Path], version: str) -> dict[str, Any]:
        """汇总一个冻结版本的公开元数据（哈希、冻结人、快照标识等）。

        Args:
            paths: 工作区文件路径表。
            version: 冻结版本名。
        """
        root = self._version_root(paths, version)
        manifest = _read_json(root / "manifest.json")
        if manifest.get("dataset_version") != version:
            raise EvidenceReviewValidationError("invalid_frozen_version")
        summary = {
            "version": version,
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "frozen_at": manifest.get("frozen_at"),
            "frozen_by": manifest.get("frozen_by"),
            "source_workspace_revision": manifest.get("source_workspace_revision"),
            "current_corpus_draft_id": manifest.get("current_corpus_draft_id"),
            "current_corpus_snapshot_sha256": manifest.get("current_corpus_snapshot_sha256"),
        }
        if (
            paths["root"].name.startswith("current-corpus-")
            or "source_document_count" in manifest
            or manifest.get("current_corpus_snapshot_sha256")
        ):
            contexts = _read_jsonl(root / "context-catalog.jsonl")
            source_document_ids = {
                str(item.get("source_document_id") or "")
                for item in contexts
                if str(item.get("source_document_id") or "")
            }
            summary["source_document_count"] = int(
                manifest.get("source_document_count")
                or len(source_document_ids)
                or manifest.get("context_count")
                or len(contexts)
            )
        return summary

    def list_version_contexts(
        self,
        org_id: str,
        dataset_id: str,
        version: str,
    ) -> dict[str, Any]:
        """返回冻结版本的受限 Context/文档目录，不包含正文。"""

        paths, _workspace, _cases = self._load(org_id, dataset_id)
        summary = self._version_summary(paths, version)
        contexts = _read_jsonl(self._version_root(paths, version) / "context-catalog.jsonl")
        documents: dict[str, dict[str, Any]] = {}
        for item in contexts:
            source_document_id = str(item.get("source_document_id") or "")
            key = source_document_id or str(item.get("context_id") or "")
            if key and key not in documents:
                documents[key] = self._bounded_context(item)
        return {
            "dataset_id": dataset_id,
            "version": version,
            "source_document_count": int(summary["source_document_count"]),
            "contexts": list(documents.values()),
        }

    @staticmethod
    def _bump(workspace: dict[str, Any]) -> int:
        """递增工作区 revision 并刷新更新时间。

        Args:
            workspace: 工作区元数据（就地更新）。
        """
        workspace["revision"] = int(workspace.get("revision") or 0) + 1
        workspace["updated_at"] = _utc_now()
        return workspace["revision"]

    @staticmethod
    def _counts(cases: list[dict[str, Any]]) -> dict[str, int]:
        """按评审状态统计用例数量，未出现的状态补零。

        Args:
            cases: 用例列表。
        """
        counter = Counter(str(case.get("review_status") or "draft") for case in cases)
        return {status: counter.get(status, 0) for status in sorted(_CASE_STATUSES)}

    def _context_map(self, paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
        """按 context_id 建立上下文目录索引。

        Args:
            paths: 工作区文件路径表。
        """
        return {
            str(item["context_id"]): item for item in _read_jsonl(paths["contexts"])
        }

    @staticmethod
    def _bounded_context(item: dict[str, Any]) -> dict[str, Any]:
        """返回对外的受限上下文视图（摘要截断到 320 字符）。

        Args:
            item: 原始上下文记录。
        """
        # 只输出元数据与截断摘要，防止大段正文经 API 泄露
        return {
            "context_id": item.get("context_id"),
            "source_document_id": item.get("source_document_id"),
            "title": item.get("title"),
            "department": item.get("department"),
            "document_version": item.get("document_version"),
            "chunk_index": item.get("chunk_index"),
            "content_sha256": item.get("content_sha256"),
            "content_excerpt": str(item.get("content_excerpt") or "")[:320],
        }

    def _decorate_case(
        self, case: dict[str, Any], contexts: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """为用例附加受限上下文摘要，并剥离内部复核部门字段。

        Args:
            case: 原始用例。
            contexts: context_id 到上下文记录的索引。
        """
        result = dict(case)
        # 内部治理字段不对外输出
        result.pop("reviewer_department_id", None)
        # 合并证据/引用上下文并按首次出现去重后生成受限摘要
        ids = list(
            dict.fromkeys(
                list(case.get("expected_evidence_context_ids") or [])
                + list(case.get("expected_citation_context_ids") or [])
            )
        )
        result["context_summaries"] = [
            self._bounded_context(contexts[item]) for item in ids if item in contexts
        ]
        return result

    def get_case(
        self, org_id: str, dataset_id: str, case_id: str
    ) -> dict[str, Any]:
        """返回组织隔离的单个用例（附受限上下文摘要）。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            case_id: 用例 ID。
        """
        paths, _workspace, cases = self._load(org_id, dataset_id)
        case = next((item for item in cases if item.get("id") == case_id), None)
        if case is None:
            raise EvidenceReviewNotFound("case_not_found")
        return self._decorate_case(case, self._context_map(paths))

    @staticmethod
    def _bounded_fixture_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
        """把 Fixture 档案归一化为受限对外视图（只保留布尔事实）。

        Args:
            profile: 原始 Fixture 档案。
        """
        fixture_values = profile.get("fixtures")
        if not isinstance(fixture_values, Mapping):
            raise EvidenceReviewValidationError("invalid_fixture_profile")
        normalized: dict[str, Any] = {}
        for name in sorted(_REQUIRED_FIXTURES):
            item = fixture_values.get(name)
            if not isinstance(item, Mapping):
                raise EvidenceReviewValidationError("invalid_fixture")
            # 旧档案数据仅作为历史输入保留：刻意不作为当前校验依据对外暴露，
            # 也绝不回传浏览器可控的租户 ID。
            normalized[name] = {
                "legacy_manual_verified": item.get("verified") is True,
                "has_legacy_configuration": bool(
                    name not in _AUTHORIZATION_FIXTURES
                    or str(item.get("tenant_id") or "").strip()
                ),
            }
        return {
            "schema_version": "evidence-gate-fixture-profile-v1",
            "status": str(profile.get("status") or "pending_operator_configuration"),
            "fixtures": normalized,
        }

    def get_fixtures(self, org_id: str, dataset_id: str) -> dict[str, Any]:
        """返回数据集的受限 Fixture 视图（仅历史信息，不含租户数据）。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
        """
        paths, workspace, _cases = self._load(org_id, dataset_id)
        profile = self._bounded_fixture_profile(_read_json(paths["fixtures"]))
        return {
            "dataset_id": dataset_id,
            "revision": workspace["revision"],
            "status": "historical_only",
            "fixtures": profile["fixtures"],
        }

    @staticmethod
    def _current_corpus_template_id(*, fixture: str = "", category: str = "") -> str:
        """生成代表性模板用例的确定性种子 ID。

        Args:
            fixture: Fixture 名称，提供时生成 Fixture 模板 ID。
            category: 标准类目名，fixture 为空时用于生成标准模板 ID。
        """
        if fixture:
            return f"current-template-fixture-{fixture}"
        return f"current-template-standard-{category}"

    @staticmethod
    def _current_corpus_template_department(
        source_ids: list[str],
        source_departments: Mapping[str, str],
    ) -> str:
        """推导模板用例的部门归属。

        Args:
            source_ids: 模板绑定的源文档 ID 列表。
            source_departments: 源文档 ID 到部门的映射。
        """
        # 优先取首个绑定源的部门；无绑定时优先 finance，否则取字典序最小部门
        if source_ids:
            return str(source_departments[source_ids[0]])
        departments = sorted({value for value in source_departments.values() if value})
        if "finance" in departments:
            return "finance"
        if departments:
            return departments[0]
        raise EvidenceReviewValidationError("current_corpus_snapshot_empty")

    @staticmethod
    def _current_corpus_source_key(title: str, department: str) -> tuple[str, str]:
        """为源重绑定生成保守的元数据匹配键。

        目录展示名可能带稳定的数字归档前缀，而复核过的旧数据包存的是文档
        本题；仅归一化该前缀可以让 ``173印章管理`` 匹配 ``印章管理``，又不
        会把无关标题视为等价。部门是键的一部分，且调用方仍要求当前语料
        中恰好唯一匹配。

        Args:
            title: 源文档标题。
            department: 源文档部门。
        """

        # NFKC 归一化 + 去归档前缀 + 去全部空白 + casefold，得到保守标题键
        normalized_title = unicodedata.normalize("NFKC", str(title or "")).strip()
        normalized_title = _SOURCE_TITLE_PREFIX.sub("", normalized_title)
        normalized_title = re.sub(r"\s+", "", normalized_title).casefold()
        normalized_department = unicodedata.normalize("NFKC", str(department or "")).strip().casefold()
        return normalized_title, normalized_department

    def get_dataset(self, org_id: str, dataset_id: str) -> dict[str, Any]:
        """返回数据集的受限汇总视图（状态计数、类目覆盖与冻结信息）。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
        """
        paths, workspace, cases = self._load(org_id, dataset_id)
        fixtures = _read_json(paths["fixtures"])
        contexts = _read_jsonl(paths["contexts"])
        categories = Counter(str(case.get("category") or "") for case in cases)
        return {
            "dataset_id": dataset_id,
            "title": workspace.get("title"),
            "revision": workspace.get("revision"),
            "status": workspace.get("status"),
            "case_count": len(cases),
            "context_count": len(contexts),
            "source_document_count": int(workspace.get("source_document_count") or len(contexts)),
            "counts": self._counts(cases),
            "category_counts": dict(sorted(categories.items())),
            "required_category_count": len(REQUIRED_EVIDENCE_GATE_CATEGORIES),
            "covered_category_count": len(set(categories) & REQUIRED_EVIDENCE_GATE_CATEGORIES),
            "fixture_status": fixtures.get("status"),
            "last_frozen_version": workspace.get("last_frozen_version"),
            "last_frozen_manifest_sha256": workspace.get("last_frozen_manifest_sha256"),
            "last_frozen_at": workspace.get("last_frozen_at"),
            "last_frozen_by": workspace.get("last_frozen_by"),
            "source_frozen_version": workspace.get("source_frozen_version"),
            "source_type": workspace.get("source_type"),
            "active": workspace.get("active", True),
            "superseded_by": workspace.get("superseded_by"),
            "current_corpus_draft_id": workspace.get("current_corpus_draft_id"),
            "current_corpus_snapshot_sha256": workspace.get("current_corpus_snapshot_sha256"),
            "updated_at": workspace.get("updated_at"),
        }

    def list_versions(
        self,
        org_id: str,
        dataset_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """按版本名倒序列出全部冻结版本的公开元数据。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
        """
        paths, _workspace, _cases = self._load(org_id, dataset_id)
        versions_root = paths["versions"]
        if not versions_root.is_dir():
            return []
        versions: list[dict[str, Any]] = []
        # 只把含 manifest.json 的子目录视为合法冻结版本
        bounded_limit = max(1, min(int(limit), 100))
        for child in sorted(versions_root.iterdir(), key=lambda item: item.name, reverse=True):
            if child.is_dir() and (child / "manifest.json").is_file():
                versions.append(self._version_summary(paths, child.name))
                if len(versions) >= bounded_limit:
                    break
        return versions

    def get_version(self, org_id: str, dataset_id: str, version: str) -> dict[str, Any]:
        """返回单个冻结版本的公开元数据。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            version: 冻结版本名。
        """
        paths, _workspace, _cases = self._load(org_id, dataset_id)
        return self._version_summary(paths, version)

    def get_frozen_evaluation_suite(
        self,
        org_id: str,
        dataset_id: str,
        version: str,
    ) -> dict[str, Any]:
        """校验已复核冻结基准包，只返回其对外身份信息（不含用例内容）。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            version: 冻结版本名。
        """
        paths, _workspace, _cases = self._load(org_id, dataset_id)
        summary = self._version_summary(paths, version)
        root = self._version_root(paths, version)
        # 安全关卡：按 manifest 哈希校验包完整性，且必须处于已复核状态
        benchmark = load_evidence_gate_benchmark(
            root / "manifest.json",
            expected_manifest_sha256=str(summary["manifest_sha256"]),
            require_reviewed=True,
        )
        return {
            "dataset_id": dataset_id,
            "version": version,
            "manifest_sha256": str(summary["manifest_sha256"]),
            "case_count": len(benchmark),
        }

    def create_draft_from_version(
        self,
        org_id: str,
        dataset_id: str,
        version: str,
        *,
        expected_revision: int,
        actor_id: str,
    ) -> dict[str, Any]:
        """从冻结版本包创建一份全新的可编辑工作区草稿。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            version: 作为来源的冻结版本名。
            expected_revision: 调用方持有的期望 revision。
            actor_id: 执行操作的管理员 ID。
        """
        with self._lock(org_id, dataset_id):
            paths, workspace, _cases = self._load(org_id, dataset_id)
            self._check_revision(workspace, expected_revision)
            if workspace.get("status") != "frozen":
                raise EvidenceReviewConflict("dataset_not_frozen")
            root = self._version_root(paths, version)
            frozen_cases = _read_jsonl(root / "cases.jsonl")
            contexts = _read_jsonl(root / "context-catalog.jsonl")
            fixtures = _read_json(root / "fixture-profile.json")
            frozen_contexts = {
                str(item.get("context_id") or ""): item
                for item in contexts
                if str(item.get("context_id") or "")
            }
            fixtures, restored_fixture_binding_count = (
                self._restore_derived_current_corpus_fixture_bindings(
                    workspace,
                    fixtures,
                    frozen_cases,
                    contexts=frozen_contexts,
                )
            )
            now = _utc_now()
            cases: list[dict[str, Any]] = []
            normalized_fixture_expectation_count = 0
            for frozen in frozen_cases:
                case = dict(frozen)
                # 早期当前语料冻结版本中，刻意设计的 BM25 单路故障用例可能仍
                # 保留旧的 `answered` 期望；而运行时在稠密与图证据仍可用时会
                # 正确返回降级的部分回答。仅归一化新的可编辑草稿，不可变的
                # 冻结源保持原样。
                if workspace.get("source_type") == "current_corpus":
                    normalized_fixture_expectation_count += (
                        self._normalize_current_corpus_fixture_expectation(case)
                    )
                case.update(
                    {
                        "review_status": "draft",
                        "case_revision": 1,
                        "last_editor_id": "system-draft-from-freeze",
                        "last_edited_at": now,
                        "submitted_at": None,
                        "reviewer_id": None,
                        "reviewer_department_id": None,
                        "reviewed_at": None,
                        "review_reason": "",
                        "rejection_reason": "",
                    }
                )
                cases.append(case)
            workspace["status"] = "authoring"
            workspace["source_frozen_version"] = version
            revision = self._bump(workspace)
            _write_jsonl(paths["cases"], cases)
            _write_jsonl(paths["contexts"], contexts)
            _write_json(paths["fixtures"], fixtures)
            _write_json(paths["workspace"], workspace)
            events = _read_jsonl(paths["events"])
            events.append(
                {
                    "event": "create_draft_from_version",
                    "dataset_id": dataset_id,
                    "version": version,
                    "actor_id": actor_id,
                    "revision": revision,
                    "normalized_fixture_expectation_count": normalized_fixture_expectation_count,
                    "restored_fixture_binding_count": restored_fixture_binding_count,
                    "recorded_at": now,
                }
            )
            _write_jsonl(paths["events"], events[-10_000:])
            # 返回与普通数据集读取一致的受限汇总，绝不向 API 调用方
            # 暴露不可变版本目录本身。
            return self.get_dataset(org_id, dataset_id)

    @staticmethod
    def _normalize_current_corpus_fixture_expectation(case: dict[str, Any]) -> int:
        """Normalize legacy smoke expectations without mutating frozen history."""

        changed = 0
        if (
            case.get("required_fixture") == "bm25_unavailable_dense_graph_available"
            and case.get("expected_response_status") != "answered"
        ):
            case["expected_response_status"] = "answered"
            changed += 1
        # 早期冒烟模板把提示注入的原因码记为通用高风险复核；当前
        # 评测契约要求使用可审计、可区分的 prompt_injection_detected。
        if (
            case.get("required_fixture") == "prompt_injection_safety_fixture"
            and "high_risk_review" in (case.get("expected_reason_codes") or [])
        ):
            case["expected_reason_codes"] = [
                "prompt_injection_detected"
                if code == "high_risk_review"
                else code
                for code in case.get("expected_reason_codes") or []
            ]
            changed += 1
        return changed

    def _mark_current_corpus_dataset_active(
        self,
        org_id: str,
        active_dataset_id: str,
        *,
        recorded_at: str,
    ) -> None:
        """Keep exactly one editable current-corpus pointer active per organization."""

        root = self.root / self._org_namespace(org_id)
        if not root.is_dir():
            return
        for workspace_path in root.glob("*/workspace.json"):
            dataset_id = workspace_path.parent.name
            if not _ID.fullmatch(dataset_id):
                continue
            if dataset_id == active_dataset_id:
                continue
            paths = self._paths(org_id, dataset_id)
            with self._lock(org_id, dataset_id):
                workspace = _read_json(paths["workspace"])
                if workspace.get("source_type") != "current_corpus":
                    continue
                workspace["active"] = False
                workspace["superseded_by"] = active_dataset_id
                workspace["updated_at"] = recorded_at
                _write_json(paths["workspace"], workspace)

    def list_datasets(self, org_id: str) -> list[dict[str, Any]]:
        """列出旧版工作区与显式创建的当前语料数据集的受限汇总。

        Args:
            org_id: 组织 ID。
        """

        # 旧版数据集固定出现在列表中，再枚举命名空间下所有合法数据集目录
        dataset_ids = set(_SOURCE_EVALUATION_DATASETS)
        root = self.root / self._org_namespace(org_id)
        if root.is_dir():
            for workspace_path in root.glob("*/workspace.json"):
                dataset_id = workspace_path.parent.name
                if _ID.fullmatch(dataset_id):
                    dataset_ids.add(dataset_id)
        # 旧版数据集排最前，其余按 ID 字典序排列
        ordered = sorted(dataset_ids, key=lambda item: (item != "evidence-gates-v1", item))
        return [self.get_dataset(org_id, dataset_id) for dataset_id in ordered]

    @staticmethod
    def _frozen_case(case: dict[str, Any]) -> dict[str, Any]:
        """把用例裁剪为冻结视图：剥离治理字段但保留部门归属。

        Args:
            case: 待冻结的用例。
        """
        # 冻结包不携带编写/复核工作流字段，仅保留部门用于后续权限判断
        frozen = {
            key: value
            for key, value in case.items()
            if key not in _AUTHORING_FIELDS
        }
        frozen["department_id"] = case["department_id"]
        return frozen

    def freeze(
        self,
        org_id: str,
        dataset_id: str,
        *,
        expected_revision: int,
        actor_id: str,
    ) -> dict[str, Any]:
        """冻结工作区：生成带哈希清单的不可变版本并锁定编写态。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            expected_revision: 调用方持有的期望 revision。
            actor_id: 执行冻结的管理员 ID。
        """
        with self._lock(org_id, dataset_id):
            paths, workspace, cases = self._load(org_id, dataset_id)
            self._check_revision(workspace, expected_revision)
            self._require_authoring(workspace)
            # ① 冻结门槛：存在用例且全部已复核通过，类目覆盖完整
            if not cases or any(
                case.get("review_status") != "approved" for case in cases
            ):
                raise EvidenceReviewConflict("review_incomplete")
            categories = {str(case.get("category") or "") for case in cases}
            if categories != REQUIRED_EVIDENCE_GATE_CATEGORIES:
                raise EvidenceReviewConflict("category_coverage_incomplete")
            # 冻结只关心内容与复核。Fixture 校验之后针对这份不可变 manifest
            # 执行，并在发布流程启动时才强制要求，绝不依赖浏览器复选框或
            # 租户字段。
            fixtures = _read_json(paths["fixtures"])
            contexts = self._context_map(paths)
            restored_fixture_binding_count = 0
            # ② 从冻结版本派生的工作区先恢复被省略的源绑定
            if workspace.get("source_frozen_version"):
                fixtures, restored_fixture_binding_count = (
                    self._restore_derived_current_corpus_fixture_bindings(
                        workspace,
                        fixtures,
                        cases,
                        contexts=contexts,
                    )
                )
            historical_profile = self._bounded_fixture_profile(fixtures)
            # ③ 逐条复验用例与 Fixture 覆盖约束，防止绕过校验写入冻结包
            for case in cases:
                validated = self._validate_case(
                    case, contexts=contexts, existing_id=case["id"]
                )
                if (
                    validated["category"] in {
                        "fully_answerable",
                        "partially_answerable",
                        "conflicting",
                        "background_only",
                        "missing_version_or_date",
                        "missing_business_record",
                        "single_branch_unavailable",
                    }
                    and not validated["reference_answer"]
                ):
                    raise EvidenceReviewConflict("reference_answer_incomplete")
            self._validate_current_corpus_fixture_coverage(
                workspace,
                paths,
                cases,
                contexts=contexts,
                fixture_profile=fixtures,
            )
            # ④ 版本名由 UTC 时间戳与期望 revision 派生，重名即冲突
            version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-r{expected_revision}"
            version_root = paths["versions"] / version
            if version_root.exists():
                raise EvidenceReviewConflict("version_already_exists")
            version_root.mkdir(parents=True)
            cases_path = version_root / "cases.jsonl"
            context_path = version_root / "context-catalog.jsonl"
            fixture_path = version_root / "fixture-profile.json"
            review_path = version_root / "review-record.json"
            automated_path = version_root / "automated-pre-review.json"
            manifest_path = version_root / "manifest.json"
            frozen_cases = [self._frozen_case(case) for case in cases]
            # ⑤ 依次写入用例/目录/Fixture/复核记录/预审记录，最后写 manifest
            # 并把工作区置为 frozen，锁定后续编写
            _write_jsonl(cases_path, frozen_cases)
            _write_jsonl(context_path, list(contexts.values()))
            _write_json(
                fixture_path,
                self._frozen_current_corpus_fixture_profile(fixtures)
                if workspace.get("source_type") == "current_corpus"
                else historical_profile,
            )
            reviewer_ids = sorted(
                {str(case["reviewer_id"]) for case in cases if case.get("reviewer_id")}
            )
            review = {
                "schema_version": EVIDENCE_GATE_REVIEW_SCHEMA,
                "dataset_version": version,
                "review_status": "approved",
                "reviewed_at": _utc_now(),
                "reviewer_ids": reviewer_ids,
                "reviewed_case_ids": [case["id"] for case in cases],
                "decision_notes": "Approved through authenticated maker-checker workbench.",
            }
            _write_json(review_path, review)
            automated = {
                "schema_version": "evidence-gate-automated-pre-review-v1",
                "status": "passed",
                "human_approval": True,
                "checks": {
                    "all_categories_covered": True,
                    "all_context_ids_exist": True,
                    "all_cases_approved": True,
                    "runtime_fixture_profile_required": False,
                },
            }
            _write_json(automated_path, automated)
            frozen_at = _utc_now()
            manifest = {
                "schema_version": EVIDENCE_GATE_MANIFEST_SCHEMA,
                "reference_answer_schema_version": REFERENCE_ANSWER_SCHEMA_VERSION,
                "dataset_version": version,
                "frozen_at": frozen_at,
                "review_status": "approved",
                "answerability_labels": sorted(REQUIRED_EVIDENCE_GATE_CATEGORIES),
                "expected_response_statuses": sorted(SUPPORTED_RESPONSE_STATUSES),
                "cases_file": cases_path.name,
                "cases_sha256": sha256_file(cases_path),
                "review_record_file": review_path.name,
                "review_record_sha256": sha256_file(review_path),
                "context_catalog_file": context_path.name,
                "context_catalog_sha256": sha256_file(context_path),
                "fixture_profile_example_file": fixture_path.name,
                "fixture_profile_example_sha256": sha256_file(fixture_path),
                "automated_pre_review_file": automated_path.name,
                "automated_pre_review_sha256": sha256_file(automated_path),
                "frozen_by": actor_id,
                "source_workspace_revision": expected_revision,
            }
            # 当前语料来源的 manifest 额外锚定草稿 ID、快照哈希与恢复计数
            if workspace.get("source_type") == "current_corpus":
                manifest["current_corpus_draft_id"] = workspace.get("current_corpus_draft_id")
                manifest["current_corpus_snapshot_sha256"] = workspace.get("current_corpus_snapshot_sha256")
                manifest["restored_fixture_binding_count"] = restored_fixture_binding_count
                manifest["source_document_count"] = int(
                    workspace.get("source_document_count") or len(contexts)
                )
                manifest["context_count"] = len(contexts)
            _write_json(manifest_path, manifest)
            manifest_sha256 = sha256_file(manifest_path)
            workspace["status"] = "frozen"
            workspace["last_frozen_version"] = version
            workspace["last_frozen_manifest_sha256"] = manifest_sha256
            workspace["last_frozen_at"] = frozen_at
            workspace["last_frozen_by"] = actor_id
            self._bump(workspace)
            _write_json(paths["fixtures"], fixtures)
            self._persist(paths, workspace, cases)
            return {
                "status": "frozen",
                "version": version,
                "manifest_sha256": manifest_sha256,
                "version_path": str(version_root),
                "revision": workspace["revision"],
            }

    def refresh_current_corpus_dataset(
        self,
        org_id: str,
        source_dataset_id: str,
        *,
        source_documents: list[Mapping[str, Any]],
        current_chunks: list[Mapping[str, Any]],
        actor_id: str,
    ) -> dict[str, Any]:
        """Create a new current-corpus workspace from the latest frozen smoke bundle.

        The source version remains immutable.  Cases and fixture contracts are copied,
        while the Context catalog is rebuilt from the authenticated live document
        catalog so a subsequent freeze records the current 41-document identity.
        """

        if not isinstance(actor_id, str) or not actor_id.strip():
            raise EvidenceReviewValidationError("invalid_actor")
        source_paths, source_workspace, _source_cases = self._load(org_id, source_dataset_id)
        # 首次生成真实语料快照时，调用方传入已冻结的题目集（通常为日常 50 条），
        # 不要求先存在一个空的 current-corpus 工作区。这样页面上的“生成快照”
        # 可以真正成为一次性入口，而不是要求管理员先手工创建隐藏草稿。
        source_is_current_corpus = source_workspace.get("source_type") == "current_corpus"
        contexts = self._current_corpus_runtime_contexts(
            source_documents,
            current_chunks,
        )
        snapshot_payload = [
            {
                "document_id": str(item.get("document_id") or ""),
                "source": str(item.get("source") or ""),
                "department": str(item.get("department") or ""),
            }
            for item in source_documents
        ]
        snapshot_sha256 = hashlib.sha256(
            json.dumps(snapshot_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        dataset_id = f"current-corpus-{snapshot_sha256[:32]}"
        paths = self._paths(org_id, dataset_id)
        with self._lock(org_id, dataset_id):
            if paths["workspace"].is_file():
                target_workspace = _read_json(paths["workspace"])
                target_cases = _read_jsonl(paths["cases"])
                target_fixtures = _read_json(paths["fixtures"])
                target_cases, rebound = self._refresh_current_corpus_case_bindings(
                    target_cases,
                    contexts,
                )
                fixture_changes = self._sync_current_corpus_fixture_bindings(
                    target_fixtures,
                    target_cases,
                )
                normalized = sum(
                    self._normalize_current_corpus_fixture_expectation(case)
                    for case in target_cases
                )
                changed = False
                if (rebound or normalized or fixture_changes) and target_workspace.get("status") == "authoring":
                    self._bump(target_workspace)
                    changed = True
                if target_workspace.get("active") is not True:
                    target_workspace["active"] = True
                    changed = True
                if target_workspace.get("superseded_by") is not None:
                    target_workspace["superseded_by"] = None
                    changed = True
                if int(target_workspace.get("source_document_count") or 0) != len(source_documents):
                    target_workspace["source_document_count"] = len(source_documents)
                    changed = True
                if changed:
                    _write_jsonl(paths["cases"], target_cases)
                    _write_jsonl(paths["contexts"], contexts)
                    _write_json(paths["fixtures"], target_fixtures)
                    _write_json(paths["workspace"], target_workspace)
                result = self.get_dataset(org_id, dataset_id)
                self._mark_current_corpus_dataset_active(
                    org_id,
                    dataset_id,
                    recorded_at=_utc_now(),
                )
                return result
            if source_is_current_corpus:
                source_version = str(source_workspace.get("last_frozen_version") or "")
                if not source_version:
                    raise EvidenceReviewConflict("source_frozen_version_required")
                source_root = self._version_root(source_paths, source_version)
            else:
                source_root = self._source_bundle(source_dataset_id)
                source_manifest = _read_json(source_root / "manifest.json")
                source_version = str(source_manifest.get("dataset_version") or source_dataset_id)
            now = _utc_now()
            source_frozen_cases = _read_jsonl(source_root / "cases.jsonl")
            if not source_is_current_corpus:
                # 当前语料数据集只承载独立的 12 条冒烟契约：先覆盖每个场景，
                # 再补足到 12 条，避免把 50 条题目集误当成语料快照本身。
                by_category: dict[str, dict[str, Any]] = {}
                by_fixture: dict[str, dict[str, Any]] = {}
                for candidate in source_frozen_cases:
                    by_category.setdefault(str(candidate.get("category") or ""), candidate)
                    by_fixture.setdefault(str(candidate.get("required_fixture") or ""), candidate)
                selected = list(by_category.values())
                selected.extend(
                    candidate for fixture_name, candidate in by_fixture.items()
                    if fixture_name in _REQUIRED_FIXTURES and candidate not in selected
                )
                selected.extend(candidate for candidate in source_frozen_cases if candidate not in selected)
                source_frozen_cases = selected[:12]
            cases = []
            for frozen_case in source_frozen_cases:
                case = dict(frozen_case)
                self._normalize_current_corpus_fixture_expectation(case)
                # 当前语料快照会换成真实的运行时 Context。即使题型/路由来自
                # 已冻结模板，答案参考也必须重新以当前题干和当前绑定核对后再由
                # 人工批准，不能把旧包里的字段当作已审核事实直接冻结。
                if str(case.get("category") or "") in _REFERENCE_ANSWER_CATEGORIES:
                    case["reference_answer"] = ""
                case.update(
                    {
                        "review_status": "draft",
                        "case_revision": 1,
                        "last_editor_id": _REFERENCE_ANSWER_CANDIDATE_EDITOR,
                        "last_edited_at": now,
                        "submitted_at": None,
                        "reviewer_id": None,
                        "reviewed_at": None,
                        "review_reason": "",
                        "rejection_reason": "",
                    }
                )
                cases.append(case)
            cases, _rebound = self._refresh_current_corpus_case_bindings(cases, contexts)
            fixture_source = source_root / "fixture-profile.json"
            if not fixture_source.is_file():
                fixture_source = source_root / "fixture-profile.example.json"
            fixtures = _read_json(fixture_source)
            self._sync_current_corpus_fixture_bindings(fixtures, cases)
            draft_id = f"ccd_{snapshot_sha256[:32]}"
            workspace = {
                "schema_version": "evidence-review-workspace-v1",
                "dataset_id": dataset_id,
                "title": f"当前语料 41 份文档（冒烟 12 条）",
                "org_id_hash": self._org_namespace(org_id),
                "revision": 1,
                "status": "authoring",
                "created_at": now,
                "updated_at": now,
                "last_frozen_version": None,
                "last_frozen_manifest_sha256": None,
                "last_frozen_at": None,
                "last_frozen_by": None,
                "source_frozen_version": source_version,
                "source_type": "current_corpus",
                "current_corpus_draft_id": draft_id,
                "current_corpus_snapshot_sha256": snapshot_sha256,
                "source_document_count": len(contexts),
                "required_fixture_names": sorted(_REQUIRED_FIXTURES),
                "created_from_actor": actor_id,
                "active": True,
            }
            _write_jsonl(paths["cases"], cases)
            _write_jsonl(paths["contexts"], contexts)
            _write_json(paths["fixtures"], fixtures)
            _write_json(paths["workspace"], workspace)
            _write_jsonl(paths["events"], [{
                "event": "current_corpus_refresh",
                "source_dataset_id": source_dataset_id,
                "source_version": source_version,
                "actor_id": actor_id,
                "source_document_count": len(contexts),
                "recorded_at": now,
                "review_status": "draft_candidates_required",
            }])
        # Frozen version directories remain untouched; only the editable pointers
        # are updated so exactly one current-corpus workspace is selectable.
        self._mark_current_corpus_dataset_active(org_id, dataset_id, recorded_at=now)
        return self.get_dataset(org_id, dataset_id)
