"""面向数据集版本的 Fixture 校验编排门面。"""

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

from evaluation.evidence_gate.benchmark import (
    EvidenceGateBenchmarkError,
    load_evidence_gate_benchmark,
)
from evaluation.fixture_validation.assertions import FixtureAssertionRuntime
from evaluation.fixture_validation.store import (
    REQUIRED_FIXTURES,
    FinanceFixtureIdentityResolver,
    FixtureValidationConflict,
    FixtureValidationError,
    FixtureValidationStore,
)


FixtureProbe = Callable[[list[Any], str, FinanceFixtureIdentityResolver], Mapping[str, list[Mapping[str, Any]]]]  # 执行探测函数：入参为用例列表、组织 ID、身份解析器，返回按 Fixture 名分组的观察结果
FixtureCorpusProbe = Callable[[Sequence[Any], str], Mapping[str, str]]  # 语料预检函数：入参为用例序列、组织 ID，返回 Fixture 名到失败原因码的映射




class FixtureValidationService:
    """面向单个不可变冻结版本，协调记录存储与真实运行时断言。"""

    def __init__(
        self,
        workspace: Any,
        *,
        finance_department_id: str,
        corpus_probe: FixtureCorpusProbe | None = None,
    ) -> None:
        """初始化校验服务。

        Args:
            workspace: 工作区对象，提供版本详情与路径信息。
            finance_department_id: 预先配置的财务部门 ID。
            corpus_probe: 可选的语料预检函数；为 None 表示不做事前语料检查。
        """
        self.workspace = workspace
        self.finance_department_id = finance_department_id
        self.corpus_probe = corpus_probe

    def _store(self, org_id: str, dataset_id: str, version: str) -> FixtureValidationStore:
        """定位指定版本的校验记录存储。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 版本号。
        """
        paths = self.workspace._paths(org_id, dataset_id)
        # 记录存放在版本目录下的专属子目录，与版本数据一同隔离
        return FixtureValidationStore(self.workspace._version_root(paths, version) / "fixture-validation-runs")

    def _assert_runtime_corpus_aligned(
        self,
        *,
        org_id: str,
        dataset_id: str,
        version: str,
        manifest_sha256: str,
        corpus_probe: FixtureCorpusProbe | None,
    ) -> None:
        """入队前检测冻结清单是否不再匹配实时目录元数据，不匹配即失败。

        这是仅读元数据的预检：刻意不看文档正文与运行时身份，
        防止历史性错配创建一条只能在后期以更大、更具误导性的结果集失败的新运行。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 版本号。
            manifest_sha256: 期望的清单 SHA-256 摘要。
            corpus_probe: 本次调用使用的语料预检函数；为 None 时回退实例默认。
        """

        # 优先使用调用方传入的预检函数，否则回退实例默认；两者皆无则跳过预检
        probe = corpus_probe or self.corpus_probe
        if probe is None:
            return
        paths = self.workspace._paths(org_id, dataset_id)
        version_root = self.workspace._version_root(paths, version)
        # ① 重新加载并校验冻结清单，失败直接判清单无效
        try:
            dataset = load_evidence_gate_benchmark(
                version_root / "manifest.json",
                expected_manifest_sha256=manifest_sha256,
                require_reviewed=True,
            )
        except EvidenceGateBenchmarkError as error:
            raise FixtureValidationError("frozen_manifest_invalid") from error
        # ② 预检自身异常按“语料不可用”冲突处理
        try:
            failures = probe(list(dataset), org_id)
        except Exception as error:
            raise FixtureValidationConflict("fixture_runtime_corpus_unavailable") from error
        if not failures:
            return
        # ③ 优先报告“未对齐”，其余失败统一归为“不可用”
        if any(code == "fixture_runtime_corpus_unaligned" for code in failures.values()):
            raise FixtureValidationConflict("fixture_runtime_corpus_unaligned")
        raise FixtureValidationConflict("fixture_runtime_corpus_unavailable")

    def start(
        self,
        *,
        org_id: str,
        dataset_id: str,
        version: str,
        expected_revision: int,
        initiated_by: str,
        corpus_probe: FixtureCorpusProbe | None = None,
    ) -> dict[str, Any]:
        """启动一次新的校验运行。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 被校验的版本号。
            expected_revision: 调用方期望的源工作区修订号，用于乐观并发控制。
            initiated_by: 发起者标识字符串。
            corpus_probe: 可选的语料预检函数；为 None 时回退实例默认。
        """
        detail = self.workspace.get_version(org_id, dataset_id, version)
        # ① 安全关卡：版本修订号与期望不一致说明工作区已变化，拒绝启动
        if int(detail.get("source_workspace_revision") or 0) != expected_revision:
            raise FixtureValidationConflict("fixture_validation_revision_conflict")
        # ② 入队前先做语料对齐预检
        self._assert_runtime_corpus_aligned(
            org_id=org_id,
            dataset_id=dataset_id,
            version=version,
            manifest_sha256=str(detail["manifest_sha256"]),
            corpus_probe=corpus_probe,
        )
        # ③ 创建 queued 状态的新运行记录
        return self._store(org_id, dataset_id, version).start(
            dataset_id=dataset_id,
            version=version,
            manifest_sha256=str(detail["manifest_sha256"]),
            source_workspace_revision=expected_revision,
            initiated_by=initiated_by,
        )

    def list(self, *, org_id: str, dataset_id: str, version: str) -> list[dict[str, Any]]:
        """列出该版本当前清单下的全部校验运行。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 版本号。
        """
        detail = self.workspace.get_version(org_id, dataset_id, version)
        return self._store(org_id, dataset_id, version).list(manifest_sha256=str(detail["manifest_sha256"]))

    def get(self, *, org_id: str, dataset_id: str, version: str, run_id: str) -> dict[str, Any]:
        """读取单条运行记录并核对其清单归属。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 版本号。
            run_id: 运行标识。
        """
        detail = self.workspace.get_version(org_id, dataset_id, version)
        record = self._store(org_id, dataset_id, version).get(run_id)
        # 安全关卡：记录所属清单与版本当前清单不一致时拒绝返回
        if record["manifest_sha256"] != detail["manifest_sha256"]:
            raise FixtureValidationError("fixture_validation_manifest_mismatch")
        return record

    def retry(
        self,
        *,
        org_id: str,
        dataset_id: str,
        version: str,
        run_id: str,
        expected_revision: int,
        initiated_by: str,
        corpus_probe: FixtureCorpusProbe | None = None,
    ) -> dict[str, Any]:
        """在失败运行的基础上重试，等价于重新发起一次新运行。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 版本号。
            run_id: 需要重试的原运行标识。
            expected_revision: 调用方期望的源工作区修订号。
            initiated_by: 发起者标识字符串。
            corpus_probe: 可选的语料预检函数；为 None 时回退实例默认。
        """
        previous = self.get(org_id=org_id, dataset_id=dataset_id, version=version, run_id=run_id)
        # ① 安全关卡：原运行所基于的修订号与期望不一致 => 工作区已变化
        if previous["source_workspace_revision"] != expected_revision:
            raise FixtureValidationConflict("fixture_validation_revision_conflict")
        # ② 只有失败的运行允许重试
        if previous["status"] != "failed":
            raise FixtureValidationConflict("fixture_validation_not_retryable")
        return self.start(
            org_id=org_id,
            dataset_id=dataset_id,
            version=version,
            expected_revision=expected_revision,
            initiated_by=initiated_by,
            corpus_probe=corpus_probe,
        )

    def fail_dispatch(self, *, org_id: str, dataset_id: str, version: str, run_id: str) -> dict[str, Any]:
        """把派发失败的运行收尾为 failed。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 版本号。
            run_id: 运行标识。
        """
        return self._store(org_id, dataset_id, version).fail(
            run_id,
            reason_code="fixture_validation_dispatch_failed",
        )

    def run(self, *, org_id: str, dataset_id: str, version: str, run_id: str) -> dict[str, Any]:
        """执行一次已排队的运行：置为 running、执行真实断言并收尾。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 版本号。
            run_id: 运行标识。
        """
        detail = self.workspace.get_version(org_id, dataset_id, version)
        store = self._store(org_id, dataset_id, version)
        # ① 状态迁移 queued -> running（幂等）
        record = store.begin(run_id)
        # ② 清单不一致时直接以失败收尾，避免在错误版本上执行断言
        if record["manifest_sha256"] != detail["manifest_sha256"]:
            return store.fail(run_id, reason_code="fixture_validation_manifest_mismatch")
        paths = self.workspace._paths(org_id, dataset_id)
        version_root = self.workspace._version_root(paths, version)
        # ③ 用 Worker 持有的财务身份构造真实断言运行时
        runtime = FixtureAssertionRuntime(
            FinanceFixtureIdentityResolver(self.finance_department_id),
            corpus_probe=self.corpus_probe,
        )
        try:
            # ④ 执行断言，取第一条失败结果的原因码作为整体失败原因
            results = runtime.run(
                manifest_path=version_root / "manifest.json",
                expected_manifest_sha256=record["manifest_sha256"],
                org_id=org_id,
            )
            reason_code = next((item["reason_code"] for item in results if item["status"] == "failed"), None)
            return store.finish(run_id, fixture_results=results, reason_code=reason_code)
        except Exception:
            # ⑤ 兜底：未预期异常以执行失败收尾，保证记录结构完整且可重试
            return store.fail(run_id, reason_code="fixture_validation_execution_failed")

    def assert_runtime_corpus_aligned(
        self,
        *,
        org_id: str,
        dataset_id: str,
        version: str,
        manifest_sha256: str,
        corpus_probe: FixtureCorpusProbe | None = None,
    ) -> None:
        """为发布流程执行有界的、仅读元数据的语料对齐预检。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 版本号。
            manifest_sha256: 期望的清单 SHA-256 摘要。
            corpus_probe: 可选的语料预检函数；为 None 时回退实例默认。
        """

        self._assert_runtime_corpus_aligned(
            org_id=org_id,
            dataset_id=dataset_id,
            version=version,
            manifest_sha256=manifest_sha256,
            corpus_probe=corpus_probe,
        )

    def latest_success(
        self, *, org_id: str, dataset_id: str, version: str, manifest_sha256: str
    ) -> dict[str, Any] | None:
        """返回精确匹配该不可变清单的最新成功运行。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 版本号。
            manifest_sha256: 要求完全匹配的清单 SHA-256 摘要。
        """

        detail = self.workspace.get_version(org_id, dataset_id, version)
        # 版本当前清单与给定摘要不一致时视为没有成功运行
        if detail["manifest_sha256"] != manifest_sha256:
            return None
        # list 已按创建时间倒序，取第一条成功记录即最新成功
        return next(
            (
                record
                for record in self._store(org_id, dataset_id, version).list(
                    manifest_sha256=manifest_sha256
                )
                if record["status"] == "succeeded"
            ),
            None,
        )

    def has_success(self, *, org_id: str, dataset_id: str, version: str, manifest_sha256: str) -> bool:
        """判断给定清单是否已有成功的校验运行。

        Args:
            org_id: 组织（租户）ID。
            dataset_id: 数据集标识。
            version: 版本号。
            manifest_sha256: 要求完全匹配的清单 SHA-256 摘要。
        """
        return self.latest_success(
            org_id=org_id,
            dataset_id=dataset_id,
            version=version,
            manifest_sha256=manifest_sha256,
        ) is not None


# 对外导出的公开 API 清单
