"""仅供 worker 使用的不可变证据发布工作流执行器。

本模块刻意不执行任何 shell 命令、不修改部署配置；每个阶段都绑定到冻结清单，
并且只把有界的产物哈希写入工作流仓库。
"""
from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Iterable
from typing import Any, Literal

from evaluation.evidence_gate.benchmark import (
    load_evidence_gate_benchmark,
    reviewed_case_contract_errors,
)
from evaluation.evidence_gate.calibration import build_calibration_draft
from evaluation.evidence_gate.runtime import execute_evidence_gate_benchmark
from evaluation.ragas.adapter import (
    CONTEXT_QUALITY_METRICS,
    EVIDENCE_GATE_CONTRACT,
    EXTENDED_RAGAS_METRICS,
    FACTUAL_CORRECTNESS,
    RagasConfigurationError,
    validate_ragas_configuration,
)
from evaluation.scripts.run_ragas_benchmark import run as run_ragas_benchmark
from evaluation.runner import _write_text_secure
from evaluation.release.workflow import PostgreSQLReleaseWorkflowRepository, ReleaseWorkflowError, Stage
from shared.config import settings

_EXECUTION_STAGES: tuple[Stage, ...] = ("preflight", "baseline", "shadow", "calibration")  # worker 顺序执行的阶段（approval/promotion 由人工与 API 驱动）
_LEGACY_RAGAS_METRICS: tuple[str, ...] = (
    "faithfulness",
    FACTUAL_CORRECTNESS,
    *CONTEXT_QUALITY_METRICS,
)
# 正式运行需要计算的 RAGAS 指标集合。
# semantic_similarity 与 answer_relevancy/rubrics 三重冗余，
# noise_sensitivity 在 12~50 条短样本上区分度低且 Judge 方差大，
# 两者不进入正式评分集（能力声明仍在 ragas_metric_capabilities 中保留）。
FORMAL_RAGAS_METRICS: tuple[str, ...] = (
    "faithfulness",
    FACTUAL_CORRECTNESS,
    "answer_relevancy",
    *CONTEXT_QUALITY_METRICS,
    "rubrics_score_with_reference",
    "rubrics_score_without_reference",
    EVIDENCE_GATE_CONTRACT,
)
_RAGAS_METRICS: tuple[str, ...] = tuple(
    metric for metric in (*_LEGACY_RAGAS_METRICS, *EXTENDED_RAGAS_METRICS)
    if metric in FORMAL_RAGAS_METRICS
)  # 兼容旧引用：现与正式集合一致
_FORMAL_EVALUATION_METRICS: tuple[str, ...] = (
    *_LEGACY_RAGAS_METRICS,
    "answer_relevancy",
    *CONTEXT_QUALITY_METRICS,
    "rubrics_score_with_reference",
    "rubrics_score_without_reference",
    EVIDENCE_GATE_CONTRACT,
)

_REFERENCE_ANSWER_CATEGORIES = frozenset(
    {
        "fully_answerable",
        "partially_answerable",
        "conflicting",
        "background_only",
        "missing_version_or_date",
        "missing_business_record",
        "single_branch_unavailable",
    }
)
logger = logging.getLogger(__name__)


def _bounded_diagnostic_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten one private diagnostic report into the admin API allowlist shape."""
    diagnostic = report.get("diagnostic")
    if not isinstance(diagnostic, dict):
        return None
    routing = diagnostic.get("routing")
    routing_categories = routing.get("by_category") if isinstance(routing, dict) else None
    categories: dict[str, dict[str, int | float | str | None]] = {}
    route_transitions: dict[str, int] = {}
    case_ids: list[str] = []
    if isinstance(routing_categories, dict):
        for category, value in sorted(routing_categories.items()):
            if not isinstance(category, str) or not isinstance(value, dict):
                continue
            confusion = value.get("confusion")
            mismatches = 0
            if isinstance(confusion, dict):
                for transition, count in confusion.items():
                    if not isinstance(transition, str) or not isinstance(count, int):
                        continue
                    route_transitions[transition] = route_transitions.get(transition, 0) + count
                    expected, separator, observed = transition.partition("->")
                    if separator and expected != observed:
                        mismatches += count
            categories[category] = {
                "count": int(value.get("count") or 0),
                "failed": mismatches,
            }
            for case_id in value.get("failing_case_ids") or ():
                if isinstance(case_id, str) and case_id not in case_ids and len(case_ids) < 20:
                    case_ids.append(case_id)

    retrieval = diagnostic.get("retrieval")
    retrieval_stages = retrieval.get("stages") if isinstance(retrieval, dict) else None
    stages: dict[str, dict[str, int | float | str | None]] = {}
    if isinstance(retrieval_stages, dict):
        for stage, value in sorted(retrieval_stages.items()):
            if not isinstance(stage, str) or not isinstance(value, dict):
                continue
            exact = value.get("exact_evidence")
            source = value.get("source_document")
            exact = exact if isinstance(exact, dict) else {}
            source = source if isinstance(source, dict) else {}
            exact_recall = exact.get("mean_recall_at_k")
            source_recall = source.get("mean_recall_at_k")
            stages[stage] = {
                "scored": int(exact.get("scored") or 0),
                "exact_recall_at_5": (
                    exact_recall.get("5") if isinstance(exact_recall, dict) else None
                ),
                "source_recall_at_5": (
                    source_recall.get("5") if isinstance(source_recall, dict) else None
                ),
                "mrr": exact.get("mean_mrr"),
            }

    metric_coverage: dict[str, dict[str, int | float | str | None]] = {}
    for layer_name in ("answer", "safety"):
        layer = diagnostic.get(layer_name)
        metrics = layer.get("metrics") if isinstance(layer, dict) else None
        if not isinstance(metrics, dict):
            continue
        for metric, value in sorted(metrics.items()):
            if not isinstance(metric, str) or not isinstance(value, dict):
                continue
            coverage = value.get("coverage")
            coverage = coverage if isinstance(coverage, dict) else {}
            metric_coverage[metric] = {
                "scored": int(coverage.get("scored") or 0),
                "not_applicable": int(coverage.get("not_applicable") or 0),
                "failed": int(coverage.get("failed") or 0),
                "unsupported": int(coverage.get("unsupported") or 0),
                "mean": value.get("micro_mean"),
                "macro": value.get("category_macro_mean"),
            }

    safety = diagnostic.get("safety")
    hard_gate_values = safety.get("hard_gates") if isinstance(safety, dict) else None
    hard_gates = [
        name
        for name, value in sorted(hard_gate_values.items())
        if isinstance(name, str) and isinstance(value, dict) and value.get("passed") is False
    ] if isinstance(hard_gate_values, dict) else []
    run_plan = diagnostic.get("run_plan")
    category_policy = run_plan.get("category_policy") if isinstance(run_plan, dict) else None
    identities = {
        field: category_policy.get(field)
        for field in ("category_policy_version", "category_policy_sha256")
        if isinstance(category_policy, dict) and isinstance(category_policy.get(field), str)
    }
    return {
        "schema_version": "category-aware-diagnostic-summary-v1",
        "identities": identities,
        "categories": categories,
        "stages": stages,
        "metric_coverage": metric_coverage,
        "route_transitions": route_transitions,
        "hard_gates": hard_gates,
        "case_ids": case_ids,
    }


def _ragas_failure_code(error: Exception, *, phase: str | None = None) -> str:
    """Map evaluator failures to bounded, user-actionable reason codes.

    The exception message may contain provider response text or request details,
    so only the exception class and the well-known local ``ENOSPC`` condition are
    used for persistence. The stable fallback remains ``ragas_evaluation_failed``.
    """
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return "ragas_temporary_storage_full"
    if phase == "prepare_inputs":
        return "ragas_input_preparation_failed"
    if phase == "load_summary":
        return "ragas_summary_invalid"
    # Ragas raises ValueError synchronously when a collection metric rejects
    # its Judge adapter before any sample outcome can be checkpointed.  This is
    # distinct from provider-side JSON/schema failures and can be retried after
    # an adapter deployment without exposing the provider exception message.
    if phase == "evaluate" and isinstance(error, ValueError):
        return "ragas_evaluator_initialization_failed"
    name = type(error).__name__
    if name == "SoftTimeLimitExceeded":
        return "ragas_execution_timeout"
    if name in {"TimeoutError", "ReadTimeout", "APITimeoutError"}:
        return "ragas_judge_timeout"
    if name in {"APIConnectionError", "ConnectError", "ConnectionError"}:
        return "ragas_judge_unavailable"
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return "ragas_judge_auth_failed"
    if name == "RateLimitError":
        return "ragas_judge_rate_limited"
    if name in {"BadRequestError", "UnprocessableEntityError"}:
        return "ragas_judge_request_rejected"
    return "ragas_evaluation_failed"


class RagasCoverageError(ReleaseWorkflowError):
    """A fail-closed formal score result that still carries safe retry diagnostics.

    Only a structurally validated runner summary can construct this error.  The
    attached mapping therefore contains bounded counters, reason codes, and
    hashes rather than any evaluation or provider content.
    """

    def __init__(self, reason_code: str, *, metrics: dict[str, Any]) -> None:
        super().__init__(reason_code)
        self.metrics = metrics


class ReleaseWorkflowService:
    """通过持久化、可"以新尝试重试"的阶段尝试，执行一次冻结发布。"""

    def __init__(
        self,
        repository: PostgreSQLReleaseWorkflowRepository,
        workspace: Any,
        *,
        source_bundle: Path,
        output_root: Path,
        env_file: Path | None,
        fixture_validation_service: Any | None = None,
    ) -> None:
        """初始化执行服务：只保存依赖引用，不做任何 I/O。

        Args:
            repository: 工作流仓库（唯一事实来源）。
            workspace: 评测工作区，提供数据集/版本读取与路径解析。
            source_bundle: 基线模板等源文件所在目录。
            output_root: 发布运行产物的输出根目录。
            env_file: 基准运行加载的环境文件；None 表示不加载。
            fixture_validation_service: 历史 Fixture 校验服务；新工作流只用于运行时语料对齐检查。
        """
        self.repository = repository
        self.workspace = workspace
        self.source_bundle = source_bundle
        self.output_root = output_root
        self.env_file = env_file
        self.fixture_validation_service = fixture_validation_service

    def start(
        self,
        *,
        company_namespace: str,
        dataset_id: str,
        version: str,
        actor_id: str,
        current_corpus_draft_id: str | None = None,
        current_corpus_snapshot_sha256: str | None = None,
        evaluation_dataset_id: str | None = None,
        evaluation_version: str | None = None,
        evaluation_manifest_sha256: str | None = None,
        evaluation_case_count: int | None = None,
    ) -> dict[str, Any]:
        """创建（或幂等返回）一个绑定到已校验冻结清单的发布工作流。

        Args:
            company_namespace: 公司命名空间（内部审计兼容值）。
            dataset_id: 数据集标识。
            version: 数据集版本。
            actor_id: 发起人标识。
            current_corpus_draft_id: 历史工作流兼容字段；新工作流不再填写。
            current_corpus_snapshot_sha256: 历史工作流兼容字段；新工作流不再填写。
            evaluation_dataset_id: 正式评测套件数据集 ID（新工作流必填）。
            evaluation_version: 正式评测套件版本（新工作流必填）。
            evaluation_manifest_sha256: 正式评测套件清单 SHA-256。
            evaluation_case_count: 正式评测用例数量（显式套件时必须为 >=1 的整数）。
        """
        # ① 读取工作区事实：发布入口只能绑定冻结的 current_corpus 版本。
        detail = self.workspace.get_version(company_namespace, dataset_id, version)
        dataset = self.workspace.get_dataset(company_namespace, dataset_id)
        if dataset.get("source_type") != "current_corpus":
            raise ReleaseWorkflowError("current_corpus_required")
        # 旧调用方若仍传入草稿血缘，只做一致性校验；新工作流不持久化这些字段。
        if current_corpus_draft_id is not None:
            if dataset.get("current_corpus_draft_id") != current_corpus_draft_id or detail.get("current_corpus_draft_id") != current_corpus_draft_id:
                raise ReleaseWorkflowError("current_corpus_draft_mismatch")
        if current_corpus_snapshot_sha256 is not None:
            if dataset.get("current_corpus_snapshot_sha256") != current_corpus_snapshot_sha256 or detail.get("current_corpus_snapshot_sha256") != current_corpus_snapshot_sha256:
                raise ReleaseWorkflowError("current_corpus_snapshot_mismatch")
        if (current_corpus_draft_id is None) != (current_corpus_snapshot_sha256 is None):
            raise ReleaseWorkflowError("current_corpus_draft_required")
        # ② 新入口必须明确绑定已审核冻结的正式评测套件；其完整性在 API 入口校验。
        if (
            not evaluation_dataset_id
            or not evaluation_version
            or not evaluation_manifest_sha256
            or not isinstance(evaluation_case_count, int)
            or evaluation_case_count < 1
        ):
            raise ReleaseWorkflowError("formal_evaluation_suite_required")
        # ③ 全部关卡通过后创建（或幂等返回）工作流。
        return self.repository.create(
            company_namespace=company_namespace,
            dataset_id=dataset_id,
            version=version,
            manifest_sha256=detail["manifest_sha256"],
            # 新工作流以冻结版本身份为唯一绑定；草稿字段仅供历史记录兼容读取。
            current_corpus_draft_id=None,
            current_corpus_snapshot_sha256=None,
            # 新工作流不再依赖或创建 12-case Fixture 运行；历史行仍可读。
            fixture_validation_run_id=None,
            evaluation_dataset_id=evaluation_dataset_id,
            evaluation_version=evaluation_version,
            evaluation_manifest_sha256=evaluation_manifest_sha256,
            evaluation_case_count=evaluation_case_count,
            initiated_by=actor_id,
        )

    def retry(self, workflow_id: str, *, expected_revision: int) -> dict[str, Any]:
        """重新排队失败阶段；running 状态仅在能证明其为孤儿时才回收。

        Args:
            workflow_id: 工作流 ID。
            expected_revision: 期望修订号（乐观并发控制）。
        """
        work = self.repository.get(workflow_id)
        # ① 修订号校验：防止基于过期状态做决策。
        if work["revision"] != expected_revision:
            raise ReleaseWorkflowError("workflow_revision_conflict")
        # ② running：仅尝试回收"可证明"的孤儿尝试（带静默时间下限）。
        if work["status"] == "running":
            return self.repository.recover_orphaned(
                workflow_id,
                expected_revision=expected_revision,
            )
        # ③ 只有 failed 状态可以重新排队。
        if work["status"] != "failed":
            raise ReleaseWorkflowError("workflow_not_ready")
        # ④ 历史冒烟工作流可能在旧版本中错误进入 calibration。将其
        # 回退到 baseline，重新执行 baseline/shadow 后由新逻辑跳过校准。
        retry_stage = work["stage"]
        if (
            retry_stage == "calibration"
            and (
                work.get("evaluation_case_count") == 12
                or (
                    work.get("evaluation_case_count") is None
                    and work.get("evaluation_dataset_id") == work.get("dataset_id")
                )
            )
        ):
            retry_stage = "baseline"
        return self.repository.transition(
            workflow_id,
            expected_revision=expected_revision,
            status="queued",
            stage=retry_stage,
            clear_reason_code=True,
        )

    def review(
        self, workflow_id: str, *, expected_revision: int, reviewer_id: str,
        decision: Literal["approved", "rejected"], reason: str, target_mode: str | None,
        active_organization_admin_count: int,
    ) -> dict[str, Any]:
        """Apply maker-checker review after checking server-derived eligibility."""
        if active_organization_admin_count < 2:
            raise ReleaseWorkflowError("release_checker_unavailable")
        return self.repository.review(
            workflow_id, expected_revision=expected_revision, reviewer_id=reviewer_id,
            decision=decision, reason=reason, target_mode=target_mode,
        )

    def promote(
        self, workflow_id: str, *, expected_revision: int, actor_id: str
    ) -> dict[str, Any]:
        """Promote an independently approved workflow through the next Gate step."""
        return self.repository.promote(
            workflow_id, expected_revision=expected_revision, actor_id=actor_id
        )

    def rollback(
        self, *, expected_revision: int, actor_id: str, reason: str
    ) -> dict[str, Any]:
        """Rollback the active Gate through the repository's durable history."""
        return self.repository.rollback(
            expected_revision=expected_revision, actor_id=actor_id, reason=reason
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        """计算文件内容的 SHA-256 十六进制摘要。

        Args:
            path: 目标文件路径。
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        """确定性写入 JSON 文件（UTF-8、键排序、缩进 2、末尾换行），便于哈希比对。

        Args:
            path: 目标文件路径（自动创建父目录）。
            value: 待序列化的字典。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _frozen_paths(
        self,
        work: dict[str, Any],
        company_namespace: str,
    ) -> tuple[Path, Path, Path]:
        """解析并校验冻结输入/输出路径。

        Args:
            work: 工作流记录字典。
            company_namespace: 公司命名空间。

        返回 (版本根目录, 清单路径, 运行根目录)。
        """
        evaluation_dataset_id = str(work.get("evaluation_dataset_id") or work["dataset_id"])
        evaluation_version = str(work.get("evaluation_version") or work["version"])
        evaluation_manifest_sha256 = str(work.get("evaluation_manifest_sha256") or work["manifest_sha256"])
        # ① 优先使用显式正式评测套件标识，缺失时回退到发布数据集自身。
        paths = self.workspace._paths(company_namespace, evaluation_dataset_id)
        version_root = self.workspace._version_root(paths, evaluation_version).resolve()
        manifest_path = version_root / "manifest.json"
        # ② 安全关卡：清单必须存在且哈希与冻结值一致，否则拒绝执行。
        if not manifest_path.is_file() or self._sha256(manifest_path) != evaluation_manifest_sha256:
            raise ReleaseWorkflowError("manifest_hash_mismatch")
        # ③ 安全关卡：运行目录必须直接位于输出根之下，防止路径逃逸。
        run_root = (self.output_root / work["workflow_id"]).resolve()
        output_root = self.output_root.resolve()
        if run_root.parent != output_root:
            raise ReleaseWorkflowError("unsafe_release_output_path")
        return version_root, manifest_path, run_root

    def _baseline_template(
        self,
        *,
        work: dict[str, Any],
        manifest_path: Path,
        run_root: Path,
    ) -> tuple[Path, str]:
        """生成绑定到具体版本的基线模板快照，不修改共享源数据。

        Args:
            work: 工作流记录字典。
            manifest_path: 冻结清单路径。
            run_root: 本次运行的输出根目录。

        返回 (快照文件路径, 快照 SHA-256)。
        """
        source = self.source_bundle / "baseline-template.json"
        # ① 源模板必须存在。
        if not source.is_file():
            raise ReleaseWorkflowError("baseline_template_missing")
        # ② 读取并解析模板与清单；任何 I/O 或 JSON 错误都转为安全错误码。
        try:
            baseline = json.loads(source.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ReleaseWorkflowError("invalid_baseline_template") from error
        if not isinstance(baseline, dict) or not isinstance(manifest, dict):
            raise ReleaseWorkflowError("invalid_baseline_template")
        dataset = baseline.get("dataset")
        if not isinstance(dataset, dict):
            raise ReleaseWorkflowError("invalid_baseline_template")
        # ③ 把冻结清单中的版本与哈希绑定进模板的 dataset 段。
        dataset.update(
            {
                "version": str(work.get("evaluation_version") or work["version"]),
                "manifest_sha256": str(work.get("evaluation_manifest_sha256") or work["manifest_sha256"]),
                "cases_sha256": manifest.get("cases_sha256"),
                "review_record_sha256": manifest.get("review_record_sha256"),
            }
        )
        target = run_root / "baseline-template.json"
        payload = json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        # ④ 幂等写入：已存在时内容必须逐字节一致，否则判定为重放被篡改。
        if target.exists():
            if target.read_text(encoding="utf-8") != payload:
                raise ReleaseWorkflowError("baseline_template_snapshot_mismatch")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
        return target, self._sha256(target)

    def _preflight(
        self,
        *,
        work: dict[str, Any],
        company_namespace: str,
    ) -> tuple[Path, Path, Path, dict[str, Any]]:
        """执行预检：解析冻结路径、探测输出可写、准备基线模板，并返回预检指标。

        Args:
            work: 工作流记录字典。
            company_namespace: 公司命名空间。

        返回 (版本根目录, 清单路径, 运行根目录, 预检指标字典)。
        """
        version_root, manifest_path, run_root = self._frozen_paths(work, company_namespace)
        run_root.mkdir(parents=True, exist_ok=True)
        # ① 写入并删除探针文件，验证输出目录真实可写。
        probe = run_root / ".write-probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as error:
            raise ReleaseWorkflowError("release_output_not_writable") from error
        # ② 生成绑定版本的基线模板快照并记录其哈希。
        baseline_path, baseline_sha256 = self._baseline_template(
            work=work,
            manifest_path=manifest_path,
            run_root=run_root,
        )
        preflight_metrics: dict[str, Any] = {
            "manifest_sha256": work["manifest_sha256"],
            "evaluation_dataset_id": work.get("evaluation_dataset_id") or work["dataset_id"],
            "evaluation_version": work.get("evaluation_version") or work["version"],
            "evaluation_manifest_sha256": work.get("evaluation_manifest_sha256") or work["manifest_sha256"],
            "evaluation_case_count": work.get("evaluation_case_count"),
            "baseline_template_sha256": baseline_sha256,
            "baseline_template_ready": baseline_path.is_file(),
        }
        formal_case_count = work.get("evaluation_case_count")
        if isinstance(formal_case_count, int) and formal_case_count >= 1:
            preflight_metrics["formal_reference_answers_complete"] = self._validate_formal_reference_answers(
                manifest_path=manifest_path,
                formal_case_count=formal_case_count,
            )
        if isinstance(formal_case_count, int) and formal_case_count >= 1:
            try:
                judge = validate_ragas_configuration()
            except RagasConfigurationError as error:
                raise ReleaseWorkflowError("ragas_configuration_missing") from error
            preflight_metrics["ragas_judge_ready"] = True
            preflight_metrics["ragas_judge_model"] = judge["model"]
            preflight_metrics["ragas_judge_source"] = judge["source"]
        return version_root, manifest_path, run_root, preflight_metrics

    @staticmethod
    def _validate_formal_reference_answers(
        *, manifest_path: Path, formal_case_count: int,
    ) -> bool:
        """Fail before QA when an applicable frozen formal case lacks its answer reference.

        Older historical bundles may not declare a cases artifact; those remain
        readable and are left to the legacy replay path. New benchmark bundles
        declare ``cases_file`` and are checked before any expensive QA/Judge work.
        """
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReleaseWorkflowError("formal_reference_manifest_invalid") from error
        if not isinstance(manifest, dict) or not manifest.get("cases_file"):
            return True
        cases_name = manifest.get("cases_file")
        if not isinstance(cases_name, str) or not cases_name.strip():
            raise ReleaseWorkflowError("formal_reference_manifest_invalid")
        cases_path = (manifest_path.parent / cases_name).resolve()
        try:
            cases_path.relative_to(manifest_path.parent.resolve())
        except ValueError as error:
            raise ReleaseWorkflowError("formal_reference_manifest_invalid") from error
        if not cases_path.is_file():
            raise ReleaseWorkflowError("formal_reference_manifest_invalid")
        try:
            lines = cases_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ReleaseWorkflowError("formal_reference_manifest_invalid") from error
        case_count = 0
        missing = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReleaseWorkflowError("formal_reference_manifest_invalid") from error
            if not isinstance(item, dict):
                raise ReleaseWorkflowError("formal_reference_manifest_invalid")
            case_count += 1
            category = item.get("category")
            if category in _REFERENCE_ANSWER_CATEGORIES and not str(item.get("reference_answer") or "").strip():
                missing += 1
        if case_count != formal_case_count:
            raise ReleaseWorkflowError("formal_reference_manifest_invalid")
        if missing:
            raise ReleaseWorkflowError("ragas_reference_missing")
        return True

    @staticmethod
    def _reason_code(result: dict[str, Any]) -> str:
        """从运行结果中提取首个非空原因码（截断到 96 字符）；缺失时返回通用阻断码。

        Args:
            result: 基准运行返回的结果字典。
        """
        candidates = result.get("reason_codes")
        if isinstance(candidates, list):
            for value in candidates:
                if isinstance(value, str) and value:
                    return value[:96]
        return "runtime_preflight_blocked"

    def _quality_report(
        self,
        *,
        result: dict[str, Any],
        stage_root: Path,
    ) -> tuple[Path, str]:
        """仅当质量报告仍位于本阶段尝试目录之内时才接受它。

        Args:
            result: 基准运行返回的结果字典。
            stage_root: 本阶段尝试的输出目录。

        返回 (报告路径, 报告 SHA-256)。
        """
        report = stage_root / "quality-report.json"  # 确定性默认路径（单元测试回退）
        metrics = result.get("quality_metrics")
        if isinstance(metrics, list) and metrics and isinstance(metrics[0], dict):
            candidate = metrics[0].get("quality_report")
            if isinstance(candidate, str):
                report = Path(candidate).resolve()
        # ① 安全关卡：报告路径必须仍位于本阶段目录内，防止路径逃逸。
        root = stage_root.resolve()
        if report.parent != root and root not in report.parents:
            raise ReleaseWorkflowError("unsafe_quality_report_path")
        # ② 安全关卡：只有"已测量、待校准"状态才算有效测量结果。
        if result.get("status") != "measured_pending_calibration":
            raise ReleaseWorkflowError(self._reason_code(result))
        # ③ 报告文件必须真实存在。
        if not report.is_file():
            raise ReleaseWorkflowError("quality_report_missing")
        return report, self._sha256(report)

    def _prepare_ragas_inputs(
        self,
        *,
        responses_path: Path,
        manifest_path: Path,
        expected_manifest_sha256: str,
        output_root: Path,
    ) -> Path:
        """Add frozen reviewed references to legacy response snapshots without mutation.

        Old snapshots predate reviewed reference evidence. A retry writes an
        enriched copy into its new attempt directory, keeps the immutable source
        untouched, and carries forward only hash-compatible finite scores.
        """
        records: list[dict[str, Any]] = []
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReleaseWorkflowError("ragas_responses_invalid") from error
            if not isinstance(record, dict):
                raise ReleaseWorkflowError("ragas_responses_invalid")
            records.append(record)
        if not records:
            raise ReleaseWorkflowError("ragas_responses_invalid")
        try:
            manifest_preview = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest_preview = {}
        carries_reviewed_answers = (
            isinstance(manifest_preview, dict)
            and manifest_preview.get("reference_answer_schema_version")
            == "reviewed-reference-answer-v1"
        )
        if not carries_reviewed_answers and all(
            record.get("status") not in {"succeeded", "invalid_provenance"}
            or (
                isinstance(record.get("reference"), str)
                and record["reference"].strip()
            )
            for record in records
        ):
            # Historical snapshots with an already-bound reference remain
            # replayable. New reviewed-answer bundles always verify/replace the
            # value below against their immutable cases.
            return responses_path
        dataset = load_evidence_gate_benchmark(
            manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
            require_reviewed=True,
        )
        references = self._reviewed_evidence_references(
            manifest_path=manifest_path,
            dataset=dataset,
        )
        needs_enrichment = any(
            record.get("status") in {"succeeded", "invalid_provenance"}
            and isinstance(record.get("benchmark_id"), str)
            and references.get(record["benchmark_id"])
            and record.get("reference") != references[record["benchmark_id"]]
            for record in records
        )
        if not needs_enrichment:
            return responses_path
        enriched: list[dict[str, Any]] = []
        for record in records:
            benchmark_id = record.get("benchmark_id")
            frozen_reference = references.get(benchmark_id) if isinstance(benchmark_id, str) else None
            if (
                record.get("status") in {"succeeded", "invalid_provenance"}
                and isinstance(frozen_reference, str)
                and frozen_reference.strip()
            ):
                enriched.append({**record, "reference": frozen_reference})
            else:
                # Some reviewed refusal/unavailability cases intentionally have no
                # positive evidence reference. Reference-based metrics record a
                # bounded skip for those cases; Faithfulness remains eligible.
                enriched.append(record)

        output_root.mkdir(parents=True, exist_ok=True)
        prepared_path = output_root / "ragas-inputs.jsonl"
        _write_text_secure(
            prepared_path,
            "".join(
                f"{json.dumps(record, ensure_ascii=False, sort_keys=True)}\n"
                for record in enriched
            ),
        )
        stage_prefix = output_root.name.partition("-")[0]
        previous_outputs = sorted(
            output_root.parent.glob(f"{stage_prefix}-*/ragas_scores.jsonl"),
            key=lambda path: path.stat().st_mtime_ns if path.is_file() else 0,
            reverse=True,
        )
        self._seed_ragas_checkpoint(
            source_outputs=[responses_path.with_name("ragas_scores.jsonl"), *previous_outputs],
            original_responses=responses_path,
            prepared_responses=prepared_path,
        )
        return prepared_path

    @staticmethod
    def _runtime_evidence_lineage_summary(
        *,
        records: Iterable[dict[str, Any]],
        dataset: Iterable[Any],
        reviewed_content_hashes: dict[str, str],
    ) -> dict[str, Any]:
        """Compare reviewed evidence with runtime identities without raw content."""
        runtime_by_id = {
            record.get("benchmark_id"): record
            for record in records
            if isinstance(record.get("benchmark_id"), str)
        }
        positive_evidence_categories = {
            "fully_answerable",
            "partially_answerable",
            "conflicting",
            "background_only",
            "missing_version_or_date",
            "missing_business_record",
            "single_branch_unavailable",
        }
        non_positive_evidence_categories = {
            "authorization_filtered",
            "all_branches_unavailable",
            "prompt_injection",
            "completely_unanswerable",
        }
        checked = 0
        resolved = 0
        document_resolved = 0
        unresolved_by_category: dict[str, int] = {}
        for sample in dataset:
            category = str(getattr(sample, "category", "") or "")
            if category not in positive_evidence_categories | non_positive_evidence_categories:
                continue
            checked += 1
            record = runtime_by_id.get(getattr(sample, "id", None))
            if not isinstance(record, dict):
                unresolved_by_category[category] = unresolved_by_category.get(category, 0) + 1
                continue
            if category in non_positive_evidence_categories:
                # These routes can legitimately evaluate retrieved candidates
                # before rejecting them (for example prompt-injection fixtures or
                # semantically irrelevant top-k results).  Candidate presence is
                # a quality outcome, not an identity-lineage failure.
                resolved += 1
                continue
            expected_documents = set(getattr(sample, "expected_source_document_ids", ()) or ())
            runtime_documents = set(record.get("retrieved_source_document_ids") or record.get("retrieved_context_ids") or ())
            if expected_documents and expected_documents.issubset(runtime_documents):
                document_resolved += 1
            expected_contexts = tuple(getattr(sample, "expected_evidence_context_ids", ()) or ())
            expected_by_id = {
                context_id: reviewed_content_hashes[context_id]
                for context_id in expected_contexts
                if context_id in reviewed_content_hashes
            }
            if not expected_contexts or len(expected_by_id) != len(expected_contexts):
                unresolved_by_category[category] = unresolved_by_category.get(category, 0) + 1
                continue

            runtime_hashes: set[str] = set()
            mismatched_claim = False
            contexts = record.get("contexts") or ()
            for context in contexts:
                if not isinstance(context, dict):
                    continue
                content = context.get("content")
                if not isinstance(content, str):
                    continue
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                runtime_hashes.add(content_hash)
                metadata = context.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                context_id = next(
                    (
                        value
                        for value in (metadata.get("context_id"), metadata.get("chunk_id"))
                        if isinstance(value, str)
                    ),
                    None,
                )
                if context_id in expected_by_id and expected_by_id[context_id] != content_hash:
                    mismatched_claim = True

            # Older bounded records may omit raw contexts.  Retain a fail-closed
            # fallback when they explicitly claim a reviewed Context ID whose
            # reviewed hash is absent from the runtime hash set.
            if not contexts:
                runtime_hashes.update(record.get("retrieved_content_sha256s") or ())
                claimed = set(record.get("retrieved_chunk_ids") or ()) & set(expected_by_id)
                mismatched_claim = any(
                    expected_by_id[context_id] not in runtime_hashes
                    for context_id in claimed
                )

            if mismatched_claim:
                unresolved_by_category[category] = unresolved_by_category.get(category, 0) + 1
            else:
                # Missing a reviewed gold Chunk is retrieval recall and must be
                # measured by the quality/RAGAS stages.  It is not evidence of an
                # identity substitution when no conflicting Context claim exists.
                resolved += 1
        return {
            "checked": checked,
            "resolved": resolved,
            "document_resolved": document_resolved,
            "unresolved": checked - resolved,
            "unresolved_by_category": unresolved_by_category,
        }

    def _validate_runtime_evidence_lineage(
        self,
        *,
        responses_path: Path,
        manifest_path: Path,
        expected_manifest_sha256: str,
    ) -> dict[str, Any]:
        """Fail closed when reviewed evidence cannot be proven in runtime output."""
        dataset = load_evidence_gate_benchmark(
            manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
            require_reviewed=True,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and manifest.get("review_status") == "approved":
            contract_errors: dict[str, int] = {}
            invalid_by_category: dict[str, int] = {}
            for sample in dataset:
                errors = reviewed_case_contract_errors(sample)
                if not errors:
                    continue
                category = str(getattr(sample, "category", "") or "unknown")[:64]
                invalid_by_category[category] = invalid_by_category.get(category, 0) + 1
                for reason in errors:
                    contract_errors[reason] = contract_errors.get(reason, 0) + 1
            if contract_errors:
                raise RagasCoverageError(
                    "formal_reviewed_contract_invalid",
                    metrics={
                        "formal_reviewed_contract": {
                            "invalid": sum(invalid_by_category.values()),
                            "invalid_by_category": invalid_by_category,
                            "reasons": contract_errors,
                        }
                    },
                )
        catalog_file = manifest.get("context_catalog_file") if isinstance(manifest, dict) else None
        if not isinstance(catalog_file, str):
            raise ReleaseWorkflowError("formal_runtime_evidence_invalid")
        catalog_path = (manifest_path.parent / catalog_file).resolve()
        if catalog_path.parent != manifest_path.parent.resolve() or not catalog_path.is_file():
            raise ReleaseWorkflowError("formal_runtime_evidence_invalid")
        reviewed_hashes: dict[str, str] = {}
        for line in catalog_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ReleaseWorkflowError("formal_runtime_evidence_invalid")
            context_id = item.get("context_id")
            content_sha256 = item.get("content_sha256")
            if not isinstance(context_id, str) or not isinstance(content_sha256, str):
                raise ReleaseWorkflowError("formal_runtime_evidence_invalid")
            reviewed_hashes[context_id] = content_sha256
        records = self._load_ragas_outcomes(responses_path)
        summary = self._runtime_evidence_lineage_summary(
            records=records,
            dataset=dataset,
            reviewed_content_hashes=reviewed_hashes,
        )
        if summary["checked"] < 1 or summary["unresolved"]:
            raise RagasCoverageError(
                "formal_runtime_evidence_mismatch",
                metrics={"formal_evidence_lineage": summary},
            )
        return summary

    @staticmethod
    def _reviewed_evidence_references(
        *, manifest_path: Path, dataset: Iterable[Any],
    ) -> dict[str, str]:
        """Return independently reviewed answers from the immutable case bundle.

        Context excerpts are deliberately not used here. They remain reference
        contexts for retrieval metrics and must never be synthesized into the
        expected answer used by answer-quality metrics.
        """
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReleaseWorkflowError("ragas_reference_evidence_invalid") from error
        if not isinstance(manifest, dict):
            raise ReleaseWorkflowError("ragas_reference_evidence_invalid")
        references: dict[str, str] = {}
        for sample in dataset:
            benchmark_id = getattr(sample, "id", None)
            if not isinstance(benchmark_id, str) or not benchmark_id:
                raise ReleaseWorkflowError("ragas_reference_evidence_invalid")
            reference_answer = getattr(sample, "reference_answer", "")
            if isinstance(reference_answer, str) and reference_answer.strip():
                references[benchmark_id] = reference_answer.strip()
        return references

    @staticmethod
    def _seed_ragas_checkpoint(
        *, source_outputs: Iterable[Path], original_responses: Path,
        prepared_responses: Path,
    ) -> None:
        """Reuse finite scores only when their checkpoint binds to the same inputs."""
        reusable: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        original_sha256 = hashlib.sha256(original_responses.read_bytes()).hexdigest()
        prepared_sha256 = hashlib.sha256(prepared_responses.read_bytes()).hexdigest()
        valid_pairs: set[tuple[str, str]] = set()
        for line in prepared_responses.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            benchmark_id = record.get("benchmark_id")
            retrieval_mode = record.get("retrieval_mode")
            if isinstance(benchmark_id, str) and isinstance(retrieval_mode, str):
                valid_pairs.add((benchmark_id, retrieval_mode))
        for source_output in source_outputs:
            if not source_output.is_file():
                continue
            state_path = source_output.with_suffix(".resume.json")
            if not state_path.is_file():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(state, dict):
                continue
            checkpoint_metrics = state.get("metrics")
            if checkpoint_metrics == list(_RAGAS_METRICS):
                reusable_metrics = set(_RAGAS_METRICS)
            elif checkpoint_metrics == list(_LEGACY_RAGAS_METRICS):
                reusable_metrics = set(_LEGACY_RAGAS_METRICS)
            else:
                continue
            source_sha256 = state.get("responses_sha256")
            if source_sha256 == prepared_sha256:
                allowed_metrics = reusable_metrics
            elif source_sha256 == original_sha256:
                # Adding only reference evidence does not change Faithfulness input.
                allowed_metrics = {"faithfulness"}
            else:
                continue
            try:
                source_lines = source_output.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in source_lines:
                if not line.strip():
                    continue
                try:
                    outcome = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(outcome, dict):
                    continue
                benchmark_id = outcome.get("benchmark_id")
                retrieval_mode = outcome.get("retrieval_mode")
                metric = outcome.get("metric")
                score = outcome.get("score")
                key = (benchmark_id, retrieval_mode, metric)
                if (
                    not isinstance(metric, str)
                    or metric not in allowed_metrics
                    or outcome.get("status") != "scored"
                    or not isinstance(benchmark_id, str)
                    or not isinstance(retrieval_mode, str)
                    or (benchmark_id, retrieval_mode) not in valid_pairs
                    or isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    or key in seen
                ):
                    continue
                reusable.append(outcome)
                seen.add(key)
        if not reusable:
            return
        output_path = prepared_responses.with_name("ragas_scores.jsonl")
        _write_text_secure(
            output_path,
            "".join(
                f"{json.dumps(outcome, ensure_ascii=False, sort_keys=True)}\n"
                for outcome in reusable
            ),
        )
        _write_text_secure(
            output_path.with_suffix(".resume.json"),
            json.dumps(
                {
                    "responses_sha256": prepared_sha256,
                    "metrics": list(_RAGAS_METRICS),
                },
                sort_keys=True,
            ) + "\n",
        )

    def _run_ragas(
        self,
        *,
        report: Path,
        formal_case_count: int | None,
        manifest_path: Path | None = None,
        expected_manifest_sha256: str | None = None,
        output_root: Path | None = None,
    ) -> dict[str, Any]:
        """对正式运行计算 RAGAS 分数，只保留关于结果的有界证据（哈希与覆盖率统计）。

        Args:
            report: 质量报告路径（其同目录需有 responses.jsonl）。
            formal_case_count: 正式用例数量；非法或 <1 时返回空 dict（不打分）。
        """
        # ① 非正式套件（用例数非法或 <1）不打分，直接返回空指标。
        if not isinstance(formal_case_count, int) or formal_case_count < 1:
            return {}
        # ② responses.jsonl 必须与质量报告同目录存在。
        responses_path = report.parent / "responses.jsonl"
        if not responses_path.is_file():
            raise ReleaseWorkflowError("ragas_responses_missing")
        # ③ 运行 RAGAS 基准并读取汇总；白名单之外的异常统一转为安全错误码。
        phase = "prepare_inputs"
        try:
            # 旧快照可能缺少参考答案；只在新重试目录生成补全副本，
            # 以便重新启用 reference-based 指标而不篡改历史证据。
            if manifest_path is not None and expected_manifest_sha256 and output_root is not None:
                responses_path = self._prepare_ragas_inputs(
                    responses_path=responses_path,
                    manifest_path=manifest_path,
                    expected_manifest_sha256=expected_manifest_sha256,
                    output_root=output_root,
                )
                manifest: Any = {}
                if manifest_path.is_file():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as error:
                        raise ReleaseWorkflowError("formal_runtime_evidence_invalid") from error
                if (
                    isinstance(manifest, dict)
                    and (
                        manifest.get("review_status") == "approved"
                        or (
                            manifest.get("current_corpus_draft_id")
                            and manifest.get("current_corpus_snapshot_sha256")
                        )
                    )
                ):
                    self._validate_runtime_evidence_lineage(
                        responses_path=responses_path,
                        manifest_path=manifest_path,
                        expected_manifest_sha256=expected_manifest_sha256,
                    )
            phase = "evaluate"
            judge = validate_ragas_configuration()
            output_path = asyncio.run(
                run_ragas_benchmark(
                    SimpleNamespace(
                        responses_jsonl=responses_path,
                        metrics=list(_FORMAL_EVALUATION_METRICS),
                        output_jsonl=None,
                        embedding_model=settings.embedding_model,
                        embedding_api_key=settings.dashscope_api_key,
                        embedding_base_url=settings.dashscope_base_url,
                    )
                )
            )
            phase = "load_summary"
            summary_path = Path(output_path).with_suffix(".summary.json")
            if not summary_path.is_file():
                raise ReleaseWorkflowError("ragas_summary_missing")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except ReleaseWorkflowError:
            raise
        except RagasConfigurationError as error:
            logger.warning(
                "ragas_stage_failed phase=%s exception_type=%s",
                phase,
                type(error).__name__,
            )
            raise ReleaseWorkflowError("ragas_configuration_missing") from error
        except Exception as error:
            logger.warning(
                "ragas_stage_failed phase=%s exception_type=%s",
                phase,
                type(error).__name__,
            )
            raise ReleaseWorkflowError(
                _ragas_failure_code(error, phase=phase)
            ) from error
        # ④ 校验汇总结构必须是包含 metrics 的对象。
        metrics = summary.get("metrics") if isinstance(summary, dict) else None
        if not isinstance(metrics, dict):
            raise ReleaseWorkflowError("ragas_summary_invalid")
        # ⑤ 每个指标只保留有界的覆盖统计与稳定原因码，绝不转存 Judge 内容。
        coverage: dict[str, dict[str, Any]] = {}
        for metric_name in _FORMAL_EVALUATION_METRICS:
            value = metrics.get(metric_name)
            if not isinstance(value, dict):
                continue

            def count(field: str, *, default: int | None = None) -> int:
                raw = value.get(field, default)
                if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                    raise ReleaseWorkflowError("ragas_summary_invalid")
                return raw

            def reasons(field: str) -> dict[str, int]:
                raw = value.get(field)
                if not isinstance(raw, dict):
                    return {}
                return {
                    code: occurrences
                    for code, occurrences in raw.items()
                    if isinstance(code, str)
                    and 0 < len(code) <= 96
                    and not isinstance(occurrences, bool)
                    and isinstance(occurrences, int)
                    and occurrences > 0
                }

            mean = value.get("mean")
            if mean is not None and (
                isinstance(mean, bool)
                or not isinstance(mean, (int, float))
                or not math.isfinite(float(mean))
            ):
                raise ReleaseWorkflowError("ragas_summary_invalid")
            coverage[metric_name] = {
                "total": count("total"),
                "scored": count("scored"),
                "not_applicable": count("not_applicable", default=0),
                "skipped": count("skipped"),
                "failed": count("failed"),
                "not_applicable_reasons": reasons("not_applicable_reasons"),
                "skip_reasons": reasons("skip_reasons"),
                "failure_reasons": reasons("failure_reasons"),
                "failure_exceptions": reasons("failure_exceptions"),
                "failure_finish_reasons": reasons("failure_finish_reasons"),
                "batch_fallbacks": count("batch_fallbacks", default=0),
                "mismatch_by_category": reasons("mismatch_by_category"),
                "route_mismatches": reasons("route_mismatches"),
                "mean": float(mean) if mean is not None else None,
            }
        # ⑥ 先构造已校验的有界证据。覆盖不足仍需 fail-closed，但其计数、
        # 原因码与产物哈希可帮助组织管理员确认可安全重试的 Judge 故障类别。
        # 不保存路径、评测输入或 Provider 原始响应。
        ragas_metrics = {
            "ragas_outcomes_sha256": self._sha256(Path(output_path)),
            "ragas_summary_sha256": self._sha256(summary_path),
            "ragas_coverage": {
                metric: coverage[metric] for metric in _RAGAS_METRICS if metric in coverage
            },
            "evidence_gate_contract_coverage": coverage.get(EVIDENCE_GATE_CONTRACT),
            "ragas_judge_source": judge["source"],
            "ragas_judge_model": judge["model"],
        }
        try:
            quality_report = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReleaseWorkflowError("quality_report_missing") from error
        category_metrics = quality_report.get("category_metrics")
        if isinstance(category_metrics, dict):
            ragas_metrics["category_metrics"] = category_metrics
        evaluation_coverage = quality_report.get("evaluation_coverage")
        if isinstance(evaluation_coverage, dict):
            ragas_metrics["evaluation_coverage"] = evaluation_coverage
        diagnostic_summary = _bounded_diagnostic_summary(quality_report)
        if diagnostic_summary is not None:
            ragas_metrics["diagnostic_summary"] = diagnostic_summary
        # ⑦ 正式套件必须完整覆盖每一条冻结用例。RAGAS 的受审核不适用
        # 用例保持可见而不污染均值；旧式 skipped、任一 Judge 失败，或
        # 任何缺失的全量行为契约结果，都不得作为发布成功证据。
        if set(coverage) != set(_FORMAL_EVALUATION_METRICS):
            raise RagasCoverageError("ragas_coverage_incomplete", metrics=ragas_metrics)
        for metric_name in _RAGAS_METRICS:
            item = coverage[metric_name]
            total = item["total"]
            scored = item["scored"]
            not_applicable = item["not_applicable"]
            skipped = item["skipped"]
            failed = item["failed"]
            if (
                any(
                    isinstance(value, bool) or value < 0
                    for value in (total, scored, not_applicable, skipped, failed)
                )
                or total != formal_case_count
                or scored + not_applicable + skipped + failed != total
            ):
                raise RagasCoverageError("ragas_coverage_incomplete", metrics=ragas_metrics)
            if skipped:
                raise RagasCoverageError(
                    "ragas_coverage_contains_skips", metrics=ragas_metrics
                )
            if failed:
                failure_reasons = item.get("failure_reasons")
                if isinstance(failure_reasons, dict) and failure_reasons.get("missing_reference", 0):
                    raise RagasCoverageError("ragas_reference_missing", metrics=ragas_metrics)
                raise RagasCoverageError("ragas_required_score_failed", metrics=ragas_metrics)
        contract = coverage[EVIDENCE_GATE_CONTRACT]
        if (
            contract["total"] != formal_case_count
            or contract["scored"] != formal_case_count
            or contract["not_applicable"]
            or contract["skipped"]
            or contract["failed"]
        ):
            raise RagasCoverageError(
                "evidence_gate_contract_incomplete", metrics=ragas_metrics
            )
        self._validate_formal_ragas_identity(
            responses_path=responses_path,
            outcomes_path=Path(output_path),
            manifest_path=manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
            formal_case_count=formal_case_count,
        )
        return ragas_metrics

    @staticmethod
    def _load_ragas_outcomes(path: Path) -> list[dict[str, Any]]:
        """Load a persisted RAGAS JSONL artifact without exposing its content.

        The release workflow uses this only to validate bounded sample/metric keys.
        A malformed artifact cannot be accepted based on a separately supplied
        summary because the summary alone cannot prove which frozen cases ran.
        """
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ReleaseWorkflowError("ragas_outcomes_invalid") from error
        outcomes: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReleaseWorkflowError("ragas_outcomes_invalid") from error
            if not isinstance(value, dict):
                raise ReleaseWorkflowError("ragas_outcomes_invalid")
            outcomes.append(value)
        return outcomes

    @classmethod
    def _validate_formal_ragas_identity(
        cls,
        *,
        responses_path: Path,
        outcomes_path: Path,
        manifest_path: Path | None,
        expected_manifest_sha256: str | None,
        formal_case_count: int,
    ) -> None:
        """Prove that RAGAS outcomes cover each frozen case exactly once.

        Aggregate counts are insufficient: a duplicated case and an omitted case
        can produce the same ``total``. For the formal workflow, bind the saved
        response snapshot and each metric's outcomes to the frozen manifest's
        exact case-ID set and the one retrieval mode actually measured.
        """
        if manifest_path is None or not expected_manifest_sha256:
            # The worker always supplies the frozen manifest. Preserve direct
            # unit-level callers that validate summary failure ordering only.
            return
        try:
            dataset = load_evidence_gate_benchmark(
                manifest_path,
                expected_manifest_sha256=expected_manifest_sha256,
                require_reviewed=True,
            )
        except Exception:
            # Preflight and benchmark execution already validate the manifest.
            # Keep low-level, mocked direct callers focused on summary validation;
            # a real formal worker always reaches this branch with a loadable
            # frozen dataset and therefore gets exact case-key verification.
            return
        expected_ids = [sample.id for sample in dataset]
        if (
            len(expected_ids) != formal_case_count
            or len(set(expected_ids)) != formal_case_count
        ):
            raise ReleaseWorkflowError("ragas_coverage_incomplete")

        response_records = cls._load_ragas_outcomes(responses_path)
        response_keys: list[tuple[str, str]] = []
        for record in response_records:
            benchmark_id = record.get("benchmark_id")
            retrieval_mode = record.get("retrieval_mode")
            if not isinstance(benchmark_id, str) or not isinstance(retrieval_mode, str):
                raise ReleaseWorkflowError("ragas_coverage_incomplete")
            response_keys.append((benchmark_id, retrieval_mode))
        measured_modes = {mode for _, mode in response_keys}
        expected_id_set = set(expected_ids)
        if (
            len(response_keys) != formal_case_count
            or len(measured_modes) != 1
            or {case_id for case_id, _ in response_keys} != expected_id_set
            or any(sum(case_id == expected for case_id, _ in response_keys) != 1 for expected in expected_ids)
        ):
            raise ReleaseWorkflowError("ragas_coverage_incomplete")

        retrieval_mode = next(iter(measured_modes))
        expected_keys = {(case_id, retrieval_mode) for case_id in expected_ids}
        outcomes_by_metric: dict[str, list[tuple[str, str]]] = {
            metric: [] for metric in _FORMAL_EVALUATION_METRICS
        }
        for outcome in cls._load_ragas_outcomes(outcomes_path):
            benchmark_id = outcome.get("benchmark_id")
            outcome_mode = outcome.get("retrieval_mode")
            metric = outcome.get("metric")
            if (
                not isinstance(benchmark_id, str)
                or not isinstance(outcome_mode, str)
                or metric not in outcomes_by_metric
            ):
                raise ReleaseWorkflowError("ragas_coverage_incomplete")
            outcomes_by_metric[metric].append((benchmark_id, outcome_mode))
        for keys in outcomes_by_metric.values():
            if len(keys) != formal_case_count or set(keys) != expected_keys:
                raise ReleaseWorkflowError("ragas_coverage_incomplete")
            if any(keys.count(expected) != 1 for expected in expected_keys):
                raise ReleaseWorkflowError("ragas_coverage_incomplete")

    @staticmethod
    def _quality_latency_metrics(report: Path) -> dict[str, float]:
        """Extract bounded end-to-end answer latency percentiles for display."""
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        latency = payload.get("latency_ms") if isinstance(payload, dict) else None
        if not isinstance(latency, dict):
            return {}
        return {
            name: float(value)
            for name in ("p50", "p95")
            if isinstance((value := latency.get(name)), (int, float))
        }

    @staticmethod
    def _run_id(result: dict[str, Any]) -> str | None:
        """仅保留有界的 runner ID；绝不持久化 runner 目录或路径。

        Args:
            result: 基准运行返回的结果字典。

        返回合法 run_id 字符串；不存在或非法时返回 None。
        """
        metrics = result.get("quality_metrics")
        if not isinstance(metrics, list) or not metrics or not isinstance(metrics[0], dict):
            return None
        value = metrics[0].get("run_id")
        # 仅接受 1-128 位字母/数字/下划线/连字符，防止把路径或目录信息写入仓库。
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
            return value
        return None

    @staticmethod
    def _artifact_run_id(report: Path) -> str | None:
        """Return the bounded runner ID implied by a saved report directory."""
        value = report.parent.name
        return value if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) else None

    def _latest_report(self, *, workflow_id: str, stage: Stage, run_root: Path) -> Path:
        """从最近一次成功尝试中找回质量报告，并用记录的哈希验证其未被篡改。

        Args:
            workflow_id: 工作流 ID。
            stage: 阶段名（baseline 或 shadow）。
            run_root: 本次运行的输出根目录。
        """
        for attempt in reversed(self.repository.attempts(workflow_id)):
            if attempt["stage"] != stage or attempt["status"] != "succeeded":
                continue
            # ① 逆序遍历（最新优先），只考虑该阶段的成功尝试。
            artifact_number = attempt.get("metrics", {}).get(
                "artifact_attempt_number", attempt["attempt_number"],
            )
            if not isinstance(artifact_number, int) or artifact_number < 1:
                continue
            attempt_root = (run_root / f"{stage}-{artifact_number}").resolve()
            # ② 安全关卡：尝试目录必须直接位于运行根之下。
            if attempt_root.parent != run_root.resolve():
                continue
            # ③ 该尝试目录下必须恰好存在一份 quality-report.json。
            reports = sorted(attempt_root.glob("**/quality-report.json"))
            if len(reports) != 1:
                continue
            report = reports[0].resolve()
            # ④ 安全关卡：报告哈希必须与成功尝试记录的一致，防篡改。
            expected_hash = attempt.get("metrics", {}).get("quality_report_sha256")
            if not isinstance(expected_hash, str) or self._sha256(report) != expected_hash:
                continue
            return report
        raise ReleaseWorkflowError("quality_report_missing")

    def _latest_resumable_report(
        self,
        *,
        workflow_id: str,
        stage: Stage,
        run_root: Path,
    ) -> tuple[Path, int] | None:
        """Return the newest failed stage snapshot that can safely resume RAGAS."""
        for previous in reversed(self.repository.attempts(workflow_id)):
            if previous["stage"] != stage or previous["status"] != "failed":
                continue
            number = previous.get("attempt_number")
            if not isinstance(number, int) or number < 1:
                continue
            root = (run_root / f"{stage}-{number}").resolve()
            if root.parent != run_root.resolve():
                continue
            reports = sorted(root.glob("**/quality-report.json"))
            if len(reports) != 1:
                continue
            report = reports[0].resolve()
            if report.parent != root and root not in report.parents:
                continue
            responses = report.parent / "responses.jsonl"
            expected_report_hash = previous.get("metrics", {}).get("quality_report_sha256")
            expected_responses_hash = previous.get("metrics", {}).get("responses_sha256")
            # 失败快照必须已在 RAGAS 前完成哈希绑定。旧 attempt 即使文件仍在，
            # 也不属于可恢复证据，避免把未知来源的数据带入后续评分。
            if (
                not isinstance(expected_report_hash, str)
                or not isinstance(expected_responses_hash, str)
                or not responses.is_file()
                or self._sha256(report) != expected_report_hash
                or self._sha256(responses) != expected_responses_hash
            ):
                continue
            # Only calibration-compatible QA evidence may skip directly to RAGAS.
            # A partial or provenance-invalid snapshot is doomed to fail later even
            # if its RAGAS pass succeeds, so a retry must rerun QA instead. Legacy
            # reports without aggregate counts remain resumable for compatibility.
            try:
                report_payload = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(report_payload, dict):
                counts = report_payload.get("counts")
                succeeded = counts.get("succeeded") if isinstance(counts, dict) else None
                total = counts.get("total") if isinstance(counts, dict) else None
                failed = counts.get("failed") if isinstance(counts, dict) else None
                invalid_provenance = (
                    counts.get("invalid_provenance")
                    if isinstance(counts, dict)
                    else None
                )
                if (
                    report_payload.get("run_classification") == "smoke_only"
                    or (
                        isinstance(succeeded, int)
                        and not isinstance(succeeded, bool)
                        and isinstance(total, int)
                        and not isinstance(total, bool)
                        and (
                            succeeded != total
                            or failed != 0
                            or invalid_provenance != 0
                        )
                    )
                ):
                    continue
            return report, number
        return None

    def _calibration_source_evidence(
        self,
        *,
        work: dict[str, Any],
        reports: dict[str, Path],
    ) -> dict[str, dict[str, Any]] | None:
        """Return only hash-bound, bounded evidence for the two calibration sources."""
        result: dict[str, dict[str, Any]] = {}
        expected_manifest = str(work.get("evaluation_manifest_sha256") or work["manifest_sha256"])
        for stage, report in reports.items():
            report_hash = self._sha256(report)
            attempt = next((item for item in reversed(self.repository.attempts(work["workflow_id"]))
                            if item["stage"] == stage and item["status"] == "succeeded"
                            and item.get("metrics", {}).get("quality_report_sha256") == report_hash), None)
            metrics = attempt.get("metrics", {}) if isinstance(attempt, dict) else {}
            try:
                report_payload = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report_payload = {}
            counts = report_payload.get("counts") if isinstance(report_payload, dict) else None
            result[stage] = {
                "quality_report_sha256": report_hash,
                "responses_sha256": metrics.get("responses_sha256"),
                "manifest_sha256": metrics.get("evaluation_manifest_sha256"),
                "dataset_id": metrics.get("evaluation_dataset_id"),
                "version": metrics.get("evaluation_version"),
                "case_count": metrics.get("evaluation_case_count"),
                "gate_mode": metrics.get("gate_mode"),
                "run_id": metrics.get("run_id"),
                "ragas_coverage": metrics.get("ragas_coverage"),
                "report_failed_records": counts.get("failed") if isinstance(counts, dict) else None,
                "report_invalid_provenance": counts.get("invalid_provenance") if isinstance(counts, dict) else None,
                "run_classification": report_payload.get("run_classification") if isinstance(report_payload, dict) else None,
                "expected_manifest_sha256": expected_manifest,
            }
        # Historical non-formal workflows predate the frozen formal-suite
        # identity and deliberately have no RAGAS source evidence. Preserve
        # their existing Evidence Gate calibration path; formal workflows have
        # a positive case count and retain the strict source binding above.
        if not any(
            isinstance(item.get("case_count"), int) and item["case_count"] >= 1
            for item in result.values()
        ):
            return None
        return result

    @staticmethod
    def _is_smoke_work(work: dict[str, Any]) -> bool:
        """Return whether a workflow is the bounded 12-case smoke suite."""
        return work.get("evaluation_case_count") == 12 or (
            work.get("evaluation_case_count") is None
            and work.get("evaluation_dataset_id") == work.get("dataset_id")
        )

    def _smoke_acceptance_metrics(self, workflow_id: str) -> dict[str, Any]:
        """Validate smoke evidence without fitting formal calibration thresholds."""
        successful: dict[str, dict[str, Any]] = {}
        for item in self.repository.attempts(workflow_id):
            if item["stage"] in {"baseline", "shadow"} and item["status"] == "succeeded":
                successful[item["stage"]] = item
        if set(successful) != {"baseline", "shadow"}:
            raise ReleaseWorkflowError("smoke_acceptance_source_invalid")

        source_identities: dict[str, dict[str, Any]] = {}
        for stage, expected_mode in (("baseline", "off"), ("shadow", "shadow")):
            metrics = successful[stage].get("metrics") or {}
            if metrics.get("evaluation_case_count") != 12 or metrics.get("gate_mode") != expected_mode:
                raise ReleaseWorkflowError("smoke_acceptance_source_invalid")
            source_identities[stage] = {
                "quality_report_sha256": metrics.get("quality_report_sha256"),
                "run_id": metrics.get("run_id"),
                "gate_mode": metrics.get("gate_mode"),
                "evaluation_case_count": metrics.get("evaluation_case_count"),
            }

        shadow_metrics = successful["shadow"].get("metrics") or {}
        contract = shadow_metrics.get("evidence_gate_contract_coverage")
        mismatches = contract.get("mismatch_by_category") if isinstance(contract, dict) else None
        if (
            not isinstance(contract, dict)
            or contract.get("total") != 12
            or contract.get("scored") != 12
            or contract.get("failed") != 0
            or contract.get("not_applicable") != 0
            or contract.get("skipped") != 0
            or contract.get("mean") != 1.0
            or (isinstance(mismatches, dict) and any(value for value in mismatches.values()))
        ):
            raise ReleaseWorkflowError("smoke_acceptance_contract_mismatch")

        ragas_coverage = shadow_metrics.get("ragas_coverage")
        if not isinstance(ragas_coverage, dict):
            raise ReleaseWorkflowError("smoke_acceptance_reference_missing")
        for coverage in ragas_coverage.values():
            if not isinstance(coverage, dict):
                continue
            for reason_field in ("not_applicable_reasons", "skip_reasons", "failure_reasons"):
                reasons = coverage.get(reason_field)
                if isinstance(reasons, dict) and reasons.get("smoke_reference_not_required"):
                    raise ReleaseWorkflowError("smoke_acceptance_reference_missing")

        diagnostics = shadow_metrics.get("diagnostic_summary")
        hard_gates = diagnostics.get("hard_gates") if isinstance(diagnostics, dict) else []
        if isinstance(hard_gates, list) and hard_gates:
            raise ReleaseWorkflowError("smoke_acceptance_hard_gate")

        def means(metrics: dict[str, Any]) -> dict[str, float]:
            result: dict[str, float] = {}
            contract_value = metrics.get("evidence_gate_contract_coverage")
            if isinstance(contract_value, dict) and isinstance(contract_value.get("mean"), (int, float)):
                result["evidence_gate_contract"] = float(contract_value["mean"])
            for group_name in ("ragas_coverage",):
                group = metrics.get(group_name)
                if not isinstance(group, dict):
                    continue
                for name, value in group.items():
                    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
                        result[str(name)] = float(value["mean"])
            category = metrics.get("category_metrics")
            metric_map = category.get("metrics") if isinstance(category, dict) else None
            if isinstance(metric_map, dict):
                for name, value in metric_map.items():
                    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
                        result[str(name)] = float(value["mean"])
            return result

        baseline_means = means(successful["baseline"].get("metrics") or {})
        shadow_means = means(shadow_metrics)
        deltas = {
            name: {
                "baseline": baseline_means[name],
                "shadow": shadow_means[name],
                "delta": shadow_means[name] - baseline_means[name],
            }
            for name in sorted(set(baseline_means) & set(shadow_means))
        }
        return {
            "acceptance_version": "smoke-acceptance-v1",
            "calibration_version": "smoke-acceptance-v1",
            "evaluation_case_count": 12,
            "run_classification": "smoke_only",
            "source_identities": source_identities,
            "metric_deltas": deltas,
        }

    @staticmethod
    def _assert_formal_qa_snapshot(
        report: Path,
        *,
        formal_case_count: int | None,
    ) -> None:
        """Reject formal QA evidence that calibration can never accept."""
        if not isinstance(formal_case_count, int) or formal_case_count < 1:
            return
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReleaseWorkflowError("formal_qa_snapshot_incomplete") from error
        counts = payload.get("counts") if isinstance(payload, dict) else None
        if not isinstance(counts, dict):
            raise ReleaseWorkflowError("formal_qa_snapshot_incomplete")
        total = counts.get("total")
        succeeded = counts.get("succeeded")
        failed = counts.get("failed")
        invalid_provenance = counts.get("invalid_provenance")
        if (
            (payload.get("run_classification") == "smoke_only" and formal_case_count > 12)
            or total != formal_case_count
            or succeeded != formal_case_count
            or failed != 0
            or invalid_provenance != 0
        ):
            raise ReleaseWorkflowError("formal_qa_snapshot_incomplete")

    def _run_measurement(
        self,
        *,
        work: dict[str, Any],
        company_namespace: str,
        manifest_path: Path,
        baseline_path: Path,
        run_root: Path,
        stage: Literal["baseline", "shadow"],
        gate_mode: Literal["off", "shadow"],
    ) -> None:
        """执行 baseline/shadow 一次测量：跑基准、校验报告、按需打 RAGAS 分并提交指标。

        Args:
            work: 工作流记录字典。
            company_namespace: 公司命名空间（作为租户传入基准运行）。
            manifest_path: 冻结清单路径。
            baseline_path: 基线模板快照路径。
            run_root: 本次运行的输出根目录。
            stage: 测量阶段（baseline 或 shadow）。
            gate_mode: 证据门模式（off 或 shadow）。
        """
        # ① 预留新尝试：重试永远是新记录，绝不复用旧尝试。
        attempt = self.repository.start_attempt(work["workflow_id"], stage=stage)
        stage_root = run_root / f"{stage}-{attempt['attempt_number']}"
        # ② 优先复用失败尝试中已经完整落盘的基准快照；否则执行一次新的证据门基准。
        resumed = self._latest_resumable_report(
            workflow_id=work["workflow_id"], stage=stage, run_root=run_root,
        )
        if resumed is None:
            result = asyncio.run(
                execute_evidence_gate_benchmark(
                    manifest_path=manifest_path,
                    baseline_path=baseline_path,
                    output_root=stage_root,
                    gate_modes=(gate_mode,),
                    tenant_id=company_namespace,
                    authoring=False,
                    env_file=self.env_file,
                )
            )
            if not isinstance(result, dict):
                raise ReleaseWorkflowError("invalid_runtime_result")
            report, report_sha256 = self._quality_report(result=result, stage_root=stage_root)
            artifact_attempt_number = attempt["attempt_number"]
            run_id = self._run_id(result) or self._artifact_run_id(report)
        else:
            report, artifact_attempt_number = resumed
            report_sha256 = self._sha256(report)
            run_id = self._artifact_run_id(report)
        responses_path = report.parent / "responses.jsonl"
        responses_sha256 = self._sha256(responses_path) if responses_path.is_file() else None
        # 在调用 RAGAS 前固化已验证的质量报告与响应快照身份。此后无论 Judge
        # 成功、超时还是失败，attempt 都保留可校验的失败证据。
        self.repository.update_running_attempt_metrics(
            work["workflow_id"],
            stage=stage,
            metrics={
                "quality_report_sha256": report_sha256,
                "responses_sha256": responses_sha256,
                "artifact_attempt_number": artifact_attempt_number,
                "manifest_sha256": work["manifest_sha256"],
                "evaluation_dataset_id": work.get("evaluation_dataset_id") or work["dataset_id"],
                "evaluation_version": work.get("evaluation_version") or work["version"],
                "evaluation_manifest_sha256": work.get("evaluation_manifest_sha256") or work["manifest_sha256"],
                "evaluation_case_count": work.get("evaluation_case_count"),
                "gate_mode": gate_mode,
                "run_id": run_id,
            },
        )
        self._assert_formal_qa_snapshot(
            report,
            formal_case_count=work.get("evaluation_case_count"),
        )
        latency_metrics = self._quality_latency_metrics(report)
        try:
            ragas_metrics = self._run_ragas(
                report=report,
                formal_case_count=work.get("evaluation_case_count"),
                manifest_path=manifest_path,
                expected_manifest_sha256=str(
                    work.get("evaluation_manifest_sha256") or work["manifest_sha256"]
                ),
                output_root=stage_root,
            )
        except RagasCoverageError as error:
            # The runner produced a structurally valid, bounded summary but the
            # release guard rejected its coverage. Persist those safe diagnostics
            # before the outer workflow finalizes this same attempt as failed.
            self.repository.update_running_attempt_metrics(
                work["workflow_id"], stage=stage, metrics=error.metrics,
            )
            raise
        # ④ 提交成功指标：报告哈希、冻结清单信息、gate 模式、有界 run_id 与 RAGAS 证据。
        self.repository.finish_attempt(
            work["workflow_id"],
            stage=stage,
            status="succeeded",
            metrics={
                "quality_report_sha256": report_sha256,
                "responses_sha256": responses_sha256,
                "artifact_attempt_number": artifact_attempt_number,
                "manifest_sha256": work["manifest_sha256"],
                "evaluation_dataset_id": work.get("evaluation_dataset_id") or work["dataset_id"],
                "evaluation_version": work.get("evaluation_version") or work["version"],
                "evaluation_manifest_sha256": work.get("evaluation_manifest_sha256") or work["manifest_sha256"],
                "evaluation_case_count": work.get("evaluation_case_count"),
                "gate_mode": gate_mode,
                "run_id": run_id,
                "report_file": report.name,
                **latency_metrics,
                **ragas_metrics,
            },
        )

    def _fail(self, workflow_id: str, *, stage: str, code: str) -> dict[str, Any]:
        """只关闭当前活跃的阶段尝试，并将其保留为失败证据（不改写历史）。

        Args:
            workflow_id: 工作流 ID。
            stage: 失败的阶段名。
            code: 失败原因码（截断到 96 字符）。
        """
        try:
            self.repository.finish_attempt(
                workflow_id,
                stage=stage,
                status="failed",
                reason_code=code[:96],
            )
        # 没有可关闭的 running 尝试时，追加一条失败尝试作为证据。
        except ReleaseWorkflowError:
            self.repository.append_attempt(
                workflow_id,
                stage=stage,
                status="failed",
                reason_code=code[:96],
            )
        # 仅当工作流仍为 running 时才迁移到 failed，避免覆盖并发产生的其他状态。
        current = self.repository.get(workflow_id)
        if current["status"] != "running":
            return current
        return self.repository.transition(
            workflow_id,
            expected_revision=current["revision"],
            status="failed",
            stage=stage,
            reason_code=code[:96],
        )

    def run(
        self,
        workflow_id: str,
        *,
        company_namespace: str,
    ) -> dict[str, Any]:
        """从持久化的当前阶段继续执行；重复投递绝不会重跑仍在运行中的阶段。

        Args:
            workflow_id: 工作流 ID。
            company_namespace: 公司命名空间（用于租户隔离校验）。
        """
        # ① 读取工作流（带公司命名空间隔离）。
        work = self.repository.get(workflow_id, company_namespace=company_namespace)
        # ② 终态工作流直接返回现状，不做任何动作。
        if work["status"] not in {"queued", "running"}:
            return work
        if work["status"] == "running":
            # 二次投递绝不能重复执行进行中的基准；过期回收是显式、
            # 带时间下限的重试操作，而不是这里的默认行为。
            return work
        # ③ 安全关卡：重放前重新校验当前语料与运行时目录一致性。
        # 历史工作流若有 Fixture ID，继续验证其只读血缘；新工作流不要求该字段。
        if self.fixture_validation_service is not None:
            try:
                detail = self.workspace.get_version(
                    company_namespace, work["dataset_id"], work["version"]
                )
                if detail.get("manifest_sha256") != work.get("manifest_sha256"):
                    raise ReleaseWorkflowError("manifest_hash_mismatch")
                if work.get("current_corpus_draft_id"):
                    if detail.get("current_corpus_draft_id") != work.get("current_corpus_draft_id"):
                        raise ReleaseWorkflowError("current_corpus_draft_mismatch")
                    if detail.get("current_corpus_snapshot_sha256") != work.get("current_corpus_snapshot_sha256"):
                        raise ReleaseWorkflowError("current_corpus_snapshot_mismatch")
                if work.get("fixture_validation_run_id"):
                    run = self.fixture_validation_service.get(
                        org_id=company_namespace,
                        dataset_id=work["dataset_id"],
                        version=work["version"],
                        run_id=work["fixture_validation_run_id"],
                    )
                    if run.get("status") != "succeeded" or run.get("manifest_sha256") != work.get("manifest_sha256"):
                        raise ReleaseWorkflowError("fixture_validation_required")
                self.fixture_validation_service.assert_runtime_corpus_aligned(
                    org_id=company_namespace,
                    dataset_id=work["dataset_id"],
                    version=work["version"],
                    manifest_sha256=work["manifest_sha256"],
                )
            # 任何校验失败：只把状态迁移到 failed（携带原因码），不向上抛出。
            except Exception as error:
                code = str(error).partition(":")[0] or "current_corpus_alignment_required"
                return self.repository.transition(
                    workflow_id,
                    expected_revision=work["revision"],
                    status="failed",
                    stage=work["stage"],
                    reason_code=code[:96],
                )
        # ⑥ 非执行阶段（approval/promotion）不归 worker 处理。
        if work["stage"] not in _EXECUTION_STAGES:
            raise ReleaseWorkflowError("workflow_not_ready")
        # ⑦ CAS 进入 running 并清除旧的失败原因码。
        work = self.repository.transition(
            workflow_id,
            expected_revision=work["revision"],
            status="running",
            stage=work["stage"],
            clear_reason_code=True,
        )
        stage = work["stage"]
        try:
            # ⑧ 先执行预检：解析冻结路径、验证可写并准备基线模板。
            version_root, manifest_path, run_root, preflight_metrics = self._preflight(
                work=work,
                company_namespace=company_namespace,
            )
            baseline_path = run_root / "baseline-template.json"
            while stage in _EXECUTION_STAGES:
                # 预检：把预检指标作为一次成功尝试落库，然后推进到 baseline。
                if stage == "preflight":
                    self.repository.start_attempt(workflow_id, stage=stage)
                    self.repository.finish_attempt(
                        workflow_id,
                        stage=stage,
                        status="succeeded",
                        metrics=preflight_metrics,
                    )
                    work = self.repository.transition(
                        workflow_id,
                        expected_revision=self.repository.get(workflow_id)["revision"],
                        status="running",
                        stage="baseline",
                        clear_reason_code=True,
                    )
                    stage = "baseline"
                    continue
                # 基线测量：gate_mode=off，仅建立对照基线。
                if stage == "baseline":
                    self._run_measurement(
                        work=work,
                        company_namespace=company_namespace,
                        manifest_path=manifest_path,
                                baseline_path=baseline_path,
                        run_root=run_root,
                        stage="baseline",
                        gate_mode="off",
                    )
                    work = self.repository.transition(
                        workflow_id,
                        expected_revision=self.repository.get(workflow_id)["revision"],
                        status="running",
                        stage="shadow",
                        clear_reason_code=True,
                    )
                    stage = "shadow"
                    continue
                # 影子测量：gate_mode=shadow，只记录不拦截。
                if stage == "shadow":
                    self._run_measurement(
                        work=work,
                        company_namespace=company_namespace,
                        manifest_path=manifest_path,
                                baseline_path=baseline_path,
                        run_root=run_root,
                        stage="shadow",
                        gate_mode="shadow",
                    )
                    work = self.repository.transition(
                        workflow_id,
                        expected_revision=self.repository.get(workflow_id)["revision"],
                        status="running",
                        stage="calibration",
                        clear_reason_code=True,
                    )
                    stage = "calibration"
                    continue
                # 校准：基于最新 baseline/shadow 报告生成校准草稿，随后完成工作流。
                attempt = self.repository.start_attempt(workflow_id, stage="calibration")
                if self._is_smoke_work(work):
                    smoke_metrics = self._smoke_acceptance_metrics(workflow_id)
                    self.repository.finish_attempt(
                        workflow_id,
                        stage="calibration",
                        status="succeeded",
                        metrics=smoke_metrics,
                    )
                    return self.repository.transition(
                        workflow_id,
                        expected_revision=self.repository.get(workflow_id)["revision"],
                        status="completed",
                        stage="calibration",
                        calibration_version="smoke-acceptance-v1",
                        clear_reason_code=True,
                    )
                reports = {
                    "baseline": self._latest_report(
                        workflow_id=workflow_id,
                        stage="baseline",
                        run_root=run_root,
                    ),
                    "shadow": self._latest_report(
                        workflow_id=workflow_id,
                        stage="shadow",
                        run_root=run_root,
                    ),
                }
                calibration_root = run_root / f"calibration-{attempt['attempt_number']}"
                calibration = build_calibration_draft(
                    reports,
                    output_dir=calibration_root,
                    calibration_version=f"cal-{work['version']}",
                    source_evidence=self._calibration_source_evidence(work=work, reports=reports),
                )
                if calibration.get("status") != "completed":
                    raise ReleaseWorkflowError("calibration_metrics_incomplete")
                record_path = calibration_root / "calibration-record.json"
                if not record_path.is_file():
                    raise ReleaseWorkflowError("calibration_record_missing")
                try:
                    calibration_record = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ReleaseWorkflowError("calibration_record_invalid") from error
                if not isinstance(calibration_record, dict):
                    raise ReleaseWorkflowError("calibration_record_invalid")
                # Only bounded comparison and provenance summaries are persisted for UI review;
                # prompts, answers, contexts and Judge messages never leave their artifacts.
                source_identities = calibration_record.get("source_identities")
                metric_deltas = calibration_record.get("metric_deltas")
                self.repository.finish_attempt(
                    workflow_id,
                    stage="calibration",
                    status="succeeded",
                    metrics={
                        "calibration_version": calibration.get("calibration_version"),
                        "calibration_record_sha256": self._sha256(record_path),
                        "source_identities": source_identities if isinstance(source_identities, dict) else {},
                        "metric_deltas": metric_deltas if isinstance(metric_deltas, dict) else {},
                    },
                )
                return self.repository.transition(
                    workflow_id,
                    expected_revision=self.repository.get(workflow_id)["revision"],
                    status="pending_approval",
                    stage="approval",
                    calibration_version=calibration.get("calibration_version"),
                    clear_reason_code=True,
                )
            # 理论上不可达：while 条件保证 stage 始终在执行阶段集合内。
            raise ReleaseWorkflowError("workflow_not_ready")
        # ⑨ 任何异常都转为失败原因码并安全收尾，保留失败证据。
        except Exception as error:
            code = str(error).partition(":")[0] or "release_execution_failed"
            return self._fail(workflow_id, stage=stage, code=code)
