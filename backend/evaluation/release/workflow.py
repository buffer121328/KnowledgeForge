"""公司级证据发布工作流的持久化仓库，以及只读的证据门（Gate）配置读取。

经认证的公司命名空间仅是内部审计兼容值：绝不从 HTTP 请求接收；
本模块不会修改环境文件、Docker、文档、问答历史或知识图谱。
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import and_, insert, select, update
from sqlalchemy.exc import IntegrityError

from infrastructure.postgres.database import DatabaseService
from infrastructure.postgres.models import (
    evidence_gate_configuration,
    evidence_gate_configuration_history,
    evidence_release_approvals,
    evidence_release_attempts,
    evidence_release_workflows,
)

WorkflowStatus = Literal["queued", "running", "completed", "pending_approval", "approved", "rejected", "promoted", "failed"]  # 工作流状态：queued 排队 / running 运行中 / completed 已完成 / pending_approval 待审批 / approved 已批准 / rejected 已拒绝 / promoted 已晋升 / failed 失败
# 工作流阶段类型：preflight→baseline→shadow→calibration→approval→promotion；
# 校准前的历史阶段取值在已持久化的历史行中仍需保持可读。
Stage = Literal["preflight", "baseline", "shadow", "calibration", "approval", "promotion"]

_ORPHANED_RUNNING_AFTER_SECONDS = 60 * 60  # 判定 running 工作流为孤儿前需静默等待的最短秒数（1 小时，覆盖 worker 硬超时）


class ReleaseWorkflowError(RuntimeError):
    """供 API 层做稳定错误映射的工作流安全异常类型。"""


def _now() -> datetime:
    """返回当前 UTC 时间。"""
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    """把时间转换为 ISO 8601 字符串；value 为 None 时返回 None。

    Args:
        value: 待转换的时间，允许为空。
    """
    return value.isoformat() if value else None


def _safe_mapping(value: Any) -> dict[str, Any]:
    """仅当 value 是 dict 时原样返回，否则返回空 dict，避免脏 JSON 数据导致崩溃。

    Args:
        value: 任意输入（通常来自数据库的 JSON 字段）。
    """
    return value if isinstance(value, dict) else {}


def _workflow(row: Any) -> dict[str, Any]:
    """把一条工作流数据库行转换为对外暴露的字典（时间统一转 ISO 字符串）。

    Args:
        row: evidence_release_workflows 的查询结果行（mapping 或等价对象）。
    """
    data = dict(row)
    return {
        "workflow_id": data["id"], "dataset_id": data["dataset_id"], "version": data["version"],
        "manifest_sha256": data["manifest_sha256"],
        "current_corpus_draft_id": data.get("current_corpus_draft_id"),
        "current_corpus_snapshot_sha256": data.get("current_corpus_snapshot_sha256"),
        "fixture_validation_run_id": data.get("fixture_validation_run_id"),
        "evaluation_dataset_id": data.get("evaluation_dataset_id"),
        "evaluation_version": data.get("evaluation_version"),
        "evaluation_manifest_sha256": data.get("evaluation_manifest_sha256"),
        "evaluation_case_count": data.get("evaluation_case_count"),
        "status": data["status"], "stage": data["stage"],
        "revision": int(data["revision"]), "initiated_by": data["initiated_by"],
        "target_mode": data.get("target_mode"),
        "calibration_version": data.get("calibration_version"),
        "reason_code": data.get("reason_code"), "created_at": _iso(data.get("created_at")),
        "updated_at": _iso(data.get("updated_at")),
    }


def _attempt(row: Any) -> dict[str, Any]:
    """把一条阶段尝试数据库行转换为对外暴露的字典。

    Args:
        row: evidence_release_attempts 的查询结果行（mapping 或等价对象）。
    """
    data = dict(row)
    return {
        "attempt_id": data["id"], "stage": data["stage"], "attempt_number": int(data["attempt_number"]),
        "status": data["status"], "reason_code": data.get("reason_code"),
        "metrics": _safe_mapping(data.get("metrics")), "started_at": _iso(data.get("started_at")),
        "finished_at": _iso(data.get("finished_at")),
    }


class PostgreSQLReleaseWorkflowRepository:
    """工作流、不可变尝试记录、审批与全局 Gate 状态的唯一事实来源。"""

    def __init__(self, database: DatabaseService) -> None:
        """初始化仓库，仅持有数据库服务引用。

        Args:
            database: PostgreSQL 数据库服务，用于获取数据库会话。
        """
        self.database = database

    def create(
        self,
        *,
        company_namespace: str,
        dataset_id: str,
        version: str,
        manifest_sha256: str,
        initiated_by: str,
        current_corpus_draft_id: str | None = None,
        current_corpus_snapshot_sha256: str | None = None,
        fixture_validation_run_id: str | None = None,
        evaluation_dataset_id: str | None = None,
        evaluation_version: str | None = None,
        evaluation_manifest_sha256: str | None = None,
        evaluation_case_count: int | None = None,
    ) -> dict[str, Any]:
        """创建（或幂等返回已存在的）发布工作流记录，初始状态 queued、阶段 preflight。

        Args:
            company_namespace: 公司命名空间（内部审计兼容值，用于数据隔离）。
            dataset_id: 数据集标识。
            version: 数据集版本。
            manifest_sha256: 冻结清单的 SHA-256，发布期间不可变。
            initiated_by: 发起人标识。
            current_corpus_draft_id: 当前语料草稿 ID，可选。
            current_corpus_snapshot_sha256: 当前语料快照的 SHA-256，可选。
            fixture_validation_run_id: 已通过的固件校验运行 ID，可选。
            evaluation_dataset_id: 正式评测套件的数据集 ID，可选。
            evaluation_version: 正式评测套件的版本，可选。
            evaluation_manifest_sha256: 正式评测套件清单的 SHA-256，可选。
            evaluation_case_count: 正式评测用例数量，可选。
        """
        now = _now(); workflow_id = f"erw_{uuid.uuid4().hex}"
        with self.database.session() as session:
            # ① 幂等检查：同数据集同版本仍有排队/运行中的工作流时，直接返回既有记录。
            active = session.execute(select(evidence_release_workflows).where(
                evidence_release_workflows.c.company_namespace == company_namespace,
                evidence_release_workflows.c.dataset_id == dataset_id,
                evidence_release_workflows.c.version == version,
                evidence_release_workflows.c.status.in_(("queued", "running")),
            )).mappings().one_or_none()
            if active:
                return _workflow(active)
            # ② 插入新工作流：状态 queued、阶段 preflight、revision=1（乐观并发基线）。
            session.execute(insert(evidence_release_workflows).values(
                id=workflow_id, company_namespace=company_namespace, dataset_id=dataset_id, version=version,
                manifest_sha256=manifest_sha256,
                current_corpus_draft_id=current_corpus_draft_id,
                current_corpus_snapshot_sha256=current_corpus_snapshot_sha256,
                fixture_validation_run_id=fixture_validation_run_id,
                evaluation_dataset_id=evaluation_dataset_id,
                evaluation_version=evaluation_version,
                evaluation_manifest_sha256=evaluation_manifest_sha256,
                evaluation_case_count=evaluation_case_count,
                initiated_by=initiated_by, status="queued", stage="preflight",
                revision=1, target_mode=None, calibration_version=None, reason_code=None, created_at=now, updated_at=now,
            ))
            # ③ 重新读回完整记录并返回。
            row = session.execute(select(evidence_release_workflows).where(evidence_release_workflows.c.id == workflow_id)).mappings().one()
        return _workflow(row)

    def get(self, workflow_id: str, *, company_namespace: str | None = None) -> dict[str, Any]:
        """按 ID 读取单个工作流；不存在时抛 ReleaseWorkflowError("workflow_not_found")。

        Args:
            workflow_id: 工作流 ID。
            company_namespace: 若提供则限定公司命名空间（公司隔离校验）。
        """
        predicate = evidence_release_workflows.c.id == workflow_id
        # 显式提供命名空间时叠加过滤，保证不跨公司读取。
        if company_namespace is not None:
            predicate = and_(predicate, evidence_release_workflows.c.company_namespace == company_namespace)
        with self.database.session() as session:
            row = session.execute(select(evidence_release_workflows).where(predicate)).mappings().one_or_none()
        if not row: raise ReleaseWorkflowError("workflow_not_found")
        return _workflow(row)

    def list(self, *, company_namespace: str, limit: int = 20) -> list[dict[str, Any]]:
        """返回按更新时间倒序、数量受限的历史列表，绝不跨越公司范围。

        Args:
            company_namespace: 公司命名空间（隔离边界）。
            limit: 请求的数量上限，会被夹紧到 [1, 100]。
        """
        bounded_limit = max(1, min(int(limit), 100))  # 夹紧数量上限，防止无界查询
        with self.database.session() as session:
            rows = session.execute(
                select(evidence_release_workflows)
                .where(evidence_release_workflows.c.company_namespace == company_namespace)
                .order_by(
                    evidence_release_workflows.c.updated_at.desc(),
                    evidence_release_workflows.c.id.desc(),
                )
                .limit(bounded_limit)
            ).mappings().all()
        return [_workflow(row) for row in rows]

    def attempts(self, workflow_id: str) -> list[dict[str, Any]]:
        """返回指定工作流的全部阶段尝试，按开始时间升序排列（不可变的审计记录）。

        Args:
            workflow_id: 工作流 ID。
        """
        with self.database.session() as session:
            rows = session.execute(select(evidence_release_attempts).where(evidence_release_attempts.c.workflow_id == workflow_id).order_by(evidence_release_attempts.c.started_at)).mappings().all()
        return [_attempt(row) for row in rows]

    def transition(
        self,
        workflow_id: str,
        *,
        expected_revision: int,
        status: str,
        stage: str,
        reason_code: str | None = None,
        calibration_version: str | None = None,
        clear_reason_code: bool = False,
    ) -> dict[str, Any]:
        """以比较并交换（CAS）方式推进工作流状态，不覆盖已批准的不可变证据字段。

        Args:
            workflow_id: 工作流 ID。
            expected_revision: 调用方持有的期望修订号（乐观并发控制）。
            status: 目标状态。
            stage: 目标阶段。
            reason_code: 新失败原因码（截断到 96 字符），可选。
            calibration_version: 校准版本号，可选；提供时写入。
            clear_reason_code: 为 True 时显式清除既有失败原因码。
        """
        now = _now()
        # ① 组装新值：revision 递增；省略的可选字段保持原值，从而保留已批准的证据。
        values: dict[str, Any] = {
            "status": status,
            "stage": stage,
            "revision": expected_revision + 1,
            "updated_at": now,
        }
        # 新阶段可通过 clear_reason_code 显式清除旧失败原因，
        # 避免过期错误被当作当前发布证据展示。
        if clear_reason_code:
            values["reason_code"] = None
        elif reason_code is not None:
            values["reason_code"] = reason_code[:96]
        if calibration_version is not None:
            values["calibration_version"] = calibration_version
        with self.database.session() as session:
            # ② CAS：仅当当前 revision 等于期望值时才更新。
            result = session.execute(
                update(evidence_release_workflows)
                .where(
                    evidence_release_workflows.c.id == workflow_id,
                    evidence_release_workflows.c.revision == expected_revision,
                )
                .values(**values)
            )
            # ③ 影响行数不为 1：说明并发冲突，拒绝本次迁移。
            if result.rowcount != 1:
                raise ReleaseWorkflowError("workflow_revision_conflict")
            row = session.execute(
                select(evidence_release_workflows).where(
                    evidence_release_workflows.c.id == workflow_id
                )
            ).mappings().one()
        return _workflow(row)

    def start_attempt(self, workflow_id: str, *, stage: str) -> dict[str, Any]:
        """在任何副作用发生之前，预留一个"以新尝试重试"的阶段尝试（状态 running）。

        Args:
            workflow_id: 工作流 ID。
            stage: 要开始的阶段名。
        """
        return self.append_attempt(workflow_id, stage=stage, status="running")

    def finish_attempt(
        self, workflow_id: str, *, stage: str, status: str,
        reason_code: str | None = None, metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """在阶段产物校验成功/失败后，提交该阶段当前 running 的尝试（写入最终状态与指标）。

        Args:
            workflow_id: 工作流 ID。
            stage: 阶段名。
            status: 最终状态（如 succeeded/failed）。
            reason_code: 失败原因码（截断到 96 字符），可选。
            metrics: 有界指标字典，可选；提供时整体写入。
        """
        now = _now()
        with self.database.session() as session:
            # ① 定位该阶段最新一条 running 尝试；缺失说明状态已被破坏。
            row = session.execute(select(evidence_release_attempts).where(
                evidence_release_attempts.c.workflow_id == workflow_id,
                evidence_release_attempts.c.stage == stage,
                evidence_release_attempts.c.status == "running",
            ).order_by(evidence_release_attempts.c.attempt_number.desc()).limit(1)).mappings().one_or_none()
            if not row: raise ReleaseWorkflowError("running_attempt_not_found")
            # ② 仅更新状态、原因码、指标与结束时间，不改动开始时间等审计字段。
            values: dict[str, Any] = {"status": status, "finished_at": now}
            if reason_code is not None: values["reason_code"] = reason_code[:96]
            if metrics is not None: values["metrics"] = metrics
            session.execute(update(evidence_release_attempts).where(evidence_release_attempts.c.id == row["id"]).values(**values))
            updated = session.execute(select(evidence_release_attempts).where(evidence_release_attempts.c.id == row["id"])).mappings().one()
        return _attempt(updated)

    def update_running_attempt_metrics(
        self, workflow_id: str, *, stage: str, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Durably bind verified artifacts to the active attempt before a later step.

        This is deliberately limited to the newest ``running`` attempt.  It
        merges bounded metadata without changing its status, so a RAGAS timeout
        leaves a hash-bound failed snapshot that can be safely checked on retry.
        """
        with self.database.session() as session:
            row = session.execute(select(evidence_release_attempts).where(
                evidence_release_attempts.c.workflow_id == workflow_id,
                evidence_release_attempts.c.stage == stage,
                evidence_release_attempts.c.status == "running",
            ).order_by(evidence_release_attempts.c.attempt_number.desc()).limit(1)).mappings().one_or_none()
            if not row:
                raise ReleaseWorkflowError("running_attempt_not_found")
            existing = row["metrics"] if isinstance(row["metrics"], dict) else {}
            merged = {**existing, **metrics}
            session.execute(update(evidence_release_attempts).where(
                evidence_release_attempts.c.id == row["id"],
                evidence_release_attempts.c.status == "running",
            ).values(metrics=merged))
            updated = session.execute(select(evidence_release_attempts).where(
                evidence_release_attempts.c.id == row["id"]
            )).mappings().one()
        return _attempt(updated)

    def append_attempt(self, workflow_id: str, *, stage: str, status: str, reason_code: str | None = None, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        """追加一条新的阶段尝试记录；历史尝试不可变，重试总是新增而非改写。

        Args:
            workflow_id: 工作流 ID。
            stage: 阶段名。
            status: 尝试初始状态；running 表示进行中，其余视为已结束。
            reason_code: 失败原因码（截断到 96 字符），可选。
            metrics: 指标字典，可选；缺省为空 dict。
        """
        now = _now(); attempt_id = f"era_{uuid.uuid4().hex}"
        with self.database.session() as session:
            # ① 计算该阶段的下一个尝试序号（当前最大序号 + 1）。
            number = int(session.execute(select(evidence_release_attempts.c.attempt_number).where(
                evidence_release_attempts.c.workflow_id == workflow_id,
                evidence_release_attempts.c.stage == stage,
            ).order_by(evidence_release_attempts.c.attempt_number.desc()).limit(1)).scalar_one_or_none() or 0) + 1
            # ② 插入新尝试；非 running 状态立即记录结束时间。
            session.execute(insert(evidence_release_attempts).values(
                id=attempt_id,
                workflow_id=workflow_id,
                stage=stage,
                attempt_number=number,
                status=status,
                reason_code=reason_code[:96] if reason_code else None,
                metrics=metrics or {},
                started_at=now,
                finished_at=None if status == "running" else now,
            ))
            row = session.execute(select(evidence_release_attempts).where(evidence_release_attempts.c.id == attempt_id)).mappings().one()
        return _attempt(row)

    def recover_orphaned(
        self,
        workflow_id: str,
        *,
        expected_revision: int,
        minimum_running_seconds: int = _ORPHANED_RUNNING_AFTER_SECONDS,
    ) -> dict[str, Any]:
        """关闭遗留 running 尝试，之后以新的不可变尝试重试。

        仅当服务器侧更新时间已超过最短运行秒数（覆盖 worker 硬超时）时，才把
        running 工作流回收为 queued，并始终新增尝试而非改写旧尝试，
        避免与仍存活的 worker 竞态。

        Args:
            workflow_id: 工作流 ID。
            expected_revision: 期望修订号（乐观并发控制）。
            minimum_running_seconds: 判定孤儿前需静默的最短运行秒数（默认 1 小时）。
        """
        now = _now()
        with self.database.session() as session:
            # ① 读取工作流并做前置校验。
            work = session.execute(
                select(evidence_release_workflows).where(
                    evidence_release_workflows.c.id == workflow_id
                )
            ).mappings().one_or_none()
            if not work:
                raise ReleaseWorkflowError("workflow_not_found")
            # ② 修订号不一致：与其他执行者并发冲突，拒绝回收。
            if work["revision"] != expected_revision:
                raise ReleaseWorkflowError("workflow_revision_conflict")
            # ③ 只有 running 状态才可能存在孤儿尝试。
            if work["status"] != "running":
                raise ReleaseWorkflowError("workflow_not_ready")
            updated_at = work["updated_at"]
            # 数据库可能返回无时区时间，统一补 UTC 再比较。
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            # ④ 静默时间不足：可能仍有 worker 存活，拒绝回收。
            if now - updated_at < timedelta(seconds=max(0, minimum_running_seconds)):
                raise ReleaseWorkflowError("workflow_still_running")
            # ⑤ CAS 回收：running → queued，写入孤儿原因码并递增 revision。
            transitioned = session.execute(
                update(evidence_release_workflows)
                .where(
                    evidence_release_workflows.c.id == workflow_id,
                    evidence_release_workflows.c.revision == expected_revision,
                    evidence_release_workflows.c.status == "running",
                )
                .values(
                    status="queued",
                    stage=work["stage"],
                    revision=expected_revision + 1,
                    reason_code="orphaned_worker_attempt",
                    updated_at=now,
                )
            )
            if transitioned.rowcount != 1:
                raise ReleaseWorkflowError("workflow_revision_conflict")
            attempt = session.execute(
                select(evidence_release_attempts)
                .where(
                    evidence_release_attempts.c.workflow_id == workflow_id,
                    evidence_release_attempts.c.stage == work["stage"],
                    evidence_release_attempts.c.status == "running",
                )
                .order_by(evidence_release_attempts.c.attempt_number.desc())
                .limit(1)
            ).mappings().one_or_none()
            # ⑥ 把遗留的 running 尝试标记为 failed，保留为失败证据（只关闭，不改写内容）。
            if attempt:
                session.execute(
                    update(evidence_release_attempts)
                    .where(evidence_release_attempts.c.id == attempt["id"])
                    .values(
                        status="failed",
                        reason_code="orphaned_worker_attempt",
                        finished_at=now,
                    )
                )
            row = session.execute(
                select(evidence_release_workflows).where(
                    evidence_release_workflows.c.id == workflow_id
                )
            ).mappings().one()
        return _workflow(row)

    def review(
        self,
        workflow_id: str,
        *,
        expected_revision: int,
        reviewer_id: str,
        decision: Literal["approved", "rejected"],
        reason: str,
        target_mode: str | None = None,
    ) -> dict[str, Any]:
        """Persist an independent review and atomically advance the workflow."""
        if not reason.strip():
            raise ReleaseWorkflowError("release_review_reason_required")
        now = _now()
        with self.database.session() as session:
            work = session.execute(select(evidence_release_workflows).where(
                evidence_release_workflows.c.id == workflow_id
            )).mappings().one_or_none()
            if not work:
                raise ReleaseWorkflowError("workflow_not_found")
            if work["revision"] != expected_revision:
                raise ReleaseWorkflowError("workflow_revision_conflict")
            if work["status"] != "pending_approval":
                raise ReleaseWorkflowError("workflow_not_ready")
            if work["initiated_by"] == reviewer_id:
                raise ReleaseWorkflowError("release_self_approval_forbidden")
            if decision == "approved" and target_mode not in {"shadow", "enforce"}:
                raise ReleaseWorkflowError("gate_transition_invalid")
            approval_id = f"era_{uuid.uuid4().hex}"
            try:
                session.execute(insert(evidence_release_approvals).values(
                    id=approval_id, workflow_id=workflow_id, reviewer_id=reviewer_id,
                    decision=decision, reason=reason.strip()[:1000], created_at=now,
                ))
            except IntegrityError as error:
                raise ReleaseWorkflowError("release_already_reviewed") from error
            values = {
                "status": decision, "stage": "approval",
                "target_mode": target_mode if decision == "approved" else None,
                "revision": expected_revision + 1, "updated_at": now,
                "reason_code": None if decision == "approved" else "release_rejected",
            }
            changed = session.execute(update(evidence_release_workflows).where(
                evidence_release_workflows.c.id == workflow_id,
                evidence_release_workflows.c.revision == expected_revision,
                evidence_release_workflows.c.status == "pending_approval",
            ).values(**values))
            if changed.rowcount != 1:
                raise ReleaseWorkflowError("workflow_revision_conflict")
            row = session.execute(select(evidence_release_workflows).where(
                evidence_release_workflows.c.id == workflow_id
            )).mappings().one()
        return _workflow(row)

    def promote(
        self, workflow_id: str, *, expected_revision: int, actor_id: str
    ) -> dict[str, Any]:
        """Activate exactly the approved next Gate mode and retain history."""
        now = _now()
        with self.database.session() as session:
            work = session.execute(select(evidence_release_workflows).where(
                evidence_release_workflows.c.id == workflow_id
            )).mappings().one_or_none()
            if not work:
                raise ReleaseWorkflowError("workflow_not_found")
            if work["revision"] != expected_revision:
                raise ReleaseWorkflowError("workflow_revision_conflict")
            if work["status"] != "approved" or work.get("target_mode") not in {"shadow", "enforce"}:
                raise ReleaseWorkflowError("release_not_approved")
            # 防御性核验：即使数据库中的工作流行被异常写入为 approved，也必须有一条
            # 非发起人的批准记录才能实际激活全局 Gate。
            approval = session.execute(select(evidence_release_approvals).where(
                evidence_release_approvals.c.workflow_id == workflow_id,
                evidence_release_approvals.c.decision == "approved",
                evidence_release_approvals.c.reviewer_id != work["initiated_by"],
            )).mappings().one_or_none()
            if not approval:
                raise ReleaseWorkflowError("release_not_approved")
            current = session.execute(select(evidence_gate_configuration).where(
                evidence_gate_configuration.c.singleton == "company"
            )).mappings().one_or_none()
            mode = current["mode"] if current else "off"
            revision = int(current["revision"]) if current else 0
            target = work["target_mode"]
            if (mode, target) not in {("off", "shadow"), ("shadow", "enforce")}:
                raise ReleaseWorkflowError("gate_transition_invalid")
            if current:
                changed = session.execute(update(evidence_gate_configuration).where(
                    evidence_gate_configuration.c.singleton == "company",
                    evidence_gate_configuration.c.revision == revision,
                ).values(mode=target, revision=revision + 1,
                    calibration_version=work.get("calibration_version"), updated_by=actor_id, updated_at=now))
                if changed.rowcount != 1:
                    raise ReleaseWorkflowError("workflow_revision_conflict")
            else:
                session.execute(insert(evidence_gate_configuration).values(
                    singleton="company", mode=target, revision=1,
                    calibration_version=work.get("calibration_version"), updated_by=actor_id, updated_at=now))
            session.execute(insert(evidence_gate_configuration_history).values(
                id=f"egh_{uuid.uuid4().hex}", action="promote", previous_mode=mode, mode=target,
                configuration_revision=revision + 1, calibration_version=work.get("calibration_version"),
                workflow_id=workflow_id, actor_id=actor_id, created_at=now))
            changed = session.execute(update(evidence_release_workflows).where(
                evidence_release_workflows.c.id == workflow_id,
                evidence_release_workflows.c.revision == expected_revision,
                evidence_release_workflows.c.status == "approved",
            ).values(status="promoted", stage="promotion", revision=expected_revision + 1, updated_at=now))
            if changed.rowcount != 1:
                raise ReleaseWorkflowError("workflow_revision_conflict")
            row = session.execute(select(evidence_release_workflows).where(
                evidence_release_workflows.c.id == workflow_id
            )).mappings().one()
        return _workflow(row)

    def rollback(
        self, *, expected_revision: int, actor_id: str, reason: str
    ) -> dict[str, Any]:
        """Rollback one durable Gate revision to its immediately previous mode.

        The configuration history is the only rollback authority.  The method
        never changes a workflow or deployment configuration; it creates a new
        monotonic Gate revision and records which activation is being reversed.
        """
        if not reason.strip():
            raise ReleaseWorkflowError("gate_rollback_reason_required")
        now = _now()
        with self.database.session() as session:
            current = session.execute(select(evidence_gate_configuration).where(
                evidence_gate_configuration.c.singleton == "company"
            )).mappings().one_or_none()
            if not current:
                raise ReleaseWorkflowError("gate_no_prior_configuration")
            revision = int(current["revision"])
            if revision != expected_revision:
                raise ReleaseWorkflowError("gate_revision_conflict")
            latest = session.execute(select(evidence_gate_configuration_history).where(
                evidence_gate_configuration_history.c.configuration_revision == revision,
                evidence_gate_configuration_history.c.mode == current["mode"],
            ).order_by(evidence_gate_configuration_history.c.created_at.desc()).limit(1)).mappings().one_or_none()
            if not latest:
                raise ReleaseWorkflowError("gate_history_missing")
            target = latest["previous_mode"]
            # Restore the latest calibration identity associated with the durable
            # preceding mode.  ``off`` intentionally has no calibration draft.
            previous_calibration = None
            if target != "off":
                previous = session.execute(select(evidence_gate_configuration_history).where(
                    evidence_gate_configuration_history.c.configuration_revision < revision,
                    evidence_gate_configuration_history.c.mode == target,
                ).order_by(evidence_gate_configuration_history.c.configuration_revision.desc()).limit(1)).mappings().one_or_none()
                if not previous:
                    raise ReleaseWorkflowError("gate_history_missing")
                previous_calibration = previous["calibration_version"]
            changed = session.execute(update(evidence_gate_configuration).where(
                evidence_gate_configuration.c.singleton == "company",
                evidence_gate_configuration.c.revision == revision,
            ).values(
                mode=target,
                revision=revision + 1,
                calibration_version=previous_calibration,
                updated_by=actor_id,
                updated_at=now,
            ))
            if changed.rowcount != 1:
                raise ReleaseWorkflowError("gate_revision_conflict")
            session.execute(insert(evidence_gate_configuration_history).values(
                id=f"egh_{uuid.uuid4().hex}",
                action="rollback",
                previous_mode=current["mode"],
                mode=target,
                configuration_revision=revision + 1,
                calibration_version=previous_calibration,
                # Retain the workflow whose activation is being reversed.
                workflow_id=latest["workflow_id"],
                actor_id=actor_id,
                created_at=now,
            ))
        return self.gate()

    def gate(self) -> dict[str, Any]:
        """读取全局（公司级单例）证据门配置；无持久化行时返回环境默认值 mode=off。"""
        with self.database.session() as session:
            row = session.execute(select(evidence_gate_configuration).where(evidence_gate_configuration.c.singleton == "company")).mappings().one_or_none()
        # 无持久化配置行：返回环境默认（来源标记 environment_default）。
        if not row: return {"mode": "off", "revision": 0, "calibration_version": None, "source": "environment_default"}
        return {"mode": row["mode"], "revision": int(row["revision"]), "calibration_version": row["calibration_version"], "updated_at": _iso(row["updated_at"]), "updated_by": row["updated_by"], "source": "persistent"}
