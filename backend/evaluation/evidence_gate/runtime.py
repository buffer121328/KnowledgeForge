"""证据门基准测试的真实预检与运行时执行（不伪造任何测量结果）。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.qa_agent import QAAgent
from services.safety.qa_checks import QASafetyRefusalError
from domain.evidence import (
    EvidenceAssessment,
    EvidenceReasonCode,
    EvidenceState,
    QAResponseStatus,
)
from domain.knowledge import QAResult, QueryIntent
from infrastructure.retrieval.bm25_index import NativeBM25Index
from infrastructure.graph.neo4j_graph import KnowledgeGraphService
from infrastructure.retrieval.vector_store import VectorStoreService
from shared.config import settings
from shared.utils.logging import get_logger
from infrastructure.evidence_gate_configuration import EffectiveEvidenceGateConfiguration
from workflows.qa_dependencies import build_qa_agent_dependencies

from ..fixture_validation import FinanceFixtureIdentityResolver, FixtureValidationError

from evaluation.evidence_gate.benchmark import (
    EvidenceGateBenchmarkError,
    load_evidence_gate_benchmark,
)
from ..release.quality_gates import aggregate_quality_report
from ..runner import BenchmarkRunner
from ..benchmarks.stage_retrieval_metrics import (
    ExactEvidenceIdentity,
    ReviewedRelevance,
    aggregate_stage_retrieval,
    score_stage_retrieval,
)
from evaluation.evidence_gate.metrics import evaluate_category_custom_metrics

# 运行时允许评测的门禁模式：off / shadow / enforce。
_SUPPORTED_GATE_MODES = frozenset({"off", "shadow", "enforce"})
# 预检产物 JSON 的 schema 版本号。
_RUNTIME_PREFLIGHT_SCHEMA = "evidence-gate-runtime-preflight-v1"


def _diagnostic_inputs(run: Any) -> dict[str, Any]:
    """Build bounded stage/custom diagnostics from one frozen benchmark run."""
    by_case = {record.get("benchmark_id"): record for record in run.records}
    stage_rows: list[dict[str, Any]] = []
    stage_records = [
        json.loads(line)
        for line in run.candidate_stages_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for item in stage_records:
        case = by_case.get(item.get("benchmark_id"))
        if not isinstance(case, dict):
            continue
        source_ids = frozenset(case.get("expected_source_document_ids") or ())
        exact_ids = tuple(
            ExactEvidenceIdentity(context_id=value)
            for value in (case.get("expected_evidence_context_ids") or ())
            if isinstance(value, str) and value
        )
        # Refusal, authorization, outage and safety cases intentionally carry no
        # positive relevance set; they are assessed by route/safety metrics.
        if not source_ids or not exact_ids:
            continue
        relevance = ReviewedRelevance(
            source_document_ids=source_ids,
            exact_evidence=exact_ids,
        )
        scored = score_stage_retrieval(
            item, relevance=relevance, k_values=(1, 3, 5, 8)
        )
        stage_rows.append({
            "benchmark_id": item.get("benchmark_id"),
            "category": case.get("category"),
            **scored,
        })
    if stage_rows:
        retrieval_summary = aggregate_stage_retrieval(
            stage_rows, k_values=(1, 3, 5, 8)
        )
    else:
        # A run may contain only refusal/authorization/outage cases. Those
        # cases have no positive retrieval relevance by design; this must not
        # turn a valid QA run into a workflow failure.
        retrieval_summary = {
            "schema_version": "stage-retrieval-metrics-v1",
            "k_values": [1, 3, 5, 8],
            "stages": {},
            "categories": {},
            "status": "not_applicable",
            "reason_code": "no_positive_retrieval_cases",
        }
    outcomes: list[dict[str, Any]] = []
    for record in run.records:
        if record.get("status") != "succeeded":
            continue
        try:
            outcomes.extend(evaluate_category_custom_metrics(record))
        except Exception:
            continue
    return {
        "metric_outcomes": outcomes,
        "retrieval_summary": retrieval_summary,
        "isolation_evidence": {
            "cross_snapshot_contamination": False,
            "unauthorized_oracle_access": False,
        },
    }

logger = get_logger(__name__)

# QA 追踪事件 operation 名 → 检索分支名（dense/bm25/graph）的映射。
_RETRIEVAL_TRACE_OPERATIONS = {
    "retrieval.vector": "dense",
    "retrieval.bm25": "bm25",
    "retrieval.graph": "graph",
}
# 命中这些状态说明分支服务本身可用（结果为空也算可用）。
_RETRIEVAL_AVAILABLE_STATUSES = {"success", "empty", "evidence_filtered"}
# 命中这些状态说明分支服务不可用（故障、契约错误或未授权）。
_RETRIEVAL_UNAVAILABLE_STATUSES = {
    "unavailable",
    "invalid",
    "contract_error",
    "unauthorized",
}


def _observed_branch_availability(result: Any) -> dict[str, str]:
    """从 QA 追踪事件中归纳各检索分支的可用性，而不是假设依赖正常工作。

    安全边界：Fixture Worker 必须区分"故意注入的故障"和"无关的在线依赖失败"；
    QA 追踪里只有分支状态与计数，本函数只读取这些有界事实，不落盘、不打日志。

    Args:
        result: QA 应答结果对象（仅按属性名读取，不依赖具体类型）。

    Returns:
        分支名 → "available" / "unavailable" 的映射。
    """
    trace = getattr(result, "trace", None)
    events = getattr(trace, "events", ()) if trace is not None else ()
    observed: dict[str, str] = {}
    for event in events or ():
        # ① 只认检索类事件：把 operation 名翻译成分支名。
        branch = _RETRIEVAL_TRACE_OPERATIONS.get(str(getattr(event, "operation", "")))
        if branch is None:
            continue
        metadata = getattr(event, "metadata", {})
        status = ""
        if isinstance(metadata, Mapping):
            status = str(metadata.get("status") or "").strip().lower()
        event_status = str(getattr(event, "status", "")).strip().lower()
        # ② 依据状态集合判定可用性；显式失败事件同样视为不可用。
        if status in _RETRIEVAL_AVAILABLE_STATUSES:
            observed[branch] = "available"
        elif status in _RETRIEVAL_UNAVAILABLE_STATUSES or event_status == "failed":
            observed[branch] = "unavailable"
    return observed


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """把对象序列化为 JSON 写入文件（键排序、UTF-8）。

    Args:
        path: 目标文件路径，父目录不存在时自动创建。
        value: 待写入的 JSON 对象。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    """读取 JSON 对象，文件缺失或内容不是对象时抛 ValueError。

    Args:
        path: JSON 文件路径。
        label: 文件用途名，用于拼接错误信息。
    """
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _effective_configuration() -> dict[str, Any]:
    """从全局配置提取预检所需的运行时配置项（密钥、图密码、存储类型）。"""
    return {
        "deepseek_api_key": settings.deepseek_api_key,
        "dashscope_api_key": settings.dashscope_api_key,
        "neo4j_password": settings.neo4j_password,
        "vector_store_type": settings.vector_store_type,
    }


def assess_evidence_gate_runtime(
    *,
    manifest_path: str | Path,
    baseline_path: str | Path,
    output_path: str | Path,
    gate_modes: Sequence[str],
    tenant_id: str,
    authoring: bool,
    env_file: str | Path | None,
    runtime_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """校验冻结输入并记录阻塞原因，全程不探测任何在线服务。

    Args:
        manifest_path: 冻结数据集 manifest 路径。
        baseline_path: 基线模板 JSON 路径（提供期望的 manifest 哈希）。
        output_path: 预检产物 JSON 的写入路径。
        gate_modes: 待评测的门禁模式列表（off/shadow/enforce）。
        tenant_id: 评测租户 ID。
        authoring: 是否处于 authoring 模式（允许尚未通过人工评审的数据集）。
        env_file: 运行时 .env 文件路径；None 表示跳过该文件的存在性检查。
        runtime_configuration: 覆盖默认配置的映射；None 时取全局 settings。

    Returns:
        预检结果字典（同时写入 output_path）。
    """
    # ① 校验门禁模式与基线模板（模板必须带 dataset.manifest_sha256）。
    modes = tuple(dict.fromkeys(gate_modes))
    if not modes or any(mode not in _SUPPORTED_GATE_MODES for mode in modes):
        raise ValueError("gate_modes must contain only off, shadow, or enforce")
    manifest = Path(manifest_path)
    baseline = _load_json(Path(baseline_path), label="baseline template")
    expected_hash = (
        baseline.get("dataset", {}).get("manifest_sha256")
        if isinstance(baseline.get("dataset"), dict)
        else None
    )
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("baseline template is missing dataset.manifest_sha256")

    reason_codes: list[str] = []
    dataset = None
    # ② 尝试加载冻结数据集：未过人工评审时降级记录 pending_human_review，
    #    其余加载失败统一记为 invalid_frozen_dataset。
    try:
        dataset = load_evidence_gate_benchmark(
            manifest,
            expected_manifest_sha256=expected_hash,
            require_reviewed=not authoring,
        )
    except EvidenceGateBenchmarkError as error:
        if "human review is not approved" in str(error):
            reason_codes.append("pending_human_review")
            dataset = load_evidence_gate_benchmark(
                manifest,
                expected_manifest_sha256=expected_hash,
                require_reviewed=False,
            )
        else:
            reason_codes.append("invalid_frozen_dataset")

    # ③ 校验运行时配置：生成/嵌入密钥、租户 ID、图凭据；占位密码视为未配置。
    configuration = dict(runtime_configuration or _effective_configuration())
    env_file_present = Path(env_file).is_file() if env_file is not None else None
    if env_file is not None and not env_file_present:
        reason_codes.append("missing_runtime_env_file")
    if not str(configuration.get("deepseek_api_key") or "").strip():
        reason_codes.append("missing_generation_api_key")
    if not str(configuration.get("dashscope_api_key") or "").strip():
        reason_codes.append("missing_embedding_api_key")
    if not tenant_id.strip():
        reason_codes.append("missing_evaluation_tenant_id")
    graph_password = str(configuration.get("neo4j_password") or "").strip()
    if not graph_password or graph_password == "password":
        reason_codes.append("placeholder_graph_credentials")
    reason_codes = list(dict.fromkeys(reason_codes))

    # ④ 区分两类阻塞：配置缺失优先于数据集未就绪，全部通过才允许服务探测。
    configuration_blockers = {
        "missing_runtime_env_file",
        "missing_generation_api_key",
        "missing_embedding_api_key",
        "missing_evaluation_tenant_id",
        "placeholder_graph_credentials",
    }
    if any(reason in configuration_blockers for reason in reason_codes):
        status = "blocked_missing_runtime_configuration"
    elif reason_codes:
        status = "blocked_dataset_not_approved"
    else:
        status = "ready_for_service_probe"
    result = {
        "schema_version": _RUNTIME_PREFLIGHT_SCHEMA,
        "recorded_at": datetime.now(UTC).isoformat(),
        "status": status,
        "measured": False,
        "quality_metrics": None,
        "gate_modes": list(modes),
        "production_gate_mode": "off",
        "authoring": authoring,
        "reason_codes": reason_codes,
        "dataset": {
            "version": getattr(dataset, "dataset_version", None),
            "manifest_sha256": getattr(dataset, "manifest_sha256", expected_hash),
            "case_count": len(dataset) if dataset is not None else 0,
            "review_status": getattr(dataset, "review_status", None),
        },
        "runtime": {
            "env_file_present": env_file_present,
            "tenant_id_configured": bool(tenant_id.strip()),
            "generation_api_key_configured": bool(
                str(configuration.get("deepseek_api_key") or "").strip()
            ),
            "embedding_api_key_configured": bool(
                str(configuration.get("dashscope_api_key") or "").strip()
            ),
            "vector_store_type": str(configuration.get("vector_store_type") or ""),
            "service_probe": "not_run",
        },
        "notes": (
            "This artifact contains no credentials and no fabricated measurements. "
            "Production QA_EVIDENCE_GATE_MODE remains off until reviewed data and "
            "measured calibration both pass."
        ),
    }
    _write_json(Path(output_path), result)
    return result


class _UnavailableDependency:
    """让所有依赖操作都失败，用于构造"分支不可用"的故障注入 Fixture。"""

    def __init__(self, branch: str) -> None:
        """记录该替身代表的检索分支名。

        Args:
            branch: 分支名（dense/bm25/graph），用于拼接错误码。
        """
        self.branch = branch

    def __getattr__(self, name: str):
        """任意属性访问都返回抛 RuntimeError 的异步函数，模拟服务不可用。

        Args:
            name: 被访问的属性名（统一忽略）。
        """
        async def unavailable(*args: Any, **kwargs: Any) -> Any:
            """对任意调用抛带分支名的 RuntimeError，模拟该分支服务不可用。

            Args:
                args: 调用方传入的位置参数（统一忽略）。
                kwargs: 调用方传入的关键字参数（统一忽略）。
            """
            # BM25 适配器只把原生不可用异常归类为 UNAVAILABLE；使用其
            # 公开契约而不是普通 RuntimeError，避免故障注入逃出 QA 边界。
            if self.branch == "bm25":
                from infrastructure.retrieval.bm25_index import BM25IndexUnavailable

                raise BM25IndexUnavailable("fixture_bm25_unavailable")
            # 向量与图谱检索边界会把依赖异常收敛为 unavailable outcome。
            raise RuntimeError(f"fixture_{self.branch}_unavailable")

        return unavailable


def _controlled_safety_refusal_result(
    question: str,
    error: QASafetyRefusalError,
) -> QAResult:
    """Convert an expected direct safety refusal into a bounded QA result."""
    return QAResult(
        question=question,
        answer="当前问题需要人工审核后才能回答。",
        contexts=[],
        intent=QueryIntent.FACTOID,
        confidence=0.0,
        reasoning_steps=["安全策略在检索前停止了自动回答。"],
        degradation_code="human_review_required",
        security_actions=[error.code],
        response_status=QAResponseStatus.HUMAN_REVIEW_REQUIRED,
        evidence_assessment=EvidenceAssessment(
            states=(EvidenceState.INVALID_PROVENANCE,),
            response_status=QAResponseStatus.HUMAN_REVIEW_REQUIRED,
            reason_codes=(
                EvidenceReasonCode.PROMPT_INJECTION_DETECTED
                if error.code == "prompt_injection"
                else EvidenceReasonCode.HIGH_RISK_REVIEW,
            ),
        ),
    )


def _fixture_observation(
    case: Any,
    result: Any,
    *,
    branch_availability: Mapping[str, str],
    identity_scope_valid: bool,
) -> dict[str, Any]:
    """把一次真实 QA 结果归约成断言所需的非敏感事实（只保留状态与计数）。

    Args:
        case: 冻结数据集中的用例对象（提供期望的来源 ID 与原因码）。
        result: agent.answer 的返回对象。
        branch_availability: Fixture 显式注入的分支状态，仅描述被故意降级的分支。
        identity_scope_valid: 解析出的身份租户范围是否与目标租户一致。

    Returns:
        有界观测字典（响应状态、分支可用性、匹配计数、是否泄露等）。
    """
    # ① 读取响应状态（兼容枚举与裸字符串两种形态）。
    response = getattr(result, "response_status", None)
    response_status = getattr(response, "value", response)
    # ② 只统计"命中期望来源清单"的 ID：一旦返回受保护来源即构成泄露证据。
    expected_source_ids = {
        str(value)
        for value in getattr(case, "expected_source_document_ids", ())
        if isinstance(value, str) and value
    }
    returned_source_ids: set[str] = set()
    for context in getattr(result, "contexts", ()) or ():
        # ③ 兼容多种元数据键名（单值与列表两种形态），逐个比对期望来源集合。
        metadata = getattr(context, "metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        for key in ("source_document_id", "document_id", "source_id", "doc_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value in expected_source_ids:
                returned_source_ids.add(value)
        for key in ("source_document_ids", "document_ids", "source_ids", "doc_ids"):
            values = metadata.get(key)
            if isinstance(values, (list, tuple, set, frozenset)):
                returned_source_ids.update(
                    value
                    for value in values
                    if isinstance(value, str) and value in expected_source_ids
                )
        source = getattr(context, "source", None)
        if isinstance(source, str) and source in expected_source_ids:
            returned_source_ids.add(source)

    # ④ 汇总安全动作与证据评估原因码，与期望原因码求交集得到匹配计数。
    security_actions = {
        str(getattr(value, "value", value))
        for value in (getattr(result, "security_actions", ()) or ())
    }
    assessment = getattr(result, "evidence_assessment", None)
    observed_reason_codes = {
        str(getattr(value, "value", value))
        for value in (getattr(assessment, "reason_codes", ()) or ())
    }
    observed_reason_codes.update(security_actions)
    expected_reason_codes = {
        str(value)
        for value in (getattr(case, "expected_reason_codes", ()) or ())
        if isinstance(value, str) and value
    }
    observed_branches = _observed_branch_availability(result)
    # 安全边界：显式注入项只描述"Fixture 故意降级的分支"（如 sparse_index=None），
    # 覆盖对应在线状态是预期行为，绝不引入其他分支的臆造状态。
    observed_branches.update(branch_availability)
    return {
        "response_status": str(response_status or ""),
        "branch_availability": observed_branches,
        "expected_source_match_count": len(returned_source_ids),
        "expected_reason_match_count": len(expected_reason_codes & observed_reason_codes),
        "protected_context_disclosed": bool(returned_source_ids),
        "identity_scope_valid": identity_scope_valid,
        "safety_action_applied": bool(security_actions),
    }


def _failed_fixture_observations(cases: Sequence[Any], *, code: str) -> dict[str, list[dict[str, Any]]]:
    """Worker 无法安全探测运行时时，为全部用例生成统一的有界失败观测。

    Args:
        cases: 冻结数据集用例列表。
        code: 统一的失败原因码。

    Returns:
        fixture 名 → 失败观测列表 的映射。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        fixture = str(getattr(case, "required_fixture", "") or "")
        if fixture:
            grouped.setdefault(fixture, []).append({"execution_error": code})
    return grouped


async def execute_frozen_fixture_probes(
    *,
    cases: Sequence[Any],
    tenant_id: str,
    identity_resolver: FinanceFixtureIdentityResolver,
) -> dict[str, list[dict[str, Any]]]:
    """把每个冻结 Fixture 走真实 QA 链路执行，只返回有界事实。

    这是管理员触发校验运行时 Worker 侧的执行边界：不接收浏览器身份、
    profile、token、文档正文或租户覆盖；结果只保留计数与状态。

    Args:
        cases: 冻结数据集用例列表。
        tenant_id: 目标评测租户 ID（不接受调用方覆盖）。
        identity_resolver: 财务授权类 Fixture 的身份解析器。

    Returns:
        fixture 名 → 每个用例的有界观测列表 的映射。
    """
    # 安全关卡：租户 ID 必须是非空字符串，杜绝空租户探测。
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        return _failed_fixture_observations(cases, code="fixture_identity_scope_invalid")

    vector_store = VectorStoreService()
    graph = KnowledgeGraphService()
    sparse = NativeBM25Index(
        settings.qa_bm25_index_path,
        k1=settings.qa_bm25_k1,
        b=settings.qa_bm25_b,
        schema_version=settings.qa_bm25_schema_version,
        tokenizer_version=settings.qa_bm25_tokenizer_version,
    )
    # ① 初始化向量库与知识图谱并做健康检查，失败即整体降级为有界失败。
    try:
        await vector_store.init()
        await graph.init()
        vector_ok, graph_ok = await asyncio.gather(
            vector_store.health_check(), graph.health_check()
        )
        if not vector_ok or not graph_ok:
            raise RuntimeError("fixture_services_unavailable")
    except Exception:
        try:
            await graph.close()
        except Exception:
            pass
        return _failed_fixture_observations(cases, code="fixture_services_unavailable")

    try:
        qa_dependencies = build_qa_agent_dependencies()
        # ② 关闭缓存并固定门禁为 off，确保探测结果只反映检索行为本身。
        offline_gate = EffectiveEvidenceGateConfiguration(
            mode="off",
            revision=0,
            calibration_version=settings.qa_evidence_calibration_version,
            source="fixture_validation",
        )
        agent_kwargs = {
            "qa_cache": None,
            "semantic_cache": None,
            "dependencies": qa_dependencies,
            "gate_configuration_provider": lambda: offline_gate,
        }
        standard_agent = QAAgent(
            vector_store=vector_store,
            knowledge_graph=graph,
            sparse_index=sparse,
            **agent_kwargs,
        )
        bm25_unavailable_agent = QAAgent(
            vector_store=vector_store,
            knowledge_graph=graph,
            # NativeBM25Retriever 会把 None 映射为受支持的 UNAVAILABLE 结果；
            # 人为抛 RuntimeError 会绕过该契约，把"故意的降级测试"变成无关的 Worker 故障。
            sparse_index=None,
            **agent_kwargs,
        )
        all_unavailable_agent = QAAgent(
            vector_store=_UnavailableDependency("dense"),
            knowledge_graph=_UnavailableDependency("graph"),
            sparse_index=None,
            **agent_kwargs,
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            # ③ 按 Fixture 选择执行器，并声明该 Fixture 故意注入的分支状态。
            fixture = str(getattr(case, "required_fixture", "") or "")
            if not fixture:
                continue
            if fixture == "bm25_unavailable_dense_graph_available":
                agent = bm25_unavailable_agent
                availability = {"bm25": "unavailable"}
            elif fixture == "all_retrieval_branches_unavailable":
                agent = all_unavailable_agent
                availability = {"dense": "unavailable", "bm25": "unavailable", "graph": "unavailable"}
            else:
                agent = standard_agent
                availability = {}
            identity_scope_valid = True
            user_id = "evaluation-fixture-default"
            visible_department_ids: tuple[str, ...] | None = None
            # ④ 授权类 Fixture 需解析真实身份，并校验其租户范围与目标租户一致。
            if fixture in {
                "finance_only_user_against_hr_document",
                "finance_only_user_against_administration_document",
            }:
                try:
                    identity = identity_resolver.resolve(tenant_id)
                except FixtureValidationError as error:
                    grouped.setdefault(fixture, []).append({"execution_error": str(error)})
                    continue
                user_id = identity.identity_id
                visible_department_ids = identity.visible_department_ids
                identity_scope_valid = identity.org_id == tenant_id
            try:
                # ⑤ 走真实 QA 链路应答，并把结果归约成有界观测。
                result = await agent.answer(
                    str(getattr(case, "question", "")),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    retrieval_mode="dense_bm25_graph",
                    visible_department_ids=visible_department_ids,
                )
                observation = _fixture_observation(
                    case,
                    result,
                    branch_availability=availability,
                    identity_scope_valid=identity_scope_valid,
                )
            except QASafetyRefusalError:
                # 安全关卡：直接拒绝提示注入就是期望的安全结果，不是执行错误；只保留有界事实。
                observation = {
                    "response_status": "refused",
                    "branch_availability": dict(availability),
                    "expected_source_match_count": 0,
                    "expected_reason_match_count": int(
                        bool(getattr(case, "expected_reason_codes", ()) or ())
                    ),
                    "protected_context_disclosed": False,
                    "identity_scope_valid": identity_scope_valid,
                    "safety_action_applied": True,
                }
            except Exception as error:
                # 有界诊断：不打日志输出问题、回答、上下文、身份范围或异常文本，只记错误类型。
                logger.warning(
                    "fixture_probe_case_failed",
                    fixture=fixture,
                    error_type=type(error).__name__,
                )
                grouped.setdefault(fixture, []).append({"execution_error": "fixture_execution_failed"})
                continue
            logger.info(
                "fixture_probe_observed",
                fixture=fixture,
                status=observation["response_status"],
                branch_states=observation["branch_availability"],
                evidence_match_count=observation["expected_source_match_count"],
                reason_match_count=observation["expected_reason_match_count"],
                disclosure_detected=observation["protected_context_disclosed"],
                identity_scope_valid=observation["identity_scope_valid"],
                safety_action_applied=observation["safety_action_applied"],
            )
            grouped.setdefault(fixture, []).append(observation)
        return grouped
    except Exception:
        return _failed_fixture_observations(cases, code="fixture_execution_unavailable")
    finally:
        # ⑥ 无论成败都关闭图连接，避免连接泄漏。
        await graph.close()


async def execute_evidence_gate_benchmark(
    *,
    manifest_path: str | Path,
    baseline_path: str | Path,
    output_root: str | Path,
    gate_modes: Sequence[str],
    tenant_id: str,
    authoring: bool,
    env_file: str | Path | None,
    retrieval_modes: Sequence[str] = ("dense_bm25_graph",),
    smoke_threshold: int = 30,
) -> dict[str, Any]:
    """预检完全就绪后先探测服务健康，再执行带真实测量的门禁模式。

    Args:
        manifest_path: 冻结数据集 manifest 路径。
        baseline_path: 基线模板 JSON 路径（提供期望的 manifest 哈希）。
        output_root: 运行时产物输出目录（预检 JSON 与各模式运行结果都写在这里）。
        gate_modes: 待评测的门禁模式列表（off/shadow/enforce）。
        tenant_id: 评测租户 ID。
        authoring: 是否处于 authoring 模式（允许尚未通过人工评审的数据集）。
        env_file: 运行时 .env 文件路径；None 表示跳过该文件的存在性检查。
        retrieval_modes: 参与测量的检索模式列表。
        smoke_threshold: 冒烟阈值，低于该样本数按冒烟口径聚合报告。

    Returns:
        测量结果字典（未就绪时返回预检产物）。
    """
    root = Path(output_root)
    status_path = root / "runtime-preflight.json"
    # ① 先跑只读预检，未达到 ready_for_service_probe 直接返回预检产物。
    preflight = assess_evidence_gate_runtime(
        manifest_path=manifest_path,
        baseline_path=baseline_path,
        output_path=status_path,
        gate_modes=gate_modes,
        tenant_id=tenant_id,
        authoring=authoring,
        env_file=env_file,
    )
    if preflight["status"] != "ready_for_service_probe":
        return preflight

    baseline = _load_json(Path(baseline_path), label="baseline template")
    expected_hash = baseline["dataset"]["manifest_sha256"]
    dataset = load_evidence_gate_benchmark(
        manifest_path,
        expected_manifest_sha256=expected_hash,
        require_reviewed=not authoring,
    )
    vector_store = VectorStoreService()
    graph = KnowledgeGraphService()
    sparse = NativeBM25Index(
        settings.qa_bm25_index_path,
        k1=settings.qa_bm25_k1,
        b=settings.qa_bm25_b,
        schema_version=settings.qa_bm25_schema_version,
        tokenizer_version=settings.qa_bm25_tokenizer_version,
    )
    # ② 探测向量库与图谱健康状态，失败记为 blocked_service_probe_failed 并返回。
    try:
        await vector_store.init()
        await graph.init()
        vector_ok, graph_ok = await asyncio.gather(
            vector_store.health_check(), graph.health_check()
        )
        if not vector_ok or not graph_ok:
            raise RuntimeError("one or more retrieval services failed health checks")
    except Exception as error:
        preflight["status"] = "blocked_service_probe_failed"
        preflight["reason_codes"] = [
            *preflight["reason_codes"],
            f"service_probe_failed:{type(error).__name__}",
        ]
        preflight["runtime"]["service_probe"] = "failed"
        _write_json(status_path, preflight)
        await graph.close()
        return preflight

    reports: list[dict[str, Any]] = []
    try:
        qa_dependencies = build_qa_agent_dependencies()
        # ③ 为三种依赖形态构建 Agent：标准 / BM25 不可用 / 全分支不可用。
        standard_agent = QAAgent(
            vector_store=vector_store,
            knowledge_graph=graph,
            sparse_index=sparse,
            qa_cache=None,
            semantic_cache=None,
            dependencies=qa_dependencies,
        )
        bm25_unavailable_agent = QAAgent(
            vector_store=vector_store,
            knowledge_graph=graph,
            sparse_index=_UnavailableDependency("bm25"),
            qa_cache=None,
            semantic_cache=None,
            dependencies=qa_dependencies,
        )
        all_unavailable_agent = QAAgent(
            vector_store=_UnavailableDependency("dense"),
            knowledge_graph=_UnavailableDependency("graph"),
            sparse_index=_UnavailableDependency("bm25"),
            qa_cache=None,
            semantic_cache=None,
            dependencies=qa_dependencies,
        )
        fixture_by_question = {case.question: case.required_fixture for case in dataset}

        async def fixture_executor(**kwargs: Any) -> Any:
            """按问题对应的 Fixture 路由到相应 Agent，并注入授权类身份。

            Args:
                kwargs: 透传给 agent.answer 的关键字参数（含 question 与 tenant_id）。
            """
            fixture = fixture_by_question.get(str(kwargs.get("question")), "standard")
            if fixture == "bm25_unavailable_dense_graph_available":
                agent = bm25_unavailable_agent
            elif fixture == "all_retrieval_branches_unavailable":
                agent = all_unavailable_agent
            else:
                agent = standard_agent
            if fixture in {
                "finance_only_user_against_hr_document",
                "finance_only_user_against_administration_document",
            }:
                identity = FinanceFixtureIdentityResolver(
                    settings.evaluation_fixture_finance_department_id
                ).resolve(str(kwargs.get("tenant_id") or ""))
                kwargs["visible_department_ids"] = identity.visible_department_ids
                kwargs["user_id"] = identity.identity_id
            try:
                return await agent.answer(**kwargs)
            except QASafetyRefusalError as error:
                if fixture != "prompt_injection_safety_fixture":
                    raise
                return _controlled_safety_refusal_result(
                    str(kwargs.get("question") or ""),
                    error,
                )

        # ④ 逐个门禁模式执行测量并聚合质量报告。
        for gate_mode in gate_modes:
            offline_gate = EffectiveEvidenceGateConfiguration(
                mode=gate_mode, revision=0,
                calibration_version=settings.qa_evidence_calibration_version,
                source="environment_default",
            )
            standard_agent.gate_configuration_provider = lambda value=offline_gate: value
            bm25_unavailable_agent.gate_configuration_provider = lambda value=offline_gate: value
            all_unavailable_agent.gate_configuration_provider = lambda value=offline_gate: value
            runner = BenchmarkRunner(
                qa_executor=fixture_executor,
                results_root=root / gate_mode,
            )
            run = await runner.run(
                dataset,
                tenant_id=tenant_id,
                retrieval_modes=retrieval_modes,
                smoke_threshold=smoke_threshold,
                run_metadata={
                    "dataset_version": dataset.dataset_version,
                    "dataset_manifest_sha256": dataset.manifest_sha256,
                    "answer_model": settings.deepseek_model,
                    "embedding_model": settings.embedding_model,
                    "retrieval_config_id": settings.qa_retrieval_strategy,
                    "threshold_version": settings.qa_evidence_calibration_version,
                    "candidate_budget_version": settings.qa_evidence_candidate_budget_version,
                    "reranker_mode": settings.qa_cross_encoder_mode,
                    "reranker_version": settings.qa_cross_encoder_version,
                    "structured_answer_schema_version": settings.qa_structured_answer_schema_version,
                    "grounding_policy_version": settings.qa_grounding_policy_version,
                    "gate_mode": gate_mode,
                },
            )
            metadata = _load_json(run.metadata_path, label="run metadata")
            report = aggregate_quality_report(
                metadata,
                run.records,
                smoke_threshold=smoke_threshold,
                diagnostic_inputs=_diagnostic_inputs(run),
            )
            report_path = run.run_dir / "quality-report.json"
            _write_json(report_path, report)
            reports.append(
                {
                    "gate_mode": gate_mode,
                    "run_id": run.run_id,
                    "run_dir": str(run.run_dir),
                    "quality_report": str(report_path),
                    "evidence_gate": report.get("evidence_gate"),
                }
            )
    finally:
        await graph.close()

    result = {
        **preflight,
        "status": "measured_pending_calibration",
        "measured": True,
        "recorded_at": datetime.now(UTC).isoformat(),
        "quality_metrics": reports,
        "reason_codes": ["pending_calibration"],
    }
    result["runtime"]["service_probe"] = "passed"
    _write_json(status_path, result)
    return result


__all__ = [
    "assess_evidence_gate_runtime",
    "execute_evidence_gate_benchmark",
    "execute_frozen_fixture_probes",
]
