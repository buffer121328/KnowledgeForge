"""经过校验的证据门基准 bundle：加载时强制哈希一致性校验与人工评审约束。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

EVIDENCE_GATE_MANIFEST_SCHEMA = "evidence-gate-benchmark-v1"  # manifest 允许的 schema 版本
EVIDENCE_GATE_REVIEW_SCHEMA = "evidence-gate-review-v1"  # 评审记录允许的 schema 版本
REFERENCE_ANSWER_SCHEMA_VERSION = "reviewed-reference-answer-v1"
REQUIRED_EVIDENCE_GATE_CATEGORIES = frozenset(  # 基准必须完整覆盖的可回答性类别（frozenset 冻结，不可变）
    {
        "fully_answerable",
        "completely_unanswerable",
        "background_only",
        "partially_answerable",
        "conflicting",
        "missing_version_or_date",
        "missing_business_record",
        "authorization_filtered",
        "single_branch_unavailable",
        "all_branches_unavailable",
        "prompt_injection",
    }
)
SUPPORTED_RESPONSE_STATUSES = frozenset(  # 允许声明的期望响应状态全集（冻结，不可变）
    {
        "answered",
        "partially_answered",
        "insufficient_evidence",
        "needs_clarification",
        "conflicting_evidence",
        "human_review_required",
        "source_unavailable",
    }
)
SUPPORTED_EVIDENCE_STATES = frozenset(  # 允许声明的期望证据状态全集（冻结，不可变）
    {
        "direct_evidence",
        "partial_evidence",
        "relevant_background",
        "conflicting_evidence",
        "irrelevant",
        "insufficient_evidence",
        "invalid_provenance",
    }
)
_BRANCH_STATUSES = frozenset({"available", "unavailable", "not_configured"})  # 检索分支可用性的合法取值
_MAX_CASES = 10_000  # 用例数量上限，防止加载超大 cases 文件
_MAX_IDENTIFIER_LENGTH = 128  # 标识符类字段的最大字符数
_MAX_QUESTION_LENGTH = 8_000  # 问题文本的最大字符数
_MAX_REFERENCE_ANSWER_LENGTH = 8_000

_REVIEWED_ROUTE_CONTRACTS: dict[str, tuple[str, str, frozenset[str]]] = {
    "fully_answerable": ("answered", "direct_evidence", frozenset({"direct_support"})),
    "partially_answerable": ("partially_answered", "partial_evidence", frozenset({"partial_support"})),
    "conflicting": ("conflicting_evidence", "conflicting_evidence", frozenset({"material_conflict"})),
    "background_only": ("insufficient_evidence", "relevant_background", frozenset({"background_only"})),
    "missing_version_or_date": ("needs_clarification", "relevant_background", frozenset({"missing_question_detail"})),
    "missing_business_record": ("insufficient_evidence", "relevant_background", frozenset({"background_only"})),
    "authorization_filtered": ("insufficient_evidence", "insufficient_evidence", frozenset({"authorized_contexts_empty"})),
    "single_branch_unavailable": ("answered", "direct_evidence", frozenset({"direct_support", "partial_dependency_unavailable"})),
    "all_branches_unavailable": ("source_unavailable", "insufficient_evidence", frozenset({"all_dependencies_unavailable"})),
    "prompt_injection": ("human_review_required", "invalid_provenance", frozenset({"prompt_injection_detected"})),
    "completely_unanswerable": ("insufficient_evidence", "insufficient_evidence", frozenset({"zero_results"})),
}
_REVIEWED_FIXTURE_CATEGORIES = {
    "frozen_corpus_absence_check": "completely_unanswerable",
    "equal_authority_conflicting_documents": "conflicting",
    "finance_only_user_against_hr_document": "authorization_filtered",
    "finance_only_user_against_administration_document": "authorization_filtered",
    "bm25_unavailable_dense_graph_available": "single_branch_unavailable",
    "all_retrieval_branches_unavailable": "all_branches_unavailable",
    "prompt_injection_safety_fixture": "prompt_injection",
}
_MISSING_INFORMATION_CATEGORIES = frozenset(
    {"partially_answerable", "background_only", "missing_version_or_date", "missing_business_record"}
)
_POSITIVE_EVIDENCE_CATEGORIES = frozenset(
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


class EvidenceGateBenchmarkError(ValueError):
    """当证据门基准不满足安全评分前提（哈希、结构或评审不合规）时抛出。"""


def reviewed_case_contract_errors(case: Any) -> tuple[str, ...]:
    """Return bounded semantic-review errors without reading raw case content."""

    def value(name: str, default: Any = None) -> Any:
        if isinstance(case, dict):
            return case.get(name, default)
        return getattr(case, name, default)

    category = str(value("category") or "")
    contract = _REVIEWED_ROUTE_CONTRACTS.get(category)
    if contract is None:
        return ("reviewed_contract_category_invalid",)
    expected_status, expected_state, required_reasons = contract
    errors: list[str] = []
    if str(value("expected_response_status") or "") != expected_status:
        errors.append("reviewed_contract_response_status_mismatch")
    states = {str(item) for item in (value("expected_evidence_states", ()) or ())}
    if states != {expected_state}:
        errors.append("reviewed_contract_evidence_state_mismatch")
    reasons = {str(item) for item in (value("expected_reason_codes", ()) or ())}
    if not required_reasons.issubset(reasons):
        errors.append("reviewed_contract_reason_code_mismatch")
    fixture = str(value("required_fixture") or "standard")
    fixture_category = _REVIEWED_FIXTURE_CATEGORIES.get(fixture)
    if fixture_category is not None and fixture_category != category:
        errors.append("reviewed_contract_fixture_category_mismatch")
    if fixture == "standard" and category in set(_REVIEWED_FIXTURE_CATEGORIES.values()):
        errors.append("reviewed_contract_fixture_missing")
    missing_fields = tuple(value("expected_missing_information_fields", ()) or ())
    if category in _MISSING_INFORMATION_CATEGORIES and not missing_fields:
        errors.append("reviewed_contract_missing_information_required")
    if category not in _MISSING_INFORMATION_CATEGORIES and missing_fields:
        errors.append("reviewed_contract_missing_information_unexpected")
    if category in _POSITIVE_EVIDENCE_CATEGORIES:
        source_ids = tuple(value("expected_source_document_ids", ()) or ())
        context_ids = tuple(value("expected_evidence_context_ids", ()) or ())
        if not source_ids or not context_ids:
            errors.append("reviewed_contract_positive_evidence_required")
        citation_ids = tuple(value("expected_citation_context_ids", ()) or ())
        if expected_status in {"answered", "partially_answered"} and not citation_ids:
            errors.append("reviewed_contract_positive_evidence_required")
    branches = value("expected_branch_availability", {}) or {}
    if category == "all_branches_unavailable" and any(
        branches.get(branch) != "unavailable" for branch in ("dense", "bm25", "graph")
    ):
        errors.append("reviewed_contract_branch_availability_mismatch")
    if category == "single_branch_unavailable" and any(
        branches.get(branch) != expected
        for branch, expected in {"dense": "available", "bm25": "unavailable", "graph": "available"}.items()
    ):
        errors.append("reviewed_contract_branch_availability_mismatch")
    return tuple(dict.fromkeys(errors))


@dataclass(frozen=True, slots=True)  # 冻结数据类：字段不可变，防止加载后被篡改
class EvidenceGateCase:
    """单条经过评审的证据门期望用例，不包含未受限的源正文（字段含义见行内注释）。"""

    id: str  # 用例唯一标识
    question: str  # 评测时向系统提出的业务问题
    category: str  # 可回答性类别，必须属于 REQUIRED_EVIDENCE_GATE_CATEGORIES
    expected_response_status: str  # 期望的响应状态（如 answered / insufficient_evidence）
    expected_evidence_states: tuple[str, ...]  # 期望的逐证据状态序列
    expected_reason_codes: tuple[str, ...]  # 期望的原因代码序列
    expected_citation_context_ids: tuple[str, ...]  # 期望被引用的上下文 ID 序列
    expected_branch_availability: dict[str, str]  # 各检索分支的期望可用性映射
    department_id: str = ""  # 用例归属部门 ID（用于租户/权限隔离场景）
    expected_source_document_ids: tuple[str, ...] = ()  # 期望涉及的源文档 ID 序列
    expected_evidence_context_ids: tuple[str, ...] = ()  # 期望检索命中的证据上下文 ID 序列
    expected_evidence_sections: tuple[str, ...] = ()  # 期望证据所在的章节名序列
    expected_missing_information_fields: tuple[str, ...] = ()  # 期望缺失的业务记录字段序列
    source_benchmark_ids: tuple[str, ...] = ()  # 来源基准的原始用例 ID 序列
    required_fixture: str = "standard"  # 运行该用例所需的测试夹具名称
    # 独立审核的参考答案，不得由 Context 摘录合成；冻结快照阶段允许暂时为空。
    reference_answer: str = ""
    notes: str = ""  # 评审备注（长度受限）

    @property
    def expected_refusal(self) -> bool:
        """返回期望路由是否为受控不作答（响应状态不是 answered/partially_answered）。"""
        return self.expected_response_status not in {"answered", "partially_answered"}

    @property
    def reference_context_ids(self) -> tuple[str, ...]:
        """返回期望引用的上下文 ID，为运行器提供基于 ID 的检索评分兼容字段。"""
        return self.expected_citation_context_ids

    @property
    def reference(self) -> str:
        """Expose the independently reviewed answer to generic runners."""
        return self.reference_answer


class EvidenceGateDataset(list[EvidenceGateCase]):
    """经过校验的基准数据集（list 子类），携带不可变的 bundle 身份标识。"""

    def __init__(
        self,
        cases: Iterable[EvidenceGateCase],
        *,
        source_path: Path,
        dataset_version: str,
        manifest_sha256: str,
        cases_sha256: str,
        review_record_sha256: str,
        review_status: str,
    ) -> None:
        """以已校验的用例和 bundle 身份信息构建数据集。

        Args:
            cases: 已通过全部校验的用例集合。
            source_path: cases 文件路径。
            dataset_version: 数据集版本号。
            manifest_sha256: manifest 文件的 SHA-256 摘要。
            cases_sha256: cases 文件的 SHA-256 摘要。
            review_record_sha256: 评审记录文件的 SHA-256 摘要。
            review_status: 人工评审状态（如 approved）。
        """
        super().__init__(cases)
        self.source_path = source_path  # cases 文件来源路径
        self.sha256 = cases_sha256  # 兼容别名：等价于 cases_sha256
        self.dataset_version = dataset_version  # 数据集版本
        self.manifest_sha256 = manifest_sha256  # manifest 哈希
        self.cases_sha256 = cases_sha256  # cases 哈希
        self.review_record_sha256 = review_record_sha256  # 评审记录哈希
        self.review_status = review_status  # 评审状态


def sha256_file(path: Path) -> str:
    """计算单个文件的小写十六进制 SHA-256 摘要。

    Args:
        path: 待哈希的文件路径。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # 按 1MiB 分块读取，避免大文件一次性载入内存
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, *, label: str) -> dict[str, Any]:
    """读取一个必须为 JSON 对象的文件，任何缺失或格式问题都以 EvidenceGateBenchmarkError 报错。

    Args:
        path: 目标文件路径。
        label: 错误信息中使用的文件名称标签。
    """
    if not path.is_file():
        raise EvidenceGateBenchmarkError(f"{label} file does not exist")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceGateBenchmarkError(f"{label} must be valid JSON") from error
    # 顶层必须是对象，拒绝数组/标量等其他 JSON 形态
    if not isinstance(value, dict):
        raise EvidenceGateBenchmarkError(f"{label} must be a JSON object")
    return value


def _required_string(value: Any, *, field: str, maximum: int = _MAX_IDENTIFIER_LENGTH) -> str:
    """校验取值为非空、去除首尾空白且长度受限的字符串，并返回规范化结果。

    Args:
        value: 待校验的原始值。
        field: 字段名（用于错误信息）。
        maximum: 允许的最大字符数，默认为标识符上限。
    """
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise EvidenceGateBenchmarkError(
            f"{field} must be a non-empty string of at most {maximum} characters"
        )
    return value.strip()  # 统一返回去空白后的值


def _string_tuple(
    value: Any,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """校验取值为字符串列表（是否允许空列表可配置）且元素唯一，返回规范化元组。

    Args:
        value: 待校验的原始值。
        field: 字段名（用于错误信息）。
        allow_empty: 是否允许空列表，默认不允许。
    """
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise EvidenceGateBenchmarkError(f"{field} must be {qualifier}")
    # 逐元素做字符串规范化校验
    normalized = tuple(
        _required_string(item, field=field)
        for item in value
    )
    # 元素必须唯一，防止重复声明
    if len(normalized) != len(set(normalized)):
        raise EvidenceGateBenchmarkError(f"{field} must contain unique values")
    return normalized


def _safe_child(root: Path, value: Any, *, field: str) -> Path:
    """把 manifest 声明的文件名解析为 root 的直接子路径，拒绝任何目录穿越。

    Args:
        root: bundle 根目录。
        value: manifest 中声明的文件名。
        field: 字段名（用于错误信息）。
    """
    name = _required_string(value, field=field)
    candidate = (root / name).resolve()
    # 安全关卡：resolve 后父目录必须恰为 root，否则视为越界/嵌套路径
    if candidate.parent != root.resolve():
        raise EvidenceGateBenchmarkError(f"{field} must name a direct bundle child")
    return candidate


def _validate_declared_artifact(
    root: Path,
    manifest: dict[str, Any],
    *,
    file_field: str,
    hash_field: str,
) -> None:
    """校验 manifest 中可选声明的补充产物：文件名与哈希必须成对出现且内容匹配。

    Args:
        root: bundle 根目录。
        manifest: 已解析的 manifest 对象。
        file_field: manifest 中产物文件名字段。
        hash_field: manifest 中产物哈希字段。
    """
    file_value = manifest.get(file_field)
    hash_value = manifest.get(hash_field)
    # 未声明则跳过（该产物为可选项）
    if file_value is None and hash_value is None:
        return
    # 文件名与哈希必须同时声明，只声明一个视为不合规
    if file_value is None or hash_value is None:
        raise EvidenceGateBenchmarkError(
            f"{file_field} and {hash_field} must be declared together"
        )
    artifact_path = _safe_child(root, file_value, field=file_field)
    expected_hash = _required_string(
        hash_value, field=hash_field, maximum=64
    )
    # 安全关卡：实际文件哈希必须与 manifest 声明一致（防篡改）
    if sha256_file(artifact_path) != expected_hash:
        raise EvidenceGateBenchmarkError(f"{hash_field} mismatch")


def _load_cases(
    path: Path,
    *,
    declared_statuses: frozenset[str],
    enforce_reviewed_contract: bool = False,
    require_reference_answers: bool = False,
) -> list[EvidenceGateCase]:
    """逐行读取 cases JSONL 文件并完成全量结构校验，返回用例列表。

    Args:
        path: cases JSONL 文件路径。
        declared_statuses: manifest 声明的期望响应状态全集，用例不得越出该集合。
    """
    if not path.is_file():
        raise EvidenceGateBenchmarkError("cases file does not exist")
    cases: list[EvidenceGateCase] = []
    identifiers: set[str] = set()
    # ① 逐行解析 JSONL，跳过空行
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceGateBenchmarkError(
                f"cases line {line_number} must be valid JSON"
            ) from error
        if not isinstance(item, dict):
            raise EvidenceGateBenchmarkError(
                f"cases line {line_number} must be a JSON object"
            )
        prefix = f"cases line {line_number}"
        # ② 校验用例 id 非空且全局唯一
        case_id = _required_string(item.get("id"), field=f"{prefix}.id")
        if case_id in identifiers:
            raise EvidenceGateBenchmarkError(f"duplicate case id: {case_id}")
        identifiers.add(case_id)
        # ③ 校验类别属于必需类别枚举
        category = _required_string(
            item.get("category"), field=f"{prefix}.category"
        )
        if category not in REQUIRED_EVIDENCE_GATE_CATEGORIES:
            raise EvidenceGateBenchmarkError(f"unsupported category: {category}")
        reference_answer = str(item.get("reference_answer") or "").strip()
        if len(reference_answer) > _MAX_REFERENCE_ANSWER_LENGTH:
            raise EvidenceGateBenchmarkError(
                f"{prefix}.reference_answer must be at most {_MAX_REFERENCE_ANSWER_LENGTH} characters"
            )
        if (
            require_reference_answers
            and category in _POSITIVE_EVIDENCE_CATEGORIES
            and not reference_answer
        ):
            raise EvidenceGateBenchmarkError(
                f"{prefix}.reference_answer is required for {category}"
            )
        # ④ 校验期望响应状态已在 manifest 中声明
        response_status = _required_string(
            item.get("expected_response_status"),
            field=f"{prefix}.expected_response_status",
        )
        if response_status not in declared_statuses:
            raise EvidenceGateBenchmarkError(
                f"undeclared expected response status: {response_status}"
            )
        # ⑤ 校验期望证据状态属于支持枚举
        states = _string_tuple(
            item.get("expected_evidence_states"),
            field=f"{prefix}.expected_evidence_states",
        )
        unknown_states = set(states) - SUPPORTED_EVIDENCE_STATES
        if unknown_states:
            raise EvidenceGateBenchmarkError(
                f"unsupported evidence states: {sorted(unknown_states)}"
            )
        # ⑥ 校验分支可用性映射非空且取值合法
        branch_value = item.get("expected_branch_availability")
        if not isinstance(branch_value, dict) or not branch_value:
            raise EvidenceGateBenchmarkError(
                f"{prefix}.expected_branch_availability must be a non-empty object"
            )
        branch_availability: dict[str, str] = {}
        for branch, status in branch_value.items():
            branch_name = _required_string(
                branch, field=f"{prefix}.expected_branch_availability"
            )
            branch_status = _required_string(
                status, field=f"{prefix}.expected_branch_availability.{branch_name}"
            )
            if branch_status not in _BRANCH_STATUSES:
                raise EvidenceGateBenchmarkError(
                    f"unsupported branch availability: {branch_status}"
                )
            branch_availability[branch_name] = branch_status
        # ⑦ 组装冻结数据类用例（notes 截断到 2000 字符）
        cases.append(
            EvidenceGateCase(
                id=case_id,
                question=_required_string(
                    item.get("question"),
                    field=f"{prefix}.question",
                    maximum=_MAX_QUESTION_LENGTH,
                ),
                reference_answer=reference_answer,
                category=category,
                expected_response_status=response_status,
                expected_evidence_states=states,
                expected_reason_codes=_string_tuple(
                    item.get("expected_reason_codes"),
                    field=f"{prefix}.expected_reason_codes",
                ),
                expected_citation_context_ids=_string_tuple(
                    item.get("expected_citation_context_ids"),
                    field=f"{prefix}.expected_citation_context_ids",
                    allow_empty=True,
                ),
                expected_branch_availability=branch_availability,
                department_id=str(item.get("department_id") or "").strip(),
                expected_source_document_ids=_string_tuple(
                    item.get("expected_source_document_ids") or [],
                    field=f"{prefix}.expected_source_document_ids",
                    allow_empty=True,
                ),
                expected_evidence_context_ids=_string_tuple(
                    item.get("expected_evidence_context_ids") or [],
                    field=f"{prefix}.expected_evidence_context_ids",
                    allow_empty=True,
                ),
                expected_evidence_sections=_string_tuple(
                    item.get("expected_evidence_sections") or [],
                    field=f"{prefix}.expected_evidence_sections",
                    allow_empty=True,
                ),
                expected_missing_information_fields=_string_tuple(
                    item.get("expected_missing_information_fields") or [],
                    field=f"{prefix}.expected_missing_information_fields",
                    allow_empty=True,
                ),
                source_benchmark_ids=_string_tuple(
                    item.get("source_benchmark_ids") or [],
                    field=f"{prefix}.source_benchmark_ids",
                    allow_empty=True,
                ),
                required_fixture=_required_string(
                    item.get("required_fixture") or "standard",
                    field=f"{prefix}.required_fixture",
                ),
                notes=str(item.get("notes") or "")[:2_000],
            )
        )
        # ⑧ 数量安全关卡：超过上限立即失败，防止资源耗尽
        if len(cases) > _MAX_CASES:
            raise EvidenceGateBenchmarkError(
                f"cases exceed the bounded limit of {_MAX_CASES}"
            )
    if not cases:
        raise EvidenceGateBenchmarkError("cases file contains no cases")
    # ⑨ 覆盖率关卡：类别必须与必需类别集合完全一致（不多不少）
    categories = {case.category for case in cases}
    if categories != REQUIRED_EVIDENCE_GATE_CATEGORIES:
        missing = sorted(REQUIRED_EVIDENCE_GATE_CATEGORIES - categories)
        extra = sorted(categories - REQUIRED_EVIDENCE_GATE_CATEGORIES)
        raise EvidenceGateBenchmarkError(
            f"category coverage mismatch; missing={missing}, extra={extra}"
        )
    if enforce_reviewed_contract:
        for case in cases:
            if errors := reviewed_case_contract_errors(case):
                raise EvidenceGateBenchmarkError(f"{case.id}: {errors[0]}")
    return cases


def _validate_review(
    review: dict[str, Any],
    *,
    dataset_version: str,
    case_ids: set[str],
    require_reviewed: bool,
) -> str:
    """校验人工评审记录，在要求已评审时核对批准状态与逐用例覆盖，返回评审状态。

    Args:
        review: 已解析的评审记录对象。
        dataset_version: 数据集版本号，必须与评审记录一致。
        case_ids: 全部用例 ID 集合，reviewed_case_ids 必须与其完全相等。
        require_reviewed: 是否要求数据集已通过人工评审批准。
    """
    # ① schema 与数据集版本必须匹配
    if review.get("schema_version") != EVIDENCE_GATE_REVIEW_SCHEMA:
        raise EvidenceGateBenchmarkError("unsupported review schema_version")
    if review.get("dataset_version") != dataset_version:
        raise EvidenceGateBenchmarkError("review dataset_version mismatch")
    status = _required_string(review.get("review_status"), field="review_status")
    # ② 不要求已评审时仅返回状态，跳过后续批准校验
    if not require_reviewed:
        return status
    # ③ 评审安全关卡：状态必须是 approved，且评审时间与评审人齐备
    if status != "approved":
        raise EvidenceGateBenchmarkError("human review is not approved")
    _required_string(review.get("reviewed_at"), field="reviewed_at")
    reviewer_ids = set(
        _string_tuple(review.get("reviewer_ids"), field="reviewer_ids")
    )
    if not reviewer_ids:
        raise EvidenceGateBenchmarkError("reviewer_ids must not be empty")
    # ④ 覆盖关卡：评审记录必须逐一覆盖每个用例，不多不少
    reviewed_case_ids = set(
        _string_tuple(review.get("reviewed_case_ids"), field="reviewed_case_ids")
    )
    if reviewed_case_ids != case_ids:
        raise EvidenceGateBenchmarkError("reviewed_case_ids must cover every case")
    return status


def load_evidence_gate_benchmark(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    require_reviewed: bool = True,
) -> EvidenceGateDataset:
    """在 bundle 哈希与人工评审校验全部通过后才加载并返回基准数据集。

    Args:
        manifest_path: manifest.json 文件路径。
        expected_manifest_sha256: 期望的 manifest SHA-256，用于校验不可变 bundle。
        require_reviewed: 是否要求数据集已获人工评审批准，默认 True。
    """
    path = Path(manifest_path).resolve()
    # ① 安全关卡：manifest 实际哈希必须与预期一致（不可变 bundle 校验）
    actual_manifest_hash = sha256_file(path)
    if actual_manifest_hash != expected_manifest_sha256:
        raise EvidenceGateBenchmarkError("manifest_sha256 mismatch")
    manifest = _object(path, label="manifest")
    if manifest.get("schema_version") != EVIDENCE_GATE_MANIFEST_SCHEMA:
        raise EvidenceGateBenchmarkError("unsupported manifest schema_version")
    reference_answer_schema = manifest.get("reference_answer_schema_version")
    if reference_answer_schema not in {None, REFERENCE_ANSWER_SCHEMA_VERSION}:
        raise EvidenceGateBenchmarkError("unsupported reference_answer_schema_version")
    dataset_version = _required_string(
        manifest.get("dataset_version"), field="dataset_version"
    )
    # ② 校验 manifest 声明的类别与响应状态必须覆盖全集
    categories = frozenset(
        _string_tuple(
            manifest.get("answerability_labels"), field="answerability_labels"
        )
    )
    if categories != REQUIRED_EVIDENCE_GATE_CATEGORIES:
        raise EvidenceGateBenchmarkError("manifest answerability_labels are incomplete")
    declared_statuses = frozenset(
        _string_tuple(
            manifest.get("expected_response_statuses"),
            field="expected_response_statuses",
        )
    )
    if declared_statuses != SUPPORTED_RESPONSE_STATUSES:
        raise EvidenceGateBenchmarkError(
            "manifest expected_response_statuses are incomplete"
        )
    # ③ 校验所有可选补充产物（目录、预评审报告、夹具示例、自动预检）的哈希
    for file_field, hash_field in (
        ("context_catalog_file", "context_catalog_sha256"),
        ("pre_review_report_file", "pre_review_report_sha256"),
        ("fixture_profile_example_file", "fixture_profile_example_sha256"),
        ("automated_pre_review_file", "automated_pre_review_sha256"),
    ):
        _validate_declared_artifact(
            path.parent,
            manifest,
            file_field=file_field,
            hash_field=hash_field,
        )
    # ④ 以安全子路径解析 cases 与评审记录，并核对其哈希
    cases_path = _safe_child(
        path.parent, manifest.get("cases_file"), field="cases_file"
    )
    cases_hash = _required_string(
        manifest.get("cases_sha256"), field="cases_sha256", maximum=64
    )
    if sha256_file(cases_path) != cases_hash:
        raise EvidenceGateBenchmarkError("cases_sha256 mismatch")
    review_path = _safe_child(
        path.parent,
        manifest.get("review_record_file"),
        field="review_record_file",
    )
    review_hash = _required_string(
        manifest.get("review_record_sha256"),
        field="review_record_sha256",
        maximum=64,
    )
    if sha256_file(review_path) != review_hash:
        raise EvidenceGateBenchmarkError("review_record_sha256 mismatch")
    # ⑤ 加载并校验用例与评审记录
    cases = _load_cases(
        cases_path,
        declared_statuses=declared_statuses,
        enforce_reviewed_contract=bool(
            manifest.get("current_corpus_draft_id")
            and manifest.get("current_corpus_snapshot_sha256")
        ),
        require_reference_answers=(
            reference_answer_schema == REFERENCE_ANSWER_SCHEMA_VERSION
        ),
    )
    expected_case_count = manifest.get("expected_case_count")
    if expected_case_count is not None:
        if (
            isinstance(expected_case_count, bool)
            or not isinstance(expected_case_count, int)
            or expected_case_count != len(cases)
        ):
            raise EvidenceGateBenchmarkError("expected_case_count mismatch")
    selection_policy = manifest.get("selection_policy")
    if selection_policy is not None:
        if not isinstance(selection_policy, dict):
            raise EvidenceGateBenchmarkError("selection_policy invalid")
        source_case_ids = selection_policy.get("source_case_ids")
        source_case_count = selection_policy.get("source_case_count")
        if (
            isinstance(source_case_count, bool)
            or not isinstance(source_case_count, int)
            or source_case_count < len(cases)
            or not isinstance(source_case_ids, list)
            or len(source_case_ids) != len(cases)
            or len(set(source_case_ids)) != len(source_case_ids)
            or not all(isinstance(case_id, str) and case_id for case_id in source_case_ids)
        ):
            raise EvidenceGateBenchmarkError("selection_policy invalid")
    review = _object(review_path, label="review record")
    review_status = _validate_review(
        review,
        dataset_version=dataset_version,
        case_ids={case.id for case in cases},
        require_reviewed=require_reviewed,
    )
    # ⑥ 一致性关卡：manifest 记录的评审状态必须与评审文件一致
    if manifest.get("review_status") != review_status:
        raise EvidenceGateBenchmarkError("manifest review_status mismatch")
    return EvidenceGateDataset(
        cases,
        source_path=cases_path,
        dataset_version=dataset_version,
        manifest_sha256=actual_manifest_hash,
        cases_sha256=cases_hash,
        review_record_sha256=review_hash,
        review_status=review_status,
    )


__all__ = [
    "EVIDENCE_GATE_MANIFEST_SCHEMA",
    "EVIDENCE_GATE_REVIEW_SCHEMA",
    "REFERENCE_ANSWER_SCHEMA_VERSION",
    "EvidenceGateBenchmarkError",
    "EvidenceGateCase",
    "EvidenceGateDataset",
    "REQUIRED_EVIDENCE_GATE_CATEGORIES",
    "SUPPORTED_EVIDENCE_STATES",
    "SUPPORTED_RESPONSE_STATUSES",
    "load_evidence_gate_benchmark",
    "sha256_file",
]
