"""证据数据集的受控编写、maker-checker 双人复核与冻结。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import unicodedata
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.evidence_gate.benchmark import (
    EVIDENCE_GATE_MANIFEST_SCHEMA,
    EVIDENCE_GATE_REVIEW_SCHEMA,
    REFERENCE_ANSWER_SCHEMA_VERSION,
    REQUIRED_EVIDENCE_GATE_CATEGORIES,
    SUPPORTED_EVIDENCE_STATES,
    SUPPORTED_RESPONSE_STATUSES,
    load_evidence_gate_benchmark,
    reviewed_case_contract_errors,
    sha256_file,
)



_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
# 租户 ID 格式：字母/数字开头，最长 128，允许字母、数字、下划线、点、冒号与连字符
_TENANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
# 授权类（越权访问）Fixture 名称：历史配置必须带租户 ID 才算完成
_AUTHORIZATION_FIXTURES = (
    "finance_only_user_against_hr_document",
    "finance_only_user_against_administration_document",
)
# 用例 ID 格式：字母/数字开头，最长 128，允许字母、数字、下划线、点、冒号与连字符
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
# 组织命名空间目录名格式：组织 ID 的 sha256 前 24 位十六进制，不在磁盘暴露原始 ID
_ORG_NAMESPACE = re.compile(r"^[0-9a-f]{24}$")
# 用例评审状态全集：草稿 / 待复核 / 已通过
_CASE_STATUSES = frozenset({"draft", "pending_review", "approved"})
# 复核决定全集：通过 / 驳回
_DECISIONS = frozenset({"approve", "reject"})
# 当前语料工作区必须覆盖的 Fixture 名称全集
_REQUIRED_FIXTURES = frozenset(
    {
        "frozen_corpus_absence_check",
        "equal_authority_conflicting_documents",
        "finance_only_user_against_hr_document",
        "finance_only_user_against_administration_document",
        "bm25_unavailable_dense_graph_available",
        "all_retrieval_branches_unavailable",
        "prompt_injection_safety_fixture",
    }
)
# 服务端治理的编写/复核字段：用例校验时先剥离再由服务端重置，禁止客户端直接写入
_AUTHORING_FIELDS = frozenset(
    {
        "review_status",
        "case_revision",
        "last_editor_id",
        "department_id",
        "last_edited_at",
        "submitted_at",
        "reviewer_id",
        "reviewer_department_id",
        "reviewed_at",
        "review_reason",
        "rejection_reason",
        "context_summaries",
    }
)

# 代表性模板导入支持的标准类目（按此顺序逐类导入）
_CURRENT_CORPUS_TEMPLATE_STANDARD_CATEGORIES = (
    "fully_answerable",
    "partially_answerable",
    "background_only",
    "missing_version_or_date",
    "missing_business_record",
)
# 模板导入返回的 imported/skipped 列表长度上限 = 必需 Fixture 数 + 标准类目数
_CURRENT_CORPUS_TEMPLATE_IMPORT_LIMIT = (
    len(_REQUIRED_FIXTURES) + len(_CURRENT_CORPUS_TEMPLATE_STANDARD_CATEGORIES)
)
_SOURCE_EVALUATION_DATASETS = frozenset({"evidence-gates-v1", "evidence-gates-v1-routine"})
# 候选答案只能补齐以下会进入答案质量评分的类别；拒答、权限和不可用路由
# 使用其既有的行为契约，不应凭空编造语义答案。
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
_REFERENCE_ANSWER_CANDIDATE_SCHEMA = "evidence-reference-answer-candidates-v1"
_REFERENCE_ANSWER_CANDIDATE_EDITOR = "system-reference-answer-candidate-v1"
_REFERENCE_ANSWER_CONTRACT_FIELDS = (
    "question",
    "category",
    "expected_response_status",
    "expected_evidence_states",
    "expected_reason_codes",
    "expected_source_document_ids",
    "expected_evidence_sections",
    "expected_missing_information_fields",
    "required_fixture",
    "expected_branch_availability",
)
# 目录标题开头的数字归档前缀（如 “173印章管理” 中的 “173”），仅用于保守的标题对齐
_SOURCE_TITLE_PREFIX = re.compile(r"^\s*\d{1,4}[\s._-]*")


class EvidenceReviewError(ValueError):
    """受限工作区操作的基础错误。"""

    code = "evidence_review_error"


class EvidenceReviewNotFound(EvidenceReviewError):
    """请求范围内看不到目标数据集或用例时抛出。"""

    code = "not_found"


class EvidenceReviewConflict(EvidenceReviewError):
    """revision 乐观并发冲突与状态机冲突时抛出。"""

    code = "conflict"


class EvidenceReviewPermissionError(EvidenceReviewError):
    """已认证账号的部门职责不足（越权）时抛出。"""

    code = "forbidden"


class EvidenceReviewValidationError(EvidenceReviewError):
    """用例或 Fixture 输入非法时抛出。"""

    code = "validation_error"


def _utc_now() -> str:
    """返回当前 UTC 时间的 ISO-8601 字符串。"""
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """以原子替换方式把 JSON 对象写入文件。

    Args:
        path: 目标文件路径，父目录不存在时自动创建。
        value: 要序列化写入的 JSON 对象。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写入带进程/线程标识的临时文件后原子替换，避免并发读到半截内容
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """以原子替换方式把记录列表逐行写入 JSONL 文件。

    Args:
        path: 目标文件路径，父目录不存在时自动创建。
        records: 要写入的 JSON 对象列表，每行一条。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写入带进程/线程标识的临时文件后原子替换，避免并发读到半截内容
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    """读取并解析 JSON 文件，非对象结构视为非法。

    Args:
        path: 要读取的 JSON 文件路径。
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    # 安全关卡：顶层必须是 JSON 对象，防止结构漂移进入后续治理逻辑
    if not isinstance(value, dict):
        raise EvidenceReviewValidationError("invalid_workspace_json")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """逐行读取 JSONL 文件并返回对象列表（忽略空行）。

    Args:
        path: 要读取的 JSONL 文件路径。
    """
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # 跳过空行；非对象行视为非法
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise EvidenceReviewValidationError("invalid_workspace_jsonl")
        records.append(value)
    return records


def _string_list(value: Any, field: str, *, allow_empty: bool = True) -> list[str]:
    """校验并归一化字符串列表：去空白、去重、限制单项长度。

    Args:
        value: 待校验的原始值，必须是字符串列表。
        field: 字段名，用于拼接校验错误码。
        allow_empty: 为 False 时空列表也视为非法。
    """
    if not isinstance(value, list) or (not allow_empty and not value):
        raise EvidenceReviewValidationError(f"invalid_{field}")
    normalized: list[str] = []
    # 去重时保持首次出现顺序；单项去空白后不得超过 256 字符
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item.strip()) > 256:
            raise EvidenceReviewValidationError(f"invalid_{field}")
        if item.strip() not in normalized:
            normalized.append(item.strip())
    return normalized


