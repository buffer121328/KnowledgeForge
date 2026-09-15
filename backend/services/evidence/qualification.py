"""Deterministic evidence qualification for authorized QA retrieval contexts."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from domain.evidence import (
    EvidenceAssessment,
    EvidenceReasonCode,
    EvidenceState,
    MissingInformation,
    QAResponseStatus,
)
from domain.knowledge import QueryIntent, RetrievedContext
from domain.retrieval import RetrievalStatus


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")
_CRITICAL_VALUE_RE = re.compile(
    r"(?:v(?:ersion)?\s*)?\d+(?:\.\d+)*(?:%|万元|亿元|元|万|亿|年|月|日|版)?",
    re.IGNORECASE,
)
_QUESTION_NOISE = frozenset(
    "的了呢吗啊是什么多少哪些哪个如何怎么请问告诉说明相关当前该本"
)
_HIGH_RISK_TERMS = frozenset(
    {"财务", "报销", "薪酬", "工资", "人事", "解雇", "法务", "合同", "合规", "安全", "权限"}
)
_CONFLICT_DIMENSION_TERMS = frozenset(
    {"金额", "预算", "日期", "时间", "生效", "版本", "上限", "下限", "比例", "编号", "条款"}
)
_CONFLICT_QUERY_MARKERS = frozenset({"两份", "两项", "相矛盾", "矛盾", "冲突", "优先级"})
_DYNAMIC_QUERY_MARKERS = frozenset(
    {
        "名单",
        "记录",
        "全文",
        "某员工",
        "某项",
        "某采购单",
        "某采购计划",
        "某订单",
        "完成情况",
        "执行结果",
        "审批结果",
        "签收",
        "违规事件",
        "盘点结果",
        "最终执行数据",
    }
)
_UNAVAILABLE_RECORD_MARKERS = frozenset(
    {"尚未发布", "尚未产生", "尚未发生", "未来", "下一年度", "未生成"}
)
_QUESTION_PART_SEPARATORS = ("同时", "以及", "并且", "；", ";")


def context_identity(context: RetrievedContext, index: int) -> str:
    """Return the bounded context identity shared by qualification and grounding."""
    metadata = context.metadata or {}
    # A canonical physical chunk ID is stronger than an adapter-local context ID.
    for key in ("chunk_id", "context_id", "id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value[:128]
    digest = hashlib.sha256(
        f"{context.source}\0{context.content}".encode("utf-8")
    ).hexdigest()[:20]
    return f"ctx_{index}_{digest}"


@dataclass(frozen=True, slots=True)
class EvidenceQualificationPolicy:
    """Validated deterministic thresholds and version identities."""

    background_threshold: float = 0.35
    gray_zone_lower: float = 0.45
    gray_zone_upper: float = 0.65
    direct_threshold: float = 0.75
    candidate_limit: int = 8
    policy_version: str = "evidence-composite-v2"
    calibration_version: str = "qualification-boundaries-v2"

    def __post_init__(self) -> None:
        values = (
            self.background_threshold,
            self.gray_zone_lower,
            self.gray_zone_upper,
            self.direct_threshold,
        )
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("evidence thresholds must be between zero and one")
        if values != tuple(sorted(values)):
            raise ValueError(
                "evidence thresholds must satisfy background <= gray lower <= gray upper <= direct"
            )
        if self.candidate_limit <= 0 or self.candidate_limit > 50:
            raise ValueError("candidate_limit must be between 1 and 50")


class EvidenceQualifier:
    """Classify evidence without relying on optional model runtimes."""

    def __init__(self, policy: EvidenceQualificationPolicy | None = None) -> None:
        self.policy = policy or EvidenceQualificationPolicy()

    def assess(
        self,
        question: str,
        contexts: Sequence[RetrievedContext],
        *,
        intent: QueryIntent,
        branch_statuses: Mapping[str, RetrievalStatus | str],
        reranker_scores: Mapping[str, float] | None = None,
    ) -> EvidenceAssessment:
        """Return a bounded response route for current authorized contexts."""
        del intent  # Intent remains part of the stable call contract for future policies.
        statuses = tuple(self._status(value) for value in branch_statuses.values())
        unavailable = sum(status is RetrievalStatus.UNAVAILABLE for status in statuses)
        if not contexts:
            if statuses and unavailable == len(statuses):
                return self._assessment(
                    EvidenceState.INSUFFICIENT_EVIDENCE,
                    QAResponseStatus.SOURCE_UNAVAILABLE,
                    (EvidenceReasonCode.ALL_DEPENDENCIES_UNAVAILABLE,),
                )
            if statuses and unavailable and not any(
                status in {RetrievalStatus.SUCCESS, RetrievalStatus.EMPTY}
                for status in statuses
            ):
                return self._assessment(
                    EvidenceState.INSUFFICIENT_EVIDENCE,
                    QAResponseStatus.SOURCE_UNAVAILABLE,
                    (EvidenceReasonCode.ALL_DEPENDENCIES_UNAVAILABLE,),
                )
            return self._assessment(
                EvidenceState.INSUFFICIENT_EVIDENCE,
                QAResponseStatus.INSUFFICIENT_EVIDENCE,
                (EvidenceReasonCode.ZERO_RESULTS,),
            )

        bounded = list(contexts[: self.policy.candidate_limit])
        valid: list[tuple[str, RetrievedContext]] = []
        for index, context in enumerate(bounded):
            context_id = self._context_id(context, index)
            if context.source.strip() and context.content.strip():
                valid.append((context_id, context))
        evaluated_ids = tuple(context_id for context_id, _ in valid)
        if not valid:
            return self._assessment(
                EvidenceState.INVALID_PROVENANCE,
                QAResponseStatus.INSUFFICIENT_EVIDENCE,
                (EvidenceReasonCode.INVALID_CONTEXT_PROVENANCE,),
            )

        valid_contexts = [context for _, context in valid]
        combined_content = "\n".join(context.content for context in valid_contexts)
        partial_outage = bool(unavailable and unavailable < max(len(statuses), 1))
        conflict = self._has_material_conflict(question, valid_contexts)
        # A question that explicitly asks how two authoritative/conflicting
        # documents should be handled must not silently fall through to a
        # normal answer merely because lexical coverage is high.  Require
        # independent source documents before asserting a material conflict.
        if not conflict and self._is_explicit_conflict_question(question):
            source_ids = {
                str((context.metadata or {}).get("source_document_id") or context.source).strip()
                for _, context in valid
            }
            conflict = len(source_ids) >= 2
        if self._requires_clarification(question, combined_content):
            return self._assessment(
                EvidenceState.RELEVANT_BACKGROUND,
                QAResponseStatus.NEEDS_CLARIFICATION,
                (EvidenceReasonCode.MISSING_QUESTION_DETAIL,),
                evaluated_ids=evaluated_ids,
            )
        if conflict:
            reasons = [EvidenceReasonCode.MATERIAL_CONFLICT]
            response_status = QAResponseStatus.CONFLICTING_EVIDENCE
            if self._is_high_risk(question):
                reasons.append(EvidenceReasonCode.HIGH_RISK_REVIEW)
                response_status = QAResponseStatus.HUMAN_REVIEW_REQUIRED
            if partial_outage:
                reasons.append(EvidenceReasonCode.PARTIAL_DEPENDENCY_UNAVAILABLE)
            return self._assessment(
                EvidenceState.CONFLICTING_EVIDENCE,
                response_status,
                tuple(reasons),
                evaluated_ids=evaluated_ids,
                supporting_ids=evaluated_ids,
            )

        requests_dynamic_record = self._requests_dynamic_record(question)
        unavailable_record = requests_dynamic_record and any(
            marker in question for marker in _UNAVAILABLE_RECORD_MARKERS
        )
        if unavailable_record:
            # A policy or template cannot prove a future record that the question
            # itself says has not been published/created.  This is a normal
            # no-evidence route; sensitive words in the policy title must not
            # promote it to human review.
            return self._assessment(
                EvidenceState.INSUFFICIENT_EVIDENCE,
                QAResponseStatus.INSUFFICIENT_EVIDENCE,
                (EvidenceReasonCode.ZERO_RESULTS,),
                evaluated_ids=evaluated_ids,
            )

        missing_dynamic_record = requests_dynamic_record and not self._has_requested_record(
            question, combined_content
        )
        static_question = self._static_question_part(question)
        static_support = (
            self._qualified_coverage(static_question, combined_content)
            if static_question
            else 0.0
        )
        if missing_dynamic_record and static_support >= self.policy.gray_zone_lower:
            supporting_ids = tuple(
                context_id
                for context_id, context in valid
                if self._qualified_coverage(static_question, context.content)
                >= self.policy.background_threshold
            )
            return self._assessment(
                EvidenceState.PARTIAL_EVIDENCE,
                QAResponseStatus.PARTIALLY_ANSWERED,
                (EvidenceReasonCode.PARTIAL_SUPPORT,),
                evaluated_ids=evaluated_ids,
                supporting_ids=supporting_ids or (evaluated_ids[0],),
                missing_information=(self._missing_dynamic_information(question),),
            )

        scored = [
            (self._qualified_coverage(question, context.content), context_id)
            for context_id, context in valid
        ]
        scored.sort(reverse=True)
        best_lexical_score = scored[0][0]
        aggregate_coverage = self._qualified_coverage(
            question,
            "\n".join(context.content for _, context in valid),
        )
        branch_support = self._branch_support(
            [context for _, context in valid],
            statuses=statuses,
        )
        provenance_support = max(
            self._provenance_support(context) for _, context in valid
        )
        # A direct route may be established by one self-contained chunk or by a
        # set of canonically identified chunks that jointly cover the question.
        # Missing critical dimensions remain capped by _qualified_coverage, so
        # retrieval agreement cannot manufacture an absent amount/date/version.
        base_score = max(
            best_lexical_score,
            min(
                1.0,
                aggregate_coverage * 0.85
                + branch_support * 0.10
                + provenance_support * 0.05,
            ),
        )
        reranker_supported_ids = self._reranker_supported_ids(
            valid,
            question=question,
            reranker_scores=reranker_scores,
        )
        reranker_strengthened = bool(
            base_score < self.policy.direct_threshold and reranker_supported_ids
        )
        best_score = (
            self.policy.direct_threshold if reranker_strengthened else base_score
        )
        supporting_ids = tuple(
            context_id
            for score, context_id in scored
            if score >= self.policy.background_threshold
        )
        outage_reason = (
            (EvidenceReasonCode.PARTIAL_DEPENDENCY_UNAVAILABLE,)
            if partial_outage
            else ()
        )

        if best_score >= self.policy.direct_threshold:
            reasons = [EvidenceReasonCode.DIRECT_SUPPORT]
            if reranker_strengthened:
                reasons.append(EvidenceReasonCode.RERANKER_SUPPORT)
            reasons.extend(outage_reason)
            return self._assessment(
                EvidenceState.DIRECT_EVIDENCE,
                QAResponseStatus.ANSWERED,
                tuple(reasons),
                evaluated_ids=evaluated_ids,
                supporting_ids=(
                    reranker_supported_ids
                    if reranker_strengthened
                    else supporting_ids or (scored[0][1],)
                ),
            )
        if best_score >= self.policy.gray_zone_lower:
            response_status = QAResponseStatus.PARTIALLY_ANSWERED
            reasons = [EvidenceReasonCode.PARTIAL_SUPPORT, *outage_reason]
            if self._is_high_risk(question):
                response_status = QAResponseStatus.HUMAN_REVIEW_REQUIRED
                reasons.append(EvidenceReasonCode.HIGH_RISK_REVIEW)
            return self._assessment(
                EvidenceState.PARTIAL_EVIDENCE,
                response_status,
                tuple(reasons),
                evaluated_ids=evaluated_ids,
                supporting_ids=supporting_ids or (scored[0][1],),
                missing_information=(
                    MissingInformation(
                        field="unsupported_question_parts",
                        description="当前资料只能支持问题的一部分，需要补充更直接的记录。",
                    ),
                ),
            )
        if best_score >= self.policy.background_threshold:
            return self._assessment(
                EvidenceState.RELEVANT_BACKGROUND,
                QAResponseStatus.INSUFFICIENT_EVIDENCE,
                (EvidenceReasonCode.BACKGROUND_ONLY, *outage_reason),
                evaluated_ids=evaluated_ids,
            )
        return self._assessment(
            EvidenceState.IRRELEVANT,
            QAResponseStatus.INSUFFICIENT_EVIDENCE,
            (EvidenceReasonCode.IRRELEVANT_CONTEXT, *outage_reason),
            evaluated_ids=evaluated_ids,
        )

    def _assessment(
        self,
        state: EvidenceState,
        response_status: QAResponseStatus,
        reasons: tuple[EvidenceReasonCode, ...],
        *,
        evaluated_ids: tuple[str, ...] = (),
        supporting_ids: tuple[str, ...] = (),
        missing_information: tuple[MissingInformation, ...] = (),
    ) -> EvidenceAssessment:
        return EvidenceAssessment(
            states=(state,),
            response_status=response_status,
            reason_codes=reasons,
            evaluated_context_ids=evaluated_ids,
            supporting_context_ids=supporting_ids,
            missing_information=missing_information,
            policy_version=self.policy.policy_version,
            calibration_version=self.policy.calibration_version,
        )

    @staticmethod
    def _status(value: RetrievalStatus | str) -> RetrievalStatus:
        return value if isinstance(value, RetrievalStatus) else RetrievalStatus(value)

    @staticmethod
    def _context_id(context: RetrievedContext, index: int) -> str:
        return context_identity(context, index)

    def _qualified_coverage(self, question: str, content: str) -> float:
        """Cap lexical coverage when requested factual dimensions are absent."""
        coverage = self._coverage(question, content)
        required_dimensions = {
            term for term in _CONFLICT_DIMENSION_TERMS if term in question
        }
        if required_dimensions:
            covered_dimensions = {
                term for term in required_dimensions if term in content
            }
            if not covered_dimensions:
                coverage = min(coverage, self.policy.background_threshold)
            elif covered_dimensions != required_dimensions:
                coverage = min(coverage, self.policy.gray_zone_upper)
        # Lexical overlap with a policy document is not evidence of a current
        # business record.  Dynamic requests require an explicit record-like
        # marker in the retrieved text; otherwise keep them in the background
        # or partial zone so they cannot be routed as fully answered.
        if (
            self._requests_dynamic_record(question)
            and ("仅凭" in question or not self._has_record_marker(content))
        ):
            return min(coverage, self.policy.background_threshold)
        return coverage

    @classmethod
    def _is_explicit_conflict_question(cls, question: str) -> bool:
        return any(marker in question for marker in _CONFLICT_QUERY_MARKERS)

    @classmethod
    def _requests_dynamic_record(cls, question: str) -> bool:
        return any(marker in question for marker in _DYNAMIC_QUERY_MARKERS)

    @staticmethod
    def _static_question_part(question: str) -> str:
        """Return the independently answerable clause before a dynamic request."""
        positions = [
            question.find(separator)
            for separator in _QUESTION_PART_SEPARATORS
            if separator in question
        ]
        if not positions:
            return ""
        return question[: min(positions)].strip(" ，,？?")

    @staticmethod
    def _has_requested_record(question: str, content: str) -> bool:
        """Require query-specific record evidence, not generic policy wording."""
        if "名单" in question and any(term in question for term in ("空缺岗位", "空缺职位")):
            return bool(
                re.search(
                    r"(?:空缺|在招|招聘)(?:岗位|职位)(?:名单|清单|列表|明细|\s*[:：])",
                    content,
                )
            )
        if "某员工" in question:
            return "员工编号" in content and any(
                marker in content for marker in ("当前环节", "入职状态", "试用状态")
            )
        if "完成情况" in question:
            return any(
                marker in content
                for marker in ("实际完成", "已完成", "完成日期", "完成率")
            )
        if "最终执行数据" in question or "执行结果" in question:
            return any(
                marker in content
                for marker in ("实际执行", "执行日期", "执行金额", "执行统计")
            )
        # 具体单据/事件类动态问题（违规事件、审批结果、某订单/某项的时间或
        # 结果）不能用通用词表兜底：制度条文本身常含"记录/报表/审批结果"等
        # 词，会把"流程要求"误判为"业务记录存在"。这类问题必须命中与问题
        # 对应的记录性证据（实例编号/日期/结果值），否则视为记录缺失。
        if "违规事件" in question:
            return bool(
                re.search(r"违规事件.{0,12}(?:编号|日期|名单|处理结果)", content)
            )
        if "审批结果" in question:
            return bool(
                re.search(r"审批结果[:：]\s*\S+", content)
                or re.search(r"(?:审批|审核).{0,10}(?:通过|驳回|批准|同意)[^，。]{0,10}(?:日期|意见)", content)
            )
        if re.search(r"某(?:订单|项|采购|员工|合同)", question) and (
            "何时" in question or "哪个节点" in question or "结果是什么" in question
        ):
            return bool(
                re.search(r"(?:订单|单号|编号).{0,8}(?:\d{4,}|[A-Z]{2,}-?\d+)", content)
                or re.search(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}", content)
            )
        return EvidenceQualifier._has_record_marker(content)

    @staticmethod
    def _missing_dynamic_information(question: str) -> MissingInformation:
        """Describe the bounded current-record slot required by the question."""
        if "名单" in question and any(term in question for term in ("空缺岗位", "空缺职位")):
            field = "current_vacancy_business_record"
            description = "缺少截至指定日期的实际空缺岗位名单。"
        elif "招聘" in question and "完成情况" in question:
            field = "current_recruitment_progress_record"
            description = "缺少本月各招聘环节的实际进度记录。"
        elif "某员工" in question and any(term in question for term in ("入职", "试用", "环节")):
            field = "employee_current_onboarding_record"
            description = "缺少该员工当前入职或试用环节的业务记录。"
        else:
            field = "current_business_record"
            description = "缺少问题所要求的当前实际业务记录。"
        return MissingInformation(field=field, description=description)

    @staticmethod
    def _requires_clarification(question: str, content: str) -> bool:
        asks_version = "版本" in question or "哪一版" in question
        asks_effective_date = "仍然生效" in question or "现行" in question
        if not (asks_version or asks_effective_date):
            return False
        values = EvidenceQualifier._dimension_values("版本", content)
        dates = EvidenceQualifier._dimension_values("日期", content)
        return (asks_version and not values) or (asks_effective_date and not dates)

    @staticmethod
    def _has_record_marker(content: str) -> bool:
        return any(
            marker in content
            for marker in (
                "记录",
                "名单",
                "台账",
                "报表",
                "订单号",
                "资产编号",
                "签收",
                "已开具",
                "完成情况",
                "执行结果",
                "审批结果",
                "违规事件",
                "盘点结果",
            )
        )

    @staticmethod
    def _provenance_support(context: RetrievedContext) -> float:
        """Score only bounded identity completeness, never source text content."""
        metadata = context.metadata or {}
        chunk_id = str(
            metadata.get("chunk_id") or metadata.get("context_id") or ""
        ).strip()
        document_id = str(
            metadata.get("source_document_id") or metadata.get("doc_id") or ""
        ).strip()
        if chunk_id and document_id:
            return 1.0
        if chunk_id:
            return 0.5
        return 0.0

    def _reranker_supported_ids(
        self,
        valid: Sequence[tuple[str, RetrievedContext]],
        *,
        question: str,
        reranker_scores: Mapping[str, float] | None,
    ) -> tuple[str, ...]:
        """Return canonical candidates with finite high reranker support.

        Reranker support may strengthen only a lexically anchored candidate with
        complete canonical provenance.  It never manufactures support for a
        missing business record or an unrelated candidate.
        """
        if not reranker_scores or self._requests_dynamic_record(question):
            return ()
        supported: list[str] = []
        for context_id, context in valid:
            raw_score = reranker_scores.get(context_id)
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(raw_score)
                or raw_score < 0.85
                or raw_score > 1.0
            ):
                continue
            if self._provenance_support(context) < 1.0:
                continue
            if self._qualified_coverage(question, context.content) < self.policy.background_threshold:
                continue
            supported.append(context_id)
        return tuple(supported)

    @staticmethod
    def _branch_support(
        contexts: Sequence[RetrievedContext],
        *,
        statuses: Sequence[RetrievalStatus],
    ) -> float:
        """Return normalized cross-branch agreement from RRF provenance metadata."""
        available = max(
            1,
            sum(
                status in {RetrievalStatus.SUCCESS, RetrievalStatus.EMPTY}
                for status in statuses
            ),
        )
        maximum = 0
        for context in contexts:
            fusion = (context.metadata or {}).get("fusion")
            ranks = fusion.get("source_ranks") if isinstance(fusion, dict) else None
            if isinstance(ranks, dict):
                maximum = max(
                    maximum,
                    sum(
                        isinstance(branch, str)
                        and isinstance(rank, int)
                        and not isinstance(rank, bool)
                        and rank >= 1
                        for branch, rank in ranks.items()
                    ),
                )
        if maximum == 0:
            maximum = 1
        return min(maximum / available, 1.0)

    @classmethod
    def _coverage(cls, question: str, content: str) -> float:
        question_tokens = cls._tokens(question)
        if not question_tokens:
            return 0.0
        content_tokens = cls._tokens(content)
        return len(question_tokens & content_tokens) / len(question_tokens)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        raw = [token.lower() for token in _TOKEN_RE.findall(value)]
        return {
            token
            for token in raw
            if token not in _QUESTION_NOISE and (len(token) > 1 or token.isdigit() or "\u4e00" <= token <= "\u9fff")
        }

    @staticmethod
    def _is_high_risk(question: str) -> bool:
        return any(term in question for term in _HIGH_RISK_TERMS)

    @staticmethod
    def _has_material_conflict(
        question: str, contexts: Sequence[RetrievedContext]
    ) -> bool:
        dimensions = {
            term for term in _CONFLICT_DIMENSION_TERMS if term in question
        }
        for dimension in dimensions:
            value_sets = [
                EvidenceQualifier._dimension_values(dimension, context.content)
                for context in contexts
                if dimension in context.content
            ]
            non_empty = [values for values in value_sets if values]
            if len(non_empty) < 2:
                continue
            shared = set.intersection(*non_empty)
            distinct = set.union(*non_empty) - shared
            if len(distinct) >= 2:
                return True
        return False

    @staticmethod
    def _dimension_values(dimension: str, content: str) -> set[str]:
        """Extract values compatible with one requested factual dimension."""
        values = {match.lower() for match in _CRITICAL_VALUE_RE.findall(content)}
        if dimension in {"金额", "预算", "上限", "下限", "比例"}:
            return {
                value
                for value in values
                if value.endswith(("%", "元", "万元", "亿元", "万", "亿"))
            }
        if dimension in {"日期", "时间", "生效"}:
            return {
                value for value in values if value.endswith(("年", "月", "日"))
            }
        if dimension == "版本":
            return {
                value
                for value in values
                if value.startswith("v") or value.endswith("版")
            }
        return values
