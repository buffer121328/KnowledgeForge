"""ATDD-style tests for deterministic evidence qualification."""

from __future__ import annotations

from services.evidence.qualification import EvidenceQualificationPolicy, EvidenceQualifier
from domain.evidence import EvidenceReasonCode, EvidenceState, QAResponseStatus
from domain.knowledge import QueryIntent, RetrievedContext
from domain.retrieval import RetrievalStatus


def context(content: str, *, source: str = "policy.md", context_id: str = "ctx_1") -> RetrievedContext:
    return RetrievedContext(
        content=content,
        source=source,
        score=0.01,
        retrieval_type="vector",
        metadata={"context_id": context_id},
    )


def fused_context(
    content: str,
    *,
    context_id: str,
    branches: tuple[str, ...],
) -> RetrievedContext:
    return RetrievedContext(
        content=content,
        source="policy.md",
        score=1.0,
        retrieval_type="vector",
        metadata={
            "context_id": context_id,
            "chunk_id": context_id,
            "source_document_id": "doc-1",
            "fusion": {
                "source_ranks": {
                    branch: index + 1 for index, branch in enumerate(branches)
                },
            },
        },
    )


def qualifier() -> EvidenceQualifier:
    return EvidenceQualifier(
        EvidenceQualificationPolicy(
            background_threshold=0.2,
            gray_zone_lower=0.35,
            gray_zone_upper=0.65,
            direct_threshold=0.7,
            policy_version="evidence-v1",
            calibration_version="test-v1",
        )
    )


def test_direct_evidence_authorizes_answer() -> None:
    result = qualifier().assess(
        "员工年度培训预算是多少",
        [context("员工年度培训预算是100万元，适用于2026年度。")],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.primary_state is EvidenceState.DIRECT_EVIDENCE
    assert result.response_status is QAResponseStatus.ANSWERED
    assert result.generation_allowed is True


def test_current_policy_fact_is_not_treated_as_business_record_request() -> None:
    result = qualifier().assess(
        "当前员工年度培训预算是多少",
        [context("员工年度培训预算是100万元，适用于2026年度。")],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.response_status is QAResponseStatus.ANSWERED


def test_canonical_chunk_id_wins_over_adapter_context_id() -> None:
    value = RetrievedContext(
        content="员工年度培训预算是100万元。",
        source="policy.md",
        score=1.0,
        retrieval_type="vector",
        metadata={
            "context_id": "adapter-context",
            "chunk_id": "canonical-chunk",
        },
    )
    result = qualifier().assess(
        "员工年度培训预算是多少",
        [value],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.evaluated_context_ids == ("canonical-chunk",)


def test_background_only_does_not_authorize_fact_answer() -> None:
    result = qualifier().assess(
        "员工年度培训预算是多少",
        [context("公司重视员工培训，并每年组织多种学习活动。")],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.primary_state is EvidenceState.RELEVANT_BACKGROUND
    assert result.response_status is QAResponseStatus.INSUFFICIENT_EVIDENCE
    assert EvidenceReasonCode.BACKGROUND_ONLY in result.reason_codes


def test_partial_coverage_returns_partial_answer_route() -> None:
    result = qualifier().assess(
        "员工年度培训预算和生效日期是什么",
        [context("员工年度培训预算是100万元。")],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.primary_state is EvidenceState.PARTIAL_EVIDENCE
    assert result.response_status is QAResponseStatus.PARTIALLY_ANSWERED


def test_material_conflict_stops_automatic_answer() -> None:
    result = qualifier().assess(
        "员工年度培训预算金额是多少",
        [
            context("员工年度培训预算金额是100万元。", context_id="ctx_1"),
            context("员工年度培训预算金额是120万元。", source="revision.md", context_id="ctx_2"),
        ],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.primary_state is EvidenceState.CONFLICTING_EVIDENCE
    assert result.response_status is QAResponseStatus.CONFLICTING_EVIDENCE


def test_high_risk_conflict_requires_human_review() -> None:
    result = qualifier().assess(
        "财务报销金额上限是多少",
        [
            context("财务报销金额上限是5000元。", context_id="ctx_1"),
            context("财务报销金额上限是8000元。", source="revision.md", context_id="ctx_2"),
        ],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.response_status is QAResponseStatus.HUMAN_REVIEW_REQUIRED
    assert EvidenceReasonCode.HIGH_RISK_REVIEW in result.reason_codes


def test_invalid_provenance_is_not_supporting_evidence() -> None:
    result = qualifier().assess(
        "员工年度培训预算是多少",
        [context("员工年度培训预算是100万元。", source="")],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.primary_state is EvidenceState.INVALID_PROVENANCE
    assert result.response_status is QAResponseStatus.INSUFFICIENT_EVIDENCE


def test_successful_zero_retrieval_is_a_no_answer_not_an_outage() -> None:
    result = qualifier().assess(
        "不存在的记录",
        [],
        intent=QueryIntent.FACTOID,
        branch_statuses={
            "vector": RetrievalStatus.EMPTY,
            "bm25": RetrievalStatus.EMPTY,
            "graph": RetrievalStatus.EMPTY,
        },
    )

    assert result.primary_state is EvidenceState.INSUFFICIENT_EVIDENCE
    assert result.response_status is QAResponseStatus.INSUFFICIENT_EVIDENCE
    assert result.reason_codes == (EvidenceReasonCode.ZERO_RESULTS,)


def test_partial_branch_failure_can_answer_with_direct_evidence() -> None:
    result = qualifier().assess(
        "员工年度培训预算是多少",
        [context("员工年度培训预算是100万元。")],
        intent=QueryIntent.FACTOID,
        branch_statuses={
            "vector": RetrievalStatus.SUCCESS,
            "graph": RetrievalStatus.UNAVAILABLE,
        },
    )

    assert result.response_status is QAResponseStatus.ANSWERED
    assert EvidenceReasonCode.PARTIAL_DEPENDENCY_UNAVAILABLE in result.reason_codes


def test_total_dependency_failure_is_distinct_from_no_answer() -> None:
    result = qualifier().assess(
        "员工年度培训预算是多少",
        [],
        intent=QueryIntent.FACTOID,
        branch_statuses={
            "vector": RetrievalStatus.UNAVAILABLE,
            "bm25": RetrievalStatus.UNAVAILABLE,
            "graph": RetrievalStatus.UNAVAILABLE,
        },
    )

    assert result.response_status is QAResponseStatus.SOURCE_UNAVAILABLE
    assert result.reason_codes == (EvidenceReasonCode.ALL_DEPENDENCIES_UNAVAILABLE,)


def test_qualification_does_not_require_cross_encoder_signal() -> None:
    result = qualifier().assess(
        "员工年度培训预算是多少",
        [context("员工年度培训预算是100万元。")],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
        reranker_scores=None,
    )

    assert result.response_status is QAResponseStatus.ANSWERED


def test_finite_reranker_support_can_strengthen_anchored_canonical_evidence() -> None:
    boosted = EvidenceQualifier(
        EvidenceQualificationPolicy(
            background_threshold=0.2,
            gray_zone_lower=0.35,
            gray_zone_upper=0.65,
            direct_threshold=0.9,
            policy_version="evidence-v2",
            calibration_version="test-v2",
        )
    )
    value = fused_context(
        "员工培训申请由部门负责人审批。",
        context_id="training-approval",
        branches=("vector", "bm25"),
    )
    result = boosted.assess(
        "培训申请的批准人是谁",
        [value],
        intent=QueryIntent.FACTOID,
        branch_statuses={
            "vector": RetrievalStatus.SUCCESS,
            "bm25": RetrievalStatus.SUCCESS,
        },
        reranker_scores={"training-approval": 0.95},
    )

    assert result.response_status is QAResponseStatus.ANSWERED
    assert EvidenceReasonCode.RERANKER_SUPPORT in result.reason_codes


def test_joint_canonical_evidence_can_authorize_answer() -> None:
    result = qualifier().assess(
        "年度培训预算金额和生效日期是什么",
        [
            fused_context(
                "年度培训预算金额是100万元。",
                context_id="chunk-1",
                branches=("vector", "bm25"),
            ),
            fused_context(
                "该预算生效日期为2026年1月1日。",
                context_id="chunk-2",
                branches=("vector", "graph"),
            ),
        ],
        intent=QueryIntent.FACTOID,
        branch_statuses={
            "vector": RetrievalStatus.SUCCESS,
            "bm25": RetrievalStatus.SUCCESS,
            "graph": RetrievalStatus.SUCCESS,
        },
    )

    assert result.primary_state is EvidenceState.DIRECT_EVIDENCE
    assert result.response_status is QAResponseStatus.ANSWERED


def test_cross_branch_agreement_cannot_replace_a_missing_critical_dimension() -> None:
    result = qualifier().assess(
        "年度培训预算金额和生效日期是什么",
        [
            fused_context(
                "年度培训预算金额是100万元，各系统记录一致。",
                context_id="chunk-1",
                branches=("vector", "bm25", "graph"),
            ),
        ],
        intent=QueryIntent.FACTOID,
        branch_statuses={
            "vector": RetrievalStatus.SUCCESS,
            "bm25": RetrievalStatus.SUCCESS,
            "graph": RetrievalStatus.SUCCESS,
        },
    )

    assert result.primary_state is EvidenceState.PARTIAL_EVIDENCE
    assert result.response_status is QAResponseStatus.PARTIALLY_ANSWERED


def test_dynamic_business_record_request_is_not_answered_by_policy_text() -> None:
    result = qualifier().assess(
        "仅凭资产盘点制度能否确认某项固定资产的当前盘点结果",
        [context("制度规定固定资产盘点范围、方式和数据准确性要求。")],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.primary_state is EvidenceState.RELEVANT_BACKGROUND
    assert result.response_status is QAResponseStatus.INSUFFICIENT_EVIDENCE


def test_missing_version_metadata_cannot_become_direct_answer() -> None:
    result = qualifier().assess(
        "哪一版财务制度截至2026年8月3日仍然生效",
        [context("财务制度列出了财务主管、主管会计和出纳的岗位职责。")],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.response_status is QAResponseStatus.NEEDS_CLARIFICATION


def test_date_without_available_record_remains_insufficient_evidence() -> None:
    result = qualifier().assess(
        "截至2026年8月14日能否给出下一年度尚未发布的最终执行数据",
        [context("消防安全管理制度规定员工职责和设备维护要求。")],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )
    assert result.response_status is QAResponseStatus.INSUFFICIENT_EVIDENCE


def test_unpublished_future_record_is_not_escalated_by_high_risk_policy_text() -> None:
    result = qualifier().assess(
        "截至2026年8月14日，《消防安全管理制度》能否给出下一年度尚未发布的最终执行数据？",
        [
            context(
                "消防安全管理制度要求员工遵守防火安全制度并保存检查记录。",
                context_id="fire-policy",
            ),
            context(
                "本月工作完成情况和下月工作计划应填写工作汇报表。",
                context_id="report-template",
            ),
        ],
        intent=QueryIntent.FACTOID,
        branch_statuses={
            "vector": RetrievalStatus.SUCCESS,
            "bm25": RetrievalStatus.SUCCESS,
            "graph": RetrievalStatus.EMPTY,
        },
    )

    assert result.primary_state is EvidenceState.INSUFFICIENT_EVIDENCE
    assert result.response_status is QAResponseStatus.INSUFFICIENT_EVIDENCE
    assert result.reason_codes == (EvidenceReasonCode.ZERO_RESULTS,)
    assert result.missing_information == ()


def test_supported_policy_and_missing_current_record_returns_specific_partial_route() -> None:
    result = qualifier().assess(
        "招聘制度适用于哪些岗位，同时截至2026年8月3日公司实际空缺岗位名单是什么？",
        [
            context(
                "本办法适用于公司所有岗位员工的招聘工作。",
                context_id="scope",
            ),
            context(
                "当公司出现职位空缺时，应首先在公司内部进行招聘。",
                context_id="policy",
            ),
        ],
        intent=QueryIntent.FACTOID,
        branch_statuses={
            "vector": RetrievalStatus.SUCCESS,
            "bm25": RetrievalStatus.SUCCESS,
            "graph": RetrievalStatus.EMPTY,
        },
    )

    assert result.primary_state is EvidenceState.PARTIAL_EVIDENCE
    assert result.response_status is QAResponseStatus.PARTIALLY_ANSWERED
    assert result.reason_codes == (EvidenceReasonCode.PARTIAL_SUPPORT,)
    assert [item.field for item in result.missing_information] == [
        "current_vacancy_business_record"
    ]


def test_explicit_two_document_conflict_is_not_answered_by_lexical_overlap() -> None:
    result = qualifier().assess(
        "两份同等权威但规则相矛盾的绩效制度当前应执行哪一份",
        [
            context("绩效制度规定年度考核结果作为晋升依据。", source="a.md", context_id="a"),
            context("绩效制度规定晋升必须通过人力资源部评估。", source="b.md", context_id="b"),
        ],
        intent=QueryIntent.FACTOID,
        branch_statuses={"vector": RetrievalStatus.SUCCESS},
    )

    assert result.response_status is QAResponseStatus.CONFLICTING_EVIDENCE
