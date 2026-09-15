"""复合「政策 + 记录」题的路由契约回归（OpenSpec reclass-composite-policy-record-cases）。

复合题（静态制度/流程部分可独立作答 + 动态业务记录部分缺失）按评审决定
归入 ``partially_answerable``，期望 ``partially_answered``；有据回答静态
部分并声明记录缺失必须通过契约，而不是计入拒答幻觉。
"""

from __future__ import annotations

from evaluation.evidence_gate.benchmark import (
    EvidenceGateCase,
    reviewed_case_contract_errors,
)


def _composite_case(**overrides: object) -> EvidenceGateCase:
    """构造一条复合「政策+记录」题基线（partially_answerable 形态）。"""
    base: dict[str, object] = {
        "id": "eg-company-demo-20",
        "question": "采购流程从部门申请到订单完结经过哪些节点，同时某采购单当前处于哪个节点？",
        "category": "partially_answerable",
        "expected_response_status": "partially_answered",
        "expected_evidence_states": ("partial_evidence",),
        "expected_reason_codes": ("partial_support",),
        "expected_citation_context_ids": ("doc-011#chunk-0",),
        "expected_branch_availability": {"dense": "available", "bm25": "available", "graph": "not_configured"},
        "expected_source_document_ids": ("doc-011",),
        "expected_evidence_context_ids": ("doc-011#chunk-0",),
        "expected_evidence_sections": ("采购管理流程",),
        "expected_missing_information_fields": ("current_procurement_order_record",),
    }
    base.update(overrides)
    return EvidenceGateCase(**base)  # type: ignore[arg-type]


def test_composite_policy_record_case_passes_as_partially_answerable() -> None:
    """复合题按 partially_answerable 声明时必须通过路由契约。"""
    case = _composite_case()
    assert reviewed_case_contract_errors(case) == ()


def test_composite_case_left_in_refusal_category_fails_contract() -> None:
    """复合题若仍留在拒答类（背景/缺记录）并期望部分回答，契约必须拒绝。"""
    mismatched_status = _composite_case(
        category="missing_business_record",
    )
    errors = reviewed_case_contract_errors(mismatched_status)
    assert "reviewed_contract_response_status_mismatch" in errors


def test_partially_answerable_contract_requires_missing_information_field() -> None:
    """partially_answerable 必须声明缺失记录字段，否则契约拒绝。"""
    case = _composite_case(expected_missing_information_fields=())
    errors = reviewed_case_contract_errors(case)
    assert "reviewed_contract_missing_information_required" in errors


def test_partially_answerable_contract_requires_partial_evidence_state() -> None:
    """partially_answerable 的证据状态必须是 partial_evidence。"""
    case = _composite_case(
        expected_evidence_states=("relevant_background",),
        expected_reason_codes=("background_only",),
    )
    errors = reviewed_case_contract_errors(case)
    assert "reviewed_contract_evidence_state_mismatch" in errors
    assert "reviewed_contract_reason_code_mismatch" in errors


def test_composite_case_with_cited_static_part_is_not_refusal() -> None:
    """复合题期望部分回答时必须绑定引用上下文（可答部分有据）。"""
    case = _composite_case(expected_citation_context_ids=())
    errors = reviewed_case_contract_errors(case)
    assert "reviewed_contract_positive_evidence_required" in errors
