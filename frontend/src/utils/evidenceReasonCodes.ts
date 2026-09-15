/** Chinese presentation labels for the stable Evidence Reason Code contract. */
export const EVIDENCE_REASON_CODE_LABELS = {
  direct_support: '直接证据支持',
  partial_support: '部分证据支持',
  background_only: '仅背景相关',
  material_conflict: '实质性冲突',
  irrelevant_context: '上下文不相关',
  zero_results: '无检索结果',
  authorized_contexts_empty: '授权范围内无上下文',
  invalid_context_provenance: '上下文溯源无效',
  partial_dependency_unavailable: '部分依赖不可用',
  all_dependencies_unavailable: '全部依赖不可用',
  missing_question_detail: '缺少问题细节',
  high_risk_review: '高风险人工复核',
  unknown_citation: '未知引用',
  unsupported_claim: '不受支持的主张',
  unsupported_critical_value: '关键数值无支持',
  structured_output_invalid: '结构化输出无效',
} as const

export type EvidenceReasonCode = keyof typeof EVIDENCE_REASON_CODE_LABELS

export const EVIDENCE_REASON_CODE_OPTIONS = Object.entries(EVIDENCE_REASON_CODE_LABELS).map(
  ([value, label]) => ({ value, label }),
)

/** Keep unknown persisted codes visible for backwards-compatible case editing. */
export function evidenceReasonCodeLabel(reasonCode: string): string {
  return EVIDENCE_REASON_CODE_LABELS[reasonCode as EvidenceReasonCode] ?? reasonCode
}
