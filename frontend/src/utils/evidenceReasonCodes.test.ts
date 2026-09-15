import { describe, expect, it } from 'vitest'
import {
  EVIDENCE_REASON_CODE_LABELS,
  EVIDENCE_REASON_CODE_OPTIONS,
  evidenceReasonCodeLabel,
} from './evidenceReasonCodes'

describe('evidence reason-code labels', () => {
  it('exposes every governed code as a selectable Chinese label', () => {
    expect(EVIDENCE_REASON_CODE_OPTIONS).toHaveLength(16)
    expect(EVIDENCE_REASON_CODE_OPTIONS).toContainEqual({
      value: 'direct_support', label: '直接证据支持',
    })
    expect(EVIDENCE_REASON_CODE_LABELS.high_risk_review).toBe('高风险人工复核')
    expect(evidenceReasonCodeLabel('unknown_future_code')).toBe('unknown_future_code')
  })
})
