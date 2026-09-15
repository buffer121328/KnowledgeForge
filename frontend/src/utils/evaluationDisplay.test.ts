import { describe, expect, it } from 'vitest'

import { evaluationReasonLabel } from './evaluationDisplay'

describe('evaluationReasonLabel', () => {
  it('localizes category-based RAGAS applicability reasons', () => {
    expect(evaluationReasonLabel('ordinary_answer_metric_invalid_for_background_only'))
      .toBe('仅背景信息题型按评测策略不适用普通答案指标')
  })

  it('explains an administrator-paused workflow', () => {
    expect(evaluationReasonLabel('paused_by_admin'))
      .toBe('管理员已暂停该评测；已有基线结果已保留，影子阶段未继续执行。')
  })

  it('explains why smoke evaluation skips calibration', () => {
    expect(evaluationReasonLabel('smoke_only_calibration_not_required'))
      .toBe('冒烟测试无需执行指标校准；基线和影子评测结果已保留。')
  })

  it('explains fail-closed smoke acceptance failures', () => {
    expect(evaluationReasonLabel('smoke_acceptance_contract_mismatch'))
      .toBe('冒烟验收失败：存在预期路由不匹配，不能进入审批或 Gate 晋级。')
    expect(evaluationReasonLabel('smoke_acceptance_reference_missing'))
      .toBe('冒烟验收失败：可回答样本缺少已审核参考答案，关键质量指标未实际执行。')
  })
})
