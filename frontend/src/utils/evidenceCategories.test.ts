import { describe, expect, it } from 'vitest'
import { EVIDENCE_CATEGORY_LABELS, evidenceCategoryLabel } from './evidenceCategories'

describe('evidence category labels', () => {
  it('maps every governed evidence category to a Chinese label', () => {
    expect(EVIDENCE_CATEGORY_LABELS).toEqual({
      fully_answerable: '完全可回答',
      completely_unanswerable: '完全不可回答',
      background_only: '仅背景相关',
      partially_answerable: '部分可回答',
      conflicting: '证据冲突',
      missing_version_or_date: '缺少版本或日期',
      missing_business_record: '缺少业务记录',
      authorization_filtered: '权限过滤',
      single_branch_unavailable: '单路检索不可用',
      all_branches_unavailable: '全部检索不可用',
      prompt_injection: '提示词注入',
    })
    expect(evidenceCategoryLabel('future_category')).toBe('future_category')
  })
})
