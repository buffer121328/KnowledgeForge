import { describe, expect, it } from 'vitest'
import type { EvidenceDatasetSummary } from '@/types'
import { evaluationDatasetLabel, evaluationSuiteLabel, evaluationVersionCompactLabel, evaluationVersionLabel } from './evaluationDatasetLabels'

function dataset(overrides: Partial<EvidenceDatasetSummary>): EvidenceDatasetSummary {
  return {
    dataset_id: 'unknown-dataset',
    revision: 1,
    status: 'frozen',
    case_count: 0,
    context_count: 0,
    counts: {},
    category_counts: {},
    required_category_count: 0,
    covered_category_count: 0,
    ...overrides,
  }
}

describe('evaluationDatasetLabel', () => {
  it('localizes the known 50-case and 100-case suites', () => {
    expect(evaluationDatasetLabel(dataset({ dataset_id: 'evidence-gates-v1-routine', case_count: 50 })))
      .toBe('评测题目集 · 标准评测 · 50 条题目 · 已冻结')
    expect(evaluationDatasetLabel(dataset({ dataset_id: 'evidence-gates-v1', case_count: 100, status: 'authoring' })))
      .toBe('评测题目集 · 全量评测 · 100 条题目 · 编辑中')
  })

  it('shows current-corpus datasets as a Chinese business label without an internal ID', () => {
    expect(evaluationDatasetLabel(dataset({
      dataset_id: 'current-corpus-7146f01a4c334407b3927ccebc96d127',
      source_type: 'current_corpus',
      case_count: 12,
      source_document_count: 41,
    }))).toBe('评测题目集 · 冒烟测试 · 12 条题目 · 已冻结')
  })

  it('uses compact release-suite labels and Chinese-time version labels', () => {
    expect(evaluationSuiteLabel(dataset({ source_type: 'current_corpus', case_count: 12 }))).toBe('冒烟测试 · 12 条')
    expect(evaluationSuiteLabel(dataset({ dataset_id: 'evidence-gates-v1-routine', case_count: 50 }))).toBe('标准评测 · 50 条')
    expect(evaluationSuiteLabel(dataset({ dataset_id: 'evidence-gates-v1', case_count: 100 }))).toBe('全量评测 · 100 条')
    expect(evaluationVersionCompactLabel('20260815T173052Z-r5')).toBe('第 5 版 · 2026-08-16 01:30:52')
    expect(evaluationVersionLabel('20260815T173052Z-r5')).toBe('第 5 版 · 2026-08-16 01:30:52（20260815T173052Z-r5）')
  })

  it('falls back to the server title and then the raw ID', () => {
    expect(evaluationDatasetLabel(dataset({ title: '专项评测集', case_count: 8 })))
      .toBe('专项评测集 · 8 条题目 · 已冻结')
    expect(evaluationDatasetLabel(dataset({ dataset_id: 'future-suite', status: 'ready' })))
      .toBe('future-suite · 0 条题目 · ready')
  })
})
