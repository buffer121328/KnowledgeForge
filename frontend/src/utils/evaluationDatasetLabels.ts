import type { EvidenceDatasetSummary } from '@/types'
import { formatChinaStandardTime } from '@/utils/time'

const DATASET_STATUS_LABELS: Record<string, string> = {
  authoring: '编辑中',
  draft: '草稿',
  frozen: '已冻结',
  pending_review: '待复核',
}

function datasetName(dataset: EvidenceDatasetSummary): string {
  if (dataset.dataset_id === 'evidence-gates-v1-routine') return '评测题目集 · 标准评测'
  if (dataset.dataset_id === 'evidence-gates-v1') return '评测题目集 · 全量评测'
  if (dataset.source_type === 'current_corpus' || dataset.dataset_id.startsWith('current-corpus-')) {
    return '评测题目集 · 冒烟测试'
  }
  return dataset.title?.trim() || dataset.dataset_id
}

/** Build a localized, distinguishable label without changing the stable dataset ID value. */
export function evaluationDatasetLabel(dataset: EvidenceDatasetSummary): string {
  const status = DATASET_STATUS_LABELS[dataset.status] ?? dataset.status
  return `${datasetName(dataset)} · ${dataset.case_count} 条题目 · ${status}`
}

/** Label the case bundle independently from the corpus snapshot selector. */
export function evaluationSuiteLabel(dataset: EvidenceDatasetSummary): string {
  if (dataset.source_type === 'current_corpus' || dataset.dataset_id.startsWith('current-corpus-')) {
    return `冒烟测试 · ${dataset.case_count} 条`
  }
  if (dataset.dataset_id === 'evidence-gates-v1-routine') return '标准评测 · 50 条'
  if (dataset.dataset_id === 'evidence-gates-v1') return '全量评测 · 100 条'
  return `${datasetName(dataset)} · ${dataset.case_count} 条`
}

/** Render a compact Chinese business version label for selectors. */
export function evaluationVersionCompactLabel(version: string | null | undefined): string {
  if (!version) return '—'
  const match = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z-r(\d+)$/.exec(version)
  if (!match) return '冻结版本'
  const [, year, month, day, hour, minute, second, revision] = match
  const frozenAt = formatChinaStandardTime(`${year}-${month}-${day}T${hour}:${minute}:${second}Z`)
  return `第 ${revision} 版 · ${frozenAt}`
}

/** Render the business label and retain the raw version for technical details. */
export function evaluationVersionLabel(version: string | null | undefined): string {
  if (!version) return '—'
  return `${evaluationVersionCompactLabel(version)}（${version}）`
}
