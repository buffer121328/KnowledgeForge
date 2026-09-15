import { useCallback, useEffect, useMemo, useState } from 'react'
import { App } from 'antd'
import axios from 'axios'
import { evaluationApi } from '@/api/evaluation'
import type {
  EvidenceBulkActionResponse,
  EvidenceBulkSelectionMode,
  EvidenceCase,
  EvidenceDatasetSummary,
  EvidenceFreezeResult,
  EvidenceFrozenVersion,
  EvidenceReferenceAnswerCandidateResult,
  EvidenceReviewer,
  EvidenceReviewStatus,
} from '@/types'

const PAGE_SIZE = 20
const WORKBENCH_DATASET_IDS = ['evidence-gates-v1', 'evidence-gates-v1-routine'] as const

/** Keep the authoring selector limited to full, standard, and current smoke suites. */
export function selectWorkbenchDatasets(items: EvidenceDatasetSummary[]): EvidenceDatasetSummary[] {
  const full = items.find((item) => item.dataset_id === WORKBENCH_DATASET_IDS[0])
  const standard = items.find((item) => item.dataset_id === WORKBENCH_DATASET_IDS[1])
  const smoke = items
    .filter((item) => item.source_type === 'current_corpus' && item.active !== false && item.case_count === 12)
    .sort((left, right) => String(right.updated_at || '').localeCompare(String(left.updated_at || '')))[0]
  return [full, standard, smoke].filter((item): item is EvidenceDatasetSummary => Boolean(item))
}

function errorCode(error: unknown): string {
  if (!axios.isAxiosError(error)) return ''
  const detail = error.response?.data?.detail
  return typeof detail === 'object' && detail && 'code' in detail ? String(detail.code) : ''
}

/** Own the organization-scoped dataset workbench state and optimistic mutations. */
export function useEvidenceDatasets() {
  const { message } = App.useApp()
  const [datasets, setDatasets] = useState<EvidenceDatasetSummary[]>([])
  const [dataset, setDataset] = useState<EvidenceDatasetSummary | null>(null)
  const [cases, setCases] = useState<EvidenceCase[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [category, setCategory] = useState('')
  const [reviewStatus, setReviewStatus] = useState<EvidenceReviewStatus | ''>('')
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [mutating, setMutating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [conflict, setConflict] = useState(false)
  const [lastFreeze, setLastFreeze] = useState<EvidenceFreezeResult | null>(null)
  const [frozenVersions, setFrozenVersions] = useState<EvidenceFrozenVersion[]>([])
  const datasetId = dataset?.dataset_id ?? datasets[0]?.dataset_id ?? ''
  const refreshSummary = useCallback(async (id: string) => {
    const [current, versions] = await Promise.all([
      evaluationApi.getDataset(id),
      evaluationApi.listFrozenVersions(id),
    ])
    setDataset(current)
    setFrozenVersions(versions)
    setDatasets((items) => items.map((item) => (item.dataset_id === id ? current : item)))
    return current
  }, [])

  const loadCases = useCallback(async (id: string, targetPage = 1) => {
    const result = await evaluationApi.getCases(id, {
      page: targetPage,
      pageSize: PAGE_SIZE,
      category,
      reviewStatus,
      query,
    })
    setCases(result.cases)
    setTotal(result.total)
    setPage(result.page)
    setDataset((current) => current ? { ...current, revision: result.revision } : current)
    return result
  }, [category, query, reviewStatus])

  const selectDataset = useCallback(async (id: string) => {
    if (!id || id === dataset?.dataset_id) return
    setLoading(true)
    setError(null)
    setCategory('')
    setReviewStatus('')
    setQuery('')
    try {
      await Promise.all([loadCases(id, 1), refreshSummary(id)])
    } catch {
      setError('切换数据集失败，请检查权限或服务状态。')
    } finally {
      setLoading(false)
    }
  }, [dataset?.dataset_id, loadCases, refreshSummary])

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await evaluationApi.listDatasets()
      const workbenchDatasets = selectWorkbenchDatasets(result.datasets)
      setDatasets(workbenchDatasets)
      const first = workbenchDatasets[0]
      setDataset(first ?? null)
      if (first) {
        await Promise.all([loadCases(first.dataset_id, 1), refreshSummary(first.dataset_id)])
      } else {
        setCases([])
        setFrozenVersions([])
      }
    } catch {
      setError('数据集工作台加载失败，请检查权限或服务状态。')
    } finally {
      setLoading(false)
    }
  }, [loadCases, refreshSummary])

  useEffect(() => { void load() }, [load])

  const reloadCases = useCallback(async (targetPage = 1) => {
    if (!datasetId) return
    setLoading(true)
    try { await loadCases(datasetId, targetPage) }
    catch { message.error('加载样本失败') }
    finally { setLoading(false) }
  }, [datasetId, loadCases, message])

  const applyMutation = useCallback(async (
    operation: () => Promise<{ revision: number; case: EvidenceCase }>,
    successMessage: string,
  ) => {
    setMutating(true)
    setConflict(false)
    try {
      const result = await operation()
      setCases((items) => {
        const exists = items.some((item) => item.id === result.case.id)
        return exists
          ? items.map((item) => item.id === result.case.id ? result.case : item)
          : [result.case, ...items]
      })
      setDataset((current) => current ? { ...current, revision: result.revision } : current)
      message.success(successMessage)
      if (datasetId) void refreshSummary(datasetId)
      return result.case
    } catch (error) {
      if (errorCode(error) === 'revision_conflict') {
        setConflict(true)
        message.warning('工作区已被其他员工或部门负责人更新，请刷新后重试。')
      } else if (errorCode(error) === 'maker_checker_violation') {
        message.warning('制作者不能批准自己的最后一次编辑，请由本部门其他负责人复核。')
      } else if (errorCode(error) === 'dataset_frozen') {
        message.warning('该数据集已冻结，不能再修改；请基于冻结版本创建新草稿。')
      } else {
        message.error(`操作失败${errorCode(error) ? `：${errorCode(error)}` : ''}`)
      }
      throw error
    } finally {
      setMutating(false)
    }
  }, [datasetId, message, refreshSummary])

  const saveCase = useCallback((value: EvidenceCase, isNew = false) => {
    if (!dataset || !datasetId) return Promise.reject(new Error('dataset unavailable'))
    return applyMutation(
      () => isNew
        ? evaluationApi.createCase(datasetId, dataset.revision, value)
        : evaluationApi.updateCase(datasetId, value.id, dataset.revision, value),
      isNew ? '样本已创建' : '样本已保存，旧审核状态已失效',
    )
  }, [applyMutation, dataset, datasetId])

  const deleteCase = useCallback(async (value: EvidenceCase) => {
    if (!dataset || !datasetId) return Promise.reject(new Error('dataset unavailable'))
    setMutating(true)
    setConflict(false)
    try {
      const result = await evaluationApi.deleteCase(datasetId, value.id, dataset.revision)
      setCases((items) => items.filter((item) => item.id !== result.deleted_case_id))
      setDataset((current) => current ? { ...current, revision: result.revision } : current)
      const refreshed = await loadCases(datasetId, page)
      if (page > 1 && refreshed.total > 0 && refreshed.cases.length === 0) {
        await loadCases(datasetId, Math.max(1, page - 1))
      }
      await refreshSummary(datasetId)
      message.success('样本已删除；历史冻结版本不受影响')
    } catch (error) {
      if (errorCode(error) === 'revision_conflict') {
        setConflict(true)
        message.warning('工作区已被其他员工或部门负责人更新，请刷新后再继续。')
      } else {
        message.error(`删除样本失败${errorCode(error) ? `：${errorCode(error)}` : ''}`)
      }
      throw error
    } finally {
      setMutating(false)
    }
  }, [dataset, datasetId, loadCases, message, page, refreshSummary])

  const loadReviewers = useCallback((value: EvidenceCase): Promise<EvidenceReviewer[]> => {
    if (!datasetId) return Promise.reject(new Error('dataset unavailable'))
    return evaluationApi.listReviewers(datasetId, value.id)
  }, [datasetId])

  const submitCase = useCallback((value: EvidenceCase, reviewerId: string) => {
    if (!dataset || !datasetId) return Promise.reject(new Error('dataset unavailable'))
    return applyMutation(
      () => evaluationApi.submitCase(datasetId, value.id, dataset.revision, reviewerId),
      '样本已提交复核',
    )
  }, [applyMutation, dataset, datasetId])

  const reviewCase = useCallback((value: EvidenceCase, decision: 'approve' | 'reject', reason: string) => {
    if (!dataset || !datasetId) return Promise.reject(new Error('dataset unavailable'))
    return applyMutation(
      () => evaluationApi.reviewCase(datasetId, value.id, dataset.revision, decision, reason),
      decision === 'approve' ? '样本已批准' : '样本已退回草稿',
    )
  }, [applyMutation, dataset, datasetId])

  const applyBulkMutation = useCallback(async (
    action: 'submit' | 'approve',
    selectionMode: EvidenceBulkSelectionMode,
    caseIds: string[],
  ): Promise<EvidenceBulkActionResponse> => {
    if (!dataset || !datasetId) throw new Error('dataset unavailable')
    setMutating(true)
    setConflict(false)
    try {
      const payload = selectionMode === 'selected'
        ? {
            expected_revision: dataset.revision,
            selection_mode: selectionMode,
            case_ids: caseIds,
          }
        : {
            expected_revision: dataset.revision,
            selection_mode: selectionMode,
            case_ids: [],
            category: category || undefined,
            review_status: reviewStatus || undefined,
            query: query || undefined,
          }
      const result = action === 'submit'
        ? await evaluationApi.bulkSubmitCases(datasetId, payload)
        : await evaluationApi.bulkApproveCases(datasetId, payload)
      setDataset((current) => current ? { ...current, revision: result.revision } : current)

      let refreshFailed = false
      try {
        const refreshedPage = await loadCases(datasetId, page)
        if (page > 1 && refreshedPage.total > 0 && refreshedPage.cases.length === 0) {
          await loadCases(datasetId, Math.max(1, Math.ceil(refreshedPage.total / PAGE_SIZE)))
        }
        await refreshSummary(datasetId)
      } catch {
        refreshFailed = true
      }

      const actionLabel = action === 'submit' ? '提交复核' : '通过'
      if (refreshFailed) {
        message.warning(`批量${actionLabel}已执行，但列表刷新失败，请手动刷新确认。`)
      } else if (result.processed_count === 0) {
        message.warning(`没有可批量${actionLabel}的样本；跳过 ${result.skipped_count} 条。`)
      } else if (result.skipped_count > 0) {
        message.warning(`已批量${actionLabel} ${result.processed_count} 条，跳过 ${result.skipped_count} 条。`)
      } else {
        message.success(`已批量${actionLabel} ${result.processed_count} 条样本。`)
      }
      return result
    } catch (error) {
      if (errorCode(error) === 'revision_conflict') {
        setConflict(true)
        message.warning('工作区已被其他员工或部门负责人更新，请刷新后重试。')
      } else {
        message.error(`批量操作失败${errorCode(error) ? `：${errorCode(error)}` : ''}`)
      }
      throw error
    } finally {
      setMutating(false)
    }
  }, [category, dataset, datasetId, loadCases, message, page, query, refreshSummary, reviewStatus])

  const bulkSubmit = useCallback(
    (selectionMode: EvidenceBulkSelectionMode, caseIds: string[]) =>
      applyBulkMutation('submit', selectionMode, caseIds),
    [applyBulkMutation],
  )

  const bulkApprove = useCallback(
    (selectionMode: EvidenceBulkSelectionMode, caseIds: string[]) =>
      applyBulkMutation('approve', selectionMode, caseIds),
    [applyBulkMutation],
  )

  const freezeDataset = useCallback(async () => {
    if (!dataset || !datasetId) return
    setMutating(true)
    try {
      const result = await evaluationApi.freezeDataset(datasetId, dataset.revision)
      setLastFreeze(result)
      setDataset((current) => current ? { ...current, revision: result.revision, status: 'frozen', last_frozen_version: result.version } : current)
      message.success('数据集已冻结；尚未运行 baseline，也未改变生产 Gate 模式')
      await refreshSummary(datasetId)
    } catch (error) {
      if (errorCode(error) === 'revision_conflict') setConflict(true)
      message.error(`冻结被阻止${errorCode(error) ? `：${errorCode(error)}` : ''}`)
      throw error
    } finally { setMutating(false) }
  }, [dataset, datasetId, message, refreshSummary])

  const createDraftFromVersion = useCallback(async (version: string) => {
    if (!dataset || !datasetId) return
    setMutating(true)
    try {
      const drafted = await evaluationApi.createDraftFromVersion(datasetId, version, dataset.revision)
      setDataset(drafted)
      setDatasets((items) => items.map((item) => item.dataset_id === datasetId ? drafted : item))
      setLastFreeze(null)
      await Promise.all([loadCases(datasetId, 1), refreshSummary(datasetId)])
      message.success(`已基于冻结版本 ${version} 创建新草稿。`)
      return drafted
    } catch (error) {
      if (errorCode(error) === 'revision_conflict') setConflict(true)
      message.error(`创建草稿失败${errorCode(error) ? `：${errorCode(error)}` : ''}`)
      throw error
    } finally { setMutating(false) }
  }, [dataset, datasetId, loadCases, message, refreshSummary])

  const rebindToCurrentCorpus = useCallback(async () => {
    if (!dataset || !datasetId) return
    setMutating(true)
    setConflict(false)
    try {
      const rebound = await evaluationApi.rebindToCurrentCorpus(datasetId, dataset.revision)
      setDataset(rebound)
      setDatasets((items) => items.map((item) => item.dataset_id === datasetId ? rebound : item))
      await Promise.all([loadCases(datasetId, 1), refreshSummary(datasetId)])
      message.success('已用当前语料的真实 Chunk 重绑定；所有样本已回到草稿待复核状态。')
      return rebound
    } catch (error) {
      if (errorCode(error) === 'revision_conflict') setConflict(true)
      message.error(`重绑定被阻止${errorCode(error) ? `：${errorCode(error)}` : ''}`)
      throw error
    } finally { setMutating(false) }
  }, [dataset, datasetId, loadCases, message, refreshSummary])

  const populateReferenceAnswerCandidates = useCallback(async (): Promise<EvidenceReferenceAnswerCandidateResult | undefined> => {
    if (!dataset || !datasetId) return undefined
    setMutating(true)
    setConflict(false)
    try {
      const result = await evaluationApi.populateReferenceAnswerCandidates(datasetId, dataset.revision)
      setDataset((current) => current ? { ...current, revision: result.revision } : current)
      await Promise.all([loadCases(datasetId, 1), refreshSummary(datasetId)])
      if (result.populated_case_ids.length > 0) {
        message.success(`已生成 ${result.populated_case_ids.length} 条候选参考答案，请人工复核后再冻结。`)
      } else {
        message.info('没有需要生成的候选参考答案；现有人工编辑内容已保留。')
      }
      return result
    } catch (error) {
      if (errorCode(error) === 'revision_conflict') setConflict(true)
      message.error(`生成候选参考答案失败${errorCode(error) ? `：${errorCode(error)}` : ''}`)
      throw error
    } finally {
      setMutating(false)
    }
  }, [dataset, datasetId, loadCases, message, refreshSummary])

  const counts = useMemo(() => dataset?.counts ?? {}, [dataset])

  return {
    bulkApprove, bulkSubmit, cases, category, conflict, counts, createDraftFromVersion,
    dataset, datasets, deleteCase, error, freezeDataset, frozenVersions, lastFreeze, load, loadReviewers,
    loading, mutating, page, populateReferenceAnswerCandidates, query, rebindToCurrentCorpus, reloadCases, reviewCase, reviewStatus, saveCase, setCategory, setPage,
    selectDataset, setQuery, setReviewStatus, submitCase, total,
  }
}
