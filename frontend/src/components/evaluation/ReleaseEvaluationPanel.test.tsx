// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { App } from 'antd'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import ReleaseEvaluationPanel from './ReleaseEvaluationPanel'

const api = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  listFrozenVersions: vi.fn(),
  listSnapshotContexts: vi.fn(),
  listReleaseWorkflows: vi.fn(),
  startReleaseWorkflow: vi.fn(),
  getReleaseWorkflow: vi.fn(),
  listReleaseAttempts: vi.fn(),
  retryReleaseWorkflow: vi.fn(),
  getEvidenceGate: vi.fn(),
  reviewReleaseWorkflow: vi.fn(),
  promoteReleaseWorkflow: vi.fn(),
  rollbackEvidenceGate: vi.fn(),
}))
vi.mock('@/api/evaluation', () => ({ evaluationApi: api }))

const targetDataset = {
  dataset_id: 'current-corpus-production',
  title: '冻结当前语料',
  revision: 42,
  status: 'frozen',
  source_type: 'current_corpus',
  case_count: 12,
  context_count: 790,
  source_document_count: 41,
  counts: {},
  category_counts: {},
  required_category_count: 7,
  covered_category_count: 7,
}
const targetVersion = {
  version: '20260816T085851Z-r8',
  manifest_sha256: 'c'.repeat(64),
  frozen_at: '2026-08-16T00:00:00+00:00',
  frozen_by: 'company-admin-a',
  source_workspace_revision: 42,
}
const formalDataset = {
  dataset_id: 'evidence-gates-v1',
  title: '正式评测集',
  revision: 19,
  status: 'frozen',
  source_type: 'formal_evaluation',
  case_count: 100,
  context_count: 36,
  counts: { approved: 100, draft: 0, pending_review: 0 },
  category_counts: {},
  required_category_count: 11,
  covered_category_count: 11,
  fixture_status: 'ready',
}
const standardDataset = {
  ...formalDataset,
  dataset_id: 'evidence-gates-v1-routine',
  title: '标准评测集',
  case_count: 50,
  counts: { approved: 50, draft: 0, pending_review: 0 },
}
const formalVersion = {
  version: '20260815T010203Z-r19',
  manifest_sha256: 'e'.repeat(64),
  frozen_at: '2026-08-15T00:00:00+00:00',
  frozen_by: 'company-admin-a',
  source_workspace_revision: 19,
}

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(() => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() })),
  })
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  Object.defineProperty(window, 'ResizeObserver', { value: ResizeObserverMock })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function seed(overrides: Record<string, unknown> = {}) {
  api.listDatasets.mockResolvedValue({ datasets: [targetDataset, formalDataset] })
  api.listSnapshotContexts.mockResolvedValue({ dataset_id: targetDataset.dataset_id, version: targetVersion.version, source_document_count: 41, contexts: [] })
  api.listReleaseWorkflows.mockResolvedValue([])
  api.listFrozenVersions.mockImplementation((datasetId: string) => Promise.resolve(datasetId === formalDataset.dataset_id ? [formalVersion] : [targetVersion]))
  api.listReleaseAttempts.mockResolvedValue([])
  api.startReleaseWorkflow.mockResolvedValue({
    workflow_id: 'erw-1',
    dataset_id: targetDataset.dataset_id,
    version: targetVersion.version,
    manifest_sha256: targetVersion.manifest_sha256,
    current_corpus_draft_id: null,
    current_corpus_snapshot_sha256: null,
    fixture_validation_run_id: null,
    evaluation_dataset_id: formalDataset.dataset_id,
    evaluation_version: formalVersion.version,
    evaluation_manifest_sha256: formalVersion.manifest_sha256,
    evaluation_case_count: 100,
    status: 'queued',
    stage: 'preflight',
    revision: 1,
    initiated_by: 'company-admin-a',
  })
  api.retryReleaseWorkflow.mockResolvedValue({})
  api.getEvidenceGate.mockResolvedValue({ mode: 'off', revision: 0, source: 'persistent' })
  api.reviewReleaseWorkflow.mockResolvedValue({})
  api.promoteReleaseWorkflow.mockResolvedValue({})
  api.rollbackEvidenceGate.mockResolvedValue({ mode: 'off', revision: 2, source: 'persistent' })
  Object.assign(api, overrides)
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((next) => { resolve = next })
  return { promise, resolve }
}

describe('ReleaseEvaluationPanel', () => {
  it('starts independent bootstrap requests together and loads Gate only once', async () => {
    const datasets = deferred<{ datasets: Array<typeof targetDataset | typeof formalDataset> }>()
    seed({ listDatasets: vi.fn().mockReturnValue(datasets.promise) })

    render(<App><ReleaseEvaluationPanel /></App>)

    await waitFor(() => {
      expect(api.listDatasets).toHaveBeenCalledTimes(1)
      expect(api.listReleaseWorkflows).toHaveBeenCalledTimes(1)
      expect(api.getEvidenceGate).toHaveBeenCalledTimes(1)
    })
    datasets.resolve({ datasets: [targetDataset, formalDataset] })

    expect(await screen.findByText('被测语料：当前真实入库文档（41 份）')).toBeTruthy()
    expect(api.getEvidenceGate).toHaveBeenCalledTimes(1)
    expect(api.getReleaseWorkflow).not.toHaveBeenCalled()
    await waitFor(() => {
      expect(api.listFrozenVersions).toHaveBeenCalledWith(targetDataset.dataset_id, 1)
      expect(api.listFrozenVersions).toHaveBeenCalledWith(formalDataset.dataset_id, 1)
    })
  })

  it('deep-refreshes the selected workflow detail and attempts from the server', async () => {
    seed()
    const attempt = {
      attempt_id: 'era-refresh', stage: 'baseline', attempt_number: 1, status: 'failed',
      reason_code: 'ragas_reference_missing', metrics: {},
      started_at: '2026-08-29T00:00:00Z', finished_at: '2026-08-29T00:01:00Z',
    }
    render(<App><ReleaseEvaluationPanel /></App>)
    fireEvent.click(await screen.findByRole('button', { name: /一键运行至完成/ }))
    await screen.findByText('发布评测记录')
    const updated = {
      ...(await api.startReleaseWorkflow.mock.results[0].value),
      status: 'failed', reason_code: 'ragas_reference_missing', revision: 2,
    }
    api.listReleaseWorkflows.mockResolvedValue([updated])
    api.getReleaseWorkflow.mockResolvedValue(updated)
    api.listReleaseAttempts.mockResolvedValue([attempt])
    await waitFor(() => expect(screen.getAllByRole('button').some((button) => button.textContent?.trim() === '刷新')).toBe(true))
    fireEvent.click(screen.getAllByRole('button').find((button) => button.textContent?.trim() === '刷新')!)

    await waitFor(() => {
      expect(api.getReleaseWorkflow).toHaveBeenCalledWith('erw-1')
      expect(api.listReleaseAttempts).toHaveBeenCalledWith('erw-1')
      expect(screen.getAllByText(/RAGAS 评分输入缺少冻结参考答案/).length).toBeGreaterThan(0)
    })
  })

  it('lists only the full, standard, and smoke question suites', async () => {
    const oldCorpus = {
      ...targetDataset,
      dataset_id: 'current-corpus-old-40',
      active: false,
      source_document_count: 40,
    }
    const unsupportedSuite = {
      ...formalDataset,
      dataset_id: 'future-suite',
      case_count: 25,
    }
    seed({
      listDatasets: vi.fn().mockResolvedValue({
        datasets: [oldCorpus, targetDataset, unsupportedSuite, standardDataset, formalDataset],
      }),
    })
    render(<App><ReleaseEvaluationPanel /></App>)

    fireEvent.mouseDown(await screen.findByRole('combobox', { name: '评测题目集' }))
    const optionLabels = (await screen.findAllByRole('option')).map((option) => option.textContent)
    expect(optionLabels).toEqual([
      '全量评测 · 100 条',
      '标准评测 · 50 条',
      '冒烟测试 · 12 条',
    ])
    expect(screen.queryByText(/40 份文档/)).toBeNull()
    expect(screen.queryByText(/41 份文档.*评测题目集/)).toBeNull()
  })

  it('starts directly from a frozen current-corpus version and the 100-case suite', async () => {
    seed()
    render(<App><ReleaseEvaluationPanel /></App>)

    expect(await screen.findByText('被测语料：当前真实入库文档（41 份）')).toBeTruthy()
    expect(screen.getAllByText('全量评测 · 100 条').length).toBeGreaterThan(0)
    expect(screen.queryByText(/回答与检索使用的知识库内容快照/)).toBeNull()
    expect(screen.queryByText(/当前选择 100 条正式评测集/)).toBeNull()
    expect(screen.queryByRole('combobox', { name: '发布评测当前语料草稿' })).toBeNull()
    expect(screen.queryByText(/Fixture/)).toBeNull()
    expect(screen.queryByText(/冻结快照文档目录/)).toBeNull()
    expect(api.listSnapshotContexts).not.toHaveBeenCalled()

    const launchButton = screen.getByRole('button', { name: /一键运行至完成/ }) as HTMLButtonElement
    await waitFor(() => expect(launchButton.disabled).toBe(false))
    fireEvent.click(launchButton)
    await waitFor(() => expect(api.startReleaseWorkflow).toHaveBeenCalledWith(
      targetDataset.dataset_id,
      targetVersion.version,
      formalDataset.dataset_id,
      formalVersion.version,
    ))
    expect(await screen.findByText('发布评测记录')).toBeTruthy()
    expect(screen.queryByText(/工作流 erw-1/)).toBeNull()
    expect(screen.getByText('全量评测 · 100 条题目')).toBeTruthy()
  })

  it('does not infer a launch target when no frozen current-corpus version exists', async () => {
    seed()
    api.listDatasets.mockResolvedValue({ datasets: [formalDataset] })

    render(<App><ReleaseEvaluationPanel /></App>)

    expect(await screen.findByText('暂无已冻结的当前语料版本，请先完成审核并冻结版本。')).toBeTruthy()
    expect(api.startReleaseWorkflow).not.toHaveBeenCalled()
  })

  it('allows the backend to safely recover a stale running workflow', async () => {
    const running = {
      workflow_id: 'erw-running', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
      manifest_sha256: targetVersion.manifest_sha256, status: 'running', stage: 'shadow', revision: 7,
      initiated_by: 'company-admin-a', evaluation_dataset_id: formalDataset.dataset_id,
      evaluation_version: formalVersion.version, evaluation_case_count: 100,
      created_at: '2026-08-27T14:00:00Z', updated_at: '2026-08-27T14:00:00Z',
    }
    const queued = { ...running, status: 'queued', revision: 8, reason_code: 'orphaned_worker_attempt' }
    seed({
      listReleaseWorkflows: vi.fn().mockResolvedValue([running]),
      getReleaseWorkflow: vi.fn().mockResolvedValue(running),
      retryReleaseWorkflow: vi.fn().mockResolvedValue(queued),
    })

    render(<App><ReleaseEvaluationPanel /></App>)

    const recoverButton = await screen.findByRole('button', { name: '检查并恢复中断任务' })
    expect((recoverButton as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(recoverButton)
    await waitFor(() => expect(api.retryReleaseWorkflow).toHaveBeenCalledWith('erw-running', 7))
  })

  it('does not show deprecated 12-case Fixture identity in historical workflows', async () => {
    const historical = {
      workflow_id: 'erw-history',
      dataset_id: targetDataset.dataset_id,
      version: targetVersion.version,
      manifest_sha256: targetVersion.manifest_sha256,
      current_corpus_draft_id: 'ccd_' + 'a'.repeat(32),
      current_corpus_snapshot_sha256: 'b'.repeat(64),
      fixture_validation_run_id: 'fvr_' + 'd'.repeat(32),
      evaluation_dataset_id: formalDataset.dataset_id,
      evaluation_version: formalVersion.version,
      evaluation_case_count: 100,
      status: 'failed',
      stage: 'calibration',
      revision: 4,
      initiated_by: 'company-admin-a',
      reason_code: 'calibration_metrics_incomplete',
      created_at: '2026-08-16T08:00:00Z',
      updated_at: '2026-08-16T08:10:00Z',
    }
    seed({
      listReleaseWorkflows: vi.fn().mockResolvedValue([historical]),
      getReleaseWorkflow: vi.fn().mockResolvedValue(historical),
    })

    render(<App><ReleaseEvaluationPanel /></App>)
    expect(await screen.findByText('发布评测记录')).toBeTruthy()
    expect(screen.queryByText(/工作流 erw-history/)).toBeNull()
    expect(screen.getByText(historical.current_corpus_draft_id)).toBeTruthy()
    expect(screen.queryByText(historical.fixture_validation_run_id)).toBeNull()
    expect(screen.queryByText(/历史 Fixture 运行/)).toBeNull()
  })

  it('opens a concrete result and renders Asia/Shanghai time without a zone suffix', async () => {
    const onViewRuns = vi.fn()
    const completed = {
      workflow_id: 'erw-result', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
      manifest_sha256: targetVersion.manifest_sha256, current_corpus_draft_id: null,
      current_corpus_snapshot_sha256: null, fixture_validation_run_id: null, status: 'completed',
      stage: 'calibration', revision: 2, initiated_by: 'company-admin-a',
      created_at: '2026-08-16T09:00:42Z', updated_at: '2026-08-16T09:04:05Z', calibration_version: 'cal-v1',
      evaluation_dataset_id: formalDataset.dataset_id, evaluation_version: formalVersion.version, evaluation_case_count: 100,
    }
    seed({
      listReleaseWorkflows: vi.fn().mockResolvedValue([completed]),
      getReleaseWorkflow: vi.fn().mockResolvedValue(completed),
      listReleaseAttempts: vi.fn().mockResolvedValue([{
        attempt_id: 'era-baseline', stage: 'baseline', attempt_number: 1, status: 'succeeded',
        metrics: { run_id: '20260816T090043Z-fb09fc37' },
        started_at: '2026-08-16T09:00:43Z', finished_at: '2026-08-16T09:01:37Z',
      }]),
    })
    render(<App><ReleaseEvaluationPanel onViewRuns={onViewRuns} /></App>)
    fireEvent.click(await screen.findByRole('button', { name: '查看运行结果 20260816T090043Z-fb09fc37' }))
    expect(onViewRuns).toHaveBeenCalledWith('20260816T090043Z-fb09fc37')
    expect(screen.queryByText(/中国标准时间/)).toBeNull()
    expect(screen.getAllByText(/2026-08-16 17:00:42/).length).toBeGreaterThan(0)
  })
})

it('explains RAGAS worker timeout and labels answer latency scope', async () => {
  const failed = {
    workflow_id: 'erw-timeout', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
    manifest_sha256: targetVersion.manifest_sha256, current_corpus_draft_id: null,
    current_corpus_snapshot_sha256: null, fixture_validation_run_id: null, status: 'failed',
    stage: 'baseline', revision: 2, initiated_by: 'company-admin-a',
    reason_code: 'ragas_execution_timeout', created_at: '2026-08-17T16:42:45Z', updated_at: '2026-08-17T17:32:46Z',
    evaluation_dataset_id: formalDataset.dataset_id, evaluation_version: formalVersion.version, evaluation_case_count: 100,
  }
  seed({
    listReleaseWorkflows: vi.fn().mockResolvedValue([failed]),
    getReleaseWorkflow: vi.fn().mockResolvedValue(failed),
    listReleaseAttempts: vi.fn().mockResolvedValue([{
      attempt_id: 'era-timeout', stage: 'baseline', attempt_number: 1, status: 'failed',
      reason_code: 'ragas_execution_timeout', metrics: { p50: 4180.415 },
      started_at: '2026-08-17T16:42:47Z', finished_at: '2026-08-17T17:32:46Z',
    }]),
  })
  render(<App><ReleaseEvaluationPanel /></App>)
  expect((await screen.findAllByText(/RAGAS 评分超过 Worker 的执行时间上限/)).length).toBeGreaterThan(0)
  expect(screen.queryByText(/P50 端到端回答延迟（毫秒）/)).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: /查看详细评测数据与审计记录/ }))
  expect(screen.getByText(/P50 端到端回答延迟（毫秒）/)).toBeTruthy()
})


it('explains an all-unscored RAGAS run instead of showing an unmapped reason', async () => {
  const failed = {
    workflow_id: 'erw-unscored', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
    manifest_sha256: targetVersion.manifest_sha256, current_corpus_draft_id: null,
    current_corpus_snapshot_sha256: null, fixture_validation_run_id: null, status: 'failed',
    stage: 'baseline', revision: 2, initiated_by: 'company-admin-a',
    reason_code: 'ragas_coverage_unscored', created_at: '2026-08-17T16:42:45Z', updated_at: '2026-08-17T17:32:46Z',
    evaluation_dataset_id: formalDataset.dataset_id, evaluation_version: formalVersion.version, evaluation_case_count: 100,
  }
  seed({
    listReleaseWorkflows: vi.fn().mockResolvedValue([failed]),
    getReleaseWorkflow: vi.fn().mockResolvedValue(failed),
    listReleaseAttempts: vi.fn().mockResolvedValue([{
      attempt_id: 'era-unscored', stage: 'baseline', attempt_number: 1, status: 'failed',
      reason_code: 'ragas_coverage_unscored', metrics: {},
      started_at: '2026-08-17T16:42:47Z', finished_at: '2026-08-17T17:32:46Z',
    }]),
  })
  render(<App><ReleaseEvaluationPanel /></App>)
  expect((await screen.findAllByText(/RAGAS 没有产生有效评分/)).length).toBeGreaterThan(0)
  expect(screen.queryByText(/未映射原因：ragas_coverage_unscored/)).toBeNull()
})


it('explains a RAGAS input preparation failure without exposing an internal exception', async () => {
  const failed = {
    workflow_id: 'erw-input-prepare', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
    manifest_sha256: targetVersion.manifest_sha256, current_corpus_draft_id: null,
    current_corpus_snapshot_sha256: null, fixture_validation_run_id: null, status: 'failed',
    stage: 'baseline', revision: 2, initiated_by: 'company-admin-a',
    reason_code: 'ragas_input_preparation_failed', created_at: '2026-08-18T08:16:25Z', updated_at: '2026-08-18T08:24:41Z',
    evaluation_dataset_id: formalDataset.dataset_id, evaluation_version: formalVersion.version, evaluation_case_count: 100,
  }
  seed({
    listReleaseWorkflows: vi.fn().mockResolvedValue([failed]),
    getReleaseWorkflow: vi.fn().mockResolvedValue(failed),
    listReleaseAttempts: vi.fn().mockResolvedValue([{
      attempt_id: 'era-input-prepare', stage: 'baseline', attempt_number: 1, status: 'failed',
      reason_code: 'ragas_input_preparation_failed', metrics: {},
      started_at: '2026-08-18T08:16:25Z', finished_at: '2026-08-18T08:24:41Z',
    }]),
  })
  render(<App><ReleaseEvaluationPanel /></App>)
  expect((await screen.findAllByText(/RAGAS 评分输入准备失败/)).length).toBeGreaterThan(0)
  expect(screen.queryByText(/AttributeError/)).toBeNull()
  expect(screen.queryByText(/未映射原因：ragas_input_preparation_failed/)).toBeNull()
})


it('explains that incomplete formal QA stops before RAGAS scoring', async () => {
  const failed = {
    workflow_id: 'erw-qa-incomplete', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
    manifest_sha256: targetVersion.manifest_sha256, current_corpus_draft_id: null,
    current_corpus_snapshot_sha256: null, fixture_validation_run_id: null, status: 'failed',
    stage: 'baseline', revision: 2, initiated_by: 'company-admin-a',
    reason_code: 'formal_qa_snapshot_incomplete', created_at: '2026-08-18T08:16:25Z', updated_at: '2026-08-18T08:24:41Z',
    evaluation_dataset_id: formalDataset.dataset_id, evaluation_version: formalVersion.version, evaluation_case_count: 100,
  }
  seed({
    listReleaseWorkflows: vi.fn().mockResolvedValue([failed]),
    getReleaseWorkflow: vi.fn().mockResolvedValue(failed),
    listReleaseAttempts: vi.fn().mockResolvedValue([{
      attempt_id: 'era-qa-incomplete', stage: 'baseline', attempt_number: 1, status: 'failed',
      reason_code: 'formal_qa_snapshot_incomplete', metrics: {},
      started_at: '2026-08-18T08:16:25Z', finished_at: '2026-08-18T08:24:41Z',
    }]),
  })
  render(<App><ReleaseEvaluationPanel /></App>)
  expect((await screen.findAllByText(/正式问答快照不完整/)).length).toBeGreaterThan(0)
  expect(screen.queryByText(/未映射原因：formal_qa_snapshot_incomplete/)).toBeNull()
})


it('shows maker-checker controls and submits a bounded approval reason', async () => {
  const pending = {
    workflow_id: 'erw-pending', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
    manifest_sha256: targetVersion.manifest_sha256, status: 'pending_approval', stage: 'approval',
    revision: 3, initiated_by: 'company-admin-a', evaluation_dataset_id: formalDataset.dataset_id,
    evaluation_version: formalVersion.version, evaluation_case_count: 100,
  }
  seed({
    listReleaseWorkflows: vi.fn().mockResolvedValue([pending]),
    getReleaseWorkflow: vi.fn().mockResolvedValue(pending),
    reviewReleaseWorkflow: vi.fn().mockResolvedValue({ ...pending, status: 'approved', revision: 4, target_mode: 'shadow' }),
  })
  render(<App><ReleaseEvaluationPanel /></App>)
  const reason = await screen.findByRole('textbox', { name: '审批或回滚原因' })
  fireEvent.change(reason, { target: { value: '独立管理员复核通过' } })
  fireEvent.click(screen.getByRole('button', { name: '独立审批' }))
  await waitFor(() => expect(api.reviewReleaseWorkflow).toHaveBeenCalledWith(
    'erw-pending', 'approve', 3, '独立管理员复核通过', 'shadow',
  ))
})

it('shows RAGAS applicability separately from the all-case behavior contract', async () => {
  const failed = {
    workflow_id: 'erw-applicability', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
    manifest_sha256: targetVersion.manifest_sha256, current_corpus_draft_id: null,
    current_corpus_snapshot_sha256: null, fixture_validation_run_id: null, status: 'failed',
    stage: 'baseline', revision: 2, initiated_by: 'company-admin-a',
    reason_code: 'ragas_required_score_failed', created_at: '2026-08-19T09:00:00Z', updated_at: '2026-08-19T09:10:00Z',
    evaluation_dataset_id: formalDataset.dataset_id, evaluation_version: formalVersion.version, evaluation_case_count: 100,
  }
  seed({
    listReleaseWorkflows: vi.fn().mockResolvedValue([failed]),
    getReleaseWorkflow: vi.fn().mockResolvedValue(failed),
    listReleaseAttempts: vi.fn().mockResolvedValue([{
      attempt_id: 'era-applicability', stage: 'baseline', attempt_number: 1, status: 'failed',
      reason_code: 'ragas_required_score_failed',
      metrics: {
        gate_mode: 'off',
        evaluation_case_count: 100,
        ragas_judge_model: 'deepseek-v4-flash',
        evaluation_coverage: { ragas_planned_outcomes: 80, custom_planned_outcomes: 34 },
        ragas_coverage: {
          faithfulness: {
            total: 100, scored: 79, not_applicable: 20, skipped: 0, failed: 1, mean: 0.61,
            not_applicable_reasons: { ordinary_answer_metric_invalid_for_background_only: 20 },
            failure_reasons: { judge_structured_output_retry_exhausted: 1 },
            failure_exceptions: { InstructorRetryException: 1 },
            failure_finish_reasons: { length: 1 },
            batch_fallbacks: 1,
          },
        },
        evidence_gate_contract_coverage: {
          total: 100, scored: 100, not_applicable: 0, skipped: 0, failed: 0, mean: 0.99,
          mismatch_by_category: { fully_answerable: 1 },
        },
        category_metrics: {
          case_count: 100,
          outcome_count: 34,
          metrics: {
            hallucination_rate: {
              total: 20, scored: 20, mean: 0.1, direction: 'maximum', hard_gate: false,
            },
            injection_detection: {
              total: 4, scored: 4, mean: 1, direction: 'minimum', hard_gate: true,
            },
            grounding_correctness: {
              total: 10, scored: 0, not_applicable: 10, mean: null, direction: 'minimum', hard_gate: false,
              not_applicable_reasons: { evaluation_stage_disabled: 10 },
            },
          },
        },
      },
      started_at: '2026-08-19T09:00:00Z', finished_at: '2026-08-19T09:10:00Z',
    }]),
  })

  render(<App><ReleaseEvaluationPanel /></App>)

  expect(await screen.findByText('评测已产出结果，但仍有评分失败')).toBeTruthy()
  expect(screen.getByText('样本 100/100')).toBeTruthy()
  expect(screen.getByText('RAGAS 计划 80/80')).toBeTruthy()
  expect(screen.getByText('类别专项 34/34')).toBeTruthy()
  expect(screen.getByText('评分失败 1')).toBeTruthy()
  expect(screen.queryByText('RAGAS 适用性覆盖')).toBeNull()
  expect(screen.queryByText('正式行为契约覆盖')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: /查看详细评测数据与审计记录/ }))
  expect(screen.getByText('RAGAS 适用性覆盖')).toBeTruthy()
  expect(screen.getByText('正式行为契约覆盖')).toBeTruthy()
  expect(screen.getByText(/100\/100 条样本已有类别专项评测/)).toBeTruthy()
  expect(screen.getByText(/RAGAS 策略结果 80\/100；类别策略应执行 80 项；按类别规则不执行 20；评分失败 1/)).toBeTruthy()
  expect(screen.getByText(/不代表样本漏评/)).toBeTruthy()
  expect(screen.getByText(/适用 80\/100（已评分 79；评分失败 1/)).toBeTruthy()
  expect(screen.getByText(/不适用 20（仅背景信息题型按评测策略不适用普通答案指标 20/)).toBeTruthy()
  expect(screen.getByText(/异常类型（Instructor 结构化校验重试耗尽 1）/)).toBeTruthy()
  expect(screen.getByText(/完成原因（达到输出长度上限 1）/)).toBeTruthy()
  expect(screen.getByText(/批量回退 1/)).toBeTruthy()
  expect(screen.getByText(/契约不匹配（完全可回答 1）/)).toBeTruthy()
  expect(screen.getByText(/适用 100\/100（已评分 100；评分失败 0）；不适用 0；契约不匹配（完全可回答 1）；适用项均值 0.9900/)).toBeTruthy()
  expect(screen.getByText(/本次 RAGAS 评分模型：deepseek-v4-flash/)).toBeTruthy()
  expect(screen.getByText(/幻觉率：已评分 20\/20；均值 0.1000（越低越好，0–1）/)).toBeTruthy()
  expect(screen.getByText(/提示词注入识别率：已评分 4\/4；均值 1.0000（越高越好，0–1；安全硬门）/)).toBeTruthy()
  expect(screen.getByText(/依据充分正确率：已评分 0\/10；不适用 10（本次对照模式未执行该指标依赖的生成或依据校验阶段 10）；均值 —/)).toBeTruthy()
  expect(screen.getByText(/本次为证据门关闭的对照基线/)).toBeTruthy()
})

it('renders category, stage, applicability, root cause, and hard gates separately', async () => {
  const completed = {
    workflow_id: 'erw-diagnostic', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
    manifest_sha256: targetVersion.manifest_sha256, status: 'completed', stage: 'calibration',
    revision: 2, initiated_by: 'company-admin-a', evaluation_dataset_id: formalDataset.dataset_id,
    evaluation_version: formalVersion.version, evaluation_case_count: 100,
  }
  seed({
    listReleaseWorkflows: vi.fn().mockResolvedValue([completed]),
    getReleaseWorkflow: vi.fn().mockResolvedValue(completed),
    listReleaseAttempts: vi.fn().mockResolvedValue([{
      attempt_id: 'era-diagnostic', stage: 'baseline', attempt_number: 1, status: 'succeeded', metrics: {},
      diagnostic_summary: {
        availability: 'available', schema_version: 'category-aware-diagnostic-summary-v1',
        policy_identities: {}, variant_identities: {},
        categories: { fully_answerable: { count: 10, coverage: 0.8 } },
        stages: { retrieval: { coverage: 0.75 } },
        metric_coverage: { faithfulness: { scored: 8, not_applicable: 2 } },
        route_transitions: { 'answered->refused': 1 }, hard_gates: ['authorization_leakage'], case_ids: ['case-1'],
        root_cause: {
          decision: 'qualification_or_routing', reason_code: 'oracle_route_material_repair', paired_count: 30,
          recommend_embedding_change: false, compared_variants: ['oracle_context', 'oracle_route'], supporting_hashes: [],
        },
      },
    }]),
  })

  render(<App><ReleaseEvaluationPanel /></App>)

  const detailsButton = await screen.findByRole('button', { name: /查看详细评测数据与审计记录/ })
  expect(screen.queryByText('诊断四层结果')).toBeNull()
  fireEvent.click(detailsButton)
  expect(screen.getByText('诊断四层结果')).toBeTruthy()
  expect(screen.getByText(/类别 · 完全可回答/)).toBeTruthy()
  expect(screen.getByText(/阶段 · 检索/)).toBeTruthy()
  expect(screen.getByText(/指标适用性 · 忠实度/)).toBeTruthy()
  expect(screen.getByText(/根因：/)).toBeTruthy()
  expect(screen.getByText(/Hard gate/)).toBeTruthy()
})

it('does not call a category-only low score a route regression when the formal contract passed', async () => {
  const completed = {
    workflow_id: 'erw-contract-passed', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
    manifest_sha256: targetVersion.manifest_sha256, status: 'completed', stage: 'calibration',
    revision: 2, initiated_by: 'company-admin-a', evaluation_dataset_id: formalDataset.dataset_id,
    evaluation_version: formalVersion.version, evaluation_case_count: 12,
  }
  seed({
    listReleaseWorkflows: vi.fn().mockResolvedValue([completed]),
    getReleaseWorkflow: vi.fn().mockResolvedValue(completed),
    listReleaseAttempts: vi.fn().mockResolvedValue([{
      attempt_id: 'era-contract-passed', stage: 'baseline', attempt_number: 1, status: 'succeeded',
      metrics: {
        evidence_gate_contract_coverage: { total: 12, scored: 12, failed: 0, not_applicable: 0, mean: 1 },
        category_metrics: {
          metrics: {
            authorization_route: { mean: 0 },
            missing_field_accuracy: { mean: 0 },
          },
        },
      },
    }]),
  })

  render(<App><ReleaseEvaluationPanel /></App>)

  expect(await screen.findByText('重点关注')).toBeTruthy()
  expect(screen.getByText(/正式行为契约已通过/)).toBeTruthy()
  expect(screen.queryByText('部分拒答、澄清、权限或缺记录题型的路由正确率偏低。')).toBeNull()
})

it('labels a historical attempt without diagnostics instead of showing zero', async () => {
  const historical = {
    workflow_id: 'erw-no-diagnostic', dataset_id: targetDataset.dataset_id, version: targetVersion.version,
    manifest_sha256: targetVersion.manifest_sha256, status: 'completed', stage: 'calibration',
    revision: 2, initiated_by: 'company-admin-a', evaluation_dataset_id: formalDataset.dataset_id,
    evaluation_version: formalVersion.version, evaluation_case_count: 100,
  }
  seed({
    listReleaseWorkflows: vi.fn().mockResolvedValue([historical]),
    getReleaseWorkflow: vi.fn().mockResolvedValue(historical),
    listReleaseAttempts: vi.fn().mockResolvedValue([{
      attempt_id: 'era-history', stage: 'baseline', attempt_number: 1, status: 'succeeded', metrics: {},
      diagnostic_summary: {
        availability: 'not_available', schema_version: null, policy_identities: {}, variant_identities: {},
        categories: {}, stages: {}, metric_coverage: {}, route_transitions: {}, hard_gates: [], case_ids: [], root_cause: null,
      },
    }]),
  })

  render(<App><ReleaseEvaluationPanel /></App>)
  const detailsButton = await screen.findByRole('button', { name: /查看详细评测数据与审计记录/ })
  expect(screen.queryByText('本次运行无该诊断')).toBeNull()
  fireEvent.click(detailsButton)
  expect(screen.getByText('本次运行无该诊断')).toBeTruthy()
  expect(screen.queryByText(/诊断四层结果.*0/)).toBeNull()
})
