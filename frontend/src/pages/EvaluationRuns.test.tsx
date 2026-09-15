// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { App } from 'antd'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { evaluationApi } from '@/api/evaluation'
import EvaluationRuns from './EvaluationRuns'

vi.mock('@/api/evaluation', () => ({
  evaluationApi: {
    listRuns: vi.fn(),
    getRun: vi.fn(),
    getRecords: vi.fn(),
    getAnomalies: vi.fn(),
    getRunReport: vi.fn(),
    submitSpotCheck: vi.fn(),
  },
}))

const listRuns = vi.mocked(evaluationApi.listRuns)
const getRun = vi.mocked(evaluationApi.getRun)
const getRecords = vi.mocked(evaluationApi.getRecords)
const getAnomalies = vi.mocked(evaluationApi.getAnomalies)
const getRunReport = vi.mocked(evaluationApi.getRunReport)

const runSummary = {
  run_id: 'run-001',
  started_at: '2026-08-01T10:00:00Z',
  incomplete: false,
  retrieval_modes: ['hybrid'],
  run_classification: 'smoke_only',
  record_count: 1,
  completed: 1,
  failed: 0,
  invalid_provenance: 0,
  skipped_lines: 0,
}

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

/** Render the page wrapped in the antd App provider. */
function renderPage() {
  return render(
    <App>
      <EvaluationRuns />
    </App>,
  )
}

describe('EvaluationRuns', () => {
  /** Default anomaly and report fetches for detail views. */
  function mockReport(overrides: Record<string, unknown> = {}) {
    getRunReport.mockResolvedValue({
      run_id: 'run-001',
      run_classification: 'smoke_only',
      counts: { total: 12, succeeded: 12, failed: 0, invalid_provenance: 0 },
      sections: {
        ragas: [
          { key: 'faithfulness', kind: 'ragas', unit: 'score', value: 0.8664, scored: 3, total: 12 },
        ],
        evidence: [
          { key: 'conflict_recognition_rate', kind: 'evidence', unit: 'percent', value: 100 },
        ],
        retrieval: [
          { key: 'id_based_context_recall', kind: 'retrieval', unit: 'percent', value: 77.78 },
        ],
        safety: [
          { key: 'unauthorized_citation_rate', kind: 'safety', unit: 'percent', value: 0, direction: 'lower_is_better' },
        ],
      },
      advisories: [],
      ...overrides,
    })
  }

  /** Default anomaly fetch: no anomalies. */
  function mockNoAnomalies() {
    getAnomalies.mockResolvedValue({
      run_id: 'run-001',
      total_records: 1,
      anomaly_count: 0,
      spot_checked: 0,
      low_score_threshold: 0.5,
      anomalies: [],
    })
  }

  it('opens an initial release run and displays China Standard Time', async () => {
    mockNoAnomalies()
    mockReport()
    listRuns.mockResolvedValue({ runs: [runSummary] })
    getRun.mockResolvedValue({
      run_id: runSummary.run_id,
      incomplete: false,
      metadata: { started_at: runSummary.started_at },
      quality_report: null,
      ragas_summaries: {},
    })
    getRecords.mockResolvedValue({
      run_id: runSummary.run_id,
      page: 1,
      page_size: 20,
      total: 0,
      skipped_lines: 0,
      records: [],
    })

    render(<App><EvaluationRuns initialRunId={runSummary.run_id} /></App>)

    expect(await screen.findByText('2026-08-01 18:00:00')).toBeTruthy()
    expect(getRun).toHaveBeenCalledWith(runSummary.run_id)
  })

  it('shows the empty state when there are no runs', async () => {
    listRuns.mockResolvedValue({ runs: [] })
    renderPage()

    expect(await screen.findByText(/暂无评测运行/)).toBeTruthy()
    expect(screen.getByText('RAGAS 指标状态')).toBeTruthy()
    expect(screen.getByText('忠实度')).toBeTruthy()
    expect(screen.getByText('事实正确性')).toBeTruthy()
    expect(screen.getByText('上下文精确率')).toBeTruthy()
    expect(screen.getByText('上下文召回率')).toBeTruthy()
  })

  it('renders the run list with the returned run id', async () => {
    listRuns.mockResolvedValue({ runs: [runSummary] })
    renderPage()

    expect(await screen.findByText('run-001')).toBeTruthy()
    expect(screen.getByText('冒烟评测')).toBeTruthy()
  })

  it('opens the detail and renders metrics and per-question records', async () => {
    mockNoAnomalies()
    mockReport()
    listRuns.mockResolvedValue({ runs: [runSummary] })
    getRun.mockResolvedValue({
      run_id: 'run-001',
      incomplete: false,
      metadata: {
        started_at: '2026-08-01T10:00:00Z',
        benchmark_source: 'manual-bench.json',
        benchmark_sha256: 'abc123',
        code_revision: 'deadbeef',
        retrieval_modes: ['hybrid'],
      },
      quality_report: {
        run_classification: 'baseline',
        metrics: { accuracy: 0.85 },
        by_category: {
          background_only: {
            counts: { total: 8, completed: 8, failed: 0, invalid_provenance: 8 },
          },
        },
      },
      ragas_summaries: {
        ragas_faithfulness: {
          metrics: {
            faithfulness: { total: 10, scored: 8, failed: 1, skipped: 1, mean: 0.72 },
          },
        },
      },
    })
    getRecords.mockResolvedValue({
      run_id: 'run-001',
      page: 1,
      page_size: 20,
      total: 1,
      skipped_lines: 0,
      records: [
        {
          benchmark_id: 'b1',
          retrieval_mode: 'hybrid',
          category: 'fully_answerable',
          status: 'succeeded',
          exception: null,
          refused: false,
          latency_ms: 123,
          question: '问题A',
          response: '回答B',
          context_count: 1,
          contexts: [],
          retrieved_context_ids: [],
          reference_context_ids: [],
          ragas_scores: { faithfulness: 0.85 },
          ragas_metric_states: {
            faithfulness: { status: 'scored', reason_code: null },
            factual_correctness: { status: 'not_applicable', reason_code: 'ordinary_answer_metric_invalid_for_fully_answerable' },
            context_recall: { status: 'failed', reason_code: 'judge_error' },
          },
        },
      ],
    })
    renderPage()

    fireEvent.click(await screen.findByText('查看详情'))

    expect(await screen.findByText('评测结果详情')).toBeTruthy()
    expect(await screen.findByText('评测报告')).toBeTruthy()
    // antd Statistic 会把数字拆成多个 span,用整段文本匹配
    const metricValue = await screen.findAllByText(
      (_content, element) => element?.textContent === '0.8664',
    )
    expect(metricValue.length).toBeGreaterThan(0)
    expect(await screen.findByText('冲突识别率')).toBeTruthy()
    const conflictValue = await screen.findAllByText(
      (_content, element) => element?.textContent === '100.00%',
    )
    expect(conflictValue.length).toBeGreaterThan(0)
    // 检索/安全分区默认折叠为面板,展开后可见指标
    fireEvent.click(await screen.findByText(/检索质量（确定性）/))
    expect(await screen.findByText('上下文召回率（ID 口径）')).toBeTruthy()
    const recallValue = await screen.findAllByText(
      (_content, element) => element?.textContent === '77.78%',
    )
    expect(recallValue.length).toBeGreaterThan(0)
    expect(await screen.findByText('忠实度：0.8500')).toBeTruthy()
    expect(await screen.findByText('事实正确性：不适用')).toBeTruthy()
    expect(await screen.findByText('上下文召回率：评分失败')).toBeTruthy()
    expect(screen.getByText('问题A')).toBeTruthy()
    expect(screen.getByText('回答B')).toBeTruthy()
    expect(screen.getAllByText('混合检索')).toHaveLength(2)
    expect(screen.getByText('完全可回答')).toBeTruthy()
    expect(screen.getByText('仅背景信息')).toBeTruthy()
    expect(screen.queryByText('fully_answerable')).toBeNull()
    expect(screen.queryByText('background_only')).toBeNull()
    await waitFor(() => expect(getRun).toHaveBeenCalledWith('run-001'))
  })

  it('shows the spot-check panel with anomalies when present', async () => {
    mockReport()
    listRuns.mockResolvedValue({ runs: [runSummary] })
    getRun.mockResolvedValue({
      run_id: 'run-001',
      incomplete: false,
      metadata: {},
      quality_report: null,
      ragas_summaries: {},
    })
    getRecords.mockResolvedValue({
      run_id: 'run-001', page: 1, page_size: 20, total: 0, skipped_lines: 0, records: [],
    })
    getAnomalies.mockResolvedValue({
      run_id: 'run-001',
      total_records: 12,
      anomaly_count: 2,
      spot_checked: 1,
      low_score_threshold: 0.5,
      anomalies: [
        {
          benchmark_id: 'case-conflict',
          retrieval_mode: 'dense_bm25_graph',
          category: 'conflicting',
          anomaly_codes: ['ragas_low_score', 'evidence_recall_gap'],
          status: 'succeeded',
          exception: null,
          question: '冲突题',
          response: '冲突回答',
          ragas_scores: { context_recall: 0.25 },
          spot_check: null,
        },
        {
          benchmark_id: 'case-ok',
          retrieval_mode: 'dense_bm25_graph',
          category: 'fully_answerable',
          anomaly_codes: ['contract_mismatch'],
          status: 'succeeded',
          question: '范围题',
          response: '范围回答',
          ragas_scores: { factual_correctness: 0.4 },
          spot_check: {
            verdict: 'judge_error',
            note: '评分器误判',
            reviewer_id: 'user-1',
            reviewed_at: '2026-08-30T05:00:00Z',
          },
        },
      ],
    })

    renderPage()
    fireEvent.click(await screen.findByText('查看详情'))

    expect(await screen.findByText('异常人工抽检')).toBeTruthy()
    expect(screen.getByText('2')).toBeTruthy()
    expect(await screen.findByText('打开抽检面板（2 条）')).toBeTruthy()
  })

  it('shows the success alert when a run has no anomalies', async () => {
    listRuns.mockResolvedValue({ runs: [runSummary] })
    getRun.mockResolvedValue({ run_id: 'run-001', incomplete: false, metadata: {}, quality_report: null, ragas_summaries: {} })
    getRecords.mockResolvedValue({ run_id: 'run-001', page: 1, page_size: 20, total: 0, skipped_lines: 0, records: [] })
    mockNoAnomalies()
    mockReport()

    renderPage()
    fireEvent.click(await screen.findByText('查看详情'))

    expect(await screen.findByText('本次运行没有检测到需要人工抽检的异常样本')).toBeTruthy()
  })

  it('shows an error alert when the run list request fails', async () => {
    listRuns.mockRejectedValue(new Error('boom'))
    renderPage()

    const alerts = await screen.findAllByText(/加载评测结果失败/)
    expect(alerts.length).toBeGreaterThan(0)
  })
})
