// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import Dashboard from './Dashboard'

const mocks = vi.hoisted(() => ({
  readiness: vi.fn(),
  requestTrends: vi.fn(),
  stats: vi.fn(),
  taskList: vi.fn(),
  user: { username: 'cheng' },
}))

vi.mock('@/api/admin', () => ({
  adminApi: {
    readiness: mocks.readiness,
    requestTrends: mocks.requestTrends,
    stats: mocks.stats,
  },
  taskApi: {
    list: mocks.taskList,
  },
}))

vi.mock('@/stores/auth', () => ({
  useAuthStore: () => ({ user: mocks.user }),
}))

vi.mock('recharts', () => ({
  Area: () => null,
  AreaChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CartesianGrid: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Tooltip: () => null,
  XAxis: () => null,
  YAxis: () => null,
}))

const nonZeroStats = {
  vector_store: { total_vectors: 12 },
  knowledge_graph: { nodes: 198, edges: 3, status: 'ok' },
  request_trend: [],
  request_detail: {
    ai: { count: 2, error_count: 1, total_latency_ms: 300, methods: { POST: 2 }, routes: { '/api/v1/qa/ask': { count: 2, error_count: 1, last_status: 200 } } },
    system_api: { count: 3, error_count: 0, total_latency_ms: 90, methods: { GET: 3 }, routes: { '/api/v1/docs': { count: 3, error_count: 0, last_status: 200 } } },
  },
}

const liveTrend = {
  window: '60m' as const,
  timezone: 'UTC' as const,
  generated_at: '2026-08-09T03:27:00Z',
  points: [{ timestamp: '2026-08-09T03:27:00Z', requests: 5, qa: 2, errors: 1, average_latency_ms: 78, details: {} }],
  summary: {
    requests: 5,
    qa: 2,
    errors: 1,
    average_latency_ms: 78,
    details: nonZeroStats.request_detail,
  },
}

const emptyStats = {
  vector_store: { total_vectors: 0 },
  knowledge_graph: { nodes: 0, edges: 0, status: 'empty' },
  request_trend: [],
}

function statValue(title: string): string {
  const titleElement = screen.getByText(title)
  const card = titleElement.closest('.ant-card')
  if (!(card instanceof HTMLElement)) throw new Error(`missing card for ${title}`)
  const value = within(card).getByText(/\d+/)
  return value.textContent ?? ''
}

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(() => ({
      addEventListener: vi.fn(),
      addListener: vi.fn(),
      dispatchEvent: vi.fn(),
      matches: false,
      media: '',
      onchange: null,
      removeEventListener: vi.fn(),
      removeListener: vi.fn(),
    })),
  })
  vi.stubGlobal('ResizeObserver', class {
    observe = vi.fn()
    unobserve = vi.fn()
    disconnect = vi.fn()
  })
})

describe('Dashboard statistics refresh', () => {
  afterEach(() => cleanup())

  beforeEach(() => {
    mocks.stats.mockReset()
    mocks.taskList.mockReset()
    mocks.readiness.mockReset()
    mocks.requestTrends.mockReset()
    mocks.stats.mockResolvedValue(nonZeroStats)
    mocks.taskList.mockResolvedValue([])
    mocks.readiness.mockResolvedValue({
      status: 'ready',
      components: {
        vector_store: 'ready',
        knowledge_graph: 'ready',
        security_state: 'ready',
        task_broker: 'ready',
      },
    })
    mocks.requestTrends.mockResolvedValue(liveTrend)
  })

  it('replaces old graph counts when the operator refreshes after a data reset', async () => {
    render(<Dashboard />)

    await waitFor(() => expect(statValue('图谱节点')).toBe('198'))
    expect(statValue('图谱关系')).toBe('3')

    mocks.stats.mockResolvedValueOnce(emptyStats)
    fireEvent.click(screen.getByRole('button', { name: /刷新统计/ }))

    await waitFor(() => expect(statValue('图谱节点')).toBe('0'))
    expect(statValue('图谱关系')).toBe('0')
    expect(mocks.stats).toHaveBeenCalledTimes(2)
  })


  it('refreshes only the lightweight trend when the dashboard becomes visible again', async () => {
    render(<Dashboard />)

    await waitFor(() => expect(statValue('图谱节点')).toBe('198'))
    fireEvent(document, new Event('visibilitychange'))

    await waitFor(() => expect(mocks.requestTrends).toHaveBeenCalledTimes(2))
    expect(statValue('图谱节点')).toBe('198')
    expect(mocks.stats).toHaveBeenCalledTimes(1)
  })

  it('polls every 15 seconds only while the page is visible', async () => {
    vi.useFakeTimers()
    let visibility: DocumentVisibilityState = 'visible'
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => visibility })
    render(<Dashboard />)
    await act(async () => { await Promise.resolve(); await Promise.resolve() })
    expect(mocks.requestTrends).toHaveBeenCalledTimes(1)

    await act(async () => { vi.advanceTimersByTime(15_000); await Promise.resolve() })
    expect(mocks.requestTrends).toHaveBeenCalledTimes(2)

    visibility = 'hidden'
    await act(async () => { vi.advanceTimersByTime(30_000); await Promise.resolve() })
    expect(mocks.requestTrends).toHaveBeenCalledTimes(2)

    visibility = 'visible'
    fireEvent(document, new Event('visibilitychange'))
    await act(async () => { await Promise.resolve() })
    expect(mocks.requestTrends).toHaveBeenCalledTimes(3)
    vi.useRealTimers()
  })

  it('shows AI/system API detail and uses task descriptions instead of task IDs', async () => {
    mocks.taskList.mockResolvedValueOnce([{ task_id: 'opaque-task-id', status: 'QUEUED', ready: false, description: '知识更新：finance/policy.docx' }])
    render(<Dashboard />)

    expect(await screen.findByText('AI / 智能问答请求')).toBeTruthy()
    expect(screen.getByText('系统 API 请求')).toBeTruthy()
    expect(screen.getByText('知识更新：finance/policy.docx')).toBeTruthy()
    expect(screen.queryByText('opaque-task-id')).toBeNull()
  })

  it('keeps stale trend data on a polling error and clears the warning after recovery', async () => {
    render(<Dashboard />)
    expect(await screen.findByText('AI / 智能问答请求')).toBeTruthy()

    mocks.requestTrends.mockRejectedValueOnce(new Error('temporary'))
    fireEvent.click(screen.getByRole('button', { name: /刷新统计/ }))
    expect(await screen.findByText(/图表保留最近一次成功数据/)).toBeTruthy()
    expect(screen.getByText('AI / 智能问答请求')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /刷新统计/ }))
    await waitFor(() => expect(screen.queryByText(/图表保留最近一次成功数据/)).toBeNull())
  })

})
