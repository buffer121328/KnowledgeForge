// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react'
import { message } from 'antd'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { graphApi } from '@/api/graph'
import { useKnowledgeGraph } from './useKnowledgeGraph'

vi.mock('antd', async () => {
  const actual = await vi.importActual<typeof import('antd')>('antd')
  return { ...actual, message: { error: vi.fn(), warning: vi.fn() } }
})

vi.mock('@/api/graph', () => ({
  graphApi: {
    companyOverview: vi.fn(), department: vi.fn(), subgraph: vi.fn(), types: vi.fn(),
    paths: vi.fn(), evidence: vi.fn(),
  },
}))

const overview = {
  company_id: 'tenant-a', company_name: '示例公司',
  departments: [
    { department_id: 'finance', name: '财务部', document_count: 5 },
    { department_id: 'hr', name: '人事部', document_count: 3 },
  ],
  nodes: [
    { id: 'company:tenant-a', label: '示例公司', type: 'Company' },
    { id: 'department:finance', label: '财务部', type: 'Department', document_count: 5 },
    { id: 'department:hr', label: '人事部', type: 'Department', document_count: 3 },
    { id: 'document:budget', label: '预算.docx', type: 'Document' },
    { id: 'entity:ada', label: 'Ada', type: 'Person' },
    { id: 'relationclaim:uses', label: 'USES', type: 'RelationClaim' },
  ],
  edges: [
    { source: 'company:tenant-a', target: 'department:finance', label: 'HAS_DEPARTMENT', path_count: 5 },
    { source: 'company:tenant-a', target: 'department:hr', label: 'HAS_DEPARTMENT', path_count: 3 },
    { source: 'department:finance', target: 'department:hr', label: 'CROSS_DEPARTMENT', path_count: 2, claim_ids: ['claim-a', 'claim-b'] },
    { source: 'department:finance', target: 'entity:ada', label: 'HAS_ENTITY' },
  ],
  status: 'ok',
}

const companyOverview = vi.mocked(graphApi.companyOverview)
const department = vi.mocked(graphApi.department)
describe('useKnowledgeGraph', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    companyOverview.mockResolvedValue(overview)
    department.mockResolvedValue({
      nodes: [
        { id: 'department:finance', label: '财务部', type: 'Department' },
        { id: 'entity:budget', label: '年度预算', type: 'Concept', description: '年度费用规划' },
        { id: 'entity:ada', label: 'Ada', type: 'Person', description: '预算负责人' },
        { id: 'entity:system', label: '财务系统', type: 'Technology' },
        { id: 'document:budget', label: '预算.docx', type: 'Document' },
        { id: 'relationclaim:owns', label: 'OWNS', type: 'RelationClaim' },
      ],
      edges: [
        { source: 'department:finance', target: 'entity:budget', label: 'MANAGES' },
        { source: 'entity:budget', target: 'entity:ada', label: 'APPROVED_BY' },
        { source: 'entity:ada', target: 'entity:system', label: 'USES' },
        { source: 'entity:budget', target: 'document:budget', label: 'FROM' },
      ],
      status: 'ok',
    })
  })

  it('opens a lightweight department network and returns to the company overview', async () => {
    const { result } = renderHook(() => useKnowledgeGraph())
    await waitFor(() => expect(result.current.companyName).toBe('示例公司'))

    await act(async () => { await result.current.enterDepartment('finance') })

    await waitFor(() => expect(department).toHaveBeenCalledWith('finance', false, expect.any(AbortSignal)))
    expect(result.current.scope).toBe('department')
    expect(result.current.visibleNodes.map((node) => node.type)).toEqual([
      'Concept', 'Person', 'Technology',
    ])
    expect(result.current.visibleEdges).toEqual([
      { source: 'entity:budget', target: 'entity:ada', label: 'APPROVED_BY' },
      { source: 'entity:ada', target: 'entity:system', label: 'USES' },
    ])

    await act(async () => { await result.current.handleReset() })
    await waitFor(() => expect(result.current.scope).toBe('company'))
  })

  it('filters department entities by keyword and type without leaving dangling relationships', async () => {
    const { result } = renderHook(() => useKnowledgeGraph())
    await waitFor(() => expect(result.current.companyName).toBe('示例公司'))
    await act(async () => { await result.current.enterDepartment('finance') })

    act(() => result.current.setEntityQuery('负责人'))
    expect(result.current.visibleNodes.map((node) => node.id)).toEqual([
      'entity:ada',
    ])
    expect(result.current.visibleEdges).toEqual([])

    act(() => {
      result.current.setEntityQuery('')
      result.current.setEntityType('Concept')
    })
    expect(result.current.visibleNodes.map((node) => node.id)).toEqual([
      'entity:budget',
    ])
    expect(result.current.visibleEdges).toEqual([])
    expect(department).toHaveBeenCalledTimes(1)
  })

  it('filters relationship labels, clears hidden selection, and restores locally', async () => {
    const { result } = renderHook(() => useKnowledgeGraph())
    await waitFor(() => expect(result.current.companyName).toBe('示例公司'))
    await act(async () => { await result.current.enterDepartment('finance') })

    act(() => result.current.setSelected(result.current.nodes.find((node) => node.id === 'entity:budget') ?? null))
    act(() => result.current.setRelationType('USES'))

    await waitFor(() => expect(result.current.selected).toBeNull())
    expect(result.current.visibleNodes.map((node) => node.id)).toEqual([
      'entity:ada', 'entity:system',
    ])
    expect(result.current.visibleEdges).toEqual([
      { source: 'entity:ada', target: 'entity:system', label: 'USES' },
    ])
    expect(result.current.hasActiveFilters).toBe(true)

    act(() => result.current.clearFilters())
    expect(result.current.visibleNodes).toEqual(result.current.nodes)
    expect(result.current.visibleEdges).toEqual(result.current.edges)
    expect(result.current.hasActiveFilters).toBe(false)
    expect(department).toHaveBeenCalledTimes(1)
  })

  it('loads only the company overview and defensively retains its organizational network', async () => {
    const { result } = renderHook(() => useKnowledgeGraph())

    await waitFor(() => expect(result.current.companyName).toBe('示例公司'))
    expect(result.current.departments[0].document_count).toBe(5)
    expect(result.current.visibleNodes).toEqual([
      { id: 'company:tenant-a', label: '示例公司', type: 'Company' },
      { id: 'department:finance', label: '财务部', type: 'Department', document_count: 5 },
      { id: 'department:hr', label: '人事部', type: 'Department', document_count: 3 },
    ])
    expect(result.current.visibleEdges).toEqual([
      { source: 'company:tenant-a', target: 'department:finance', label: 'HAS_DEPARTMENT', path_count: 5 },
      { source: 'company:tenant-a', target: 'department:hr', label: 'HAS_DEPARTMENT', path_count: 3 },
      { source: 'department:finance', target: 'department:hr', label: 'CROSS_DEPARTMENT', path_count: 2, claim_ids: ['claim-a', 'claim-b'] },
    ])
    expect(graphApi.department).not.toHaveBeenCalled()
    expect(graphApi.subgraph).not.toHaveBeenCalled()
    expect(graphApi.types).not.toHaveBeenCalled()
    expect(graphApi.paths).not.toHaveBeenCalled()
    expect(graphApi.evidence).not.toHaveBeenCalled()
  })

  it('clears stale graph counts when the authoritative overview is empty', async () => {
    const { result } = renderHook(() => useKnowledgeGraph())
    await waitFor(() => expect(result.current.stats).toEqual({ nodes: 3, edges: 3 }))

    companyOverview.mockResolvedValueOnce({
      company_id: 'tenant-a',
      company_name: '',
      departments: [],
      nodes: [],
      edges: [],
      status: 'empty',
    })
    await act(async () => { await result.current.handleReset() })

    await waitFor(() => expect(result.current.stats).toEqual({ nodes: 0, edges: 0 }))
    expect(result.current.visibleNodes).toEqual([])
    expect(result.current.visibleEdges).toEqual([])
    expect(result.current.departments).toEqual([])
    expect(result.current.selected).toBeNull()
    expect(result.current.status).toBe('empty')
  })

  it('cancels stale scope requests so earlier responses cannot overwrite the latest view', async () => {
    let resolveFirst: ((value: typeof overview) => void) | undefined
    companyOverview
      .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve }))
      .mockResolvedValueOnce({ ...overview, company_name: '最新公司' })
    const { result } = renderHook(() => useKnowledgeGraph())

    act(() => result.current.handleReset())
    await waitFor(() => expect(result.current.companyName).toBe('最新公司'))
    await act(async () => resolveFirst?.({ ...overview, company_name: '过期公司' }))

    expect(result.current.companyName).toBe('最新公司')
  })

  it('exposes a permission-safe state without retaining unauthorized graph data', async () => {
    companyOverview.mockRejectedValue({ response: { status: 403 } })
    const { result } = renderHook(() => useKnowledgeGraph())

    await waitFor(() => expect(result.current.status).toBe('permission-denied'))
    expect(result.current.visibleNodes).toEqual([])
    expect(message.error).toHaveBeenCalledWith('无权查看该图谱范围')
  })
})
