// @vitest-environment jsdom
import { render, screen } from '@testing-library/react'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import GraphView from './GraphView'

const { useKnowledgeGraphMock } = vi.hoisted(() => ({
  useKnowledgeGraphMock: vi.fn(),
}))

vi.mock('@/components/GraphCanvas', () => ({
  GraphCanvas: () => <div data-testid="organizational-graph" />,
}))

vi.mock('./useKnowledgeGraph', () => ({
  useKnowledgeGraph: useKnowledgeGraphMock,
}))

const baseGraphState = {
  clearFilters: vi.fn(),
  colorMap: { Company: '#d4380d', Department: '#08979c' },
  companyName: '示例公司',
  departmentName: '',
  edges: [
    { source: 'company:tenant-a', target: 'department:finance', label: 'HAS_DEPARTMENT' },
    { source: 'company:tenant-a', target: 'department:hr', label: 'HAS_DEPARTMENT' },
    { source: 'department:finance', target: 'department:hr', label: 'CROSS_DEPARTMENT', path_count: 2 },
  ],
  enterDepartment: vi.fn(),
  entityQuery: '',
  entityType: 'all',
  entityTypes: ['Company', 'Department'],
  filterEntityTypes: [],
  handleReset: vi.fn(),
  hasActiveFilters: false,
  loading: false,
  nodes: [
    { id: 'company:tenant-a', label: '示例公司', type: 'Company' },
    { id: 'department:finance', label: '财务部', type: 'Department' },
    { id: 'department:hr', label: '人事部', type: 'Department' },
  ],
  relationType: 'all',
  relationTypes: [],
  selected: null,
  setEntityQuery: vi.fn(),
  setEntityType: vi.fn(),
  setRelationType: vi.fn(),
  setSelected: vi.fn(),
  stats: { nodes: 3, edges: 3 },
  status: 'ok',
  scope: 'company' as const,
  visibleEdges: [
    { source: 'company:tenant-a', target: 'department:finance', label: 'HAS_DEPARTMENT' },
    { source: 'company:tenant-a', target: 'department:hr', label: 'HAS_DEPARTMENT' },
    { source: 'department:finance', target: 'department:hr', label: 'CROSS_DEPARTMENT', path_count: 2 },
  ],
  visibleNodes: [
    { id: 'department:finance', label: '财务部', type: 'Department' },
    { id: 'department:hr', label: '人事部', type: 'Department' },
  ],
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
})

beforeEach(() => {
  useKnowledgeGraphMock.mockReturnValue(baseGraphState)
})

describe('GraphView organizational network', () => {
  it('omits department filters from the company overview', () => {
    render(<GraphView />)

    expect(screen.getByTestId('organizational-graph')).toBeTruthy()
    expect(screen.queryByLabelText('筛选实体关键词')).toBeNull()
    expect(screen.getByText('公司组织架构 / 跨部门关联网络')).toBeTruthy()
    expect(screen.getByText('组织关系 / 跨部门关联')).toBeTruthy()
    expect(screen.getAllByText('跨部门').length).toBeGreaterThan(0)
  })

  it('shows entity and relationship filters inside a department', () => {
    useKnowledgeGraphMock.mockReturnValue({
      ...baseGraphState,
      departmentName: '财务部',
      entityTypes: ['Department', 'Concept', 'Person'],
      filterEntityTypes: ['Concept', 'Person'],
      relationTypes: ['APPROVED_BY', 'MANAGES'],
      scope: 'department',
    })

    render(<GraphView />)

    expect(screen.getByLabelText('筛选实体关键词')).toBeTruthy()
    expect(screen.getAllByLabelText('筛选实体类型')).toHaveLength(2)
    expect(screen.getAllByLabelText('筛选关系类型')).toHaveLength(2)
    expect(screen.getByText('部门内部关系')).toBeTruthy()
  })
})
