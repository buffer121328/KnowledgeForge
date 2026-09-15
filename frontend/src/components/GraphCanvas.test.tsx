// @vitest-environment jsdom
import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { GraphEdge, GraphNode } from '@/api/graph'
import { GraphCanvas } from './GraphCanvas'

afterEach(() => cleanup())

vi.mock('./KnowledgeGraph3D', () => ({
  KnowledgeGraph3D: () => <div data-testid="knowledge-graph-3d" />,
}))

const nodes: GraphNode[] = [
  { id: 'person-1', label: 'Ada', type: 'Person', description: 'Engineer' },
  { id: 'tech-1', label: 'Three.js', type: 'Technology', description: '3D renderer' },
  { id: 'org-1', label: 'OpenAI', type: 'Organization' },
]

const edges: GraphEdge[] = [
  { source: 'person-1', target: 'tech-1', label: 'USES' },
  { source: 'org-1', target: 'person-1', label: 'EMPLOYS' },
]

describe('GraphCanvas selected entity details', () => {
  it('shows stable identifiers and directly connected relationships', async () => {
    render(
      <GraphCanvas
        colorMap={{ Person: '#1677ff', Technology: '#722ed1', Organization: '#52c41a' }}
        edges={edges}
        entityTypes={['Person', 'Technology', 'Organization']}
        loading={false}
        nodes={nodes}
        onNodeSelect={vi.fn()}
        selected={nodes[0]}
        status="ok"
      />,
    )

    expect(await screen.findByText('实体详情')).toBeTruthy()
    expect(screen.getByText('person-1')).toBeTruthy()
    expect(screen.getByText('Ada —使用→ Three.js（技术）')).toBeTruthy()
    expect(screen.getByText('OpenAI（组织） —雇佣→ Ada')).toBeTruthy()
  })

  it('explains shape, size, and motion instead of relying on color alone', async () => {
    render(
      <GraphCanvas
        colorMap={{ Company: '#d4380d', Department: '#08979c' }}
        edges={edges}
        entityTypes={['Company', 'Department']}
        loading={false}
        nodes={nodes}
        onNodeSelect={vi.fn()}
        selected={null}
        status="ok"
      />,
    )

    expect(await screen.findByTestId('knowledge-graph-3d')).toBeTruthy()
    expect(within(screen.getByTestId('graph-legend-Company')).getByText('多面体')).toBeTruthy()
    expect(within(screen.getByTestId('graph-legend-Department')).getByText('立方体')).toBeTruthy()
    expect(screen.queryByTestId('graph-legend-Document')).toBeNull()
    expect(screen.queryByTestId('graph-legend-RelationClaim')).toBeNull()
    expect(screen.getAllByText(/形状.*实体类型/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/大小.*可见关系数/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/流光.*关系方向/).length).toBeGreaterThan(0)
  })

  it('explains the company hierarchy, automatic orbit, and multi-hop links', async () => {
    render(
      <GraphCanvas
        colorMap={{ Company: '#d4380d', Department: '#08979c' }}
        edges={[
          { source: 'company:tenant-a', target: 'department:finance', label: 'HAS_DEPARTMENT' },
          { source: 'department:finance', target: 'department:hr', label: 'CROSS_DEPARTMENT', path_count: 2 },
        ]}
        entityTypes={['Company', 'Department']}
        loading={false}
        nodes={[
          { id: 'company:tenant-a', label: '示例公司', type: 'Company' },
          { id: 'department:finance', label: '财务部', type: 'Department' },
          { id: 'department:hr', label: '人事部', type: 'Department' },
        ]}
        onNodeSelect={vi.fn()}
        selected={{ id: 'department:finance', label: '财务部', type: 'Department' }}
        status="ok"
      />,
    )

    expect(await screen.findByTestId('knowledge-graph-3d')).toBeTruthy()
    expect(screen.getByText('财务部 —跨部门多跳关联→ 人事部（部门）')).toBeTruthy()
    expect(screen.getByText(/公司节点位于顶部/)).toBeTruthy()
    expect(screen.getByText(/实线：公司与部门组织关系/)).toBeTruthy()
    expect(screen.getByText(/细虚线：部门间接关系/)).toBeTruthy()
    expect(screen.getByText('慢速环绕：已启用')).toBeTruthy()
  })

})
