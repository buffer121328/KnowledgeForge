// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { LineDashedMaterial, type Mesh, type Object3D } from 'three'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { GraphEdge, GraphNode } from '@/api/graph'
import { KnowledgeGraph3D } from './KnowledgeGraph3D'

interface RenderGraphLink extends GraphEdge {
  isIndirect?: boolean
  renderId?: string
  renderIndex?: number
  renderTotal?: number
}

interface ForceGraphProps {
  controlType?: 'orbit' | 'trackball' | 'fly'
  graphData?: { nodes: Array<GraphNode & { fx?: number; fy?: number; fz?: number }>; links: RenderGraphLink[] }
  linkDirectionalParticles?: number | ((link: RenderGraphLink) => number)
  linkThreeObject?: (link: RenderGraphLink) => Object3D
  nodeThreeObject?: (node: GraphNode) => Object3D
  onEngineStop?: () => void
  onNodeHover?: (node: GraphNode | null) => void
  onNodeClick?: (node: GraphNode, event: MouseEvent) => void
}

const forceGraph = vi.hoisted(() => {
  const controls = {
    autoRotate: false,
    autoRotateSpeed: 0,
    dampingFactor: 0,
    enableDamping: false,
    target: { set: vi.fn() },
    update: vi.fn(),
  }
  return {
    controls,
    methods: {
      controls: vi.fn(() => controls),
      lights: vi.fn(),
      pauseAnimation: vi.fn(),
      refresh: vi.fn(),
      zoomToFit: vi.fn(),
    },
    props: null as ForceGraphProps | null,
  }
})

vi.mock('react-force-graph-3d', async () => {
  const React = await import('react')

  const MockForceGraph3D = React.forwardRef(function MockForceGraph3D(
    props: ForceGraphProps,
    ref: React.ForwardedRef<typeof forceGraph.methods>,
  ) {
    React.useImperativeHandle(ref, () => forceGraph.methods)
    forceGraph.props = props
    return <div data-testid="force-graph-3d" />
  })

  return { default: MockForceGraph3D }
})

class ResizeObserverMock {
  constructor(private readonly callback: ResizeObserverCallback) {}

  observe() {
    this.callback([], this as unknown as ResizeObserver)
  }

  disconnect() {}
}

const node: GraphNode = {
  id: 'person-1',
  label: 'Ada',
  type: 'Person',
  description: 'Engineer',
}
const technology: GraphNode = { id: 'tech-1', label: 'Three.js', type: 'Technology' }
const organization: GraphNode = { id: 'org-1', label: 'OpenAI', type: 'Organization' }
const location: GraphNode = { id: 'location-1', label: 'Shanghai', type: 'Location' }
const edge: GraphEdge = { source: 'person-1', target: 'tech-1', label: 'USES' }
const employment: GraphEdge = { source: 'org-1', target: 'person-1', label: 'EMPLOYS' }
const unrelated: GraphEdge = { source: 'org-1', target: 'location-1', label: 'LOCATED_IN' }

describe('KnowledgeGraph3D', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('ResizeObserver', ResizeObserverMock)
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(640)
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(420)
    forceGraph.methods.pauseAnimation.mockReset()
    forceGraph.methods.refresh.mockReset()
    forceGraph.methods.zoomToFit.mockReset()
    forceGraph.methods.controls.mockReset().mockReturnValue(forceGraph.controls)
    forceGraph.methods.lights.mockReset()
    forceGraph.controls.autoRotate = false
    forceGraph.controls.autoRotateSpeed = 0
    forceGraph.controls.dampingFactor = 0
    forceGraph.controls.enableDamping = false
    forceGraph.controls.target.set.mockReset()
    forceGraph.controls.update.mockReset()
    forceGraph.props = null
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('pins a company above departments and expands indirect path links', () => {
    const company: GraphNode = { id: 'company:tenant-a', label: '示例公司', type: 'Company' }
    const finance: GraphNode = { id: 'department:finance', label: '财务部', type: 'Department' }
    const hr: GraphNode = { id: 'department:hr', label: '人事部', type: 'Department' }
    const overviewEdges: GraphEdge[] = [
      { source: company.id, target: finance.id, label: 'HAS_DEPARTMENT' },
      { source: company.id, target: hr.id, label: 'HAS_DEPARTMENT' },
      { source: finance.id, target: hr.id, label: 'CROSS_DEPARTMENT', path_count: 3 },
    ]

    render(
      <KnowledgeGraph3D
        colorMap={{ Company: '#d4380d', Department: '#08979c' }}
        edges={overviewEdges}
        nodes={[company, finance, hr]}
        onNodeSelect={vi.fn()}
        selected={null}
      />,
    )

    const graphData = forceGraph.props?.graphData
    expect(graphData?.nodes).toHaveLength(3)
    const renderedCompany = graphData?.nodes.find((node) => node.id === company.id)
    const renderedFinance = graphData?.nodes.find((node) => node.id === finance.id)
    expect(renderedCompany?.fy).toBeGreaterThan(renderedFinance?.fy ?? 0)
    expect(renderedCompany?.fx).toBe(0)
    expect(graphData?.links).toHaveLength(5)
    expect(graphData?.links.filter((link) => link.isIndirect)).toHaveLength(3)

    const indirectLink = graphData?.links.find((link) => link.isIndirect)
    const linkObject = indirectLink && forceGraph.props?.linkThreeObject?.(indirectLink)
    expect((linkObject as { material?: unknown } | undefined)?.material).toBeInstanceOf(LineDashedMaterial)

    const createNodeObject = forceGraph.props?.nodeThreeObject
    const companyObject = createNodeObject?.(company)
    const departmentObject = createNodeObject?.(finance)
    expect(companyObject?.scale.x).toBeGreaterThan(departmentObject?.scale.x ?? 0)
  })

  it('uses slow OrbitControls camera rotation for the organization overview', () => {
    const company: GraphNode = { id: 'company:tenant-a', label: '示例公司', type: 'Company' }
    const finance: GraphNode = { id: 'department:finance', label: '财务部', type: 'Department' }
    render(
      <KnowledgeGraph3D
        colorMap={{ Company: '#d4380d', Department: '#08979c' }}
        edges={[{ source: company.id, target: finance.id, label: 'HAS_DEPARTMENT' }]}
        nodes={[company, finance]}
        onNodeSelect={vi.fn()}
        selected={null}
      />,
    )

    expect(forceGraph.props?.controlType).toBe('orbit')
    expect(forceGraph.controls.autoRotate).toBe(true)
    expect(forceGraph.controls.autoRotateSpeed).toBeGreaterThan(0)
    expect(forceGraph.controls.autoRotateSpeed).toBeLessThan(1)
    expect(forceGraph.controls.enableDamping).toBe(true)
    expect(forceGraph.controls.target.set).toHaveBeenCalledWith(0, 0, 0)
    expect(forceGraph.controls.update).toHaveBeenCalled()
    expect(forceGraph.methods.lights).toHaveBeenCalledWith(
      expect.arrayContaining([expect.objectContaining({ isLight: true })]),
    )
  })

  it('keeps slow OrbitControls camera rotation enabled inside a department graph', () => {
    render(
      <KnowledgeGraph3D
        colorMap={{ Person: '#1677ff', Technology: '#722ed1' }}
        edges={[edge]}
        nodes={[node, technology]}
        onNodeSelect={vi.fn()}
        selected={null}
      />,
    )

    expect(forceGraph.props?.controlType).toBe('orbit')
    expect(forceGraph.controls.autoRotate).toBe(true)
    expect(forceGraph.controls.autoRotateSpeed).toBeGreaterThan(0)
    expect(forceGraph.controls.autoRotateSpeed).toBeLessThan(1)
    expect(forceGraph.controls.enableDamping).toBe(true)
  })

  it('keeps the graph ref after hover updates so reset view still fits the current graph', async () => {
    render(
      <KnowledgeGraph3D
        colorMap={{ Person: '#1677ff', Technology: '#722ed1' }}
        edges={[edge]}
        nodes={[node, technology]}
        onNodeSelect={vi.fn()}
        selected={null}
      />,
    )

    expect(screen.getByTestId('force-graph-3d')).toBeTruthy()
    expect(forceGraph.props?.onNodeHover).toBeTypeOf('function')

    act(() => {
      vi.advanceTimersByTime(100)
    })
    forceGraph.methods.zoomToFit.mockClear()

    act(() => {
      forceGraph.props?.onNodeHover?.(node)
    })

    fireEvent.click(screen.getByRole('button', { name: '重置视角' }))
    act(() => {
      vi.runOnlyPendingTimers()
    })

    expect(forceGraph.methods.zoomToFit).toHaveBeenCalledWith(600, 50)
  })

  it('does not shrink a user zoom when the force engine stops after initial fit', async () => {
    render(
      <KnowledgeGraph3D
        colorMap={{ Person: '#1677ff', Technology: '#722ed1' }}
        edges={[edge]}
        nodes={[node, technology]}
        onNodeSelect={vi.fn()}
        selected={null}
      />,
    )

    act(() => {
      vi.advanceTimersByTime(100)
    })
    forceGraph.methods.zoomToFit.mockClear()

    act(() => {
      forceGraph.props?.onEngineStop?.()
    })

    expect(forceGraph.methods.zoomToFit).not.toHaveBeenCalled()
  })

  it('renders distinct silhouettes and scales visible hubs above leaf nodes', () => {
    render(
      <KnowledgeGraph3D
        colorMap={{ Person: '#1677ff', Technology: '#722ed1', Organization: '#52c41a' }}
        edges={[edge, employment]}
        nodes={[node, technology, organization]}
        onNodeSelect={vi.fn()}
        selected={null}
      />,
    )

    const createNodeObject = forceGraph.props?.nodeThreeObject
    expect(createNodeObject).toBeTypeOf('function')

    const personObject = createNodeObject?.(node)
    const organizationObject = createNodeObject?.(organization)
    const personSurface = personObject?.children.find(
      (child) => child.userData.role === 'surface',
    ) as Mesh
    const organizationSurface = organizationObject?.children.find(
      (child) => child.userData.role === 'surface',
    ) as Mesh

    expect(personSurface.geometry.type).toBe('SphereGeometry')
    expect(organizationSurface.geometry.type).toBe('BoxGeometry')
    expect(personObject?.scale.x).toBeGreaterThan(organizationObject?.scale.x ?? 0)
  })

  it('reuses node objects while updating hover emphasis in place', () => {
    render(
      <KnowledgeGraph3D
        colorMap={{ Person: '#1677ff', Technology: '#722ed1' }}
        edges={[edge]}
        nodes={[node, technology]}
        onNodeSelect={vi.fn()}
        selected={null}
      />,
    )

    const initialObject = forceGraph.props?.nodeThreeObject?.(node)
    const initialHalo = initialObject?.children.find((child) => child.userData.role === 'halo')
    expect(initialHalo?.visible).toBe(false)

    act(() => {
      forceGraph.props?.onNodeHover?.(node)
    })

    const focusedObject = forceGraph.props?.nodeThreeObject?.(node)
    const focusedHalo = focusedObject?.children.find((child) => child.userData.role === 'halo')
    expect(focusedObject).toBe(initialObject)
    expect(focusedHalo?.visible).toBe(true)
  })

  it('animates only relationships incident to the selected node', () => {
    const { rerender } = render(
      <KnowledgeGraph3D
        colorMap={{
          Person: '#1677ff',
          Technology: '#722ed1',
          Organization: '#52c41a',
          Location: '#13c2c2',
        }}
        edges={[edge, employment, unrelated]}
        nodes={[node, technology, organization, location]}
        onNodeSelect={vi.fn()}
        selected={node}
      />,
    )

    const selectedParticles = forceGraph.props?.linkDirectionalParticles
    expect(typeof selectedParticles).toBe('function')
    expect(typeof selectedParticles === 'function' ? selectedParticles(edge) : selectedParticles).toBe(2)
    expect(
      typeof selectedParticles === 'function' ? selectedParticles(unrelated) : selectedParticles,
    ).toBe(0)

    rerender(
      <KnowledgeGraph3D
        colorMap={{
          Person: '#1677ff',
          Technology: '#722ed1',
          Organization: '#52c41a',
          Location: '#13c2c2',
        }}
        edges={[edge, employment, unrelated]}
        nodes={[node, technology, organization, location]}
        onNodeSelect={vi.fn()}
        selected={null}
      />,
    )

    const unfocusedParticles = forceGraph.props?.linkDirectionalParticles
    expect(typeof unfocusedParticles === 'function' ? unfocusedParticles(edge) : unfocusedParticles).toBe(
      0,
    )
  })

  it('opens only department nodes on a double-click while retaining normal selection', () => {
    const department = { id: 'department:finance', label: '财务部', type: 'Department' }
    const selectNode = vi.fn()
    const openDepartment = vi.fn()
    render(
      <KnowledgeGraph3D
        colorMap={{ Department: '#08979c', Person: '#1677ff' }}
        edges={[]}
        nodes={[department, node]}
        onNodeSelect={selectNode}
        onDepartmentOpen={openDepartment}
        selected={null}
      />,
    )

    forceGraph.props?.onNodeClick?.(department, { detail: 1 } as MouseEvent)
    forceGraph.props?.onNodeClick?.(department, { detail: 2 } as MouseEvent)
    forceGraph.props?.onNodeClick?.(node, { detail: 2 } as MouseEvent)

    expect(selectNode).toHaveBeenCalledTimes(3)
    expect(openDepartment).toHaveBeenCalledTimes(1)
    expect(openDepartment).toHaveBeenCalledWith(department)
  })
})
