import { AimOutlined } from '@ant-design/icons'
import { Button, Tooltip } from 'antd'
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type MutableRefObject,
} from 'react'
import ForceGraph3D, {
  type ForceGraphMethods,
  type LinkObject,
  type NodeObject,
} from 'react-force-graph-3d'
import * as THREE from 'three'
import type { GraphEdge, GraphNode } from '@/api/graph'
import {
  buildVisibleDegreeMap,
  colorOf,
  NODE_SHAPE_DETAILS,
  shapeOf,
  toForceGraphData,
  type GraphNodeShape,
} from '@/utils/graphVisualization'
import { formatGraphRelation } from '@/utils/graphLabels'
import { entityTypeLabels, labelOf } from '@/utils/labels'

interface KnowledgeGraph3DProps {
  colorMap: Record<string, string>
  edges: GraphEdge[]
  nodes: GraphNode[]
  onDepartmentOpen?: (node: GraphNode) => void
  onNodeSelect: (node: GraphNode) => void
  selected: GraphNode | null
}

interface RenderGraphEdge extends GraphEdge {
  isIndirect: boolean
  renderId: string
  renderIndex: number
  renderTotal: number
}

type PositionedGraphNode = GraphNode & { fx?: number; fy?: number; fz?: number }
type RenderLine = THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial | THREE.LineDashedMaterial>
type RenderNode = NodeObject<GraphNode>
type RenderLink = LinkObject<GraphNode, RenderGraphEdge>
type GraphRef = ForceGraphMethods<GraphNode, RenderGraphEdge>

interface OrbitControlsLike {
  autoRotate: boolean
  autoRotateSpeed: number
  dampingFactor: number
  enableDamping: boolean
  target?: { set: (x: number, y: number, z: number) => unknown }
  update?: () => void
}

type Endpoint = string | number | { id?: string | number } | null | undefined

interface NodeVisualState {
  color: string
  degree: number
  faded: boolean
  hovered: boolean
  nodeType: GraphNode['type']
  organizationOverview: boolean
  selected: boolean
}

const GRAPH_BACKGROUND = 'rgba(2, 8, 23, 0)'
const FADED_NODE_COLOR = '#64748b'
const FADED_LINK_COLOR = '#334155'
const RELATION_COLORS = ['#93c5fd', '#5eead4', '#c4b5fd', '#fcd34d', '#fda4af', '#7dd3fc']
const GRAPH_AUTO_ROTATE_SPEED = 0.35

const NODE_GEOMETRIES: Record<GraphNodeShape, THREE.BufferGeometry> = {
  sphere: new THREE.SphereGeometry(5.1, 22, 16),
  cube: new THREE.BoxGeometry(8.2, 8.2, 8.2),
  octahedron: new THREE.OctahedronGeometry(6.2, 0),
  prism: new THREE.CylinderGeometry(5.2, 5.2, 8.6, 6),
  torus: new THREE.TorusGeometry(4.7, 1.75, 12, 28),
  cone: new THREE.ConeGeometry(5.5, 9.2, 7),
  icosahedron: new THREE.IcosahedronGeometry(5.9, 0),
  document: new THREE.BoxGeometry(7.8, 9.8, 1.8),
  tetrahedron: new THREE.TetrahedronGeometry(6.2, 0),
}
const NODE_HALO_GEOMETRY = new THREE.SphereGeometry(7.4, 16, 10)

/** Normalize a graph link endpoint to its node ID. */
function endpointId(endpoint: Endpoint): string | undefined {
  if (typeof endpoint === 'object' && endpoint !== null) {
    return endpoint.id === undefined || endpoint.id === null ? undefined : String(endpoint.id)
  }
  return endpoint === undefined || endpoint === null ? undefined : String(endpoint)
}

/** Choose the display color for a graph relation. */
function relationColor(label: string): string {
  let hash = 0
  for (let index = 0; index < label.length; index += 1) {
    hash = (hash * 31 + label.charCodeAt(index)) | 0
  }
  return RELATION_COLORS[Math.abs(hash) % RELATION_COLORS.length]
}

/** Produce a gentle deterministic curve so relationship depth remains readable. */
function relationCurvature(link: RenderLink): number {
  const sourceId = endpointId(link.source)
  const targetId = endpointId(link.target)
  if (sourceId && sourceId === targetId) return 0.45

  let hash = 0
  const key = `${sourceId ?? ''}-${link.label}-${targetId ?? ''}`
  for (let index = 0; index < key.length; index += 1) {
    hash = (hash * 31 + key.charCodeAt(index)) | 0
  }
  const magnitude = 0.045 + (Math.abs(hash) % 4) * 0.025
  return hash % 2 === 0 ? magnitude : -magnitude
}

/** Create a restrained studio-light rig that remains legible on the navy graph surface. */
function createGraphLights(): THREE.Light[] {
  const hemisphere = new THREE.HemisphereLight('#dbeafe', '#020617', 1.2)
  const key = new THREE.DirectionalLight('#eff6ff', 1.35)
  key.position.set(140, 180, 120)
  const cyanFill = new THREE.PointLight('#38bdf8', 1.05, 900)
  cyanFill.position.set(-190, 90, 170)
  const warmRim = new THREE.PointLight('#f59e0b', 0.58, 760)
  warmRim.position.set(190, -80, 120)
  return [hemisphere, key, cyanFill, warmRim]
}

/** Identify the concise Company/Department overview that needs a fixed organization layout. */
function isCompanyOverview(nodes: GraphNode[]): boolean {
  return nodes.length > 0
    && nodes.some((node) => node.type === 'Company')
    && nodes.every((node) => node.type === 'Company' || node.type === 'Department')
}

/** Place the organization root and departments in deterministic balanced hierarchy rows. */
function positionCompanyHierarchy(nodes: GraphNode[]): PositionedGraphNode[] {
  const company = nodes.find((node) => node.type === 'Company')
  if (!company) return nodes.map((node) => ({ ...node }))

  const departments = nodes
    .filter((node) => node.type === 'Department')
    .sort((left, right) => left.label.localeCompare(right.label, 'zh-Hans-CN') || left.id.localeCompare(right.id))
  const positionById = new Map<string, Pick<PositionedGraphNode, 'fx' | 'fy' | 'fz'>>()
  positionById.set(company.id, { fx: 0, fy: 180, fz: 0 })

  departments.forEach((department, index) => {
    const level = Math.floor(Math.log2(index + 1))
    const firstIndex = (2 ** level) - 1
    const indexInLevel = index - firstIndex
    const countInLevel = Math.min(2 ** level, departments.length - firstIndex)
    const horizontalGap = 170
    const x = countInLevel === 1 ? 0 : (indexInLevel - (countInLevel - 1) / 2) * horizontalGap
    positionById.set(department.id, { fx: x, fy: 54 - level * 116, fz: level % 2 === 0 ? 0 : 20 })
  })

  return nodes.map((node) => ({ ...node, ...positionById.get(node.id) }))
}

/** Expand aggregate cross-department path counts without mutating API-level relationship state. */
function expandCompanyOverviewLinks(
  nodes: GraphNode[],
  edges: GraphEdge[],
): RenderGraphEdge[] {
  const typeById = new Map(nodes.map((node) => [node.id, node.type] as const))
  return edges.flatMap((edge, edgeIndex) => {
    const indirect = edge.label === 'CROSS_DEPARTMENT'
      && typeById.get(edge.source) === 'Department'
      && typeById.get(edge.target) === 'Department'
    const total = indirect ? Math.max(0, Math.floor(edge.path_count ?? 1)) : 1
    if (total === 0) return []

    return Array.from({ length: total }, (_, renderIndex) => ({
      ...edge,
      isIndirect: indirect,
      renderId: `${edge.source}-${edge.label}-${edge.target}-${edgeIndex}-${renderIndex}`,
      renderIndex,
      renderTotal: total,
    }))
  })
}

/** Create one renderer-managed ownership or dotted indirect relationship line. */
function createLinkObject(link: RenderGraphEdge): RenderLine {
  const material = link.isIndirect
    ? new THREE.LineDashedMaterial({
      color: '#7dd3fc',
      dashSize: 2.2,
      gapSize: 1.8,
      opacity: 0.68,
      transparent: true,
    })
    : new THREE.LineBasicMaterial({ color: '#60a5fa', opacity: 0.72, transparent: true })
  const line = new THREE.Line(new THREE.BufferGeometry(), material)
  line.userData = { renderId: link.renderId }
  return line
}

/** Position a custom link and offset aggregate dotted paths into separate gentle curves. */
function updateLinkPosition<NodeType, LinkType>(
  object: THREE.Object3D,
  coords: { start: { x: number; y: number; z: number }; end: { x: number; y: number; z: number } },
  link: LinkObject<NodeType, LinkType>,
): boolean {
  if (!(object instanceof THREE.Line)) return false
  const line = object as RenderLine
  const renderLink = link as unknown as RenderGraphEdge
  const start = new THREE.Vector3(coords.start.x, coords.start.y, coords.start.z)
  const end = new THREE.Vector3(coords.end.x, coords.end.y, coords.end.z)
  const points = [start, end]

  if (renderLink.isIndirect) {
    const midpoint = start.clone().add(end).multiplyScalar(0.5)
    const direction = end.clone().sub(start)
    const normal = new THREE.Vector3(-direction.y, direction.x, 0)
    if (normal.lengthSq() < 0.001) normal.set(0, 1, 0)
    normal.normalize()
    const centeredIndex = renderLink.renderIndex - (renderLink.renderTotal - 1) / 2
    midpoint.addScaledVector(normal, centeredIndex * 12 + (centeredIndex >= 0 ? 14 : -14))
    midpoint.z += 10 + Math.abs(centeredIndex) * 5
    points.splice(0, points.length, ...new THREE.QuadraticBezierCurve3(start, midpoint, end).getPoints(20))
  }

  line.geometry.setFromPoints(points)
  line.geometry.computeBoundingSphere()
  if (line.material instanceof THREE.LineDashedMaterial) line.computeLineDistances()
  return true
}

/** Update custom relationship color and opacity as the current focus changes. */
function applyLinkVisualState(line: RenderLine, color: string, faded: boolean, indirect: boolean): void {
  line.material.color.set(color)
  line.material.opacity = faded ? 0.12 : indirect ? 0.62 : 0.72
}

/** Dispose renderer-owned link geometry and material resources. */
function disposeLinkObject(line: RenderLine): void {
  line.geometry.dispose()
  line.material.dispose()
}

/** Escape HTML-sensitive characters before building tooltip markup. */
function escapeHtml(value: string): string {
  return value.replace(
    /[&<>"']/g,
    (character) =>
      ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      })[character] ?? character,
  )
}

/** Calculate a bounded visual scale from visible degree, scope, and interaction focus. */
function nodeVisualScale(
  degree: number,
  nodeType: GraphNode['type'],
  organizationOverview: boolean,
  selected: boolean,
  hovered: boolean,
): number {
  const structuralScale = 0.82 + Math.min(Math.sqrt(Math.max(0, degree)) * 0.14, 0.48)
  const organizationScale = organizationOverview
    ? nodeType === 'Company'
      ? 2.05
      : nodeType === 'Department'
        ? 1.68
        : 1
    : 1
  const scale = structuralScale * organizationScale
  if (selected) return scale * 1.16
  if (hovered) return scale * 1.1
  return scale
}

/** Update a cached 3D entity silhouette for the latest focus and degree state. */
function applyNodeVisualState(
  group: THREE.Group,
  shape: GraphNodeShape,
  state: NodeVisualState,
): void {
  const surface = group.children.find((child) => child.userData.role === 'surface') as
    | THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>
    | undefined
  const outline = group.children.find((child) => child.userData.role === 'outline') as
    | THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>
    | undefined
  const halo = group.children.find((child) => child.userData.role === 'halo') as
    | THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>
    | undefined
  const displayColor = state.faded ? FADED_NODE_COLOR : state.color
  const baseColor = new THREE.Color(displayColor)

  if (surface) {
    surface.material.color.set(baseColor)
    surface.material.emissive.set(state.faded ? FADED_NODE_COLOR : state.color)
    surface.material.emissiveIntensity = state.selected
      ? 0.58
      : state.hovered
        ? 0.4
        : state.faded
          ? 0.04
          : state.organizationOverview
            ? 0.28
            : 0.16
    surface.material.metalness = state.faded ? 0.02 : state.organizationOverview ? 0.24 : 0.16
    surface.material.opacity = state.faded ? 0.38 : 0.96
    surface.material.roughness = state.faded ? 0.82 : state.organizationOverview ? 0.28 : 0.38
  }

  if (outline) {
    const outlineColor = baseColor
      .clone()
      .lerp(new THREE.Color('#dbeafe'), state.faded ? 0.08 : 0.34)
    outline.material.color.set(state.selected || state.hovered ? '#ffffff' : outlineColor)
    outline.material.opacity = state.selected
      ? 0.64
      : state.hovered
        ? 0.48
        : state.faded
          ? 0.08
          : 0.2
  }

  if (halo) {
    const organizationGlow = state.organizationOverview
      && (state.nodeType === 'Company' || state.nodeType === 'Department')
    halo.material.color.set(state.color)
    halo.material.opacity = state.selected
      ? 0.25
      : state.hovered
        ? 0.17
        : organizationGlow
          ? state.nodeType === 'Company' ? 0.08 : 0.05
          : 0
    halo.visible = state.selected || state.hovered || organizationGlow
  }

  const scale = nodeVisualScale(
    state.degree,
    state.nodeType,
    state.organizationOverview,
    state.selected,
    state.hovered,
  )
  group.scale.setScalar(scale)
  group.userData = {
    degree: state.degree,
    faded: state.faded,
    shape,
    visualScale: scale,
  }
}

/** Build a cached lit 3D entity silhouette with outline and interaction halo. */
function createNodeObject(node: RenderNode, state: NodeVisualState): THREE.Group {
  const shape = shapeOf(node.type)
  const group = new THREE.Group()
  const surface = new THREE.Mesh(
    NODE_GEOMETRIES[shape],
    new THREE.MeshStandardMaterial({
      flatShading: shape !== 'sphere' && shape !== 'torus',
      transparent: true,
    }),
  )
  surface.userData.role = 'surface'
  surface.castShadow = false
  surface.receiveShadow = false

  const outline = new THREE.Mesh(
    NODE_GEOMETRIES[shape],
    new THREE.MeshBasicMaterial({ transparent: true, wireframe: true }),
  )
  outline.userData.role = 'outline'
  outline.scale.setScalar(1.045)

  const halo = new THREE.Mesh(
    NODE_HALO_GEOMETRY,
    new THREE.MeshBasicMaterial({
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0,
      side: THREE.BackSide,
      transparent: true,
      wireframe: true,
    }),
  )
  halo.userData.role = 'halo'
  halo.visible = false

  group.add(surface, outline, halo)

  if (shape === 'torus') group.rotation.x = Math.PI / 5
  if (shape === 'document') group.rotation.z = -0.12
  if (shape === 'octahedron' || shape === 'tetrahedron') group.rotation.z = Math.PI / 10

  applyNodeVisualState(group, shape, state)
  return group
}

/** Dispose per-node materials while retaining shared geometry definitions. */
function disposeNodeObject(object: THREE.Object3D): void {
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return
    const materials = Array.isArray(child.material) ? child.material : [child.material]
    materials.forEach((material) => material.dispose())
  })
}

/** Render the interactive Three.js knowledge-graph surface. */
export function KnowledgeGraph3D({
  colorMap,
  edges,
  nodes,
  onDepartmentOpen,
  onNodeSelect,
  selected,
}: KnowledgeGraph3DProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const graphRef = useRef<GraphRef | null>(null)
  const [size, setSize] = useState({ width: 0, height: 500 })
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null)

  const companyOverview = useMemo(() => isCompanyOverview(nodes), [nodes])
  const graphData = useMemo(() => {
    const baseData = toForceGraphData(nodes, edges)
    return {
      nodes: companyOverview ? positionCompanyHierarchy(baseData.nodes) : baseData.nodes,
      links: companyOverview
        ? expandCompanyOverviewLinks(baseData.nodes, baseData.links)
        : baseData.links.map((edge, index) => ({
          ...edge,
          isIndirect: false,
          renderId: `${edge.source}-${edge.label}-${edge.target}-${index}-0`,
          renderIndex: 0,
          renderTotal: 1,
        })),
    }
  }, [companyOverview, edges, nodes])
  const nodeById = useMemo(
    () => new Map(graphData.nodes.map((node) => [node.id, node] as const)),
    [graphData.nodes],
  )
  const degreeById = useMemo(
    () => buildVisibleDegreeMap(graphData.nodes, graphData.links),
    [graphData.links, graphData.nodes],
  )
  const focusRootIds = useMemo(() => {
    const ids = new Set<string>()
    if (selected?.id) ids.add(selected.id)
    if (hoveredNodeId) ids.add(hoveredNodeId)
    return ids
  }, [hoveredNodeId, selected?.id])
  const focusedIds = useMemo(() => {
    if (focusRootIds.size === 0) return focusRootIds

    const ids = new Set(focusRootIds)
    graphData.links.forEach((link) => {
      const sourceId = endpointId(link.source)
      const targetId = endpointId(link.target)
      if (sourceId && focusRootIds.has(sourceId) && targetId) ids.add(targetId)
      if (targetId && focusRootIds.has(targetId) && sourceId) ids.add(sourceId)
    })
    return ids
  }, [focusRootIds, graphData.links])

  useEffect(() => {
    const element = containerRef.current
    if (!element) return

    /** Synchronize the graph canvas size with its container. */
    const updateSize = () => {
      setSize({
        width: element.clientWidth,
        height: element.clientHeight,
      })
    }

    updateSize()
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(updateSize)
      observer.observe(element)
      return () => observer.disconnect()
    }

    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  const setGraphRef = useCallback((instance: GraphRef | null) => {
    graphRef.current = instance
    if (!instance) return

    const controls = instance.controls() as OrbitControlsLike
    controls.autoRotate = true
    controls.autoRotateSpeed = GRAPH_AUTO_ROTATE_SPEED
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.target?.set(0, 0, 0)
    controls.update?.()
    instance.lights(createGraphLights())
  }, [])

  const fitView = useCallback(() => {
    setHoveredNodeId(null)
    window.setTimeout(() => {
      graphRef.current?.zoomToFit(600, 50)
    }, 0)
  }, [])

  useEffect(() => {
    if (!graphData.nodes.length || size.width <= 0) return

    const timer = window.setTimeout(() => {
      graphRef.current?.zoomToFit(600, 50)
    }, 80)
    return () => window.clearTimeout(timer)
  }, [graphData, size.width])

  useEffect(
    () => () => {
      graphRef.current?.pauseAnimation()
      graphRef.current = null
    },
    [],
  )

  const linkObjects = useMemo(
    () => new Map(
      companyOverview
        ? graphData.links.map((link) => [link.renderId, createLinkObject(link)] as const)
        : [],
    ),
    [companyOverview, graphData.links],
  )

  const nodeObjects = useMemo(
    () =>
      new Map(
        graphData.nodes.map((node) => [
          node.id,
          createNodeObject(node, {
            color: colorOf(node.type, colorMap),
            degree: degreeById.get(node.id) ?? 0,
            faded: false,
            hovered: false,
            nodeType: node.type,
            organizationOverview: companyOverview,
            selected: false,
          }),
        ] as const),
      ),
    [colorMap, companyOverview, degreeById, graphData.nodes],
  )

  useLayoutEffect(() => {
    graphData.nodes.forEach((node) => {
      const object = nodeObjects.get(node.id)
      if (!object) return
      applyNodeVisualState(object, shapeOf(node.type), {
        color: colorOf(node.type, colorMap),
        degree: degreeById.get(node.id) ?? 0,
        faded: focusRootIds.size > 0 && !focusedIds.has(node.id),
        hovered: node.id === hoveredNodeId,
        nodeType: node.type,
        organizationOverview: companyOverview,
        selected: node.id === selected?.id,
      })
    })
    graphRef.current?.refresh()
  }, [
    colorMap,
    companyOverview,
    degreeById,
    focusRootIds.size,
    focusedIds,
    graphData.nodes,
    hoveredNodeId,
    nodeObjects,
    selected?.id,
  ])

  useEffect(
    () => () => {
      nodeObjects.forEach(disposeNodeObject)
    },
    [nodeObjects],
  )

  useEffect(
    () => () => {
      linkObjects.forEach(disposeLinkObject)
    },
    [linkObjects],
  )

  const getLinkThreeObject = useCallback(
    (link: RenderLink) => linkObjects.get(link.renderId) ?? new THREE.Group(),
    [linkObjects],
  )

  const getNodeThreeObject = useCallback(
    (node: RenderNode) => {
      const id = node.id === undefined || node.id === null ? undefined : String(node.id)
      return (id && nodeObjects.get(id)) || new THREE.Group()
    },
    [nodeObjects],
  )

  const getNodeLabel = useCallback(
    (node: RenderNode) => {
      const id = node.id === undefined || node.id === null ? undefined : String(node.id)
      const label = escapeHtml(node.label)
      const type = escapeHtml(labelOf(entityTypeLabels, node.type))
      const shape = NODE_SHAPE_DETAILS[shapeOf(node.type)].label
      const degree = id ? (degreeById.get(id) ?? 0) : 0
      const description = node.description
        ? `<br/><span style="color:#64748b">${escapeHtml(node.description.slice(0, 140))}</span>`
        : ''
      return `<div style="max-width:280px"><strong>${label}</strong><br/>类型：${type} · 形状：${shape}<br/>可见关系：${degree}${description}</div>`
    },
    [degreeById],
  )

  const isFocusedLink = useCallback(
    (link: RenderLink) => {
      if (focusRootIds.size === 0) return false
      const sourceId = endpointId(link.source)
      const targetId = endpointId(link.target)
      return Boolean(
        (sourceId !== undefined && focusRootIds.has(sourceId)) ||
          (targetId !== undefined && focusRootIds.has(targetId)),
      )
    },
    [focusRootIds],
  )

  const getLinkColor = useCallback(
    (link: RenderLink) => {
      if (focusRootIds.size === 0 || isFocusedLink(link)) return relationColor(link.label)
      return FADED_LINK_COLOR
    },
    [focusRootIds.size, isFocusedLink],
  )

  const getLinkWidth = useCallback(
    (link: RenderLink) => {
      if (focusRootIds.size === 0) return 1.5
      return isFocusedLink(link) ? 2.8 : 0.65
    },
    [focusRootIds.size, isFocusedLink],
  )

  const getLinkParticles = useCallback(
    (link: RenderLink) => (link.isIndirect ? 0 : isFocusedLink(link) ? 2 : 0),
    [isFocusedLink],
  )

  useLayoutEffect(() => {
    linkObjects.forEach((line, renderId) => {
      const link = graphData.links.find((item) => item.renderId === renderId)
      if (!link) return
      applyLinkVisualState(
        line,
        getLinkColor(link),
        focusRootIds.size > 0 && !isFocusedLink(link),
        link.isIndirect,
      )
    })
  }, [focusRootIds.size, getLinkColor, graphData.links, isFocusedLink, linkObjects])

  const handleNodeHover = useCallback((node: RenderNode | null) => {
    setHoveredNodeId(node?.id === undefined || node?.id === null ? null : String(node.id))
  }, [])

  const handleNodeClick = useCallback(
    (node: RenderNode, event: MouseEvent) => {
      const id = node.id === undefined || node.id === null ? undefined : String(node.id)
      if (!id) return
      const selectedNode = nodeById.get(id)
      if (!selectedNode) return
      onNodeSelect(selectedNode)
      if (event.detail === 2 && selectedNode.type === 'Department') onDepartmentOpen?.(selectedNode)
    },
    [nodeById, onDepartmentOpen, onNodeSelect],
  )

  const hasRenderableGraph = graphData.nodes.length > 0 && size.width > 0 && size.height > 0

  return (
    <div ref={containerRef} style={{ position: 'absolute', inset: 0 }}>
      {hasRenderableGraph && (
        <>
          <ForceGraph3D<GraphNode, RenderGraphEdge>
            // react-force-graph types only declare object refs; callback refs are required here for React 19.
            ref={setGraphRef as unknown as MutableRefObject<GraphRef | undefined>}
            controlType="orbit"
            graphData={graphData}
            width={size.width}
            height={size.height}
            backgroundColor={GRAPH_BACKGROUND}
            showNavInfo={false}
            nodeId="id"
            nodeThreeObject={getNodeThreeObject}
            nodeThreeObjectExtend={false}
            nodeLabel={getNodeLabel}
            linkSource="source"
            linkTarget="target"
            linkColor={getLinkColor}
            linkWidth={getLinkWidth}
            linkOpacity={0.68}
            linkLabel={(link) => escapeHtml(`${formatGraphRelation(link.label)}${link.path_count && link.path_count > 1 ? ` · ${link.path_count} 条路径` : ''}`)}
            linkThreeObject={companyOverview ? getLinkThreeObject : undefined}
            linkThreeObjectExtend={false}
            linkPositionUpdate={companyOverview ? updateLinkPosition : undefined}
            linkCurvature={companyOverview ? 0 : relationCurvature}
            linkDirectionalArrowLength={(link) =>
              link.isIndirect
                ? 2.8
                : focusRootIds.size === 0 || isFocusedLink(link) ? 4.8 : 3.2
            }
            linkDirectionalArrowRelPos={1}
            linkDirectionalArrowColor={getLinkColor}
            linkDirectionalParticles={getLinkParticles}
            linkDirectionalParticleSpeed={0.006}
            linkDirectionalParticleWidth={1.8}
            linkDirectionalParticleColor={getLinkColor}
            linkDirectionalParticleResolution={8}
            warmupTicks={30}
            cooldownTicks={120}
            cooldownTime={2500}
            d3AlphaDecay={0.04}
            d3VelocityDecay={0.32}
            enableNavigationControls
            enableNodeDrag
            showPointerCursor
            onNodeHover={handleNodeHover}
            onNodeClick={handleNodeClick}
            onBackgroundClick={() => setHoveredNodeId(null)}
          />
          <div style={{ position: 'absolute', top: 14, right: 14, zIndex: 3 }}>
            <Tooltip title="重置视角">
              <Button
                aria-label="重置视角"
                icon={<AimOutlined />}
                onClick={fitView}
                shape="circle"
                style={{
                  backdropFilter: 'blur(12px)',
                  background: 'rgba(255, 255, 255, 0.86)',
                  borderColor: 'rgba(148, 163, 184, 0.35)',
                  boxShadow: '0 8px 24px rgba(15, 23, 42, 0.12)',
                }}
              />
            </Tooltip>
          </div>
        </>
      )}
    </div>
  )
}
