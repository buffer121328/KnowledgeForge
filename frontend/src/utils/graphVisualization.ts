import type { GraphEdge, GraphNode } from '@/api/graph'

export type { GraphEdge, GraphNode }

export interface ForceGraphData {
  nodes: GraphNode[]
  links: GraphEdge[]
}

export type GraphNodeShape =
  | 'sphere'
  | 'cube'
  | 'octahedron'
  | 'prism'
  | 'torus'
  | 'cone'
  | 'icosahedron'
  | 'document'
  | 'tetrahedron'

interface GraphNodeShapeDetails {
  glyph: string
  label: string
}

export const FALLBACK_COLORS: Record<string, string> = {
  Person: '#1677ff',
  Organization: '#52c41a',
  Technology: '#722ed1',
  Product: '#faad14',
  Concept: '#eb2f96',
  Location: '#13c2c2',
  Event: '#fa541c',
  Document: '#2f54eb',
  Company: '#d4380d',
  Department: '#08979c',
  RelationClaim: '#531dab',
}

export const NODE_SHAPE_DETAILS: Record<GraphNodeShape, GraphNodeShapeDetails> = {
  sphere: { glyph: '●', label: '球体' },
  cube: { glyph: '◆', label: '立方体' },
  octahedron: { glyph: '⬢', label: '八面体' },
  prism: { glyph: '⬣', label: '棱柱' },
  torus: { glyph: '◉', label: '圆环' },
  cone: { glyph: '▲', label: '锥体' },
  icosahedron: { glyph: '✦', label: '多面体' },
  document: { glyph: '▰', label: '文档片' },
  tetrahedron: { glyph: '△', label: '四面体' },
}

const EXTRA_PALETTE = ['#13c2c2', '#eb2f96', '#fa8c16', '#a0d911', '#2f54eb', '#f5222d', '#722ed1']

const KNOWN_TYPE_SHAPES: Record<string, GraphNodeShape> = {
  Person: 'sphere',
  Organization: 'cube',
  Technology: 'octahedron',
  Product: 'prism',
  Concept: 'torus',
  Location: 'cone',
  Event: 'icosahedron',
  Document: 'document',
  Company: 'icosahedron',
  Department: 'cube',
  RelationClaim: 'torus',
}

const EXTRA_SHAPES: GraphNodeShape[] = [
  'tetrahedron',
  'icosahedron',
  'prism',
  'torus',
  'octahedron',
  'cone',
  'cube',
  'sphere',
]

/** Build a deterministic signed hash for graph visual fallback choices. */
function stableHash(value: string): number {
  let hash = 0
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) | 0
  }
  return hash
}

/** Build a stable color lookup for graph entity types. */
export function buildColorMap(entityTypes: string[]): Record<string, string> {
  const colorMap = { ...FALLBACK_COLORS }
  entityTypes.forEach((type, index) => {
    if (!colorMap[type]) colorMap[type] = EXTRA_PALETTE[index % EXTRA_PALETTE.length]
  })
  return colorMap
}

/** Return the configured color for an entity type. */
export function colorOf(type: string, known: Record<string, string>): string {
  if (known[type]) return known[type]
  if (FALLBACK_COLORS[type]) return FALLBACK_COLORS[type]
  return EXTRA_PALETTE[Math.abs(stableHash(type)) % EXTRA_PALETTE.length]
}

/** Return a deterministic 3D silhouette for a graph entity type. */
export function shapeOf(type: string): GraphNodeShape {
  return KNOWN_TYPE_SHAPES[type] ?? EXTRA_SHAPES[Math.abs(stableHash(type)) % EXTRA_SHAPES.length]
}

/** Count incident relationships whose endpoints both exist in the visible node collection. */
export function buildVisibleDegreeMap(
  nodes: GraphNode[],
  edges: GraphEdge[],
): Map<string, number> {
  const degree = new Map<string, number>(nodes.map((node) => [node.id, 0]))

  edges.forEach((edge) => {
    if (!degree.has(edge.source) || !degree.has(edge.target)) return
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1)
    if (edge.target !== edge.source) {
      degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1)
    }
  })

  return degree
}

/**
 * Convert API graph data to the `{ nodes, links }` shape expected by the 3D renderer.
 * The copies prevent the renderer's x/y/z simulation fields from mutating page state.
 */
export function toForceGraphData(
  nodes: GraphNode[] = [],
  edges: GraphEdge[] = [],
): ForceGraphData {
  const graphNodes = nodes.map((node) => ({ ...node }))
  const nodeIds = new Set(graphNodes.map((node) => node.id))
  const links = edges
    .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
    .map((edge) => ({ ...edge }))

  return { nodes: graphNodes, links }
}

/** Filter graph nodes and edges while preserving valid relationships. */
export function filterGraphByType(
  nodes: GraphNode[],
  edges: GraphEdge[],
  nodeType: string,
): { visibleNodes: GraphNode[]; visibleEdges: GraphEdge[] } {
  const visibleNodes = nodes.filter((node) => nodeType === 'all' || node.type === nodeType)
  const visibleIds = new Set(visibleNodes.map((node) => node.id))
  const visibleEdges = edges.filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
  )

  return { visibleNodes, visibleEdges }
}
