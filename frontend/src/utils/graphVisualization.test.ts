import { describe, expect, it } from 'vitest'
import type { GraphEdge, GraphNode } from '@/api/graph'
import {
  buildVisibleDegreeMap,
  filterGraphByType,
  shapeOf,
  toForceGraphData,
} from './graphVisualization'

const nodes: GraphNode[] = [
  { id: 'person-1', label: 'Ada', type: 'Person', description: 'Engineer' },
  { id: 'tech-1', label: 'Three.js', type: 'Technology' },
]

const edges: GraphEdge[] = [
  { source: 'person-1', target: 'tech-1', label: 'uses' },
  { source: 'missing', target: 'tech-1', label: 'invalid-source' },
  { source: 'person-1', target: 'missing', label: 'invalid-target' },
]

describe('toForceGraphData', () => {
  it('preserves presentation fields and maps valid relationships', () => {
    const result = toForceGraphData(nodes, edges)

    expect(result.nodes).toEqual(nodes)
    expect(result.links).toEqual([edges[0]])
    expect(result.nodes).not.toBe(nodes)
    expect(result.nodes[0]).not.toBe(nodes[0])
  })

  it('filters edges whose endpoints are not in the node collection', () => {
    const result = toForceGraphData(nodes, edges)

    expect(result.links).toHaveLength(1)
    expect(result.links[0]).toMatchObject({ source: 'person-1', target: 'tech-1' })
  })

  it('returns empty collections for empty input', () => {
    expect(toForceGraphData([], [])).toEqual({ nodes: [], links: [] })
    expect(toForceGraphData([], edges)).toEqual({ nodes: [], links: [] })
  })
})

describe('graph visual language', () => {
  it('maps known entity types to distinct deterministic silhouettes', () => {
    expect(shapeOf('Person')).toBe('sphere')
    expect(shapeOf('Organization')).toBe('cube')
    expect(shapeOf('Technology')).toBe('octahedron')
    expect(new Set(['Person', 'Organization', 'Technology'].map(shapeOf)).size).toBe(3)
  })

  it('keeps unknown entity silhouettes stable', () => {
    expect(shapeOf('ResearchTopic')).toBe(shapeOf('ResearchTopic'))
    expect(shapeOf('ResearchTopic')).not.toBeUndefined()
  })

  it('counts only relationships whose endpoints are visible', () => {
    const degree = buildVisibleDegreeMap(nodes, edges)

    expect(degree.get('person-1')).toBe(1)
    expect(degree.get('tech-1')).toBe(1)
    expect(degree.has('missing')).toBe(false)
  })
})

describe('filterGraphByType', () => {
  it('filters nodes and keeps only edges between visible nodes', () => {
    const result = filterGraphByType(nodes, [edges[0]], 'Person')

    expect(result.visibleNodes).toEqual([nodes[0]])
    expect(result.visibleEdges).toEqual([])
  })
})
