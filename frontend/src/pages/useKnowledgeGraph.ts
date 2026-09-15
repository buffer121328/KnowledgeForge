import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { message } from 'antd'
import {
  graphApi,
  type GraphDepartment,
  type GraphEdge,
  type GraphNode,
} from '@/api/graph'
import { buildColorMap } from '@/utils/graphVisualization'
import { departmentDisplayName } from '@/utils/departmentLabels'

const COMPANY_LEGEND_TYPES = ['Company', 'Department']
const DEPARTMENT_HIDDEN_TYPES = new Set(['Company', 'Department', 'Document', 'RelationClaim'])
const DEPARTMENT_CONTEXT_TYPES = new Set<string>()

/** Keep the company overview limited to the nodes that form the organizational network. */
function projectNetwork(
  nodes: GraphNode[],
  edges: GraphEdge[],
  allowedTypes: Set<string>,
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const scopedNodes = nodes.filter((node) => allowedTypes.has(node.type))
  const scopedIds = new Set(scopedNodes.map((node) => node.id))

  return {
    nodes: scopedNodes,
    edges: edges.filter(
      (edge) => scopedIds.has(edge.source) && scopedIds.has(edge.target),
    ),
  }
}

/** Keep the company root and ordinary organizational links in the overview. */
function projectCompanyNetwork(data: {
  company_id?: string
  company_name?: string
  departments?: GraphDepartment[]
  edges?: GraphEdge[]
  nodes?: GraphNode[]
}): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const departments = data.departments ?? []
  const departmentNodes = new Map<string, GraphNode>()
  ;(data.nodes ?? [])
    .filter((node) => node.type === 'Department')
    .forEach((node) => departmentNodes.set(node.id, node))
  departments.forEach((department) => {
    const id = `department:${department.department_id}`
    if (!departmentNodes.has(id)) {
      departmentNodes.set(id, {
        id,
        label: departmentDisplayName(department.department_id, department.name),
        type: 'Department',
        department_id: department.department_id,
        document_count: department.document_count,
      })
    }
  })

  const sourceCompany = (data.nodes ?? []).find((node) => node.type === 'Company')
  if (!sourceCompany && departmentNodes.size === 0) return { nodes: [], edges: [] }

  const companyNode = sourceCompany ?? {
    id: `company:${data.company_id || 'current'}`,
    label: data.company_name || data.company_id || '公司',
    type: 'Company',
    company_id: data.company_id,
  }
  const organizationalNodes = [companyNode, ...Array.from(departmentNodes.values())]
  const organizationalIds = new Set(organizationalNodes.map((node) => node.id))
  return {
    nodes: organizationalNodes,
    edges: (data.edges ?? []).filter((edge) => (
      organizationalIds.has(edge.source)
      && organizationalIds.has(edge.target)
    )),
  }
}

/** Return a permission-safe page status from a company-overview failure. */
function graphFailureStatus(error: unknown): string {
  const status = (error as { response?: { status?: number } })?.response?.status
  return status === 403 || status === 404 ? 'permission-denied' : 'error'
}

/** Manage the company and department network while cancelling stale overview requests. */
export function useKnowledgeGraph() {
  const [loading, setLoading] = useState(false)
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [status, setStatus] = useState<string>('ok')
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [companyName, setCompanyName] = useState('')
  const [departments, setDepartments] = useState<GraphDepartment[]>([])
  const [scope, setScope] = useState<'company' | 'department'>('company')
  const [departmentName, setDepartmentName] = useState('')
  const [entityQuery, setEntityQuery] = useState('')
  const [entityType, setEntityType] = useState('all')
  const [relationType, setRelationType] = useState('all')
  const mainRequest = useRef<AbortController | null>(null)

  /** Load the authorized company overview without requesting fine-grained graph projections. */
  const loadCompanyOverview = useCallback(async (): Promise<void> => {
    mainRequest.current?.abort()
    const controller = new AbortController()
    mainRequest.current = controller
    setLoading(true)
    setSelected(null)

    try {
      const data = await graphApi.companyOverview(controller.signal)
      if (controller.signal.aborted) return
      const organizationalNetwork = projectCompanyNetwork(data)
      setNodes(organizationalNetwork.nodes)
      setEdges(organizationalNetwork.edges)
      setDepartments(data.departments ?? [])
      setCompanyName(data.company_name ?? data.company_id ?? '')
      setDepartmentName('')
      setEntityQuery('')
      setEntityType('all')
      setRelationType('all')
      setScope('company')
      setStatus(data.status || 'ok')
    } catch (error: unknown) {
      if (controller.signal.aborted) return
      setNodes([])
      setEdges([])
      setStatus(graphFailureStatus(error))
      message.error(
        graphFailureStatus(error) === 'permission-denied'
          ? '无权查看该图谱范围'
          : '加载公司图谱失败',
      )
    } finally {
      if (mainRequest.current === controller) setLoading(false)
    }
  }, [])

  /** Load one authorized department as a lightweight internal entity network. */
  const enterDepartment = useCallback(async (departmentId: string): Promise<void> => {
    mainRequest.current?.abort()
    const controller = new AbortController()
    mainRequest.current = controller
    setLoading(true)
    setSelected(null)
    try {
      const data = await graphApi.department(departmentId, false, controller.signal)
      if (controller.signal.aborted) return
      const departmentNetwork = projectNetwork(
        data.nodes ?? [],
        data.edges ?? [],
        new Set((data.nodes ?? []).filter((node) => !DEPARTMENT_HIDDEN_TYPES.has(node.type)).map((node) => node.type)),
      )
      setNodes(departmentNetwork.nodes)
      setEdges(departmentNetwork.edges)
      setDepartmentName(
        (data.nodes ?? []).find((node) => node.type === 'Department')?.label ?? departmentId,
      )
      setEntityQuery('')
      setEntityType('all')
      setRelationType('all')
      setScope('department')
      setStatus(data.status || 'ok')
    } catch (error: unknown) {
      if (controller.signal.aborted) return
      setNodes([])
      setEdges([])
      setStatus(graphFailureStatus(error))
      message.error(graphFailureStatus(error) === 'permission-denied' ? '无权查看该部门' : '加载部门图谱失败')
    } finally {
      if (mainRequest.current === controller) setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadCompanyOverview()
    return () => mainRequest.current?.abort()
  }, [loadCompanyOverview])

  /** Return the graph to the current authorized company overview. */
  const handleReset = (): void => { void loadCompanyOverview() }
  const entityTypes = useMemo(
    () => scope === 'company' ? COMPANY_LEGEND_TYPES : Array.from(new Set(nodes.map((node) => node.type))),
    [nodes, scope],
  )
  const filterEntityTypes = useMemo(
    () => scope === 'department'
      ? Array.from(new Set(
        nodes
          .filter((node) => !DEPARTMENT_CONTEXT_TYPES.has(node.type))
          .map((node) => node.type),
      )).sort()
      : [],
    [nodes, scope],
  )
  const relationTypes = useMemo(
    () => scope === 'department'
      ? Array.from(new Set(edges.map((edge) => edge.label))).sort()
      : [],
    [edges, scope],
  )
  const visibleNetwork = useMemo(() => {
    if (scope !== 'department') return { nodes, edges }

    const normalizedQuery = entityQuery.trim().toLocaleLowerCase()
    const candidateNodes = nodes.filter((node) => {
      if (DEPARTMENT_CONTEXT_TYPES.has(node.type)) return true
      const matchesType = entityType === 'all' || node.type === entityType
      const searchableText = `${node.label} ${node.description ?? ''}`.toLocaleLowerCase()
      return matchesType && (!normalizedQuery || searchableText.includes(normalizedQuery))
    })
    const candidateIds = new Set(candidateNodes.map((node) => node.id))
    const filteredEdges = edges.filter((edge) => (
      candidateIds.has(edge.source)
      && candidateIds.has(edge.target)
      && (relationType === 'all' || edge.label === relationType)
    ))

    if (relationType === 'all') return { nodes: candidateNodes, edges: filteredEdges }

    const connectedIds = new Set<string>()
    candidateNodes.forEach((node) => {
      if (DEPARTMENT_CONTEXT_TYPES.has(node.type)) connectedIds.add(node.id)
    })
    filteredEdges.forEach((edge) => {
      connectedIds.add(edge.source)
      connectedIds.add(edge.target)
    })
    return {
      nodes: candidateNodes.filter((node) => connectedIds.has(node.id)),
      edges: filteredEdges,
    }
  }, [edges, entityQuery, entityType, nodes, relationType, scope])
  const visibleNodeIds = useMemo(
    () => new Set(visibleNetwork.nodes.map((node) => node.id)),
    [visibleNetwork.nodes],
  )
  useEffect(() => {
    setSelected((current) => current && !visibleNodeIds.has(current.id) ? null : current)
  }, [visibleNodeIds])

  const clearFilters = useCallback((): void => {
    setEntityQuery('')
    setEntityType('all')
    setRelationType('all')
  }, [])

  /** Persist a manual correction to one extracted entity and keep the local graph in sync. */
  const updateEntity = useCallback(async (
    node: GraphNode,
    values: { label?: string; type?: string; description?: string },
  ): Promise<void> => {
    const response = await graphApi.updateEntity({ node_id: node.id, ...values })
    setNodes((current) => current.map((item) => (item.id === node.id ? response.node : item)))
    setSelected((current) => (current?.id === node.id ? response.node : current))
    message.success('实体已更新')
  }, [])

  /** Persist a manual correction to one extracted relation claim and keep the local graph in sync. */
  const updateRelation = useCallback(async (edge: GraphEdge, relationType: string): Promise<void> => {
    const claimId = edge.claim_id || edge.claim_ids?.[0]
    if (!claimId) {
      message.warning('这条聚合关系没有可编辑的关系声明')
      return
    }
    const response = await graphApi.updateRelation({ claim_id: claimId, relation_type: relationType })
    setEdges((current) => current.map((item) => {
      const itemClaimIds = new Set([item.claim_id, ...(item.claim_ids ?? [])].filter(Boolean))
      if (!itemClaimIds.has(claimId)) return item
      return {
        ...item,
        label: item.claim_id === claimId ? response.relation_type : item.label,
        relation_details: item.relation_details?.map((detail) => (
          detail.claim_id === claimId ? { ...detail, relation_type: response.relation_type } : detail
        )),
        relation_types: Array.from(new Set([
          ...(item.relation_types ?? []).filter((type) => type !== edge.label),
          response.relation_type,
        ])),
      }
    }))
    message.success('关系已更新')
  }, [])
  const hasActiveFilters = entityQuery.trim() !== '' || entityType !== 'all' || relationType !== 'all'
  const colorMap = useMemo(() => buildColorMap(entityTypes), [entityTypes])

  return {
    clearFilters,
    colorMap,
    companyName,
    departmentName,
    departments,
    edges,
    entityQuery,
    entityType,
    entityTypes,
    enterDepartment,
    filterEntityTypes,
    handleReset,
    hasActiveFilters,
    loading,
    nodes,
    relationType,
    relationTypes,
    selected,
    setEntityQuery,
    setEntityType,
    setRelationType,
    setSelected,
    updateEntity,
    updateRelation,
    stats: { nodes: visibleNetwork.nodes.length, edges: visibleNetwork.edges.length },
    status,
    scope,
    visibleEdges: visibleNetwork.edges,
    visibleNodes: visibleNetwork.nodes,
  }
}
