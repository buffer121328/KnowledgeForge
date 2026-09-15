import client from './client'

export interface GraphNode {
  id: string
  label: string
  type: string
  description?: string
  company_id?: string
  department_id?: string
  document_count?: number
}

export interface GraphRelationDetail {
  claim_id: string
  relation_type: string
  head_name: string
  head_type?: string
  tail_name: string
  tail_type?: string
}

export interface GraphEdge {
  source: string
  target: string
  label: string
  claim_id?: string
  path_count?: number
  claim_ids?: string[]
  relation_types?: string[]
  relation_details?: GraphRelationDetail[]
}

export interface GraphDepartment {
  department_id: string
  name: string
  document_count: number
}

export interface GraphSubgraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
  status: string
  scope?: string
  department_id?: string
}

export interface CompanyGraphOverview extends GraphSubgraph {
  company_id: string
  company_name: string
  departments: GraphDepartment[]
}

export interface GraphPathResponse {
  paths: Array<{ nodes?: Array<Record<string, unknown>>; relations?: string[] }>
  path_count: number
  status: string
}

export interface GraphEvidenceItem {
  evidence_id: string
  doc_id: string
  chunk_id: string
  department_id: string
  confidence?: number
  extractor?: string
  model?: string
  extractor_version?: string
  display_name?: string
  provenance_source_filename?: string
  relative_path?: string
}

export interface GraphEvidenceResponse {
  claim_id: string
  evidence: GraphEvidenceItem[]
  status: string
}

export const graphApi = {
  /** Fetch a filtered legacy entity subgraph. */
  subgraph: (params?: { q?: string; entity_type?: string; limit?: number }, signal?: AbortSignal) =>
    client.get<GraphSubgraph>('/graph/subgraph', { params, signal }).then((response) => response.data),

  /** Fetch the tenant company and department overview used as the default graph scope. */
  companyOverview: (signal?: AbortSignal) =>
    client.get<CompanyGraphOverview>('/graph/company-overview', { signal }).then((response) => response.data),

  /** Fetch one authorized department projection with optional cross-department evidence. */
  department: (departmentId: string, includeCrossDepartment = false, signal?: AbortSignal) =>
    client.get<GraphSubgraph>(`/graph/departments/${departmentId}`, {
      params: { include_cross_department: includeCrossDepartment }, signal,
    }).then((response) => response.data),

  /** Fetch finite concrete paths between two visible graph nodes. */
  paths: (params: {
    from_id: string
    to_id: string
    department_id?: string
    include_cross_department?: boolean
    max_hops?: number
    limit?: number
  }, signal?: AbortSignal) =>
    client.get<GraphPathResponse>('/graph/paths', { params, signal }).then((response) => response.data),

  /** Fetch source document/chunk evidence for one tenant-visible relation claim. */
  evidence: (claimId: string, signal?: AbortSignal) =>
    client.get<GraphEvidenceResponse>(`/graph/claims/${claimId}/evidence`, { signal }).then((response) => response.data),

  /** Apply a user correction to one extracted entity. */
  updateEntity: (payload: { node_id: string; label?: string; type?: string; description?: string }) =>
    client.patch<{ node: GraphNode; status: string }>('/graph/entities', payload).then((response) => response.data),

  /** Apply a user correction to one extracted relation claim. */
  updateRelation: (payload: { claim_id: string; relation_type: string }) =>
    client.patch<{ claim_id: string; relation_type: string; status: string }>('/graph/relations', payload).then((response) => response.data),

  /** Fetch the available graph entity types. */
  types: (signal?: AbortSignal) => client.get<string[]>('/graph/types', { signal }).then((response) => response.data),
}
