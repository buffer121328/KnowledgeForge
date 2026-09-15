/** 认证相关类型 */
export interface User {
  user_id: string
  username: string
  display_name?: string
  role: 'organization_admin' | 'admin' | 'editor' | 'viewer' | 'api_user'
  org_id: string
  permissions: string[]
  department_id?: string | null
  is_department_manager?: boolean
}

export interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface LoginRequest {
  username: string
  password: string
}

/** 文档相关类型 */
export interface IngestResponse {
  file_name: string
  chunks_count: number
  entities_count: number
  relations_count: number
  status: string
  task_id?: string
  doc_id?: string
  client_file_id?: string
  relative_path?: string
  department_id?: string
  display_name?: string
  provenance_source_filename?: string
  version?: number
  error_code?: string
  message?: string
}

export interface DepartmentItem {
  department_id: string
  company_id: string
  name: string
  normalized_key: string
  status: string
}

export interface FolderManifestFile {
  client_file_id: string
  relative_path: string
  department_id: string
  external_source_id?: string
  display_name?: string
  provenance_source_filename?: string
  metadata?: Record<string, unknown>
}

export interface FolderUploadManifest {
  root_folder_name: string
  company_id: string
  department_mappings: Record<string, string>
  files: FolderManifestFile[]
  upload_id?: string
}

export interface DocListItem {
  doc_id: string
  file_name: string
  source: string
  file_reference: string
  doc_type: string
  chunks_count: number
  tenant_id?: string
  company_id?: string
  department_id?: string
  folder_path?: string
  relative_path?: string
  uploaded_filename?: string
  display_name?: string
  provenance_source_filename?: string
  content_sha256?: string
  version?: number
  ingest_status?: string
  authority?: string
  review_status?: string
  sensitivity?: string
  entities_count?: number
  relations_count?: number
  created_at?: string
  updated_at?: string
  error_code?: string
  retryable?: boolean
  legacy?: boolean
  legacy_name_unresolved?: boolean
  ingest_stage?: string
  ingest_stage_index?: number
  ingest_stage_total?: number
  ingest_stage_label?: string
  processing_step?: string
  processing_step_index?: number
  processing_step_total?: number
  processing_step_label?: string
}

export type FolderPreviewStatus = 'ready' | 'ignored' | 'unsupported' | 'oversized' | 'duplicate' | 'invalid-path'

export interface FolderPreviewFile {
  clientFileId: string
  file: File
  originalPath: string
  relativePath: string
  folderKey: string
  status: FolderPreviewStatus
  reason: string
}

export interface FolderUploadPreview {
  rootFolderName: string
  files: FolderPreviewFile[]
  departmentCounts: Record<string, number>
  readyCount: number
  issueCount: number
}

export interface IngestProgressItem {
  doc_id: string
  client_file_id?: string
  file_name: string
  status: string
  ingest_stage: string
  ingest_stage_index: number
  ingest_stage_total: number
  ingest_stage_label: string
  processing_step: string
  processing_step_index: number
  processing_step_total: number
  processing_step_label: string
  error_code?: string
  message?: string
}

export interface IngestProgressResponse {
  upload_id: string
  total_count: number
  completed_count: number
  failed_count: number
  stage_index: number
  stage_total: number
  stage_label: string
  processing_step: string
  processing_step_index: number
  processing_step_total: number
  processing_step_label: string
  terminal: boolean
  status: 'processing' | 'success' | 'failed'
  items: IngestProgressItem[]
}

/** 文档上传本地历史记录 */
export interface DocumentUploadRecord {
  key: string
  fileName: string
  docId: string
  chunksCount: number
  entitiesCount: number
  relationsCount: number
  status: 'success' | 'processing' | 'failed'
  uploadedAt: string
  relativePath?: string
  departmentId?: string
  taskId?: string
  errorMessage?: string
  retryable?: boolean
}

export interface ChunkItem {
  chunk_id: string
  content: string
  chunk_index: number
  doc_id: string
  doc_type: string
  source: string
  tenant_id?: string
}

/** 问答相关类型 */
export interface QuestionRequest {
  question: string
  conversation_id?: string
  retrieval_mode?: 'vector' | 'hybrid'
  semantic_bypass_token?: string
}

export interface RetrievedSource {
  content: string
  source: string
  score: number
  type: 'vector' | 'graph' | 'hybrid'
  document_id?: string
  chunk_id?: string
  chunk_index?: number | null
}

export type QADegradationCode = 'graph_retrieval_unavailable' | 'vector_retrieval_unavailable' | 'insufficient_verified_evidence'

export type QAResponseStatus = 'answered' | 'partially_answered' | 'insufficient_evidence' | 'needs_clarification' | 'conflicting_evidence' | 'human_review_required' | 'source_unavailable'

export interface QAAnswerClaim {
  claim_id: string
  text: string
  citation_ids: string[]
  material: boolean
}

export interface QAAnswerCitation {
  citation_id: string
  source: string
  content: string
  document_id?: string
  chunk_id?: string
  chunk_index?: number | null
  highlight?: string
}

export interface QAMissingInformation {
  field: string
  description: string
}

export interface QAGroundingResult {
  passed: boolean
  accepted_claim_ids: string[]
  rejected_claim_ids: string[]
  reason_codes: string[]
  policy_version: string
}

export interface QuestionResponse {
  status: 'answered'
  question: string
  answer: string
  confidence: number
  intent: 'factoid' | 'analytical' | 'comparative' | 'procedural' | 'exploratory'
  sources: RetrievedSource[]
  reasoning_steps: string[]
  degradation_code?: QADegradationCode
  conversation_id?: string
  qa_run_id?: string
  cache_hit_type: 'none' | 'exact' | 'semantic_confirmed'
  history_saved: boolean
  warning_code?: string
  response_status?: QAResponseStatus | null
  evidence_state?: string | null
  evidence_reason_codes?: string[]
  claims?: QAAnswerClaim[]
  citations?: QAAnswerCitation[]
  missing_information?: QAMissingInformation[]
  grounding_result?: QAGroundingResult | null
  grounding_passed?: boolean | null
  policy_version?: string | null
}

export interface SemanticConfirmationResponse {
  status: 'semantic_confirmation_required'
  similar_question: string
  similarity: number
  cached_at: string
  confirmation_token: string
}

export type QAAskResponse = QuestionResponse | SemanticConfirmationResponse

export interface QAConversation {
  id: string
  title: string
  status: string
  created_at: string
  updated_at: string
}

export interface QAConversationPage {
  items: QAConversation[]
  next_cursor?: string
}

export interface QAConversationDetail {
  conversation: QAConversation
  messages: Array<{
    id: string
    sequence: number
    role: 'user' | 'assistant'
    content: string
    created_at: string
  }>
  runs: Array<Record<string, unknown> & { id: string }>
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  sources?: RetrievedSource[]
  confidence?: number
  intent?: string
  reasoning_steps?: string[]
  degradation_code?: QADegradationCode
  qa_run_id?: string
  response_status?: QAResponseStatus
  evidence_state?: string
  evidence_reason_codes?: string[]
  claims?: QAAnswerClaim[]
  citations?: QAAnswerCitation[]
  missing_information?: QAMissingInformation[]
  grounding_result?: QAGroundingResult
  policy_version?: string
}

/** 用户管理类型 */
export interface UserListItem {
  user_id: string
  username: string
  display_name: string
  email: string
  role: string
  org_id: string
  department_id?: string | null
  is_department_manager?: boolean
  is_active: boolean
  created_at?: string
}

export interface UserCreateRequest {
  username: string
  password: string
  display_name?: string
  email?: string
  role: string
  org_id?: string
  department_id?: string | null
  is_department_manager?: boolean
}

/** 角色管理类型 */
export interface Role {
  role_id: string
  name: string
  display_name: string
  description: string
  permissions: string[]
  is_builtin: boolean
  user_count: number
}

/** API request trend point in an explicit UTC bucket. */
export interface RequestTrendPoint {
  timestamp: string
  requests: number
  qa: number
  errors: number
  average_latency_ms: number
  details?: Record<string, RequestAggregate>
}

export interface RequestTrendResponse {
  window: '60m' | '24h'
  timezone: 'UTC'
  generated_at: string
  points: RequestTrendPoint[]
  summary: {
    requests: number
    qa: number
    errors: number
    average_latency_ms: number
    details: Record<string, RequestAggregate>
  }
}

export interface RequestAggregate {
  count?: number
  error_count?: number
  total_latency_ms?: number
  methods?: Record<string, number>
  routes?: Record<string, { count: number; error_count: number; last_status: number }>
}

/** 系统统计类型 */
export type ReadinessComponent = 'ready' | 'unavailable'

export interface ReadinessReport {
  status: 'ready' | 'not_ready'
  components: {
    vector_store: ReadinessComponent
    knowledge_graph: ReadinessComponent
    security_state: ReadinessComponent
    task_broker: ReadinessComponent
  }
}

export interface SystemStats {
  vector_store: {
    total_vectors: number
    collections: number
    [key: string]: unknown
  }
  knowledge_graph: {
    nodes: number
    edges: number
    total_entities?: number
    total_relations?: number
    status?: string
    [key: string]: unknown
  }
  request_trend?: RequestTrendPoint[]
  request_detail?: Record<'ai' | 'system_api', RequestAggregate>
}

/** 异步任务类型 */
export interface TaskStatus {
  task_id: string
  status: string
  ready: boolean
  successful?: boolean
  result?: Record<string, unknown>
  error?: string
  description?: string
}

/** Webhook 类型 */
export interface Webhook {
  id: string
  url: string
  events: string[]
  secret?: string | null
  is_active: boolean
  created_at: string
  failure_count: number
  last_triggered_at: string | null
  last_response_code?: number | null
}

/** 审计日志类型 */
export interface AuditLog {
  audit_id: string
  timestamp: string
  user_id: string
  username: string
  action: string
  resource: string
  result: string
  ip: string
  user_agent: string
  org_id: string
  metadata: Record<string, unknown>
  prev_hash: string
  hash: string
}

/** API 统一响应 */
export interface ApiResponse<T = unknown> {
  data?: T
  detail?: string
}

/** 评测相关类型 */
export interface EvaluationRunSummary {
  run_id: string
  started_at?: string | null
  incomplete?: boolean
  retrieval_modes?: string[]
  run_classification?: string | null
  record_count?: number
  completed?: number
  failed?: number
  invalid_provenance?: number
  skipped_lines?: number
}

export interface EvaluationRunList {
  runs: EvaluationRunSummary[]
}

export interface EvaluationRecordItem {
  benchmark_id?: string | null
  retrieval_mode?: string | null
  category?: string | null
  status?: string | null
  exception?: string | null
  refused?: boolean | null
  latency_ms?: number | null
  question?: string | null
  response?: string | null
  context_count?: number
  contexts?: Array<{
    rank?: number | null
    source?: string | null
    score?: number | null
    retrieval_type?: string | null
    source_document_id?: string | null
  }>
  retrieved_context_ids?: string[]
  reference_context_ids?: string[]
  ragas_scores?: Record<string, number>
  ragas_metric_states?: Record<string, { status: string; reason_code?: string | null }>
}

export interface EvaluationRecords {
  run_id: string
  page?: number
  page_size?: number
  total?: number
  skipped_lines?: number
  records: EvaluationRecordItem[]
}

export interface EvaluationRunDetail {
  run_id: string
  incomplete?: boolean
  metadata?: Record<string, unknown>
  quality_report?: Record<string, unknown> | null
  ragas_summaries?: Record<string, unknown>
}

/** 人工抽检结论的有界枚举。 */
export type SpotCheckVerdict = 'judge_error' | 'system_issue' | 'confirmed_ok' | 'needs_data_fix'

/** 单条异常样本及其人工抽检状态。 */
export interface EvaluationSpotCheckItem {
  benchmark_id: string
  retrieval_mode?: string | null
  category?: string | null
  anomaly_codes: string[]
  status?: string | null
  exception?: string | null
  expected_response_status?: string | null
  observed_response_status?: string | null
  expected_evidence_states?: string[] | null
  observed_evidence_state?: string | null
  expected_evidence_context_ids?: string[]
  retrieved_context_ids?: string[]
  question?: string | null
  response?: string | null
  ragas_scores?: Record<string, number>
  spot_check?: {
    verdict: SpotCheckVerdict
    note: string
    reviewer_id: string
    reviewed_at: string
  } | null
}

/** 报告中的单个指标条目。 */
export interface EvaluationReportMetricEntry {
  key: string
  kind: string
  unit: string
  value?: number | null
  scored?: number | null
  total?: number | null
  failed?: number | null
  direction?: string | null
}

/** 报告注意事项条目。 */
export interface EvaluationReportAdvisory {
  level: string
  message: string
}

/** 单次运行的综合评测报告。 */
export interface EvaluationRunReportResponse {
  run_id: string
  incomplete?: boolean
  run_classification?: string | null
  counts?: Record<string, number | null>
  sections?: Record<string, EvaluationReportMetricEntry[]>
  advisories?: EvaluationReportAdvisory[]
}

/** 单次运行的异常样本清单。 */
export interface EvaluationSpotCheckResponse {
  run_id: string
  total_records: number
  skipped_lines?: number
  anomaly_count: number
  spot_checked: number
  low_score_threshold: number
  anomalies: EvaluationSpotCheckItem[]
}

export type EvidenceReviewStatus = 'draft' | 'pending_review' | 'approved'

export interface EvidenceDatasetSummary {
  dataset_id: string
  title?: string | null
  revision: number
  status: string
  case_count: number
  context_count: number
  counts: Record<string, number>
  category_counts: Record<string, number>
  required_category_count: number
  covered_category_count: number
  fixture_status?: string | null
  last_frozen_version?: string | null
  last_frozen_manifest_sha256?: string | null
  last_frozen_at?: string | null
  last_frozen_by?: string | null
  last_frozen_by_display_name?: string | null
  source_frozen_version?: string | null
  source_type?: string | null
  source_document_count?: number
  active?: boolean
  superseded_by?: string | null
  current_corpus_draft_id?: string | null
  current_corpus_snapshot_sha256?: string | null
  updated_at?: string | null
}

export interface EvidenceDatasetList {
  datasets: EvidenceDatasetSummary[]
}

export interface EvidenceFrozenVersion {
  version: string
  manifest_sha256: string
  frozen_at?: string | null
  frozen_by?: string | null
  frozen_by_display_name?: string | null
  source_workspace_revision?: number | null
  source_document_count?: number
  current_corpus_draft_id?: string | null
  current_corpus_snapshot_sha256?: string | null
}

export interface EvidenceFrozenVersionList {
  versions: EvidenceFrozenVersion[]
}

export interface EvidenceContextSummary {
  context_id: string
  source_document_id?: string | null
  title?: string | null
  department?: string | null
  document_version?: number | null
  chunk_index?: number | null
  content_sha256?: string | null
  content_excerpt: string
}

export interface EvidenceSnapshotContextList {
  dataset_id: string
  version: string
  source_document_count: number
  contexts: EvidenceContextSummary[]
}

export interface EvidenceReviewer {
  user_id: string
  username: string
  display_name: string
}

export interface EvidenceCase {
  id: string
  question: string
  reference_answer: string
  category: string
  expected_response_status: string
  expected_evidence_states: string[]
  expected_reason_codes: string[]
  expected_source_document_ids: string[]
  expected_evidence_context_ids: string[]
  expected_evidence_sections: string[]
  expected_citation_context_ids: string[]
  expected_missing_information_fields: string[]
  required_fixture: string
  review_status: EvidenceReviewStatus
  case_revision: number
  last_editor_id?: string | null
  last_editor_display_name?: string | null
  department_id: string
  last_edited_at?: string | null
  submitted_at?: string | null
  reviewer_id?: string | null
  reviewer_display_name?: string | null
  reviewed_at?: string | null
  review_reason?: string
  rejection_reason?: string
  context_summaries: EvidenceContextSummary[]
  [key: string]: unknown
}

export interface EvidenceCasePage {
  dataset_id: string
  revision: number
  page: number
  page_size: number
  total: number
  cases: EvidenceCase[]
}

export interface EvidenceMutationResponse {
  revision: number
  case: EvidenceCase
}

export interface EvidenceDeleteResponse {
  revision: number
  deleted_case_id: string
}

export type EvidenceBulkSelectionMode = 'selected' | 'filtered'

export interface EvidenceBulkActionRequest {
  expected_revision: number
  selection_mode: EvidenceBulkSelectionMode
  case_ids: string[]
  category?: string
  review_status?: EvidenceReviewStatus
  query?: string
}

export interface EvidenceBulkSkippedItem {
  case_id: string
  code: string
}

/** Document-grounded candidate answers added to an editable review workspace. */
export interface EvidenceReferenceAnswerCandidateResult {
  dataset_id: string
  revision: number
  populated_case_ids: string[]
  skipped: EvidenceBulkSkippedItem[]
}

export interface EvidenceBulkActionResponse {
  revision: number
  matched_count: number
  processed_count: number
  skipped_count: number
  processed_case_ids: string[]
  skipped_items: EvidenceBulkSkippedItem[]
}

export interface EvidenceFixtureItem {
  legacy_manual_verified: boolean
  has_legacy_configuration: boolean
}

export interface EvidenceFixtureProfile {
  dataset_id: string
  revision: number
  status: 'historical_only' | 'pending_operator_configuration' | 'ready'
  fixtures: Record<string, EvidenceFixtureItem>
}

export type EvidenceFixtureValidationStatus = 'queued' | 'running' | 'succeeded' | 'failed'

export interface EvidenceFixtureValidationResult {
  fixture: string
  status: 'passed' | 'failed'
  reason_code: string
  started_at: string
  finished_at: string
  summary: string
}

export interface EvidenceFixtureValidationRun {
  run_id: string
  dataset_id: string
  version: string
  manifest_sha256: string
  source_workspace_revision: number
  status: EvidenceFixtureValidationStatus
  reason_code?: string | null
  fixture_results: EvidenceFixtureValidationResult[]
  created_at: string
  started_at?: string | null
  finished_at?: string | null
}

export interface EvidenceFreezeResult {
  status: string
  version: string
  manifest_sha256: string
  revision: number
}

export interface EvidenceReleaseWorkflow {
  workflow_id: string
  dataset_id: string
  version: string
  manifest_sha256: string
  current_corpus_draft_id?: string | null
  current_corpus_snapshot_sha256?: string | null
  fixture_validation_run_id?: string | null
  evaluation_dataset_id?: string | null
  evaluation_version?: string | null
  evaluation_manifest_sha256?: string | null
  evaluation_case_count?: number | null
  status: string
  stage: string
  revision: number
  initiated_by: string
  target_mode?: 'shadow' | 'enforce' | null
  initiated_by_display_name?: string
  calibration_version?: string | null
  reason_code?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface EvidenceGateConfiguration {
  mode: 'off' | 'shadow' | 'enforce'
  revision: number
  calibration_version?: string | null
  source: string
  updated_at?: string | null
  updated_by?: string | null
}

export interface EvidenceReleaseAttempt {
  attempt_id: string
  stage: string
  attempt_number: number
  status: string
  reason_code?: string | null
  metrics: Record<string, unknown>
  diagnostic_summary?: EvidenceDiagnosticSummary
  started_at?: string | null
  finished_at?: string | null
}

export interface EvidenceDiagnosticRootCause {
  decision: string
  reason_code: string
  paired_count?: number | null
  effect_direction?: string | null
  risk?: string | null
  recommend_embedding_change: boolean
  compared_variants: string[]
  supporting_hashes: string[]
}

export interface EvidenceDiagnosticSummary {
  availability: 'available' | 'not_available' | 'unsupported_schema'
  schema_version?: string | null
  policy_identities: Record<string, string>
  variant_identities: Record<string, string>
  categories: Record<string, Record<string, number | string | null>>
  stages: Record<string, Record<string, number | string | null>>
  metric_coverage: Record<string, Record<string, number | string | null>>
  route_transitions: Record<string, number>
  hard_gates: string[]
  root_cause?: EvidenceDiagnosticRootCause | null
  case_ids: string[]
}
