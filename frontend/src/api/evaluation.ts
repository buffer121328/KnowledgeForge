import client from './client'
import type {
  EvaluationRecords,
  EvaluationRunDetail,
  EvaluationRunList,
  EvaluationRunReportResponse,
  EvaluationSpotCheckResponse,
  SpotCheckVerdict,
  EvidenceCase,
  EvidenceCasePage,
  EvidenceDeleteResponse,
  EvidenceBulkActionRequest,
  EvidenceBulkActionResponse,
  EvidenceDatasetList,
  EvidenceDatasetSummary,
  EvidenceFixtureValidationRun,
  EvidenceFreezeResult,
  EvidenceFrozenVersion,
  EvidenceFrozenVersionList,
  EvidenceMutationResponse,
  EvidenceReviewer,
  EvidenceReleaseAttempt,
  EvidenceReleaseWorkflow,
  EvidenceGateConfiguration,
  EvidenceSnapshotContextList,
  EvidenceReferenceAnswerCandidateResult,
} from '@/types'

export interface EvidenceCaseFilters {
  page?: number
  pageSize?: number
  category?: string
  reviewStatus?: string
  query?: string
}

export const evaluationApi = {
  /** Fetch the list of evaluation runs. */
  listRuns: () => client.get<EvaluationRunList>('/evaluation/runs').then((r) => r.data),

  /** Fetch the detail of one evaluation run. */
  getRun: (runId: string) => client.get<EvaluationRunDetail>(`/evaluation/runs/${encodeURIComponent(runId)}`).then((r) => r.data),

  /** Fetch one page of per-question records for an evaluation run. */
  getRecords: (runId: string, page: number, pageSize: number) =>
    client
      .get<EvaluationRecords>(`/evaluation/runs/${encodeURIComponent(runId)}/records`, {
        params: { page, page_size: pageSize }
      })
      .then((r) => r.data),

  /** Fetch the synthesized evaluation report of one run. */
  getRunReport: (runId: string) =>
    client
      .get<EvaluationRunReportResponse>(`/evaluation/runs/${encodeURIComponent(runId)}/report`)
      .then((r) => r.data),

  /** Fetch the anomaly/spot-check list of one evaluation run. */
  getAnomalies: (runId: string) =>
    client
      .get<EvaluationSpotCheckResponse>(`/evaluation/runs/${encodeURIComponent(runId)}/anomalies`)
      .then((r) => r.data),

  /** Submit one human spot-check verdict for an anomalous sample. */
  submitSpotCheck: (runId: string, benchmarkId: string, verdict: SpotCheckVerdict, note: string) =>
    client
      .post<EvaluationSpotCheckResponse>(`/evaluation/runs/${encodeURIComponent(runId)}/spot-checks`, {
        benchmark_id: benchmarkId,
        verdict,
        note,
      })
      .then((r) => r.data),

  listDatasets: () => client.get<EvidenceDatasetList>('/evaluation/datasets').then((r) => r.data),

  getDataset: (datasetId: string) => client.get<EvidenceDatasetSummary>(`/evaluation/datasets/${encodeURIComponent(datasetId)}`).then((r) => r.data),

  listFrozenVersions: (datasetId: string, limit = 100) => client.get<EvidenceFrozenVersionList>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/versions`, { params: { limit } }).then((r) => r.data.versions),

  getFrozenVersion: (datasetId: string, version: string) => client.get<EvidenceFrozenVersion>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/versions/${encodeURIComponent(version)}`).then((r) => r.data),
  listSnapshotContexts: (datasetId: string, version: string) => client.get<EvidenceSnapshotContextList>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/versions/${encodeURIComponent(version)}/contexts`).then((r) => r.data),
  refreshCurrentCorpus: (datasetId: string) => client.post<EvidenceDatasetSummary>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/refresh-current-corpus`).then((r) => r.data),
  populateReferenceAnswerCandidates: (datasetId: string, expectedRevision: number) =>
    client
      .post<EvidenceReferenceAnswerCandidateResult>(
        `/evaluation/datasets/${encodeURIComponent(datasetId)}/reference-answer-candidates`,
        { expected_revision: expectedRevision },
      )
      .then((r) => r.data),

  createDraftFromVersion: (datasetId: string, version: string, expectedRevision: number) =>
    client
      .post<EvidenceDatasetSummary>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/drafts`, {
        version,
        expected_revision: expectedRevision
      })
      .then((r) => r.data),

  rebindToCurrentCorpus: (datasetId: string, expectedRevision: number) =>
    client
      .post<EvidenceDatasetSummary>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/rebind-current-corpus`, {
        expected_revision: expectedRevision,
      })
      .then((r) => r.data),

  /** Start release from a frozen current-corpus version and the reviewed 100-case suite. */
  startReleaseWorkflow: (
    datasetId: string,
    version: string,
    evaluationDatasetId: string,
    evaluationVersion: string,
  ) =>
    client.post<EvidenceReleaseWorkflow>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/versions/${encodeURIComponent(version)}/release-workflows`, {
      evaluation_dataset_id: evaluationDatasetId,
      evaluation_version: evaluationVersion,
    }).then((r) => r.data),
  listReleaseWorkflows: (limit = 20) =>
    client.get<{ workflows: EvidenceReleaseWorkflow[] }>('/evaluation/release-workflows', { params: { limit } }).then((r) => r.data.workflows),
  getReleaseWorkflow: (workflowId: string) => client.get<EvidenceReleaseWorkflow>(`/evaluation/release-workflows/${encodeURIComponent(workflowId)}`).then((r) => r.data),
  listReleaseAttempts: (workflowId: string) => client.get<{ attempts: EvidenceReleaseAttempt[] }>(`/evaluation/release-workflows/${encodeURIComponent(workflowId)}/attempts`).then((r) => r.data.attempts),
  retryReleaseWorkflow: (workflowId: string, expectedRevision: number) =>
    client.post<EvidenceReleaseWorkflow>(`/evaluation/release-workflows/${encodeURIComponent(workflowId)}/retry`, { expected_revision: expectedRevision }).then((r) => r.data),

  reviewReleaseWorkflow: (workflowId: string, decision: 'approve' | 'reject', expectedRevision: number, reason: string, targetMode?: 'shadow' | 'enforce') =>
    client.post<EvidenceReleaseWorkflow>(`/evaluation/release-workflows/${encodeURIComponent(workflowId)}/${decision}`, {
      expected_revision: expectedRevision, reason, ...(decision === 'approve' ? { target_mode: targetMode } : {}),
    }).then((r) => r.data),
  promoteReleaseWorkflow: (workflowId: string, expectedRevision: number) =>
    client.post<EvidenceReleaseWorkflow>(`/evaluation/release-workflows/${encodeURIComponent(workflowId)}/promote`, { expected_revision: expectedRevision }).then((r) => r.data),
  getEvidenceGate: () => client.get<EvidenceGateConfiguration>('/evaluation/gate').then((r) => r.data),
  rollbackEvidenceGate: (expectedRevision: number, reason: string) =>
    client.post<EvidenceGateConfiguration>('/evaluation/gate/rollback', { expected_revision: expectedRevision, reason }).then((r) => r.data),

  getCases: (datasetId: string, filters: EvidenceCaseFilters = {}) =>
    client
      .get<EvidenceCasePage>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/cases`, {
        params: {
          page: filters.page ?? 1,
          page_size: filters.pageSize ?? 20,
          category: filters.category || undefined,
          review_status: filters.reviewStatus || undefined,
          query: filters.query || undefined
        }
      })
      .then((r) => r.data),

  createCase: (datasetId: string, expectedRevision: number, value: Partial<EvidenceCase>) =>
    client.post<EvidenceMutationResponse>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/cases`, { expected_revision: expectedRevision, case: value }).then((r) => r.data),

  updateCase: (datasetId: string, caseId: string, expectedRevision: number, value: EvidenceCase) =>
    client.put<EvidenceMutationResponse>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/cases/${encodeURIComponent(caseId)}`, { expected_revision: expectedRevision, case: value }).then((r) => r.data),

  deleteCase: (datasetId: string, caseId: string, expectedRevision: number) =>
    client.delete<EvidenceDeleteResponse>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/cases/${encodeURIComponent(caseId)}`, { data: { expected_revision: expectedRevision } }).then((r) => r.data),

  listReviewers: (datasetId: string, caseId: string) => client.get<EvidenceReviewer[]>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/cases/${encodeURIComponent(caseId)}/reviewers`).then((r) => r.data),

  submitCase: (datasetId: string, caseId: string, expectedRevision: number, reviewerId: string) =>
    client.post<EvidenceMutationResponse>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/cases/${encodeURIComponent(caseId)}/submit`, { expected_revision: expectedRevision, reviewer_id: reviewerId }).then((r) => r.data),

  reviewCase: (datasetId: string, caseId: string, expectedRevision: number, decision: 'approve' | 'reject', reason: string) =>
    client.post<EvidenceMutationResponse>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/cases/${encodeURIComponent(caseId)}/review`, { expected_revision: expectedRevision, decision, reason }).then((r) => r.data),

  bulkSubmitCases: (datasetId: string, payload: EvidenceBulkActionRequest) => client.post<EvidenceBulkActionResponse>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/cases/bulk-submit`, payload).then((r) => r.data),

  bulkApproveCases: (datasetId: string, payload: EvidenceBulkActionRequest) => client.post<EvidenceBulkActionResponse>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/cases/bulk-approve`, payload).then((r) => r.data),

  listFixtureValidations: (datasetId: string, version: string) =>
    client.get<{ runs: EvidenceFixtureValidationRun[] }>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/versions/${encodeURIComponent(version)}/fixture-validations`).then((r) => r.data.runs),

  startFixtureValidation: (datasetId: string, version: string, expectedRevision: number) =>
    client.post<EvidenceFixtureValidationRun>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/versions/${encodeURIComponent(version)}/fixture-validations`, { expected_revision: expectedRevision }).then((r) => r.data),

  retryFixtureValidation: (datasetId: string, version: string, runId: string, expectedRevision: number) =>
    client
      .post<EvidenceFixtureValidationRun>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/versions/${encodeURIComponent(version)}/fixture-validations/${encodeURIComponent(runId)}/retry`, { expected_revision: expectedRevision })
      .then((r) => r.data),

  freezeDataset: (datasetId: string, expectedRevision: number) => client.post<EvidenceFreezeResult>(`/evaluation/datasets/${encodeURIComponent(datasetId)}/freeze`, { expected_revision: expectedRevision }).then((r) => r.data)
}
