import client from './client'
import type { SystemStats, TaskStatus, Webhook, AuditLog, ReadinessReport, RequestTrendResponse } from '@/types'

export interface AuditLogFilters {
  user_id?: string
  action?: string
  start?: string
  end?: string
  limit?: number
  offset?: number
}

/** Remove the trailing API version while preserving any configured remote origin/path. */
function unversionedApiBase(baseURL = client.defaults.baseURL ?? '/api/v1'): string {
  return baseURL.replace(/\/+$/, '').replace(/\/v1$/, '')
}

export const adminApi = {
  /** Fetch dashboard statistics. */
  stats: () => client.get<SystemStats>('/admin/stats').then((r) => r.data),

  /** Fetch the lightweight organization-scoped live request trend. */
  requestTrends: (window: '60m' | '24h' = '60m') =>
    client.get<RequestTrendResponse>('/admin/request-trends', { params: { window } }).then((r) => r.data),

  /** Fetch dependency-aware runtime readiness, including intentional 503 reports. */
  readiness: () =>
    client
      .get<ReadinessReport>('/health/ready', {
        baseURL: unversionedApiBase(),
        validateStatus: (status) => status === 200 || status === 503,
      })
      .then((r) => r.data),

  /** Fetch audit logs using the supplied filters. */
  auditLogs: (params?: AuditLogFilters) =>
    client.get<AuditLog[]>('/audit/logs', { params }).then((r) => r.data),

  /** Fetch the known audit action names. */
  auditActions: () => client.get('/audit/actions').then((r) => r.data),

  /** Download audit logs using the supplied filters. */
  exportAuditLogs: (params?: AuditLogFilters) =>
    client.get('/audit/logs/export', { params, responseType: 'blob' }).then((r) => r.data),
}

export const taskApi = {
  /** Submit a document-ingestion task. */
  submit: (filePath: string) =>
    client.post('/tasks/submit', { file_path: filePath }).then((r) => r.data),

  /** Submit document-ingestion tasks in batch. */
  submitBatch: (filePaths: string[]) =>
    client.post('/tasks/submit-batch', { file_paths: filePaths }).then((r) => r.data),

  /** Submit a knowledge-update task. */
  update: (filePath: string, changeType: string) =>
    client.post('/tasks/update', { file_path: filePath, change_type: changeType }).then((r) => r.data),

  /** Fetch task details. */
  get: (taskId: string) =>
    client.get<TaskStatus>(`/tasks/${taskId}`).then((r) => r.data),

  /** Fetch task records. */
  list: (limit = 50) =>
    client.get<TaskStatus[]>('/tasks', { params: { limit } }).then((r) => r.data),

  /** Cancel the selected task. */
  cancel: (taskId: string) => client.delete(`/tasks/${taskId}`),

  /** Request cleanup of completed tasks. */
  cleanup: () => client.post('/tasks/cleanup'),
}

export const webhookApi = {
  /** Fetch webhook records. */
  list: () => client.get<Webhook[]>('/webhooks').then((r) => r.data),

  /** Fetch supported webhook event types. */
  events: () =>
    client
      .get<Array<{ value: string; label: string }>>('/webhooks/events')
      .then((r) => r.data),

  /** Create a webhook. */
  create: (data: {
    url: string
    events: string[]
    secret?: string
    is_active?: boolean
  }) => client.post<Webhook>('/webhooks', data).then((r) => r.data),

  /** Delete a webhook. */
  delete: (id: string) => client.delete(`/webhooks/${id}`),

  /** Send a test event to the selected webhook. */
  test: (id: string) =>
    client
      .post<{ success: boolean; message?: string; error?: string; status_code?: number }>(
        `/webhooks/${id}/test`,
      )
      .then((r) => r.data),
}
