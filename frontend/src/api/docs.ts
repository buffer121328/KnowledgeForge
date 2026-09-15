import client from './client'
import type {
  ChunkItem,
  DepartmentItem,
  DocListItem,
  FolderUploadManifest,
  IngestProgressResponse,
  IngestResponse,
} from '@/types'
import type { AxiosProgressEvent } from 'axios'

export const docApi = {
  /** Fetch catalog-first document records, optionally scoped to a department. */
  list: (departmentId?: string) =>
    client.get<DocListItem[]>('/docs', { params: { department_id: departmentId } }).then((r) => r.data),

  /** Fetch tenant-owned departments for folder mapping. */
  departments: (companyId?: string) =>
    client.get<DepartmentItem[]>('/docs/departments', { params: { company_id: companyId } }).then((r) => r.data),

  /** Fetch parsed chunks for a document. */
  chunks: (docId: string) =>
    client.get<ChunkItem[]>(`/docs/${docId}/chunks`).then((r) => r.data),

  /** Fetch one authenticated source file as a browser-openable Blob. */
  open: (docId: string) =>
    client.get<Blob>(`/docs/${docId}/file`, { responseType: 'blob' }).then((r) => r.data),

  /** Upload a document for ingestion. */
  upload: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return client.post<IngestResponse>('/ingest/upload', formData, { timeout: 300000 }).then((r) => r.data)
  },

  /** Upload multiple documents for ingestion with an optional progress batch identity. */
  batchUpload: (
    files: File[],
    options?: { uploadId?: string; clientFileIds?: string[]; onUploadProgress?: (event: AxiosProgressEvent) => void },
  ) => {
    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))
    if (options?.uploadId) formData.append('upload_id', options.uploadId)
    if (options?.clientFileIds) formData.append('client_file_ids', JSON.stringify(options.clientFileIds))
    return client.post<IngestResponse[]>('/ingest/batch', formData, {
      timeout: 600000,
      onUploadProgress: options?.onUploadProgress,
    }).then((r) => r.data)
  },

  /** Poll tenant-scoped backend stages for one upload batch. */
  progress: (uploadId: string, totalCount: number) =>
    client.get<IngestProgressResponse>(`/ingest/progress/${uploadId}`, { params: { total_count: totalCount } }).then((r) => r.data),

  /** Upload accepted folder members beside their confirmed logical manifest. */
  folderUpload: (files: File[], manifest: FolderUploadManifest, onUploadProgress?: (event: AxiosProgressEvent) => void) => {
    const formData = new FormData()
    files.forEach((file) => formData.append('files[]', file))
    formData.append('manifest', JSON.stringify(manifest))
    return client.post<IngestResponse[]>('/ingest/folder', formData, { timeout: 120000, onUploadProgress }).then((r) => r.data)
  },

  /** Retry one tenant-owned failed catalog document by stable document identity. */
  retry: (docId: string) =>
    client.post<IngestResponse>(`/docs/${docId}/retry`, undefined, { timeout: 600000 }).then((r) => r.data),

  /** Update a document. */
  update: (filePath: string, changeType: 'modified' | 'created' | 'deleted' = 'modified') =>
    client.post('/admin/update', { file_path: filePath, change_type: changeType }).then((r) => r.data),

  /** Delete a document. */
  remove: (docId: string) =>
    client.delete<{
      doc_id: string
      vectors_deleted: number
      entities_deleted: number
      file_deleted: boolean
      status: string
    }>(`/docs/${docId}`).then((r) => r.data),
}
