import { useCallback, useEffect, useMemo, useState } from 'react'
import { App } from 'antd'
import { isCompanyDemoSourceDocumentId } from '@/data/companyDemoCorpus'
import { taskApi } from '@/api/admin'
import { getApiErrorInfo } from '@/api/client'
import { docApi } from '@/api/docs'
import { useAuthStore } from '@/stores/auth'
import type {
  ChunkItem,
  DocListItem,
  DepartmentItem,
  DocumentUploadRecord,
  FolderUploadPreview,
  IngestResponse,
  IngestProgressResponse,
  TaskStatus,
} from '@/types'
import {
  buildFolderUploadPreview,
  createFolderUploadManifest,
  validateDocumentUploadBatch,
} from '@/utils/documentUpload'

export type DocumentChangeType = 'modified' | 'created' | 'deleted'

/** Project one durable catalog item into the upload outcome table contract. */
export function catalogDocumentToUploadRecord(
  doc: DocListItem,
  processing = false,
): DocumentUploadRecord {
  const status = processing
    ? 'processing'
    : doc.ingest_status === 'ingested'
      ? 'success'
      : doc.ingest_status === 'failed'
        ? 'failed'
        : 'processing'
  return {
    key: doc.doc_id,
    fileName: doc.uploaded_filename || doc.provenance_source_filename || doc.file_name,
    docId: doc.doc_id,
    chunksCount: doc.chunks_count || 0,
    entitiesCount: doc.entities_count || 0,
    relationsCount: doc.relations_count || 0,
    status,
    uploadedAt: doc.created_at || doc.updated_at || '',
    relativePath: doc.relative_path,
    departmentId: doc.department_id,
    errorMessage: processing ? undefined : doc.error_code,
    retryable: Boolean(doc.retryable),
  }
}

const UPLOAD_STAGE_TOTAL = 4
const UPLOAD_POLL_INTERVAL_MS = 800

/** Create a backend-validated UUID even when randomUUID is unavailable on non-secure origins. */
export function createUploadCorrelationId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID()
  }

  const bytes = new Uint8Array(16)
  if (typeof globalThis.crypto?.getRandomValues === 'function') {
    globalThis.crypto.getRandomValues(bytes)
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256)
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0'))
  return `${hex.slice(0, 4).join('')}-${hex.slice(4, 6).join('')}-${hex.slice(6, 8).join('')}-${hex.slice(8, 10).join('')}-${hex.slice(10, 16).join('')}`
}

async function pollUploadProgress(
  uploadId: string,
  totalCount: number,
  isActive: () => boolean,
  onUpdate: (progress: IngestProgressResponse) => void,
): Promise<IngestProgressResponse | null> {
  while (isActive()) {
    const current = await docApi.progress(uploadId, totalCount)
    if (!current) return null
    onUpdate(current)
    if (current.terminal) return current
    await new Promise((resolve) => window.setTimeout(resolve, UPLOAD_POLL_INTERVAL_MS))
  }
  return null
}

/** Manage documents state and related actions. */
export function useDocuments() {
  const { message } = App.useApp()
  const currentUser = useAuthStore.getState().user
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stageIndex, setStageIndex] = useState(0)
  const [stageLabel, setStageLabel] = useState('')
  const [stageFailed, setStageFailed] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<IngestProgressResponse | null>(null)
  const [uploadRejectedItems, setUploadRejectedItems] = useState<IngestResponse[]>([])
  const [uploadHint, setUploadHint] = useState('')
  const [transientUploads, setTransientUploads] = useState<DocumentUploadRecord[]>([])
  const [docs, setDocs] = useState<DocListItem[]>([])
  const [docsLoading, setDocsLoading] = useState(false)
  const [tasks, setTasks] = useState<TaskStatus[]>([])
  const [filePath, setFilePath] = useState('')
  const [changeType, setChangeType] = useState<DocumentChangeType>('modified')
  const [taskSubmitting, setTaskSubmitting] = useState(false)
  const [chunkDrawerOpen, setChunkDrawerOpen] = useState(false)
  const [activeDoc, setActiveDoc] = useState<DocListItem | null>(null)
  const [chunks, setChunks] = useState<ChunkItem[]>([])
  const [chunksLoading, setChunksLoading] = useState(false)
  const [folderPreview, setFolderPreview] = useState<FolderUploadPreview | null>(null)
  const [departmentMappings, setDepartmentMappings] = useState<Record<string, string>>({})
  const [departments, setDepartments] = useState<DepartmentItem[]>([])
  const [folderCompanyId, setFolderCompanyId] = useState<string | null>(null)
  const [folderPreparing, setFolderPreparing] = useState(false)
  const [folderError, setFolderError] = useState('')
  const [retryingDocIds, setRetryingDocIds] = useState<string[]>([])
  const [selectedDepartmentId, setSelectedDepartmentId] = useState<string>()

  const uploads = useMemo(() => {
    const catalogRows = docs.filter((doc) => isCompanyDemoSourceDocumentId(doc.doc_id)).map((doc) =>
      catalogDocumentToUploadRecord(doc, retryingDocIds.includes(doc.doc_id)),
    )
    const catalogIds = new Set(catalogRows.map((row) => row.docId))
    return [
      ...catalogRows,
      ...transientUploads.filter(
        (row) => isCompanyDemoSourceDocumentId(row.docId) && !catalogIds.has(row.docId),
      ),
    ]
  }, [docs, retryingDocIds, transientUploads])

  const loadDocs = useCallback(async () => {
    setDocsLoading(true)
    try {
      const list = await docApi.list(selectedDepartmentId)
      setDocs(list)
    } catch {
      message.error('加载文档列表失败')
    } finally {
      setDocsLoading(false)
    }
  }, [message, selectedDepartmentId])

  useEffect(() => {
    void loadDocs()
  }, [loadDocs])

  useEffect(() => {
    void docApi.departments().then(setDepartments).catch(() => setDepartments([]))
  }, [])

  const openChunks = useCallback(
    async (doc: DocListItem) => {
      setActiveDoc(doc)
      setChunkDrawerOpen(true)
      setChunksLoading(true)
      setChunks([])
      try {
        const list = await docApi.chunks(doc.doc_id)
        setChunks(list)
      } catch {
        message.error('加载分块内容失败')
      } finally {
        setChunksLoading(false)
      }
    },
    [message],
  )

  const openDocument = useCallback(
    async (doc: DocListItem) => {
      const previewWindow = window.open('about:blank', '_blank')
      try {
        const blob = await docApi.open(doc.doc_id)
        const objectUrl = URL.createObjectURL(blob)
        if (!previewWindow) {
          URL.revokeObjectURL(objectUrl)
          message.warning('浏览器阻止了新标签页，请允许弹窗后重试')
          return
        }
        previewWindow.opener = null
        previewWindow.location.href = objectUrl
        window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000)
      } catch (error) {
        previewWindow?.close()
        const detail = getApiErrorInfo(error)
        message.error(detail?.message || `打开失败: ${doc.file_name}`)
      }
    },
    [message],
  )

  const handleDelete = useCallback(
    async (doc: DocListItem) => {
      try {
        const result = await docApi.remove(doc.doc_id)
        message.success(
          `已删除 ${doc.file_name}（向量 ${result.vectors_deleted}，实体 ${result.entities_deleted}）`,
        )
        if (activeDoc?.doc_id === doc.doc_id) {
          setChunkDrawerOpen(false)
          setActiveDoc(null)
        }
        await loadDocs()
      } catch {
        message.error(`删除失败: ${doc.file_name}`)
      }
    },
    [activeDoc, loadDocs, message],
  )

  const handleUploadBatch = useCallback(
    async (files: File[]) => {
      const uploadId = createUploadCorrelationId()
      const clientFileIds = files.map(() => createUploadCorrelationId())
      let polling = true
      const applyProgress = (current: IngestProgressResponse) => {
        setUploadProgress(current)
        setStageIndex(current.stage_index)
        setStageLabel(current.stage_label)
        setStageFailed(current.status === 'failed' || current.failed_count > 0)
        setProgress(Math.round((current.stage_index / UPLOAD_STAGE_TOTAL) * 100))
        setUploadHint(current.stage_label)
      }
      setUploading(true)
      setStageIndex(1)
      setStageLabel('上传校验与落盘')
      setStageFailed(false)
      setUploadProgress(null)
      setProgress(25)
      setUploadHint(`正在上传 ${files.length} 份文档…`)
      const progressTask = pollUploadProgress(uploadId, files.length, () => polling, applyProgress)
      try {
        const results = await docApi.batchUpload(files, {
          uploadId,
          clientFileIds,
          onUploadProgress: (event) => {
            if (event.total) {
              setUploadHint(`正在上传文件（${Math.round((event.loaded / event.total) * 100)}%）…`)
            }
          },
        })
        const finalProgress = await docApi.progress(uploadId, files.length)
        if (finalProgress) applyProgress(finalProgress)
        setStageIndex(4)
        setStageLabel('入库完成')
        setStageFailed(results.some((result) => result.status === 'failed'))
        setProgress(100)
        setUploadHint(results.some((result) => result.status === 'failed') ? '部分文件处理失败' : '批量入库完成')
        const uploadedAt = new Date().toLocaleString()
        setTransientUploads((previous) => [
          ...results.map((result, index) => ({
            key: `${result.doc_id || files[index]?.name || 'document'}-${Date.now()}-${index}`,
            fileName: result.file_name,
            docId: result.doc_id || '',
            chunksCount: result.chunks_count,
            entitiesCount: result.entities_count,
            relationsCount: result.relations_count,
            status: result.status === 'success' ? 'success' as const : 'failed' as const,
            uploadedAt,
            taskId: result.task_id,
            errorMessage: result.message || result.error_code,
          })),
          ...previous,
        ])
        if (results.some((result) => result.status === 'failed')) {
          message.warning(`已完成 ${results.filter((result) => result.status === 'success').length} 份，另有文件失败`)
        } else {
          message.success(`已完成 ${results.length} 份文档入库`)
        }
        await loadDocs()
      } catch {
        setStageFailed(true)
        setUploadProgress(null)
        setUploadHint('上传或处理失败')
        message.error('批量入库失败（超时或服务异常，可稍后重试）')
        await loadDocs()
      } finally {
        polling = false
        await progressTask.catch(() => null)
        setUploading(false)
        setProgress(0)
        setStageIndex(0)
        setStageLabel('')
        setStageFailed(false)
        setUploadProgress(null)
        setUploadHint('')
      }
    },
    [loadDocs, message],
  )

  const selectUploadFiles = useCallback(
    (files: File[]): void => {
      const validation = validateDocumentUploadBatch(files)
      if (!validation.valid) {
        if (validation.reason === 'empty') {
          message.warning('请选择要上传的文件')
        } else if (validation.reason === 'too-many-files') {
          message.error(`单次最多上传 ${validation.maxFiles} 份文件`)
        } else if (validation.reason === 'unsupported') {
          message.error(
            `不支持的文件类型${validation.extension ? `「${validation.extension}」` : ''}：${validation.fileName}。请上传 PDF / Word(.doc/.docx) / Excel / CSV / 图片 / Markdown / 纯文本 / HTML`,
          )
        } else {
          message.error(`${validation.fileName} 超过 50MB 限制`)
        }
        return
      }

      void handleUploadBatch(files)
    },
    [handleUploadBatch, message],
  )

  /** Preview a browser-selected folder and load tenant departments for explicit mapping. */
  const selectFolderFiles = useCallback(
    async (files: File[]): Promise<void> => {
      const preview = buildFolderUploadPreview(files)
      setFolderPreview(preview)
      setFolderError('')
      setDepartmentMappings(
        Object.fromEntries(Object.keys(preview.departmentCounts).map((folderKey) => [folderKey, folderKey])),
      )
      if (preview.readyCount > 50) {
        setFolderError('可入库文件超过单次 50 份限制')
        return
      }
      setFolderPreparing(true)
      try {
        const records = await docApi.departments()
        setDepartments(records)
        setFolderCompanyId(records[0]?.company_id || useAuthStore.getState().user?.org_id || null)
        setDepartmentMappings((current) => {
          const next = { ...current }
          for (const folderKey of Object.keys(preview.departmentCounts)) {
            const match = records.find(
              (department) => department.normalized_key === folderKey || department.department_id === folderKey,
            )
            if (match) next[folderKey] = match.department_id
          }
          return next
        })
      } catch {
        setFolderError('无法读取部门映射（可能无权限），请稍后重试')
      } finally {
        setFolderPreparing(false)
      }
    },
    [],
  )

  /** Update one logical top-level folder to its confirmed tenant department. */
  const setDepartmentMapping = useCallback((folderKey: string, departmentId: string): void => {
    setDepartmentMappings((current) => ({ ...current, [folderKey]: departmentId }))
  }, [])

  /** Submit only ready folder members and preserve independent per-file outcomes. */
  const confirmFolderUpload = useCallback(async (): Promise<void> => {
    if (!folderPreview) return
    if (folderPreview.issueCount > 0) {
      setFolderError('请先移除不支持、重复、超限或路径无效的文件后再确认上传')
      return
    }
    const companyId = folderCompanyId || useAuthStore.getState().user?.org_id
    if (!companyId) {
      setFolderError('当前会话缺少公司/租户权限')
      return
    }
    const ready = folderPreview.files.filter((item) => item.status === 'ready')
    if (ready.length === 0 || Object.keys(folderPreview.departmentCounts).some((key) => !departmentMappings[key])) {
      setFolderError('请先确认所有部门映射，并确保至少有一份可入库文件')
      return
    }
    const uploadId = createUploadCorrelationId()
    const manifest = { ...createFolderUploadManifest(folderPreview, companyId, departmentMappings), upload_id: uploadId }
    let polling = true
    setUploading(true)
    setStageIndex(1)
    setStageLabel('上传校验与落盘')
    setStageFailed(false)
    setUploadProgress(null)
    setUploadRejectedItems([])
    setProgress(25)
    setFolderError('')
    setUploadHint(`正在按目录上传 ${ready.length} 份文档…`)
    try {
      const results = await docApi.folderUpload(ready.map((item) => item.file), manifest)
      const accepted = results.filter((result) => result.status === 'processing' || result.status === 'accepted')
      const failedResults = results.filter((result) => result.status === 'failed')
      setUploadRejectedItems(failedResults)
      let finalProgress: IngestProgressResponse | null = null
      if (accepted.length > 0) {
        try {
          finalProgress = await pollUploadProgress(uploadId, accepted.length, () => polling, (current) => {
            setUploadProgress(current)
            setStageIndex(current.stage_index)
            setStageLabel(current.stage_label)
            setStageFailed(current.status === 'failed' || current.failed_count > 0)
            setProgress(Math.round((current.stage_index / UPLOAD_STAGE_TOTAL) * 100))
            setUploadHint(current.stage_label)
          })
        } catch {
          setStageIndex(2)
          setStageLabel('文件已接收')
          setProgress(50)
          setUploadHint(
            failedResults.length > 0
              ? `已接收 ${accepted.length} 份，另有 ${failedResults.length} 份未接收；后台处理中`
              : '文件已接收，后台处理中；进度状态暂时无法刷新',
          )
          message.warning('目录已接收，但进度状态刷新失败，可稍后刷新上传记录')
        }
        if (finalProgress) {
          setStageIndex(finalProgress.stage_index)
          setStageLabel(finalProgress.stage_label)
          setProgress(Math.round((finalProgress.stage_index / UPLOAD_STAGE_TOTAL) * 100))
        }
      }
      const ingestionFailed = finalProgress?.status === 'failed' || (finalProgress?.failed_count ?? 0) > 0
      const onlyRejected = accepted.length === 0 && failedResults.length > 0
      const hasFailure = ingestionFailed || onlyRejected
      const isStillProcessing = accepted.length > 0 && !finalProgress?.terminal
      setStageFailed(hasFailure)
      setProgress(hasFailure ? Math.round(((finalProgress?.stage_index || 1) / UPLOAD_STAGE_TOTAL) * 100) : isStillProcessing ? 50 : 100)
      setUploadHint(
        hasFailure
          ? onlyRejected ? '文件未进入后台处理' : '部分文件后台处理失败'
          : isStillProcessing
            ? failedResults.length > 0
              ? `已接收 ${accepted.length} 份，另有 ${failedResults.length} 份未接收，后台处理中`
              : '目录已接收，后台处理中'
            : failedResults.length > 0
              ? `其余 ${accepted.length} 份入库完成，另有 ${failedResults.length} 份未接收`
              : '目录入库完成',
      )
      const uploadedAt = new Date().toLocaleString()
      setTransientUploads((previous) => [
        ...results.map((result, index) => ({
          key: `${result.client_file_id || result.doc_id || index}-${Date.now()}`,
          fileName: result.display_name || result.file_name,
          docId: result.doc_id || '',
          chunksCount: result.chunks_count,
          entitiesCount: result.entities_count,
          relationsCount: result.relations_count,
          status: result.status === 'success'
            ? 'success' as const
            : result.status === 'processing' || result.status === 'accepted'
              ? 'processing' as const
              : 'failed' as const,
          uploadedAt,
          relativePath: result.relative_path,
          departmentId: result.department_id,
          taskId: result.task_id,
          errorMessage: result.message || result.error_code,
        })),
        ...previous,
      ])
      const acceptedCount = results.filter(
        (result) => result.status === 'processing' || result.status === 'accepted',
      ).length
      const failures = results.filter((result) => result.status === 'failed').length
      if (acceptedCount > 0 && failures > 0) {
        message.warning(`已接收 ${acceptedCount} 份目录文档后台处理，另有 ${failures} 份未接收`)
      } else if (acceptedCount > 0) {
        message.success(`已接收 ${acceptedCount} 份目录文档，后台处理中`)
      } else if (failures > 0) {
        message.warning(`目录文档未进入后台处理：失败 ${failures} 份`)
      }
      await loadDocs()
      try {
        setTasks(await taskApi.list(20))
      } catch {
        message.warning('目录已接收，但任务列表刷新失败')
      }
    } catch (error) {
      const detail = getApiErrorInfo(error)
      const reason = detail?.message
        ? `：${detail.message}${detail.code ? `（${detail.code}）` : ''}`
        : '（超时、权限或服务异常）'
      setFolderError(`目录入库失败${reason}，预览与映射已保留，可重试`)
      message.error(detail?.message ? `目录入库失败：${detail.message}` : '目录入库失败，预览与映射已保留')
    } finally {
      polling = false
      setUploading(false)
      setProgress(0)
      setStageIndex(0)
      setStageLabel('')
      setStageFailed(false)
      setUploadProgress(null)
      setUploadHint('')
    }
  }, [departmentMappings, folderCompanyId, folderPreview, loadDocs, message])

  /** Retry one server-eligible failed document and always resynchronize catalog state. */
  const retryUpload = useCallback(async (docId: string): Promise<void> => {
    if (retryingDocIds.includes(docId)) return
    setRetryingDocIds((current) => [...current, docId])
    try {
      await docApi.retry(docId)
      message.success('文档重试成功')
    } catch {
      message.error('文档重试失败，请稍后再试')
    } finally {
      await loadDocs()
      setRetryingDocIds((current) => current.filter((item) => item !== docId))
    }
  }, [loadDocs, message, retryingDocIds])

  const refreshTasks = useCallback(async () => {
    try {
      const list = await taskApi.list(20)
      setTasks(list)
      message.success('已刷新任务列表')
    } catch {
      message.error('刷新任务列表失败')
    }
  }, [message])

  const submitUpdateTask = useCallback(async () => {
    const path = filePath.trim()
    if (!path) {
      message.warning('请选择已入库文档，或填写安全文件引用')
      return
    }
    setTaskSubmitting(true)
    try {
      const result = await taskApi.update(path, changeType)
      message.success(`已提交异步任务 ${result?.task_id || ''}`.trim())
      setFilePath('')
      await refreshTasks()
    } catch {
      message.error('提交失败：请确认任务队列可用，且文件引用属于当前租户')
    } finally {
      setTaskSubmitting(false)
    }
  }, [changeType, filePath, message, refreshTasks])

  const fillExamplePath = useCallback(
    (example: string) => {
      setFilePath(example)
      message.info('已填入示例引用，请使用当前租户真实文件引用提交')
    },
    [message],
  )

  const cancelTask = useCallback(
    async (task: TaskStatus) => {
      await taskApi.cancel(task.task_id)
      message.success('已取消')
      void refreshTasks()
    },
    [message, refreshTasks],
  )

  const closeChunks = useCallback(() => {
    setChunkDrawerOpen(false)
  }, [])

  return {
    activeDoc,
    cancelTask,
    changeType,
    currentUser,
    chunkDrawerOpen,
    chunks,
    chunksLoading,
    closeChunks,
    docs,
    docsLoading,
    departmentMappings,
    departments,
    filePath,
    folderError,
    folderPreparing,
    folderPreview,
    confirmFolderUpload,
    fillExamplePath,
    handleDelete,
    loadDocs,
    openChunks,
    openDocument,
    progress,
    stageFailed,
    stageIndex,
    stageLabel,
    uploadProgress,
    uploadRejectedItems,
    refreshTasks,
    retryingDocIds,
    retryUpload,
    selectFolderFiles,
    selectUploadFiles,
    selectedDepartmentId,
    setDepartmentMapping,
    setSelectedDepartmentId,
    setChangeType,
    setFilePath,
    submitUpdateTask,
    taskSubmitting,
    tasks,
    uploading,
    uploadHint,
    uploads,
  }
}
