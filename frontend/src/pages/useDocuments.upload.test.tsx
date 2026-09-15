// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { docApi } from '@/api/docs'
import { taskApi } from '@/api/admin'
import type { IngestResponse } from '@/types'
import { createUploadCorrelationId, useDocuments } from './useDocuments'

const companyDemoDocumentId = 'a23c6a1b-2e6f-54ca-b9bb-97359f53f349'

const message = {
  error: vi.fn(),
  success: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
}

vi.mock('antd', async () => {
  const actual = await vi.importActual<typeof import('antd')>('antd')
  return {
    ...actual,
    App: {
      ...actual.App,
      useApp: () => ({ message }),
    },
  }
})

vi.mock('@/stores/auth', () => ({
  useAuthStore: {
    getState: () => ({
      user: { user_id: 'u1', username: 'alice', role: 'editor', org_id: 'tenant-a', permissions: [] },
    }),
  },
}))

vi.mock('@/api/docs', () => ({
  docApi: {
    list: vi.fn(),
    chunks: vi.fn(),
    open: vi.fn(),
    remove: vi.fn(),
    batchUpload: vi.fn(),
    progress: vi.fn(),
    departments: vi.fn(),
    folderUpload: vi.fn(),
    retry: vi.fn(),
  },
}))

vi.mock('@/api/admin', () => ({
  taskApi: {
    list: vi.fn(),
    update: vi.fn(),
    cancel: vi.fn(),
  },
}))

function uploadFile(name: string, size = 1): File {
  const value = new File(['x'], name, { type: 'text/plain' })
  Object.defineProperty(value, 'size', { value: size })
  return value
}

const batchUpload = vi.mocked(docApi.batchUpload)
const uploadProgress = vi.mocked(docApi.progress)
const openDocumentSource = vi.mocked(docApi.open)
const listDocuments = vi.mocked(docApi.list)
const listDepartments = vi.mocked(docApi.departments)
const folderUpload = vi.mocked(docApi.folderUpload)
const retryDocument = vi.mocked(docApi.retry)
const listTasks = vi.mocked(taskApi.list)

describe('createUploadCorrelationId', () => {
  it('falls back to a valid UUID when randomUUID is unavailable', () => {
    const originalCrypto = globalThis.crypto
    Object.defineProperty(globalThis, 'crypto', {
      configurable: true,
      value: {
        getRandomValues: (bytes: Uint8Array) => {
          bytes.fill(7)
          return bytes
        },
      },
    })

    try {
      expect(createUploadCorrelationId()).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
      )
    } finally {
      Object.defineProperty(globalThis, 'crypto', { configurable: true, value: originalCrypto })
    }
  })
})

/** Build one durable catalog item returned by the synchronized document list. */
function catalogDocument(overrides: Record<string, unknown> = {}) {
  return {
    doc_id: companyDemoDocumentId,
    file_name: '预算制度',
    source: '',
    file_reference: `departments/finance/documents/${companyDemoDocumentId}/预算制度.txt`,
    doc_type: 'text',
    chunks_count: 0,
    department_id: 'finance',
    relative_path: 'finance/预算制度.txt',
    uploaded_filename: '预算制度.txt',
    ingest_status: 'failed',
    entities_count: 0,
    relations_count: 0,
    created_at: '2026-08-01T01:02:03+00:00',
    updated_at: '2026-08-01T01:03:04+00:00',
    error_code: 'document_ingest_failed',
    retryable: true,
    ...overrides,
  }
}

describe('useDocuments batch upload', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    uploadProgress.mockResolvedValue(null as never)
    listDocuments.mockResolvedValue([])
    batchUpload.mockResolvedValue([])
    openDocumentSource.mockResolvedValue(new Blob(['safe'], { type: 'text/plain' }))
    listDepartments.mockResolvedValue([])
    folderUpload.mockResolvedValue([])
    retryDocument.mockResolvedValue({
      file_name: '预算制度.txt', chunks_count: 1, entities_count: 2,
      relations_count: 3, status: 'success', doc_id: companyDemoDocumentId,
    })
  })


  it('opens an authenticated source Blob in a new tab', async () => {
    const previewWindow = {
      close: vi.fn(),
      location: { href: 'about:blank' },
      opener: window,
    }
    vi.spyOn(window, 'open').mockReturnValue(previewWindow as unknown as Window)
    const createObjectURL = vi.fn(() => 'blob:document-source')
    const revokeObjectURL = vi.fn()
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL })

    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    await act(async () => result.current.openDocument(catalogDocument()))

    expect(openDocumentSource).toHaveBeenCalledWith(companyDemoDocumentId)
    expect(previewWindow.location.href).toBe('blob:document-source')
    expect(previewWindow.opener).toBeNull()
    expect(message.error).not.toHaveBeenCalled()
  })

  it('closes the placeholder tab and reports an open failure', async () => {
    const previewWindow = {
      close: vi.fn(),
      location: { href: 'about:blank' },
      opener: window,
    }
    vi.spyOn(window, 'open').mockReturnValue(previewWindow as unknown as Window)
    openDocumentSource.mockRejectedValueOnce(new Error('source unavailable'))

    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    await act(async () => result.current.openDocument(catalogDocument()))

    expect(previewWindow.close).toHaveBeenCalledOnce()
    expect(message.error).toHaveBeenCalledWith('打开失败: 预算制度')
  })

  it('projects durable catalog rows into synchronized upload records on initial load', async () => {
    listDocuments.mockResolvedValue([catalogDocument()])

    const { result } = renderHook(() => useDocuments())

    await waitFor(() => expect(result.current.uploads).toHaveLength(1))
    expect(result.current.uploads[0]).toMatchObject({
      docId: companyDemoDocumentId,
      fileName: '预算制度.txt',
      relativePath: 'finance/预算制度.txt',
      departmentId: 'finance',
      status: 'failed',
      errorMessage: 'document_ingest_failed',
      uploadedAt: '2026-08-01T01:02:03+00:00',
      retryable: true,
    })
  })

  it('keeps non-manifest catalog data loaded while excluding it from upload records', async () => {
    const unrelated = catalogDocument({
      doc_id: 'legacy-unrelated-document',
      file_name: '历史制度.txt',
      uploaded_filename: '历史制度.txt',
    })
    listDocuments.mockResolvedValue([catalogDocument(), unrelated])

    const { result } = renderHook(() => useDocuments())

    await waitFor(() => expect(result.current.docs).toHaveLength(2))
    expect(result.current.uploads).toHaveLength(1)
    expect(result.current.uploads[0]?.docId).toBe(companyDemoDocumentId)
    expect(result.current.docs.map((document) => document.doc_id)).toContain('legacy-unrelated-document')
  })

  it('submits one request for one valid multi-file selection', async () => {
    const files = [uploadFile('one.txt'), uploadFile('two.pdf')]
    batchUpload.mockResolvedValue([
      {
        file_name: 'one.txt',
        chunks_count: 1,
        entities_count: 2,
        relations_count: 3,
        status: 'success',
        doc_id: companyDemoDocumentId,
      },
      {
        file_name: 'two.pdf',
        chunks_count: 4,
        entities_count: 5,
        relations_count: 6,
        status: 'success',
        doc_id: '9b24f3ba-9fd6-5dab-8c5c-fab973bcbaa3',
      },
    ])
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    act(() => result.current.selectUploadFiles(files))

    await waitFor(() => expect(batchUpload).toHaveBeenCalledTimes(1))
    expect(batchUpload).toHaveBeenCalledWith(
      files,
      expect.objectContaining({
        uploadId: expect.any(String),
        clientFileIds: expect.arrayContaining([expect.any(String), expect.any(String)]),
        onUploadProgress: expect.any(Function),
      }),
    )
    await waitFor(() => expect(result.current.uploads).toHaveLength(2))
  })

  it('uses a terminal backend stage response instead of a timer while the request is active', async () => {
    const files = [uploadFile('stage.txt')]
    let resolveBatch: ((value: IngestResponse[]) => void) | undefined
    batchUpload.mockReturnValue(new Promise((resolve) => { resolveBatch = resolve }))
    uploadProgress.mockResolvedValue({
      upload_id: 'upload-1',
      total_count: 1,
      completed_count: 1,
      failed_count: 0,
      stage_index: 4,
      stage_total: 4,
      stage_label: '入库完成',
      processing_step: 'store_graph',
      processing_step_index: 4,
      processing_step_total: 6,
      processing_step_label: '写知识图谱',
      terminal: true,
      status: 'success',
      items: [{
        doc_id: 'doc-stage',
        client_file_id: 'client-stage',
        file_name: 'stage.txt',
        status: 'processing',
        ingest_stage: 'processing',
        ingest_stage_index: 3,
        ingest_stage_total: 4,
        ingest_stage_label: '解析与知识抽取',
        processing_step: 'store_graph',
        processing_step_index: 4,
        processing_step_total: 6,
        processing_step_label: '写知识图谱',
        error_code: '',
        message: '',
      }],
    })
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    act(() => result.current.selectUploadFiles(files))

    await waitFor(() => expect(result.current.progress).toBe(100))
    expect(uploadProgress).toHaveBeenCalledWith(expect.any(String), 1)
    expect(result.current.stageIndex).toBe(4)
    expect(result.current.stageLabel).toBe('入库完成')
    expect(result.current.uploadProgress?.processing_step_label).toBe('写知识图谱')

    resolveBatch?.([{
      file_name: 'stage.txt', chunks_count: 1, entities_count: 0, relations_count: 0,
      status: 'success', doc_id: 'doc-stage',
    }])
    await waitFor(() => expect(result.current.uploading).toBe(false))
  })

  it('rejects 51 selected files without sending a request', async () => {
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())
    const files = Array.from({ length: 51 }, (_, index) => uploadFile(`doc-${index}.txt`))

    act(() => result.current.selectUploadFiles(files))

    expect(batchUpload).not.toHaveBeenCalled()
    expect(message.error).toHaveBeenCalledWith('单次最多上传 50 份文件')
  })

  it('rejects a batch containing an invalid file without sending a request', async () => {
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    act(() => result.current.selectUploadFiles([uploadFile('safe.txt'), uploadFile('bad.exe')]))

    expect(batchUpload).not.toHaveBeenCalled()
    expect(message.error).toHaveBeenCalled()
  })

  it('refreshes durable catalog failures after a rejected batch request', async () => {
    listDocuments
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([catalogDocument()])
    batchUpload.mockRejectedValue(new Error('request failed'))
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalledTimes(1))

    act(() => result.current.selectUploadFiles([uploadFile('预算制度.txt')]))

    await waitFor(() => expect(listDocuments).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(result.current.uploads[0]?.status).toBe('failed'))
  })

  it('retries an eligible failed catalog row and refreshes its synchronized outcome', async () => {
    listDocuments
      .mockResolvedValueOnce([catalogDocument()])
      .mockResolvedValueOnce([catalogDocument({
        ingest_status: 'ingested', chunks_count: 1, entities_count: 2,
        relations_count: 3, error_code: '', retryable: false,
      })])
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(result.current.uploads[0]?.retryable).toBe(true))

    await act(async () => result.current.retryUpload(companyDemoDocumentId))

    expect(retryDocument).toHaveBeenCalledWith(companyDemoDocumentId)
    await waitFor(() => expect(result.current.uploads[0]).toMatchObject({
      status: 'success', chunksCount: 1, entitiesCount: 2, relationsCount: 3,
      retryable: false,
    }))
    expect(message.success).toHaveBeenCalledWith('文档重试成功')
  })

  it('refreshes and preserves retryability after a retry request fails', async () => {
    listDocuments.mockResolvedValue([catalogDocument()])
    retryDocument.mockRejectedValue(new Error('retry failed'))
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(result.current.uploads[0]?.retryable).toBe(true))

    await act(async () => result.current.retryUpload(companyDemoDocumentId))

    expect(listDocuments.mock.calls.length).toBeGreaterThanOrEqual(2)
    expect(result.current.uploads[0]?.retryable).toBe(true)
    expect(message.error).toHaveBeenCalledWith('文档重试失败，请稍后再试')
  })

  it('keeps failed rows non-retryable when the server reports no retained source', async () => {
    listDocuments.mockResolvedValue([catalogDocument({ retryable: false })])

    const { result } = renderHook(() => useDocuments())

    await waitFor(() => expect(result.current.uploads[0]?.status).toBe('failed'))
    expect(result.current.uploads[0]?.retryable).toBe(false)
  })
})


/** Build a folder-selected file with a browser relative path. */
function folderUploadFile(path: string): File {
  const value = uploadFile(path.split('/').at(-1) ?? path)
  Object.defineProperty(value, 'webkitRelativePath', { value: path })
  return value
}

describe('useDocuments folder upload', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    uploadProgress.mockResolvedValue(null as never)
    listDocuments.mockResolvedValue([])
    listDepartments.mockResolvedValue([])
    listTasks.mockResolvedValue([])
  })

  it('shows accepted folder work as processing and refreshes catalog and tasks', async () => {
    folderUpload.mockResolvedValue([
      { file_name: 'a.txt', chunks_count: 0, entities_count: 0, relations_count: 0, status: 'processing', task_id: 'tsk-1', doc_id: companyDemoDocumentId, client_file_id: 'a', relative_path: 'finance/a.txt', department_id: 'finance' },
      { file_name: 'b.txt', chunks_count: 0, entities_count: 0, relations_count: 0, status: 'failed', doc_id: '9b24f3ba-9fd6-5dab-8c5c-fab973bcbaa3', client_file_id: 'b', relative_path: 'finance/b.txt', department_id: 'finance', error_code: 'parse_failed', message: '解析失败' },
    ])
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    await act(async () => result.current.selectFolderFiles([
      folderUploadFile('root/finance/a.txt'), folderUploadFile('root/finance/b.txt'),
    ]))
    await act(async () => result.current.confirmFolderUpload())

    await waitFor(() => expect(folderUpload).toHaveBeenCalledTimes(1))
    expect(folderUpload.mock.calls[0][1]).toMatchObject({ company_id: 'tenant-a', department_mappings: { finance: 'finance' } })
    await waitFor(() => expect(result.current.uploads.map((item) => item.status)).toEqual(['processing', 'failed']))
    expect(result.current.uploadRejectedItems).toMatchObject([
      { file_name: 'b.txt', error_code: 'parse_failed', message: '解析失败' },
    ])
    expect(listDocuments.mock.calls.length).toBeGreaterThanOrEqual(2)
    expect(listTasks).toHaveBeenCalledWith(20)
    expect(message.warning).toHaveBeenCalledWith('已接收 1 份目录文档后台处理，另有 1 份未接收')
    expect([...message.success.mock.calls, ...message.warning.mock.calls].flat().join(' ')).not.toContain('入库完成')
  })

  it('uses the tenant returned by department mapping instead of stale persisted auth data', async () => {
    listDepartments.mockResolvedValue([
      { department_id: 'finance', company_id: 'org_123', name: '财务部', normalized_key: 'finance', status: 'active' },
    ])
    folderUpload.mockResolvedValue([
      { file_name: 'a.txt', chunks_count: 0, entities_count: 0, relations_count: 0, status: 'processing', task_id: 'tsk-1', doc_id: companyDemoDocumentId, client_file_id: 'a', relative_path: 'finance/a.txt', department_id: 'finance' },
    ])

    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    await act(async () => result.current.selectFolderFiles([folderUploadFile('root/finance/a.txt')]))
    expect(listDepartments).toHaveBeenCalledWith()
    await act(async () => result.current.confirmFolderUpload())

    expect(folderUpload.mock.calls[0][1]).toMatchObject({
      company_id: 'org_123',
      department_mappings: { finance: 'finance' },
    })
  })

  it('does not report folder upload failure when only progress polling fails after acceptance', async () => {
    folderUpload.mockResolvedValue([
      { file_name: 'a.txt', chunks_count: 0, entities_count: 0, relations_count: 0, status: 'processing', task_id: 'tsk-1', doc_id: companyDemoDocumentId, client_file_id: 'a', relative_path: 'finance/a.txt', department_id: 'finance' },
    ])
    uploadProgress.mockRejectedValueOnce(new Error('progress unavailable'))
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    await act(async () => result.current.selectFolderFiles([folderUploadFile('root/finance/a.txt')]))
    await act(async () => result.current.confirmFolderUpload())

    expect(result.current.folderError).toBe('')
    expect(result.current.uploads[0]).toMatchObject({ status: 'processing', docId: companyDemoDocumentId })
    expect(message.warning).toHaveBeenCalledWith('目录已接收，但进度状态刷新失败，可稍后刷新上传记录')
    expect(message.error).not.toHaveBeenCalledWith('目录入库失败，预览与映射已保留')
  })

  it('keeps the folder preview after a publication failure', async () => {
    folderUpload.mockRejectedValue({
      isAxiosError: true,
      response: {
        data: {
          detail: {
            code: 'folder_ingest_publish_failed',
            message: '后台入库任务暂无法发布',
          },
        },
      },
    })
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    await act(async () => result.current.selectFolderFiles([folderUploadFile('root/finance/a.txt')]))
    await act(async () => result.current.confirmFolderUpload())

    await waitFor(() => expect(result.current.folderError).toContain('预览与映射已保留'))
    expect(result.current.folderPreview?.readyCount).toBe(1)
    expect(result.current.folderError).toContain('folder_ingest_publish_failed')
  })

  it('shows the stable backend reason while preserving the folder preview', async () => {
    folderUpload.mockRejectedValue({
      isAxiosError: true,
      response: {
        data: {
          detail: {
            code: 'upload_filename_invalid',
            message: '上传文件名无效',
          },
        },
      },
    })
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    await act(async () => result.current.selectFolderFiles([folderUploadFile('root/finance/a.txt')]))
    await act(async () => result.current.confirmFolderUpload())

    await waitFor(() => expect(result.current.folderError).toContain('上传文件名无效'))
    expect(result.current.folderError).toContain('upload_filename_invalid')
    expect(result.current.folderPreview?.readyCount).toBe(1)
  })

  it('blocks confirmation while the preview contains an unsupported file', async () => {
    const { result } = renderHook(() => useDocuments())
    await waitFor(() => expect(listDocuments).toHaveBeenCalled())

    await act(async () => result.current.selectFolderFiles([
      folderUploadFile('root/finance/a.txt'), folderUploadFile('root/finance/b.exe'),
    ]))
    await act(async () => result.current.confirmFolderUpload())

    expect(folderUpload).not.toHaveBeenCalled()
    expect(result.current.folderError).toContain('请先移除')
  })
})
