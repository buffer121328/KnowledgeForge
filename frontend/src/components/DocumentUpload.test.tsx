// @vitest-environment jsdom
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { DocumentUpload } from './DocumentUpload'

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }),
})

const progress = {
  upload_id: 'upload-1',
  total_count: 20,
  completed_count: 5,
  failed_count: 0,
  stage_index: 3,
  stage_total: 4,
  stage_label: '解析与知识抽取',
  processing_step: 'extract',
  processing_step_index: 2,
  processing_step_total: 6,
  processing_step_label: '知识抽取',
  terminal: false,
  status: 'processing' as const,
  items: [{
    doc_id: 'doc-1',
    client_file_id: 'file-1',
    file_name: '已接收.docx',
    status: 'processing',
    ingest_stage: 'processing',
    ingest_stage_index: 3,
    ingest_stage_total: 4,
    ingest_stage_label: '解析与知识抽取',
    processing_step: 'extract',
    processing_step_index: 2,
    processing_step_total: 6,
    processing_step_label: '知识抽取',
    error_code: '',
    message: '',
  }],
}

describe('DocumentUpload partial acceptance state', () => {
  it('keeps active ingestion blue while showing rejected files separately', () => {
    render(
      <DocumentUpload
        departmentMappings={{}}
        departments={[]}
        folderError=""
        folderPreparing={false}
        folderPreview={null}
        onConfirmFolderUpload={vi.fn(async () => undefined)}
        onDepartmentMappingChange={vi.fn()}
        onFilesSelected={vi.fn()}
        onFolderSelected={vi.fn(async () => undefined)}
        onRefresh={vi.fn(async () => undefined)}
        onRetry={vi.fn(async () => undefined)}
        progress={75}
        stageFailed={false}
        stageIndex={3}
        stageLabel="解析与知识抽取"
        uploadProgress={progress}
        uploadRejectedItems={[{
          file_name: '重复.docx',
          chunks_count: 0,
          entities_count: 0,
          relations_count: 0,
          status: 'failed',
          error_code: 'document_catalog_conflict',
          message: '文档目录冲突',
        }]}
        recordsLoading={false}
        retryingDocIds={[]}
        uploadHint="已接收 20 份，另有 1 份未接收，后台处理中"
        uploading
        uploads={[]}
      />,
    )

    expect(screen.getByText('有 1 份文件未进入后台处理')).toBeTruthy()
    expect(screen.getByText(/重复\.docx：文档目录冲突/)).toBeTruthy()
    expect(screen.getByText(/后台失败 0，未接收 1/)).toBeTruthy()
    expect(document.querySelector('.ant-progress-status-exception')).toBeNull()
    expect(document.querySelector('.ant-progress-status-active')).toBeTruthy()
  })
})
