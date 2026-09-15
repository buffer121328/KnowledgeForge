import { FolderOpenOutlined, InboxOutlined, ReloadOutlined } from '@ant-design/icons'
import { Alert, Button, Divider, Progress, Select, Space, Steps, Table, Tag, Typography, Upload } from 'antd'
import type { UploadProps } from 'antd'
import type {
  DepartmentItem,
  DocumentUploadRecord,
  FolderPreviewFile,
  FolderUploadPreview,
  IngestResponse,
  IngestProgressResponse,
} from '@/types'
import { DOCUMENT_UPLOAD_ACCEPT, MAX_DOCUMENT_UPLOAD_BATCH_FILES, folderPreviewStatusLabel } from '@/utils/documentUpload'
import { docStatusLabels, labelOf } from '@/utils/labels'
import { departmentOptionLabel } from '@/utils/departmentLabels'
import dayjs from 'dayjs'

const { Dragger } = Upload
const { Text } = Typography
const directoryInputAttributes = { webkitdirectory: '', directory: '' } as Record<string, string>

interface DocumentUploadProps {
  departmentMappings: Record<string, string>
  departments: DepartmentItem[]
  folderError: string
  folderPreparing: boolean
  folderPreview: FolderUploadPreview | null
  onConfirmFolderUpload: () => Promise<void>
  onDepartmentMappingChange: (folderKey: string, departmentId: string) => void
  onFilesSelected: (files: File[]) => void
  onFolderSelected: (files: File[]) => Promise<void>
  onRefresh: () => Promise<void>
  onRetry: (docId: string) => Promise<void>
  progress: number
  stageFailed: boolean
  stageIndex: number
  stageLabel: string
  uploadProgress: IngestProgressResponse | null
  uploadRejectedItems: IngestResponse[]
  recordsLoading: boolean
  retryingDocIds: string[]
  uploadHint: string
  uploading: boolean
  uploads: DocumentUploadRecord[]
}

/** Render flat and folder document ingestion with explicit preview/mapping confirmation. */
export function DocumentUpload({
  departmentMappings,
  departments,
  folderError,
  folderPreparing,
  folderPreview,
  onConfirmFolderUpload,
  onDepartmentMappingChange,
  onFilesSelected,
  onFolderSelected,
  onRefresh,
  onRetry,
  progress,
  stageFailed,
  stageIndex,
  stageLabel,
  uploadProgress,
  uploadRejectedItems,
  recordsLoading,
  retryingDocIds,
  uploadHint,
  uploading,
  uploads,
}: DocumentUploadProps) {
  /** Forward one complete flat selection and disable Ant Design's automatic upload. */
  const beforeUpload: UploadProps['beforeUpload'] = (file, fileList) => {
    if (fileList[0]?.uid === file.uid) onFilesSelected(fileList)
    return Upload.LIST_IGNORE
  }

  /** Forward the complete native folder selection while preserving webkitRelativePath. */
  const handleFolderInput = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const files = Array.from(event.target.files ?? [])
    if (files.length > 0) void onFolderSelected(files)
    event.target.value = ''
  }

  const previewColumns = [
    { title: '逻辑路径', dataIndex: 'relativePath', ellipsis: true },
    { title: '目录/部门', dataIndex: 'folderKey', width: 180 },
    {
      title: '状态', dataIndex: 'status', width: 220,
      /** Keep the short state and detailed reason inside a bounded wrapping cell. */
      render: (status: FolderPreviewFile['status'], row: FolderPreviewFile) => {
        const color = status === 'ready' ? 'green' : status === 'ignored' ? 'default' : 'red'
        return (
          <div className="upload-status-cell">
            <Tag color={color}>{folderPreviewStatusLabel(status)}</Tag>
            {row.reason ? <Text type={status === 'ignored' ? 'secondary' : 'danger'} className="upload-status-reason">{row.reason}</Text> : null}
          </div>
        )
      },
    },
  ]
  const uploadColumns = [
    { title: '文件名', dataIndex: 'fileName', ellipsis: true },
    { title: '逻辑路径', dataIndex: 'relativePath', ellipsis: true },
    { title: '部门', dataIndex: 'departmentId', width: 150 },
    { title: '分块数', dataIndex: 'chunksCount', width: 90 },
    { title: '实体数', dataIndex: 'entitiesCount', width: 90 },
    { title: '关系数', dataIndex: 'relationsCount', width: 90 },
    {
      title: '状态', dataIndex: 'status', width: 240,
      /** Keep ingestion errors readable without allowing tags to overflow the card. */
      render: (status: string, row: DocumentUploadRecord) => {
        const color = status === 'success' ? 'green' : status === 'processing' ? 'blue' : 'red'
        return (
          <div className="upload-status-cell">
            <Tag color={color}>{labelOf(docStatusLabels, status)}</Tag>
            {row.errorMessage ? <Text type="danger" className="upload-status-reason">{row.errorMessage}</Text> : null}
          </div>
        )
      },
    },
    {
      title: '上传时间', dataIndex: 'uploadedAt', width: 180,
      /** Render the server timestamp in the operator's local timezone. */
      render: (value: string) => value ? dayjs(value).format('YYYY-MM-DD HH:mm:ss') : '-',
    },
    {
      title: '操作', key: 'actions', width: 90,
      /** Render retry only when the server confirms a retained failed source is eligible. */
      render: (_value: unknown, row: DocumentUploadRecord) => row.retryable ? (
        <Button
          type="link"
          size="small"
          loading={retryingDocIds.includes(row.docId)}
          disabled={uploading}
          onClick={() => void onRetry(row.docId)}
        >
          重试
        </Button>
      ) : null,
    },
  ]
  const activeProgressItem = uploadProgress?.items
    ?.filter((item) => item.status !== 'success' && item.status !== 'failed')
    .sort((left, right) => left.processing_step_index - right.processing_step_index)[0]
    || uploadProgress?.items?.[0]

  return (
    <div>
      <Dragger multiple maxCount={MAX_DOCUMENT_UPLOAD_BATCH_FILES} accept={DOCUMENT_UPLOAD_ACCEPT}
        beforeUpload={beforeUpload} showUploadList={false} disabled={uploading}>
        <p className="ant-upload-drag-icon"><InboxOutlined /></p>
        <p className="ant-upload-text">点击或拖拽文件到此处上传</p>
        <p className="ant-upload-hint">支持 PDF、Word、Excel、CSV、图片、Markdown、纯文本、HTML；单次最多 50 份，单个最大 50MB</p>
      </Dragger>

      <Divider>或按文件夹上传</Divider>
      <Space direction="vertical" style={{ width: '100%' }} size={12}>
        <Button icon={<FolderOpenOutlined />} loading={folderPreparing} disabled={uploading}>
          <label style={{ cursor: 'pointer' }}>
            选择文件夹
            <input {...directoryInputAttributes} type="file" multiple hidden onChange={handleFolderInput} />
          </label>
        </Button>
        <Text type="secondary">浏览器会保留相对路径；仅经预览确认的文件会发送，物理服务器路径不会暴露。</Text>
        {folderError && <Alert type="error" showIcon message={folderError} />}
        {folderPreview && (
          <>
            <Alert
              type={folderPreview.issueCount > 0 ? 'warning' : 'success'}
              showIcon
              message={`根目录 ${folderPreview.rootFolderName}：可入库 ${folderPreview.readyCount}，需处理 ${folderPreview.issueCount}`}
              description={Object.entries(folderPreview.departmentCounts).map(([key, count]) => `${key}: ${count}`).join(' · ')}
            />
            <Space wrap>
              {Object.entries(folderPreview.departmentCounts).map(([folderKey, count]) => (
                <Space key={folderKey}>
                  <Text>{folderKey}（{count}）→</Text>
                  <Select
                    value={departmentMappings[folderKey]}
                    style={{ width: 220 }}
                    onChange={(value: string) => onDepartmentMappingChange(folderKey, value)}
                    options={[
                      ...departments.map((department) => ({ value: department.department_id, label: departmentOptionLabel(department) })),
                      ...(departments.some((department) => department.department_id === folderKey)
                        ? [] : [{ value: folderKey, label: `${folderKey}（新建）` }]),
                    ]}
                  />
                </Space>
              ))}
            </Space>
            <div className="folder-preview-table">
              <Table size="small" rowKey="clientFileId" columns={previewColumns}
                dataSource={folderPreview.files} pagination={{ pageSize: 8 }} scroll={{ x: 760 }} />
            </div>
            <Button type="primary"
              disabled={folderPreview.readyCount === 0 || folderPreview.issueCount > 0 || uploading}
              onClick={() => void onConfirmFolderUpload()}>
              确认映射并上传 {folderPreview.readyCount} 份文件
            </Button>
          </>
        )}
      </Space>

      {uploading && (
        <div style={{ marginTop: 16 }}>
          <Steps
            size="small"
            current={Math.max(0, Math.min(stageIndex, 4) - 1)}
            status={stageFailed ? 'error' : stageIndex >= 4 ? 'finish' : 'process'}
            items={[
              { title: '上传校验与落盘' },
              { title: '文件已接收' },
              { title: '解析与知识抽取' },
              { title: '入库完成' },
            ]}
          />
          <Progress percent={progress} status={stageFailed ? 'exception' : 'active'} />
          <Text type={stageFailed ? 'danger' : 'secondary'} style={{ display: 'block', marginTop: 8 }}>
            {stageLabel || uploadHint || '处理中…'}
          </Text>
          {uploadProgress?.items?.length ? (
            <div style={{ marginTop: 12 }}>
              <Alert
                showIcon
                type={stageFailed ? 'error' : uploadRejectedItems.length > 0 ? 'warning' : 'info'}
                message={`批次进度：已完成 ${uploadProgress.completed_count}/${uploadProgress.total_count}，后台失败 ${uploadProgress.failed_count}${uploadRejectedItems.length > 0 ? `，未接收 ${uploadRejectedItems.length}` : ''}`}
                description={`当前文件：${activeProgressItem?.file_name || '—'} · ${activeProgressItem?.processing_step_label || uploadProgress.processing_step_label}`}
              />
              {uploadRejectedItems.length > 0 ? (
                <Alert
                  showIcon
                  type="warning"
                  style={{ marginTop: 8 }}
                  message={`有 ${uploadRejectedItems.length} 份文件未进入后台处理`}
                  description={uploadRejectedItems.map((item) => `${item.file_name}：${item.message || item.error_code || '上传校验或目录冲突'}`).join('；')}
                />
              ) : null}
              <div className="upload-progress-list">
                <Table
                  size="small"
                  pagination={false}
                  rowKey={(row) => row.doc_id || row.client_file_id || row.file_name}
                  dataSource={uploadProgress.items}
                  columns={[
                    { title: '文件', dataIndex: 'file_name', ellipsis: true },
                    { title: '当前步骤', dataIndex: 'ingest_stage_label', width: 180 },
                    { title: '内部步骤', dataIndex: 'processing_step_label', width: 180 },
                    { title: '阶段', dataIndex: 'ingest_stage_index', width: 90, render: (value: number) => `${value}/${uploadProgress.stage_total}` },
                    { title: '状态', dataIndex: 'status', width: 100, render: (value: string) => <Tag color={value === 'failed' ? 'red' : value === 'success' ? 'green' : 'blue'}>{value === 'failed' ? '失败' : value === 'success' ? '完成' : '处理中'}</Tag> },
                  ]}
                />
              </div>
            </div>
          ) : null}
        </div>
      )}
      <Space style={{ marginTop: 16, marginBottom: 8 }}>
        <Button icon={<ReloadOutlined />} loading={recordsLoading} onClick={() => void onRefresh()}>
          刷新上传记录
        </Button>
        <Text type="secondary">记录与服务器文档目录同步；失败且源文件仍保留时可直接重试。</Text>
      </Space>
      <Table columns={uploadColumns} dataSource={uploads} loading={recordsLoading}
        pagination={{ pageSize: 10 }} size="small" scroll={{ x: 1280 }} />
    </div>
  )
}
