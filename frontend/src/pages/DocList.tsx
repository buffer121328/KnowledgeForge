import { ReloadOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { Alert, Button, Card, Empty, Input, Select, Space, Table, Tabs, Tag, Typography } from 'antd'
import { DocumentTable } from '@/components/DocumentTable'
import { DocumentUpload } from '@/components/DocumentUpload'
import type { TaskStatus } from '@/types'
import { taskStatusLabels, labelOf } from '@/utils/labels'
import { useAuthStore } from '@/stores/auth'
import { type DocumentChangeType, useDocuments } from './useDocuments'

const { Paragraph, Text } = Typography

const changeTypeOptions: Array<{ label: string; value: DocumentChangeType }> = [
  { value: 'modified', label: 'modified — 文件已修改，增量重建' },
  { value: 'created', label: 'created — 新文件，重新入库' },
  { value: 'deleted', label: 'deleted — 文件已删除，清理索引' },
]

export default function DocList() {
  const {
    activeDoc,
    cancelTask,
    changeType,
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
  } = useDocuments()
  const currentUser = useAuthStore((state) => state.user)
  const canWrite = Boolean(currentUser?.permissions.includes('doc:write'))

  const taskColumns = [
    { title: '任务描述', dataIndex: 'description', ellipsis: true, render: (value?: string) => value || '后台任务' },
    {
      title: '状态',
      dataIndex: 'status',
      render: (status: string) => {
        const color = status === 'SUCCESS' ? 'green' : status === 'FAILURE' ? 'red' : 'blue'
        return <Tag color={color}>{labelOf(taskStatusLabels, status)}</Tag>
      },
    },
    {
      title: '就绪',
      dataIndex: 'ready',
      render: (ready: boolean) => (ready ? '✓' : '⌛'),
    },
    {
      title: '操作',
      render: (_: unknown, task: TaskStatus) => (
        <Button size="small" danger onClick={() => void cancelTask(task)}>
          取消
        </Button>
      ),
    },
  ]

  return (
    <Card title="文档管理">
      <Tabs
        items={[
          {
            key: 'library',
            label: '已入库文档',
            children: (
              <DocumentTable
                activeDoc={activeDoc}
                chunks={chunks}
                chunksLoading={chunksLoading}
                docs={docs}
                departments={departments}
                selectedDepartmentId={selectedDepartmentId}
                drawerOpen={chunkDrawerOpen}
                loading={docsLoading}
                onCloseChunks={closeChunks}
                onDelete={handleDelete}
                onOpenDocument={openDocument}
                onOpenChunks={openChunks}
                onRefresh={loadDocs}
                onDepartmentChange={setSelectedDepartmentId}
              />
            ),
          },
          ...(canWrite ? [{
            key: 'upload',
            label: '文档上传',
            children: (
              <DocumentUpload
                departmentMappings={departmentMappings}
                departments={departments}
                folderError={folderError}
                folderPreparing={folderPreparing}
                folderPreview={folderPreview}
                onConfirmFolderUpload={confirmFolderUpload}
                onDepartmentMappingChange={setDepartmentMapping}
                onFilesSelected={selectUploadFiles}
                onFolderSelected={selectFolderFiles}
                onRefresh={loadDocs}
                onRetry={retryUpload}
                progress={progress}
                stageFailed={stageFailed}
                stageIndex={stageIndex}
                stageLabel={stageLabel}
                uploadProgress={uploadProgress}
                uploadRejectedItems={uploadRejectedItems}
                recordsLoading={docsLoading}
                retryingDocIds={retryingDocIds}
                uploadHint={uploadHint}
                uploading={uploading}
                uploads={uploads}
              />
            ),
          },
          {
            key: 'tasks',
            label: '后台增量更新',
            children: (
              <div>
                <Alert
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                  message="后台增量更新（异步任务）是什么？"
                  description={
                    <div>
                      <Paragraph style={{ marginBottom: 8 }}>
                        用于在后台（Celery）对<strong>租户已入库文件</strong>做增量更新：重新解析、刷新向量库和知识图谱。
                        任务只接受后端生成的文件引用，不接受操作系统绝对路径。
                      </Paragraph>
                      <Paragraph type="warning" style={{ marginBottom: 8 }}>
                        上传入库失败不需要使用这里：请回到「文档上传」，在失败记录右侧点击「重试」。
                      </Paragraph>
                      <Paragraph type="secondary" style={{ marginBottom: 4 }}>
                        文件引用由「已入库文档」生成，例如：
                      </Paragraph>
                      <Space wrap>
                        <Tag
                          style={{ cursor: 'pointer' }}
                          onClick={() => fillExamplePath('0123456789abcdef.docx')}
                        >
                          0123456789abcdef.docx
                        </Tag>
                        <Tag
                          style={{ cursor: 'pointer' }}
                          onClick={() => fillExamplePath('abcdef0123456789.pdf')}
                        >
                          abcdef0123456789.pdf
                        </Tag>
                      </Space>
                      <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
                        推荐：从下方下拉框选择「已入库文档」，会自动填入该文档的安全文件引用。
                        若任务列表一直为空，通常是 Celery Worker / Redis 未启动。
                      </Paragraph>
                    </div>
                  }
                />

                <Card size="small" title="提交知识更新任务" style={{ marginBottom: 16 }}>
                  <Space direction="vertical" style={{ width: '100%' }} size={12}>
                    <div>
                      <Text strong>1. 选择已入库文档（推荐）</Text>
                      <Select
                        allowClear
                        showSearch
                        placeholder="从已入库文档中选择，自动填入安全文件引用"
                        style={{ width: '100%', marginTop: 8 }}
                        optionFilterProp="label"
                        options={docs.map((doc) => ({
                          value: doc.file_reference,
                          label: `${doc.file_name}（${doc.file_reference || "不可异步更新"}）`,
                          disabled: !doc.file_reference,
                        }))}
                        onChange={(value: string | undefined) => {
                          if (value) setFilePath(value)
                        }}
                      />
                    </div>

                    <div>
                      <Text strong>2. 或手动填写文件引用</Text>
                      <Input
                        placeholder="例如：0b1eefe0_short_word_test.docx"
                        value={filePath}
                        onChange={(event) => setFilePath(event.target.value)}
                        style={{ marginTop: 8 }}
                        addonBefore="file_reference"
                      />
                    </div>

                    <div>
                      <Text strong>3. 变更类型</Text>
                      <Select<DocumentChangeType>
                        value={changeType}
                        style={{ width: 220, marginTop: 8, display: 'block' }}
                        onChange={setChangeType}
                        options={changeTypeOptions}
                      />
                    </div>

                    <Space>
                      <Button
                        type="primary"
                        icon={<ThunderboltOutlined />}
                        loading={taskSubmitting}
                        onClick={() => void submitUpdateTask()}
                      >
                        提交异步更新
                      </Button>
                      <Button icon={<ReloadOutlined />} onClick={() => void refreshTasks()}>
                        刷新任务列表
                      </Button>
                    </Space>
                  </Space>
                </Card>

                <Table
                  columns={taskColumns}
                  dataSource={tasks}
                  rowKey="task_id"
                  pagination={{ pageSize: 10 }}
                  size="small"
                  locale={{
                    emptyText: (
                      <Empty
                        description={
                          <span>
                            暂无运行中的任务
                            <br />
                            <Text type="secondary">
                              需启动 Redis + Celery Worker 后才会出现任务记录
                            </Text>
                          </span>
                        }
                      />
                    ),
                  }}
                />
              </div>
            ),
          }] : []),
        ]}
      />
    </Card>
  )
}
