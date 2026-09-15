import { DeleteOutlined, EyeOutlined, ReloadOutlined } from '@ant-design/icons'
import { Button, Card, Drawer, Empty, Popconfirm, Select, Space, Spin, Table, Tag, Typography } from 'antd'
import type { TableColumnsType } from 'antd'
import type { ChunkItem, DepartmentItem, DocListItem } from '@/types'
import { useAuthStore } from '@/stores/auth'
import { extractionAnomalyDescription } from '@/utils/documentExtractionHealth'
import { departmentDisplayName, departmentOptionLabel } from '@/utils/departmentLabels'

const { Paragraph, Text } = Typography

interface DocumentTableProps {
  activeDoc: DocListItem | null
  chunks: ChunkItem[]
  chunksLoading: boolean
  docs: DocListItem[]
  departments: DepartmentItem[]
  selectedDepartmentId?: string
  drawerOpen: boolean
  loading: boolean
  onCloseChunks: () => void
  onDelete: (doc: DocListItem) => Promise<void>
  onOpenDocument: (doc: DocListItem) => Promise<void>
  onOpenChunks: (doc: DocListItem) => Promise<void>
  onRefresh: () => Promise<void>
  onDepartmentChange: (departmentId?: string) => void
}

/** Render the document table component. */
export function DocumentTable({
  activeDoc,
  chunks,
  chunksLoading,
  docs,
  departments,
  selectedDepartmentId,
  drawerOpen,
  loading,
  onCloseChunks,
  onDelete,
  onOpenDocument,
  onOpenChunks,
  onRefresh,
  onDepartmentChange,
}: DocumentTableProps) {
  const currentUser = useAuthStore((state) => state.user)
  const departmentNames = new Map(
    departments.map((department) => [department.department_id, department.name]),
  )
  const canDelete = (row: DocListItem) => {
    if (!currentUser?.permissions.includes('doc:delete')) return false
    if (currentUser.role === 'organization_admin') return true
    if (currentUser.role !== 'admin') return false
    return Boolean(currentUser.department_id && row.department_id === currentUser.department_id)
  }
  const docColumns: TableColumnsType<DocListItem> = [
    {
      title: '显示名 / 来源名',
      key: 'names',
      width: 220,
      /** Preserve catalog display and provenance names instead of storage-derived basenames. */
      render: (_value: unknown, row: DocListItem) => (
        <Space direction="vertical" size={0}>
          <Text>{row.display_name || row.file_name}</Text>
          <Text type="secondary">{row.provenance_source_filename || row.uploaded_filename || '来源名不可恢复'}</Text>
        </Space>
      ),
    },
    { title: '逻辑路径', dataIndex: 'relative_path', ellipsis: true, width: 260 },
    {
      title: '部门', dataIndex: 'department_id', width: 150,
      render: (departmentId: string) => departmentDisplayName(
        departmentId,
        departmentNames.get(departmentId),
      ),
    },
    { title: '版本', dataIndex: 'version', width: 70 },
    {
      title: '状态', dataIndex: 'ingest_status', width: 110,
      /** Render durable lifecycle state and a separate safe extraction-health indicator. */
      render: (status: string, row: DocListItem) => {
        const anomalyDescription = extractionAnomalyDescription(row)
        return (
          <Space size={[4, 4]} wrap>
            <Tag color={row.legacy ? 'default' : status === 'ingested' ? 'green' : 'blue'}>
              {row.legacy ? 'legacy' : status || 'unknown'}
            </Tag>
            {anomalyDescription && (
              <Tag color="warning" title={`抽取异常：${anomalyDescription}`}>
                抽取异常：{anomalyDescription}
              </Tag>
            )}
          </Space>
        )
      },
    },
    { title: '分块', dataIndex: 'chunks_count', width: 70 },
    { title: '实体', dataIndex: 'entities_count', width: 70 },
    { title: '关系', dataIndex: 'relations_count', width: 70 },
    {
      title: '操作', width: 200, fixed: 'right',
      /** Render document inspection and evidence-aware deletion actions. */
      render: (_value: unknown, row: DocListItem) => (
        <Space>
          <Button
            type="link"
            disabled={!row.file_reference}
            onClick={() => void onOpenDocument(row)}
          >
            打开文档
          </Button>
          <Button type="link" icon={<EyeOutlined />} onClick={() => void onOpenChunks(row)}>查看分块</Button>
          {currentUser?.permissions.includes('doc:delete') && <Popconfirm title="确认删除该文档？" description="将删除向量分块、该文档图谱证据及本地源文件"
            disabled={!canDelete(row)}
            okText="删除" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => void onDelete(row)}>
            <Button
              type="link"
              danger
              disabled={!canDelete(row)}
              title={!canDelete(row) ? '部门负责人只能删除本部门文档' : undefined}
              icon={<DeleteOutlined />}
            >删除</Button>
          </Popconfirm>}
        </Space>
      ),
    },
  ]

  return (
    <>
      <div>
        <Space style={{ marginBottom: 16 }} wrap>
          <Button icon={<ReloadOutlined />} onClick={() => void onRefresh()} loading={loading}>
            刷新列表
          </Button>
          <Select
            allowClear
            placeholder="全部部门"
            value={selectedDepartmentId}
            onChange={onDepartmentChange}
            style={{ width: 180 }}
            options={departments.map((department) => ({ label: departmentOptionLabel(department), value: department.department_id }))}
          />
          <Text type="secondary">共 {docs.length} 份文档</Text>
        </Space>
        <Table
          columns={docColumns}
          dataSource={docs}
          rowKey="doc_id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1450 }}
          size="small"
        />
      </div>

      <Drawer
        title={activeDoc ? `分块内容 — ${activeDoc.display_name || activeDoc.file_name}` : '分块内容'}
        open={drawerOpen}
        onClose={onCloseChunks}
        width={720}
        destroyOnClose
      >
        {chunksLoading ? (
          <div style={{ textAlign: 'center', padding: 48 }}>
            <Spin />
          </div>
        ) : chunks.length === 0 ? (
          <Empty description="暂无分块" />
        ) : (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <Text type="secondary">
              文档ID: {activeDoc?.doc_id} · 共 {chunks.length} 个分块
            </Text>
            {chunks.map((chunk) => (
              <Card
                key={chunk.chunk_id}
                size="small"
                title={`分块 #${chunk.chunk_index}`}
                extra={<Text type="secondary">{chunk.chunk_id}</Text>}
              >
                <Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
                  {chunk.content}
                </Paragraph>
              </Card>
            ))}
          </Space>
        )}
      </Drawer>
    </>
  )
}
