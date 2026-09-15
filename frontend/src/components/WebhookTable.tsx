import { DeleteOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { Button, Popconfirm, Space, Table, Tag, Tooltip } from 'antd'
import type { TableColumnsType } from 'antd'
import dayjs from 'dayjs'
import type { Webhook } from '@/types'
import { labelOf, webhookEventLabels } from '@/utils/labels'

interface WebhookTableProps {
  loading: boolean
  onDelete: (id: string) => Promise<void>
  onTest: (id: string) => Promise<void>
  webhooks: Webhook[]
}

/** Render the webhook table component. */
export function WebhookTable({ loading, onDelete, onTest, webhooks }: WebhookTableProps) {
  const columns: TableColumnsType<Webhook> = [
    {
      title: 'URL',
      dataIndex: 'url',
      ellipsis: true,
      /** Render the table cell for this column. */
      render: (url: string) => (
        <Tooltip title={url}>
          <a href={url} target="_blank" rel="noreferrer">
            {url}
          </a>
        </Tooltip>
      ),
    },
    {
      title: '订阅事件',
      dataIndex: 'events',
      width: 280,
      /** Render the table cell for this column. */
      render: (events: string[]) => (
        <Space size={4} wrap>
          {(events || []).map((event) => (
            <Tag key={event} color="blue">
              {labelOf(webhookEventLabels, event)}
            </Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      width: 80,
      /** Render the table cell for this column. */
      render: (active: boolean) => (
        <Tag color={active ? 'green' : 'default'}>{active ? '启用' : '停用'}</Tag>
      ),
    },
    {
      title: '失败次数',
      dataIndex: 'failure_count',
      width: 90,
      /** Render the table cell for this column. */
      render: (count: number) => (
        <Tag color={count > 5 ? 'red' : count > 0 ? 'orange' : 'default'}>{count ?? 0}</Tag>
      ),
    },
    {
      title: '最近状态码',
      dataIndex: 'last_response_code',
      width: 100,
      /** Render the table cell for this column. */
      render: (code: number | null | undefined) =>
        code == null ? '-' : (
          <Tag color={code >= 200 && code < 300 ? 'green' : 'red'}>{code}</Tag>
        ),
    },
    {
      title: '最近触发',
      dataIndex: 'last_triggered_at',
      width: 180,
      /** Render the table cell for this column. */
      render: (timestamp: string | null) =>
        timestamp ? dayjs(timestamp).format('YYYY-MM-DD HH:mm:ss') : '-',
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      width: 180,
      /** Render the table cell for this column. */
      render: (timestamp: string) =>
        timestamp ? dayjs(timestamp).format('YYYY-MM-DD HH:mm:ss') : '-',
    },
    {
      title: '操作',
      width: 160,
      /** Render the table cell for this column. */
      render: (_: unknown, record: Webhook) => (
        <Space size={4}>
          <Tooltip title="发送测试事件">
            <Button
              size="small"
              icon={<ThunderboltOutlined />}
              onClick={() => onTest(record.id)}
            />
          </Tooltip>
          <Popconfirm title="确定删除该 Webhook？" onConfirm={() => onDelete(record.id)}>
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <Table
      columns={columns}
      dataSource={webhooks}
      rowKey="id"
      loading={loading}
      pagination={{ pageSize: 10, showSizeChanger: true }}
      size="middle"
      scroll={{ x: 1200 }}
      locale={{ emptyText: '暂无 Webhook，点击右上角新建' }}
    />
  )
}
