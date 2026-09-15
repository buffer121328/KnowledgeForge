import { useEffect, useState } from 'react'
import { Card, Table, Tag, Input, Space, Button, Select, App, DatePicker } from 'antd'
import { ReloadOutlined, DownloadOutlined, SearchOutlined } from '@ant-design/icons'
import dayjs, { type Dayjs } from 'dayjs'
import { adminApi, type AuditLogFilters } from '@/api/admin'
import type { AuditLog } from '@/types'
import { auditActionLabels, auditResultLabels, labelOf } from '@/utils/labels'

const { RangePicker } = DatePicker
export type AuditDateRange = [Dayjs | null, Dayjs | null] | null

/** Normalize the page controls into the shared audit list/export query contract. */
export function buildAuditLogFilters(
  userId: string | undefined,
  action: string | undefined,
  dateRange: AuditDateRange,
): AuditLogFilters {
  const normalizedUserId = userId?.trim()
  return {
    ...(normalizedUserId ? { user_id: normalizedUserId } : {}),
    ...(action ? { action } : {}),
    ...(dateRange?.[0] ? { start: dateRange[0].toISOString() } : {}),
    ...(dateRange?.[1] ? { end: dateRange[1].toISOString() } : {}),
  }
}

/** Render the audit logs page. */
export default function AuditLogs() {
  const { message } = App.useApp()
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [loading, setLoading] = useState(false)
  const [userId, setUserId] = useState<string | undefined>()
  const [action, setAction] = useState<string | undefined>()
  const [dateRange, setDateRange] = useState<AuditDateRange>(null)
  const [actions, setActions] = useState<string[]>([])

  /** Fetch audit logs using the active filters. */
  const fetchLogs = async () => {
    setLoading(true)
    try {
      const filters = buildAuditLogFilters(userId, action, dateRange)
      const data = await adminApi.auditLogs({ ...filters, limit: 500 })
      setLogs(data)
    } catch {
      message.error('加载审计日志失败')
    } finally {
      setLoading(false)
    }
  }

  /** Fetch the available audit action filters. */
  const fetchActions = async () => {
    try {
      const data = await adminApi.auditActions()
      setActions(data.actions ?? [])
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    void fetchLogs()
    void fetchActions()
  }, [])

  /** Export audit logs using the active filters. */
  const handleExport = async () => {
    try {
      const filters = buildAuditLogFilters(userId, action, dateRange)
      const blob = await adminApi.exportAuditLogs({ ...filters, limit: 10000 })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `audit_logs_${dayjs().format('YYYYMMDD_HHmmss')}.csv`
      a.click()
      URL.revokeObjectURL(url)
      message.success('导出成功')
    } catch {
      message.error('导出失败')
    }
  }

  const columns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      width: 180,
      /** Render the table cell for this column. */
      render: (ts: string) => dayjs(ts).format('YYYY-MM-DD HH:mm:ss'),
    },
    { title: '用户', dataIndex: 'username', width: 120 },
    { title: '用户ID', dataIndex: 'user_id', width: 120, ellipsis: true },
    {
      title: '动作',
      dataIndex: 'action',
      width: 140,
      /** Render the table cell for this column. */
      render: (value: string) => <Tag color="blue">{labelOf(auditActionLabels, value)}</Tag>,
    },
    { title: '资源', dataIndex: 'resource', ellipsis: true },
    {
      title: '结果',
      dataIndex: 'result',
      width: 90,
      /** Render the table cell for this column. */
      render: (result: string) => (
        <Tag color={result === 'success' ? 'green' : 'red'}>
          {labelOf(auditResultLabels, result)}
        </Tag>
      ),
    },
    { title: 'IP', dataIndex: 'ip', width: 140 },
    { title: '组织', dataIndex: 'org_id', width: 120 },
    {
      title: '哈希校验',
      width: 100,
      /** Render the table cell for this column. */
      render: (_: unknown, record: AuditLog) => (
        <Tag color={record.hash ? 'cyan' : 'default'}>
          {record.hash ? record.hash.slice(0, 8) : '-'}
        </Tag>
      ),
    },
  ]

  return (
    <Card
      title="审计日志"
      extra={
        <Space>
          <Button icon={<DownloadOutlined />} onClick={() => void handleExport()}>
            导出 CSV
          </Button>
          <Button type="primary" icon={<ReloadOutlined />} onClick={() => void fetchLogs()}>
            刷新
          </Button>
        </Space>
      }
    >
      <Space style={{ marginBottom: 16 }} wrap>
        <Input
          aria-label="按用户ID过滤"
          placeholder="按用户ID过滤"
          value={userId}
          onChange={(event) => setUserId(event.target.value)}
          onPressEnter={() => void fetchLogs()}
          style={{ width: 200 }}
          allowClear
        />
        <Select
          aria-label="按动作过滤"
          placeholder="按动作过滤"
          value={action}
          onChange={setAction}
          style={{ width: 180 }}
          allowClear
          options={actions.map((value) => ({ value, label: labelOf(auditActionLabels, value) }))}
        />
        <RangePicker
          showTime
          value={dateRange}
          onChange={(value) => setDateRange(value as AuditDateRange)}
        />
        <Button icon={<SearchOutlined />} onClick={() => void fetchLogs()}>
          查询
        </Button>
      </Space>

      <Table
        columns={columns}
        dataSource={logs}
        rowKey={(record) => `${record.timestamp}-${record.user_id}-${record.action}`}
        loading={loading}
        pagination={{
          pageSize: 20,
          showSizeChanger: true,
          /** Format the pagination total summary. */
          showTotal: (total) => `共 ${total} 条`,
        }}
        size="small"
        scroll={{ x: 1200 }}
      />
    </Card>
  )
}
