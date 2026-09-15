import { useCallback, useEffect, useMemo, useState } from 'react'
import { Alert, Button, Card, Col, Row, Segmented, Space, Statistic, Table, Spin, Tag, Typography } from 'antd'
import {
  DatabaseOutlined,
  ApartmentOutlined,
  FileTextOutlined,
  TeamOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area,
} from 'recharts'
import { adminApi, taskApi } from '@/api/admin'
import { useAuthStore } from '@/stores/auth'
import type { ReadinessReport, RequestTrendResponse, SystemStats, TaskStatus } from '@/types'
import { labelOf, taskStatusLabels } from '@/utils/labels'

const { Title } = Typography
const LIVE_REFRESH_MS = 15_000

/** Render the organization-governance system dashboard. */
export default function Dashboard() {
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [tasks, setTasks] = useState<TaskStatus[]>([])
  const [readiness, setReadiness] = useState<ReadinessReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [trend, setTrend] = useState<RequestTrendResponse | null>(null)
  const [trendWindow, setTrendWindow] = useState<'60m' | '24h'>('60m')
  const [trendLoading, setTrendLoading] = useState(false)
  const [trendError, setTrendError] = useState('')
  const { user } = useAuthStore()

  const loadDashboard = useCallback(async (): Promise<void> => {
    setLoading(true)
    try {
      const [systemStats, recentTasks, health] = await Promise.all([
        adminApi.stats(),
        taskApi.list(10),
        adminApi.readiness(),
      ])
      setStats(systemStats)
      setTasks(recentTasks)
      setReadiness(health)
    } finally {
      setLoading(false)
    }
  }, [])

  const loadTrend = useCallback(async (): Promise<void> => {
    setTrendLoading(true)
    try {
      setTrend(await adminApi.requestTrends(trendWindow))
      setTrendError('')
    } catch {
      setTrendError('实时趋势暂时更新失败，图表保留最近一次成功数据。')
    } finally {
      setTrendLoading(false)
    }
  }, [trendWindow])

  useEffect(() => {
    void loadDashboard()
  }, [loadDashboard])

  useEffect(() => {
    void loadTrend()
    const interval = window.setInterval(() => {
      if (document.visibilityState === 'visible') void loadTrend()
    }, LIVE_REFRESH_MS)
    const refreshWhenVisible = (): void => {
      if (document.visibilityState === 'visible') void loadTrend()
    }
    document.addEventListener('visibilitychange', refreshWhenVisible)
    return () => {
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [loadTrend])

  const trendData = useMemo(() => (trend?.points ?? []).map((point) => ({
    ...point,
    label: new Intl.DateTimeFormat(undefined, trendWindow === '60m'
      ? { hour: '2-digit', minute: '2-digit' }
      : { month: '2-digit', day: '2-digit', hour: '2-digit' }).format(new Date(point.timestamp)),
  })), [trend, trendWindow])
  const requestDetails = trend?.summary.details ?? { ai: {}, system_api: {} }
  const requestRows = (['ai', 'system_api'] as const).map((key) => {
    const detail = requestDetails[key] ?? {}
    const count = detail.count ?? 0
    return {
      key,
      category: key === 'ai' ? 'AI / 智能问答请求' : '系统 API 请求',
      count,
      errors: detail.error_count ?? 0,
      averageLatency: count ? Math.round((detail.total_latency_ms ?? 0) / count) : 0,
      methods: Object.entries(detail.methods ?? {}).map(([method, value]) => `${method} ${value}`).join(' · ') || '-',
      routes: Object.entries(detail.routes ?? {})
        .sort((left, right) => right[1].count - left[1].count)
        .slice(0, 3)
        .map(([route, value]) => `${route} (${value.count})`)
        .join('；') || '-',
    }
  })

  if (loading) {
    return <div style={{ display: 'flex', justifyContent: 'center', padding: 100 }}><Spin size="large" /></div>
  }

  return (
    <div>
      <Title level={4} style={{ marginBottom: 24 }}>
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <span>欢迎回来, {user?.username}</span>
          <Button
            aria-label="刷新统计"
            icon={<ReloadOutlined />}
            onClick={() => void Promise.allSettled([loadDashboard(), loadTrend()])}
            loading={loading || trendLoading}
          >
            刷新统计
          </Button>
        </Space>
      </Title>

      <Row gutter={[16, 16]}>
        <Col xs={12} md={6}><Card><Statistic title="向量总数" value={stats?.vector_store?.total_vectors ?? 0} prefix={<DatabaseOutlined />} valueStyle={{ color: '#1677ff' }} /></Card></Col>
        <Col xs={12} md={6}><Card><Statistic title="图谱节点" value={stats?.knowledge_graph?.nodes ?? 0} prefix={<ApartmentOutlined />} valueStyle={{ color: '#52c41a' }} /></Card></Col>
        <Col xs={12} md={6}><Card><Statistic title="图谱关系" value={stats?.knowledge_graph?.edges ?? 0} prefix={<ApartmentOutlined />} valueStyle={{ color: '#722ed1' }} /></Card></Col>
        <Col xs={12} md={6}><Card><Statistic title="活跃任务" value={tasks.length} prefix={<FileTextOutlined />} valueStyle={{ color: '#faad14' }} /></Card></Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={16}>
          <Card
            title={`API 请求趋势（${trendWindow === '60m' ? '最近 60 分钟' : '最近 24 小时'}）`}
            loading={trendLoading && !trend}
            extra={<Segmented options={[{ label: '60 分钟', value: '60m' }, { label: '24 小时', value: '24h' }]} value={trendWindow} onChange={(value) => setTrendWindow(value as '60m' | '24h')} />}
          >
            {trendError && <Alert type="warning" showIcon message={trendError} style={{ marginBottom: 12 }} />}
            <Row gutter={12} style={{ marginBottom: 12 }}>
              <Col span={8}><Statistic title="请求数" value={trend?.summary.requests ?? 0} /></Col>
              <Col span={8}><Statistic title="失败数" value={trend?.summary.errors ?? 0} valueStyle={{ color: '#ff4d4f' }} /></Col>
              <Col span={8}><Statistic title="平均耗时" value={trend?.summary.average_latency_ms ?? 0} suffix="ms" precision={1} /></Col>
            </Row>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="colorReq" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#1677ff" stopOpacity={0.6} /><stop offset="95%" stopColor="#1677ff" stopOpacity={0} /></linearGradient>
                  <linearGradient id="colorQa" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#52c41a" stopOpacity={0.5} /><stop offset="95%" stopColor="#52c41a" stopOpacity={0} /></linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" minTickGap={28} />
                <YAxis allowDecimals={false} />
                <Tooltip labelFormatter={(_label, payload) => payload?.[0]?.payload?.timestamp ?? _label} />
                <Area type="monotone" dataKey="requests" stroke="#1677ff" fill="url(#colorReq)" name="API 请求" />
                <Area type="monotone" dataKey="qa" stroke="#52c41a" fill="url(#colorQa)" name="问答请求" />
                <Area type="monotone" dataKey="errors" stroke="#ff4d4f" fillOpacity={0} name="失败请求" />
              </AreaChart>
            </ResponsiveContainer>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              组织内已认证 API 请求 · 源时间 UTC · 本地时间展示 · 页面可见时每 15 秒刷新
              {trend?.generated_at ? ` · 最近更新 ${new Date(trend.generated_at).toLocaleTimeString()}` : ''}
            </Typography.Text>
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="系统状态">
            <Statistic title="服务状态" value={readiness?.status === 'ready' ? '已就绪' : '未就绪'} valueStyle={{ color: readiness?.status === 'ready' ? '#52c41a' : '#ff4d4f' }} prefix={<TeamOutlined />} />
            <div style={{ marginTop: 16 }}>
              <Tag color="green">API 在线</Tag>
              <Tag color={readiness?.components.vector_store === 'ready' ? 'green' : 'red'}>向量库</Tag>
              <Tag color={readiness?.components.knowledge_graph === 'ready' ? 'green' : 'red'}>知识图谱</Tag>
              <Tag color={readiness?.components.security_state === 'ready' ? 'green' : 'red'}>安全状态</Tag>
              <Tag color={readiness?.components.task_broker === 'ready' ? 'green' : 'red'}>任务队列</Tag>
            </div>
          </Card>
        </Col>
      </Row>

      <Card title={`API 请求明细（${trendWindow === '60m' ? '最近 60 分钟' : '最近 24 小时'}）`} style={{ marginTop: 16 }}>
        <Table dataSource={requestRows} rowKey="key" pagination={false} size="small" columns={[
          { title: '请求类别', dataIndex: 'category' },
          { title: '请求数', dataIndex: 'count', width: 100 },
          { title: '失败数', dataIndex: 'errors', width: 100 },
          { title: '平均耗时', dataIndex: 'averageLatency', width: 120, render: (value: number) => `${value} ms` },
          { title: '方法', dataIndex: 'methods', width: 180 },
          { title: '高频路由', dataIndex: 'routes', ellipsis: true },
        ]} />
      </Card>

      <Card title="最近任务" style={{ marginTop: 16 }}>
        <Table dataSource={tasks} rowKey="task_id" pagination={false} size="small" columns={[
          { title: '任务描述', dataIndex: 'description', ellipsis: true, render: (value?: string) => value || '后台任务' },
          { title: '状态', dataIndex: 'status', render: (status: string) => <Tag color={status === 'SUCCESS' ? 'green' : status === 'FAILURE' ? 'red' : 'blue'}>{labelOf(taskStatusLabels, status)}</Tag> },
          { title: '就绪', dataIndex: 'ready', render: (ready: boolean) => (ready ? '是' : '否') },
          { title: '错误', dataIndex: 'error', render: (value?: string) => value || '-' },
        ]} />
      </Card>
    </div>
  )
}
