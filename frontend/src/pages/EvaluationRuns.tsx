import { useEffect, useMemo } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
} from 'antd'
import { ArrowLeftOutlined, ReloadOutlined } from '@ant-design/icons'
import { useEvaluationRuns } from './useEvaluationRuns'
import SpotCheckDrawer from './SpotCheckDrawer'
import EvaluationReportCard from './EvaluationReportCard'
import type { EvaluationRunSummary } from '@/types'
import { formatChinaStandardTime } from '@/utils/time'
import { evaluationLabel, evaluationStatusLabel, formatEvaluationValue } from '@/utils/evaluationDisplay'

const PAGE_SIZE = 20

/** Render a run classification tag. */
function classificationTag(value: string | null | undefined) {
  if (value === 'smoke_only') return <Tag>{evaluationLabel(value)}</Tag>
  if (value === 'baseline') return <Tag color="processing">正式评测</Tag>
  return <span>—</span>
}

/** Render retrieval modes in Chinese while retaining their raw API values. */
function retrievalModeTags(modes: string[] | undefined) {
  if (!modes || modes.length === 0) return <span>—</span>
  return (
    <Space size={4} wrap>
      {modes.map((mode) => (
        <Tag key={mode} color="blue">
          {evaluationLabel(mode)}
        </Tag>
      ))}
    </Space>
  )
}

/** Render a record status tag with a Chinese label. */
function statusTag(status: string | null | undefined) {
  if (!status) return <span>—</span>
  const color = status === 'succeeded' ? 'green' : status === 'failed' ? 'red' : status === 'invalid_provenance' ? 'orange' : 'default'
  return <Tag color={color}>{evaluationStatusLabel(status)}</Tag>
}

/**
 * Render per-metric score states: scored values, contract-inapplicable metrics,
 * and failed metrics are all visible so "—" can never hide the reason.
 */
function renderRagasScoreStates(record: {
  ragas_scores?: Record<string, number>
  ragas_metric_states?: Record<string, { status: string; reason_code?: string | null }>
}): Array<{ key: string; text: string; kind: 'scored' | 'inapplicable' | 'failed' }> {
  const states = record.ragas_metric_states ?? {}
  const scores = record.ragas_scores ?? {}
  const entries: Array<{ key: string; text: string; kind: 'scored' | 'inapplicable' | 'failed' }> = []
  for (const [metric, state] of Object.entries(states)) {
    if (state.status === 'scored') {
      const value = scores[metric]
      entries.push({
        key: metric,
        text: `${evaluationLabel(metric)}：${typeof value === 'number' ? value.toFixed(4) : '—'}`,
        kind: 'scored',
      })
    } else if (state.status === 'not_applicable') {
      entries.push({ key: metric, text: `${evaluationLabel(metric)}：不适用`, kind: 'inapplicable' })
    } else if (state.status === 'failed') {
      entries.push({ key: metric, text: `${evaluationLabel(metric)}：评分失败`, kind: 'failed' })
    }
  }
  // 兼容旧产物：没有状态明细但有分数时退回纯分数展示
  if (entries.length === 0 && Object.keys(scores).length > 0) {
    for (const [metric, value] of Object.entries(scores)) {
      entries.push({ key: metric, text: `${evaluationLabel(metric)}：${value.toFixed(4)}`, kind: 'scored' })
    }
  }
  return entries
}

/** Summarize every category/topic with its completed and failed sample counts. */
function categoryReportRows(value: unknown) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return []
  return Object.entries(value as Record<string, unknown>).flatMap(([category, summary]) => {
    if (!summary || typeof summary !== 'object' || Array.isArray(summary)) return []
    const counts = (summary as Record<string, unknown>).counts
    if (!counts || typeof counts !== 'object' || Array.isArray(counts)) return []
    const record = counts as Record<string, unknown>
    return [{
      key: category,
      category: evaluationLabel(category),
      total: formatEvaluationValue(record.total),
      completed: formatEvaluationValue(record.completed),
      failed: formatEvaluationValue(record.failed),
      invalidProvenance: formatEvaluationValue(record.invalid_provenance),
    }]
  })
}

/** Render the evaluation results page (list + run detail + per-question records). */
export default function EvaluationRuns({ embedded = false, initialRunId }: { embedded?: boolean; initialRunId?: string }) {
  const {
    backToList,
    changePage,
    detailLoading,
    fetchRuns,
    page,
    records,
    recordsLoading,
    recordsTotal,
    runs,
    runsError,
    runsLoading,
    openRun,
    selectedRun,
    anomalies,
    anomaliesLoading,
    loadAnomalies,
    submitSpotCheck,
    submittingSpotCheck,
    runReport,
    reportLoading,
    loadReport,
  } = useEvaluationRuns()

  useEffect(() => {
    if (initialRunId) void openRun(initialRunId)
  }, [initialRunId, openRun])

  // 进入详情页时拉取一次异常清单与综合报告
  useEffect(() => {
    if (!selectedRun) return
    void loadAnomalies(selectedRun.run_id)
    void loadReport(selectedRun.run_id)
  }, [selectedRun, loadAnomalies, loadReport])

  const listColumns = useMemo(
    () => [
      {
        title: '运行编号',
        dataIndex: 'run_id',
        ellipsis: true,
        /** Render the cell with a copyable run id. */
        render: (value: string) => <Typography.Text copyable>{value}</Typography.Text>,
      },
      {
        title: '开始时间',
        dataIndex: 'started_at',
        width: 180,
        /** Render the cell with a formatted local timestamp. */
        render: (value: string | null) => formatChinaStandardTime(value),
      },
      {
        title: '状态',
        dataIndex: 'incomplete',
        width: 90,
        /** Render the cell with the run completion state. */
        render: (value: boolean) =>
          value ? <Tag color="warning">未完成</Tag> : <Tag color="success">已完成</Tag>,
      },
      {
        title: '检索模式',
        dataIndex: 'retrieval_modes',
        /** Render the cell with the retrieval mode tags. */
        render: (value: string[]) => retrievalModeTags(value),
      },
      {
        title: '分类',
        dataIndex: 'run_classification',
        width: 90,
        /** Render the cell with the run classification tag. */
        render: (value: string | null) => classificationTag(value),
      },
      {
        title: '完成',
        dataIndex: 'completed',
        width: 80,
        /** Render the cell with the completed count. */
        render: (value: number) => value ?? 0,
      },
      {
        title: '失败',
        dataIndex: 'failed',
        width: 80,
        /** Render the cell with the failed count. */
        render: (value: number) => value ?? 0,
      },
      {
        title: '无效溯源',
        dataIndex: 'invalid_provenance',
        width: 90,
        /** Render the cell with the invalid provenance count. */
        render: (value: number) => value ?? 0,
      },
      {
        title: '操作',
        width: 100,
        /** Render the cell with the detail action. */
        render: (_: unknown, run: EvaluationRunSummary) => (
          <Button type="link" onClick={() => void openRun(run.run_id)}>
            查看详情
          </Button>
        ),
      },
    ],
    [openRun],
  )

  if (selectedRun) {
    const metadata = selectedRun.metadata ?? {}
    const qualityReport = selectedRun.quality_report
    const ragasSummaries = selectedRun.ragas_summaries ?? {}

    const retrievalModes = Array.isArray(metadata.retrieval_modes)
      ? (metadata.retrieval_modes as string[])
      : []

    const categoryRows = categoryReportRows(qualityReport?.by_category)

    const ragasRows = Object.entries(ragasSummaries).flatMap(([moduleKey, value]) => {
      const metrics =
        value &&
        typeof value === 'object' &&
        'metrics' in value &&
        typeof value.metrics === 'object' &&
        value.metrics !== null
          ? (value.metrics as Record<
              string,
              { total?: number; scored?: number; failed?: number; skipped?: number; mean?: number }
            >)
          : {}
      return Object.entries(metrics).map(([metric, counts]) => ({
        key: `${moduleKey}-${metric}`,
        module: moduleKey,
        metric,
        ...counts,
      }))
    })

    const recordData = records.map((record, index) => ({
      ...record,
      rowKey: `${index}`,
    }))

    const ragasColumns = [
      { title: '评分产物', dataIndex: 'module', width: 220, ellipsis: true, render: (value: string) => evaluationLabel(value) },
      { title: '指标', dataIndex: 'metric', ellipsis: true, render: (value: string) => evaluationLabel(value) },
      { title: '总数', dataIndex: 'total', width: 90 },
      { title: '已评分', dataIndex: 'scored', width: 90 },
      { title: '失败', dataIndex: 'failed', width: 90 },
      { title: '跳过', dataIndex: 'skipped', width: 90 },
      {
        title: '均值',
        dataIndex: 'mean',
        width: 110,
        /** Render the cell with the mean value. */
        render: (value: number) => (typeof value === 'number' ? value.toFixed(4) : '—'),
      },
    ]

    const recordColumns = [
      {
        title: '样本编号',
        dataIndex: 'benchmark_id',
        width: 120,
        ellipsis: true,
        /** Render the cell with the benchmark id. */
        render: (value: string | null) => value ?? '—',
      },
      {
        title: '检索模式',
        dataIndex: 'retrieval_mode',
        width: 132,
        /** Render a compact localized retrieval mode tag without clipping it. */
        render: (value: string | null) =>
          value ? (
            <Tag color="blue" style={{ marginInlineEnd: 0, whiteSpace: 'normal' }}>
              {evaluationLabel(value)}
            </Tag>
          ) : <span>—</span>,
      },
      {
        title: '类别',
        dataIndex: 'category',
        width: 176,
        /** Render the localized category with enough room for evidence-gate labels. */
        render: (value: string | null) =>
          value ? <span style={{ whiteSpace: 'normal', wordBreak: 'break-word' }}>{evaluationLabel(value)}</span> : '—',
      },
      {
        title: '状态',
        dataIndex: 'status',
        width: 116,
        /** Render the cell with the status tag. */
        render: (value: string | null) => statusTag(value),
      },
      {
        title: '问题',
        dataIndex: 'question',
        width: 360,
        ellipsis: true,
        /** Render the cell with the question. */
        render: (value: string | null) => value ?? '—',
      },
      {
        title: '回答',
        dataIndex: 'response',
        width: 420,
        ellipsis: true,
        /** Render the cell with a truncated response. */
        render: (value: string | null) =>
          value ? `${value.length > 200 ? `${value.slice(0, 200)}…` : value}` : '—',
      },
      {
        title: 'RAGAS 评分',
        dataIndex: 'ragas_scores',
        width: 320,
        /** Render per-metric states: values, contract-inapplicable, and failures. */
        render: (_: unknown, record: Record<string, unknown> & {
          ragas_scores?: Record<string, number>
          ragas_metric_states?: Record<string, { status: string; reason_code?: string | null }>
        }) => {
          const entries = renderRagasScoreStates(record)
          if (entries.length === 0) return <span>—</span>
          return (
            <Space size={4} wrap>
              {entries.map((entry) => (
                <Tag
                  key={entry.key}
                  style={{ marginInlineEnd: 0 }}
                  color={entry.kind === 'failed' ? 'red' : entry.kind === 'inapplicable' ? 'default' : undefined}
                >
                  {entry.text}
                </Tag>
              ))}
            </Space>
          )
        },
      },
      {
        title: '延迟(ms)',
        dataIndex: 'latency_ms',
        width: 100,
        /** Render the cell with the latency. */
        render: (value: number | null) => (value === null || value === undefined ? '—' : value),
      },
    ]

    return (
      <Card
        title={
          <Space>
            <Button icon={<ArrowLeftOutlined />} onClick={backToList}>
              返回列表
            </Button>
            <Typography.Text strong>评测结果详情</Typography.Text>
          </Space>
        }
      >
        <Spin spinning={detailLoading}>
          <Descriptions column={2} bordered size="small">
            <Descriptions.Item label="运行编号">{selectedRun.run_id}</Descriptions.Item>
            <Descriptions.Item label="状态">
              {selectedRun.incomplete ? <Tag color="warning">未完成</Tag> : <Tag color="success">已完成</Tag>}
            </Descriptions.Item>
            <Descriptions.Item label="开始时间">
              {formatChinaStandardTime(metadata.started_at)}
            </Descriptions.Item>
            <Descriptions.Item label="评测数据集">
              {typeof metadata.benchmark_source === 'string' ? metadata.benchmark_source : '—'}
            </Descriptions.Item>
            <Descriptions.Item label="数据指纹">
              {typeof metadata.benchmark_sha256 === 'string' ? metadata.benchmark_sha256 : '—'}
            </Descriptions.Item>
            <Descriptions.Item label="代码版本">
              {typeof metadata.code_revision === 'string' ? metadata.code_revision : '—'}
            </Descriptions.Item>
            <Descriptions.Item label="检索模式">
              {retrievalModeTags(retrievalModes)}
            </Descriptions.Item>
          </Descriptions>

          <EvaluationReportCard report={runReport} loading={reportLoading} />

          {qualityReport && categoryRows.length > 0 ? (
            <Card size="small" title="分类别明细" style={{ marginTop: 16 }}>
              <Table
                size="small"
                pagination={false}
                rowKey="key"
                dataSource={categoryRows}
                columns={[
                  { title: '主题/类别', dataIndex: 'category' },
                  { title: '样本总数', dataIndex: 'total', width: 100 },
                  { title: '已完成', dataIndex: 'completed', width: 100 },
                  { title: '失败', dataIndex: 'failed', width: 90 },
                  { title: '溯源无效', dataIndex: 'invalidProvenance', width: 110 },
                ]}
              />
            </Card>
          ) : null}

          {ragasRows.length === 0 ? (
            <Alert
              type="warning"
              showIcon
              message="未生成 RAGAS 评分"
              description="该运行没有可读取的 RAGAS outcome/summary 产物；它不能被解读为 RAGAS 质量通过。历史运行可能在 RAGAS 纳入发布流程之前完成。"
              style={{ marginTop: 16 }}
            />
          ) : (
            <Card size="small" title="RAGAS 评测汇总" style={{ marginTop: 16 }}>
              <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                {ragasRows.map((row) => (
                  <Col xs={12} lg={6} key={`metric-${row.key}`}>
                    <Statistic
                      title={evaluationLabel(row.metric)}
                      value={typeof row.mean === 'number' ? row.mean : '—'}
                      precision={4}
                      suffix={typeof row.scored === 'number' ? `${row.scored}/${row.total ?? row.scored}` : undefined}
                    />
                  </Col>
                ))}
              </Row>
              <Table
                columns={ragasColumns}
                dataSource={ragasRows}
                rowKey="key"
                size="small"
                pagination={false}
              />
            </Card>
          )}

          {anomalies ? (
            <SpotCheckDrawer
              runId={anomalies.run_id}
              anomalies={anomalies.anomalies}
              anomalyCount={anomalies.anomaly_count}
              spotChecked={anomalies.spot_checked}
              lowScoreThreshold={anomalies.low_score_threshold}
              loading={anomaliesLoading}
              submitting={submittingSpotCheck}
              onReload={() => void loadAnomalies(selectedRun.run_id)}
              onSubmit={(benchmarkId, verdict, note) =>
                void submitSpotCheck(selectedRun.run_id, benchmarkId, verdict, note)
              }
            />
          ) : null}

          <Card size="small" title="逐题结果" style={{ marginTop: 16 }}>
            <Table
              columns={recordColumns}
              dataSource={recordData}
              rowKey="rowKey"
              loading={recordsLoading}
              size="small"
              scroll={{ x: 1700 }}
              tableLayout="fixed"
              locale={{
                emptyText: <Empty description="暂无逐题评测记录" />,
              }}
              pagination={{
                current: page,
                pageSize: PAGE_SIZE,
                total: recordsTotal,
                showSizeChanger: false,
                /** Format the pagination total summary. */
                showTotal: (total) => `共 ${total} 条`,
                /** Handle the page change event. */
                onChange: (targetPage) => void changePage(targetPage),
              }}
            />
          </Card>
        </Spin>
      </Card>
    )
  }

  return (
    <Card
      className={embedded ? "governance-panel" : undefined}
      title={embedded ? "离线评测运行" : "评测结果"}
      extra={
        <Button type="primary" icon={<ReloadOutlined />} onClick={fetchRuns}>
          刷新
        </Button>
      }
    >
      {runsError ? (
        <Alert
          type="error"
          message={runsError}
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={fetchRuns}>
              重试
            </Button>
          }
        />
      ) : null}
      {!runsLoading && runs.length === 0 ? (
        <Card size="small" title="RAGAS 指标状态" style={{ marginBottom: 16 }}>
          <Alert
            type="info"
            showIcon
            message="尚未生成 RAGAS 评测结果"
            description="当前仅完成指标适配与只读展示，必须先对已冻结、已实际入库的评测语料运行离线评测，结果才会显示真实分数。"
            style={{ marginBottom: 16 }}
          />
          <Row gutter={[16, 16]}>
            {['faithfulness', 'factual_correctness', 'context_precision', 'context_recall'].map((metric) => (
              <Col xs={12} lg={6} key={metric}>
                <Statistic title={evaluationLabel(metric)} value="未测量" />
              </Col>
            ))}
          </Row>
        </Card>
      ) : null}
      <Table
        columns={listColumns}
        dataSource={runs}
        rowKey="run_id"
        loading={runsLoading}
        size="small"
        scroll={{ x: 1100 }}
        locale={{
          emptyText: <Empty description="暂无评测运行，请先按评测运行手册执行离线评测" />,
        }}
        pagination={{ pageSize: 20, showSizeChanger: false }}
        onRow={(run) => ({
          /** Open the run detail when clicking a row. */
          onClick: () => void openRun(run.run_id),
        })}
      />
    </Card>
  )
}
