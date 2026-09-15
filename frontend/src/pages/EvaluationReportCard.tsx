import { Alert, Card, Col, Collapse, Row, Space, Spin, Statistic, Tag, Tooltip, Typography } from 'antd'
import { WarningOutlined } from '@ant-design/icons'
import type { EvaluationReportAdvisory, EvaluationReportMetricEntry } from '@/types'
import { evaluationLabel, formatEvaluationValue } from '@/utils/evaluationDisplay'

/** 指标分层标题与说明。 */
const SECTION_META: Record<string, { title: string; hint: string }> = {
  ragas: {
    title: '回答质量（RAGAS 评分）',
    hint: '按类别评审契约适用的指标取均值；未覆盖的类别不参与均值，避免把"不适用"误读为低分。',
  },
  evidence: {
    title: '行为契约与拒答安全（确定性）',
    hint: '来自证据门的确定性验收指标，率统一为百分比；不依赖 LLM 裁判，逐轮稳定。',
  },
  retrieval: {
    title: '检索质量（确定性）',
    hint: '基于期望证据 ID 的检索命中统计，率统一为百分比。',
  },
  safety: {
    title: '安全红线（越界率，越低越好）',
    hint: '越权引用、跨租户泄漏、注入指令服从率必须恒为 0%，任何非零都是必须人工排查的红线。',
  },
}

/** 易误读指标的口径说明。 */
const METRIC_NOTES: Record<string, string> = {
  id_based_context_precision:
    '检回材料中"答题目标文档"的占比（定向性/信噪比，不是检索准确率）。本系统检索有意多召回候选材料供生成阶段筛选：定向率低 + 召回率高 = 召回优先策略正常工作；定向率高且召回率低才是检不准。',
  groundedness_pass_rate:
    '仅证据门 shadow/enforce 模式下产生结构化 grounding 验证；off 模式（基线对照）显示"未覆盖"。',
  factual_correctness:
    '裁判将回答拆成事实条目逐条与参考答案比对。回答比参考答案更细时，超出部分会被判"参考不支持"而拉低分值——这是口径特性，不代表回答有错；结合忠实度（是否依据上下文）共同解读。',
}

/** 指标中文名（补充 evaluationDisplay 未覆盖的 key）。 */
const METRIC_LABELS: Record<string, string> = {
  faithfulness: '忠实度',
  factual_correctness: '事实正确性',
  answer_relevancy: '回答相关性',
  context_recall: '上下文召回率',
  rubrics_score_with_reference: '参考评分（1-5）',
  groundedness_pass_rate: '有据回答通过率',
  answerable_false_refusal_rate: '可答题误拒率',
  no_answer_hallucination_rate: '拒答题幻觉率',
  conflict_recognition_rate: '冲突识别率',
  partial_answer_recognition_rate: '部分回答识别率',
  refusal_precision: '拒答精确率',
  refusal_recall: '拒答召回率',
  id_based_context_precision: '上下文定向率（ID 口径）',
  id_based_context_recall: '上下文召回率（ID 口径）',
  unauthorized_citation_rate: '越权引用率',
  cross_scope_leakage_rate: '跨部门泄漏率',
  instruction_data_leakage_rate: '注入指令服从率',
}

/** Report the localized metric name for a stable key. */
function metricLabel(key: string): string {
  return METRIC_LABELS[key] ?? evaluationLabel(key)
}

/** Render one advisory with its severity color. */
function advisoryAlert(advisory: EvaluationReportAdvisory) {
  const type = advisory.level === 'critical' ? 'error' : advisory.level === 'warning' ? 'warning' : 'info'
  return (
    <Alert key={advisory.message} type={type} showIcon={advisory.level !== 'info'} message={advisory.message} />
  )
}

/** Render one metric entry as a statistic with coverage hint. */
function metricStatistic(entry: EvaluationReportMetricEntry) {
  const isRate = entry.unit === 'percent'
  const display =
    entry.value === null || entry.value === undefined
      ? '未覆盖'
      : isRate
        ? `${entry.value.toFixed(2)}%`
        : entry.value.toFixed(4)
  const coverage =
    entry.scored !== null && entry.scored !== undefined && entry.total
      ? `${entry.scored}/${entry.total}`
      : undefined
  const danger = entry.direction === 'lower_is_better' && typeof entry.value === 'number' && entry.value > 0
  const note = METRIC_NOTES[entry.key]
  return (
    <Col xs={12} md={6} key={entry.key}>
      <Statistic
        title={
          note ? (
            <Tooltip title={note}>
              <span style={{ cursor: 'help', borderBottom: '1px dotted currentColor' }}>
                {metricLabel(entry.key)}
              </span>
            </Tooltip>
          ) : (
            metricLabel(entry.key)
          )
        }
        value={display}
        suffix={coverage ? <Typography.Text type="secondary" style={{ fontSize: 12 }}>{coverage}</Typography.Text> : undefined}
        valueStyle={danger ? { color: '#cf1322' } : undefined}
      />
    </Col>
  )
}

interface EvaluationReportCardProps {
  report: {
    run_id: string
    incomplete?: boolean
    run_classification?: string | null
    counts?: Record<string, number | null>
    sections?: Record<string, EvaluationReportMetricEntry[]>
    advisories?: EvaluationReportAdvisory[]
  } | null
  loading: boolean
}

/** 综合评测报告卡片：分层指标 + 注意事项，替代无重点的统计堆砌。 */
export default function EvaluationReportCard({ report, loading }: EvaluationReportCardProps) {
  if (loading || !report) {
    return (
      <Card size="small" title="评测报告" style={{ marginTop: 16 }}>
        <Spin spinning={loading}>
          <Typography.Text type="secondary">暂无报告数据</Typography.Text>
        </Spin>
      </Card>
    )
  }

  const counts = report.counts ?? {}
  const sections = report.sections ?? {}
  const advisories = report.advisories ?? []
  const criticalCount = advisories.filter((item) => item.level === 'critical').length

  return (
    <Card
      size="small"
      title={
        <Space>
          <span>评测报告</span>
          {report.run_classification === 'smoke_only' ? <Tag>冒烟口径</Tag> : <Tag color="processing">正式口径</Tag>}
          {criticalCount > 0 ? (
            <Tag color="red" icon={<WarningOutlined />}>
              {criticalCount} 项红线/失败
            </Tag>
          ) : null}
        </Space>
      }
      style={{ marginTop: 16 }}
    >
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <Space size={24} wrap>
          <Statistic title="样本总数" value={formatEvaluationValue(counts.total ?? 0)} />
          <Statistic title="成功" value={formatEvaluationValue(counts.succeeded ?? 0)} />
          <Statistic title="执行失败" value={formatEvaluationValue(counts.failed ?? 0)} />
          <Statistic title="溯源无效" value={formatEvaluationValue(counts.invalid_provenance ?? 0)} />
        </Space>

        {['ragas', 'evidence'].map((section) => {
            const entries = sections[section] ?? []
            if (entries.length === 0) return null
            const meta = SECTION_META[section]
            return (
              <div key={section}>
                <Space size={8} style={{ marginBottom: 8 }}>
                  <Typography.Text strong>{meta?.title ?? section}</Typography.Text>
                  <Tooltip title={meta?.hint}>
                    <Typography.Text type="secondary" style={{ fontSize: 12, cursor: 'help' }}>
                      说明
                    </Typography.Text>
                  </Tooltip>
                </Space>
                <Row gutter={[16, 16]}>
                  {entries.map(metricStatistic)}
                </Row>
              </div>
            )
          })}

        {(['retrieval', 'safety'] as const).map((section) => {
          const entries = sections[section] ?? []
          if (entries.length === 0) return null
          const meta = SECTION_META[section]
          return (
            <Collapse
              key={section}
              size="small"
              items={[
                {
                  key: section,
                  label: (
                    <Space size={8}>
                      <Typography.Text strong>{meta?.title ?? section}</Typography.Text>
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        ({entries.length} 项，点击展开)
                      </Typography.Text>
                    </Space>
                  ),
                  children: (
                    <Row gutter={[16, 16]}>
                      {entries.map(metricStatistic)}
                    </Row>
                  ),
                },
              ]}
            />
          )
        })}

        {advisories.length > 0 ? (
          <div>
            <Typography.Text strong style={{ display: 'block', marginBottom: 8 }}>
              需要注意
            </Typography.Text>
            <Space direction="vertical" size={8} style={{ width: '100%' }}>
              {advisories.map(advisoryAlert)}
            </Space>
          </div>
        ) : (
          <Alert type="success" showIcon message="本次运行没有触发任何注意事项" />
        )}
      </Space>
    </Card>
  )
}
