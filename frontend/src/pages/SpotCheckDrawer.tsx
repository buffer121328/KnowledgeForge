import { useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Drawer,
  Empty,
  Form,
  Input,
  Modal,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from 'antd'
import { CheckCircleOutlined, ReloadOutlined } from '@ant-design/icons'
import type { EvaluationSpotCheckItem, SpotCheckVerdict } from '@/types'
import { evaluationLabel, evaluationStatusLabel } from '@/utils/evaluationDisplay'
import { formatChinaStandardTime } from '@/utils/time'

const ANOMALY_LABELS: Record<string, { label: string; color: string }> = {
  scoring_failed: { label: '评测执行失败', color: 'red' },
  ragas_metric_failed: { label: '评分失败', color: 'volcano' },
  ragas_low_score: { label: '低分指标', color: 'orange' },
  contract_mismatch: { label: '行为契约不符', color: 'purple' },
  evidence_recall_gap: { label: '证据召回缺口', color: 'gold' },
}

const VERDICT_OPTIONS: Array<{ value: SpotCheckVerdict; label: string }> = [
  { value: 'judge_error', label: '评分器误判（裁判问题）' },
  { value: 'system_issue', label: '系统真实问题（检索/生成缺陷）' },
  { value: 'confirmed_ok', label: '确认无异常' },
  { value: 'needs_data_fix', label: '需要修正测试数据/参考答案' },
]

/** Render anomaly codes as colored Chinese tags. */
function anomalyTags(codes: string[]) {
  return (
    <Space size={4} wrap>
      {codes.map((code) => {
        const meta = ANOMALY_LABELS[code]
        return (
          <Tag key={code} color={meta?.color ?? 'default'} style={{ marginInlineEnd: 0 }}>
            {meta?.label ?? code}
          </Tag>
        )
      })}
    </Space>
  )
}

/** Render the stored spot-check verdict as a tag. */
function verdictTag(verdict: string) {
  const option = VERDICT_OPTIONS.find((item) => item.value === verdict)
  const color =
    verdict === 'system_issue' ? 'red' : verdict === 'needs_data_fix' ? 'orange' : verdict === 'judge_error' ? 'purple' : 'green'
  return <Tag color={color}>{option?.label ?? verdict}</Tag>
}

interface SpotCheckDrawerProps {
  runId: string
  anomalies: EvaluationSpotCheckItem[]
  anomalyCount: number
  spotChecked: number
  lowScoreThreshold: number
  loading: boolean
  submitting: boolean
  onReload: () => void
  onSubmit: (benchmarkId: string, verdict: SpotCheckVerdict, note: string) => void
}

/** 异常样本人工抽检面板：自动异常分类 + 人工复核结论。 */
export default function SpotCheckDrawer({
  runId,
  anomalies,
  anomalyCount,
  spotChecked,
  lowScoreThreshold,
  loading,
  submitting,
  onReload,
  onSubmit,
}: SpotCheckDrawerProps) {
  const [open, setOpen] = useState(false)
  const [selected, setSelected] = useState<EvaluationSpotCheckItem | null>(null)
  const [form] = Form.useForm<{ verdict: SpotCheckVerdict; note: string }>()

  const rows = useMemo(
    () => anomalies.map((item) => ({ ...item, rowKey: item.benchmark_id })),
    [anomalies],
  )

  const openDetail = (item: EvaluationSpotCheckItem) => {
    setSelected(item)
    form.setFieldsValue({
      verdict: item.spot_check?.verdict,
      note: item.spot_check?.note ?? '',
    })
  }

  const submit = async () => {
    const values = await form.validateFields()
    if (!selected) return
    onSubmit(selected.benchmark_id, values.verdict, values.note ?? '')
    setSelected(null)
  }

  const columns = [
    {
      title: '样本编号',
      dataIndex: 'benchmark_id',
      width: 240,
      ellipsis: true,
      render: (value: string) => <Typography.Text copyable>{value}</Typography.Text>,
    },
    { title: '类别', dataIndex: 'category', width: 150, render: (value: string | null) => (value ? evaluationLabel(value) : '—') },
    { title: '异常类型', dataIndex: 'anomaly_codes', width: 240, render: (codes: string[]) => anomalyTags(codes) },
    {
      title: 'RAGAS 评分',
      dataIndex: 'ragas_scores',
      width: 260,
      ellipsis: true,
      render: (scores: Record<string, number>) =>
        scores && Object.keys(scores).length > 0
          ? Object.entries(scores)
              .map(([key, value]) => `${evaluationLabel(key)}：${value.toFixed(3)}`)
              .join('；')
          : '—',
    },
    {
      title: '抽检状态',
      dataIndex: 'spot_check',
      width: 180,
      render: (value: EvaluationSpotCheckItem['spot_check']) =>
        value ? (
          <Space size={4} wrap>
            {verdictTag(value.verdict)}
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              {formatChinaStandardTime(value.reviewed_at)}
            </Typography.Text>
          </Space>
        ) : (
          <Tag>待抽检</Tag>
        ),
    },
    {
      title: '操作',
      width: 90,
      render: (_: unknown, item: EvaluationSpotCheckItem) => (
        <Button type="link" onClick={() => openDetail(item)}>
          抽检
        </Button>
      ),
    },
  ]

  return (
    <>
      <Card
        size="small"
        title={
          <Space>
            <span>异常人工抽检</span>
            <Typography.Text type="secondary" style={{ fontSize: 12, fontWeight: 'normal' }}>
              低分阈值 {lowScoreThreshold}
            </Typography.Text>
          </Space>
        }
        style={{ marginTop: 16 }}
        extra={
          <Button
            type="link"
            icon={<ReloadOutlined />}
            onClick={onReload}
          >
            刷新
          </Button>
        }
      >
        <Space size={24} style={{ marginBottom: 12 }}>
          <Statistic title="异常样本" value={anomalyCount} />
          <Statistic title="已抽检" value={spotChecked} suffix={`/ ${anomalyCount}`} />
        </Space>
        {anomalyCount === 0 ? (
          <Alert
            type="success"
            showIcon
            icon={<CheckCircleOutlined />}
            message="本次运行没有检测到需要人工抽检的异常样本"
          />
        ) : (
          <Button type="primary" onClick={() => setOpen(true)}>
            打开抽检面板（{anomalyCount} 条）
          </Button>
        )}
      </Card>

      <Drawer
        title={`异常人工抽检 · ${runId}`}
        width={1100}
        open={open}
        onClose={() => setOpen(false)}
        destroyOnClose
      >
        <Table
          columns={columns}
          dataSource={rows}
          rowKey="rowKey"
          loading={loading}
          size="small"
          pagination={{ pageSize: 20, showSizeChanger: false }}
          locale={{ emptyText: <Empty description="暂无异常样本" /> }}
        />
      </Drawer>

      <Modal
        title="人工抽检结论"
        open={selected !== null}
        onCancel={() => setSelected(null)}
        onOk={() => void submit()}
        okText="保存结论"
        confirmLoading={submitting}
        destroyOnClose
      >
        {selected ? (
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <div>
              <Typography.Text type="secondary">样本：</Typography.Text>
              <Typography.Text copyable>{selected.benchmark_id}</Typography.Text>
            </div>
            {anomalyTags(selected.anomaly_codes)}
            {selected.exception ? (
              <Alert type="error" showIcon message={`执行异常：${selected.exception}`} />
            ) : null}
            {selected.expected_response_status && selected.observed_response_status ? (
              <Alert
                type="warning"
                showIcon
                message={`响应状态：预期 ${evaluationStatusLabel(selected.expected_response_status)}，实际 ${evaluationStatusLabel(selected.observed_response_status)}`}
              />
            ) : null}
            <div>
              <Typography.Text type="secondary">问题：</Typography.Text>
              {selected.question}
              <br />
              <Typography.Text type="secondary">回答（截断）：</Typography.Text>
              <Typography.Paragraph style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>
                {selected.response || '—'}
              </Typography.Paragraph>
            </div>
            <Form form={form} layout="vertical">
              <Form.Item
                name="verdict"
                label="复核结论"
                rules={[{ required: true, message: '请选择复核结论' }]}
              >
                <Select options={VERDICT_OPTIONS} placeholder="请选择复核结论" />
              </Form.Item>
              <Form.Item name="note" label="备注">
                <Input.TextArea rows={3} maxLength={600} showCount placeholder="评分偏差原因、需要修正的数据等" />
              </Form.Item>
            </Form>
          </Space>
        ) : null}
      </Modal>
    </>
  )
}
