import { useEffect, useState } from 'react'
import dayjs from 'dayjs'
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Divider,
  Drawer,
  Form,
  Input,
  Modal,
  Select,
  Space,
  Tag,
  Typography,
} from 'antd'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { departmentDisplayName } from '@/utils/departmentLabels'
import type { EvidenceCase, EvidenceReviewer } from '@/types'
import { EVIDENCE_CATEGORY_OPTIONS, evidenceCategoryLabel } from '@/utils/evidenceCategories'
import { EVIDENCE_REASON_CODE_OPTIONS } from '@/utils/evidenceReasonCodes'

const { Paragraph, Text } = Typography

const responseStatuses = [
  'answered', 'partially_answered', 'insufficient_evidence', 'needs_clarification',
  'conflicting_evidence', 'human_review_required', 'source_unavailable',
]
const evidenceStates = [
  'direct_evidence', 'partial_evidence', 'relevant_background', 'conflicting_evidence',
  'irrelevant', 'insufficient_evidence', 'invalid_provenance',
]

interface CaseFormValues {
  id: string
  department_id: string
  question: string
  reference_answer: string
  category: string
  expected_response_status: string
  expected_evidence_states: string[]
  evidence_context_ids: string
  citation_context_ids: string
  reason_codes: string[]
  source_document_ids: string
  missing_information_fields: string
}

function splitIds(value: string | undefined): string[] {
  return (value ?? '').split(/[\n,]/).map((item) => item.trim()).filter(Boolean)
}

function statusTag(status: string) {
  const mapping: Record<string, { color: string; label: string }> = {
    draft: { color: 'default', label: '草稿' },
    pending_review: { color: 'processing', label: '待复核' },
    approved: { color: 'success', label: '已批准' },
  }
  const item = mapping[status] ?? { color: 'default', label: status }
  return <Tag color={item.color}>{item.label}</Tag>
}

function reviewDate(value?: string | null) {
  const date = value ? dayjs(value) : null
  return date?.isValid() ? date.format('YYYY年MM月DD日 HH:mm') : '—'
}

function reviewerLabel(value: EvidenceCase) {
  return String(value.reviewer_display_name || value.reviewer_id || '—')
}

function editorLabel(value: EvidenceCase) {
  if (value.last_editor_id === 'system-import') return '系统导入'
  return String(value.last_editor_id || '—')
}

export interface EvidenceCaseDrawerProps {
  open: boolean
  value: EvidenceCase | null
  currentUserId?: string
  currentDepartmentId?: string | null
  isDepartmentManager?: boolean
  isOrganizationAdmin?: boolean
  /** Frozen datasets remain visible but cannot be altered from this drawer. */
  readOnly?: boolean
  loading: boolean
  onClose: () => void
  onSave: (value: EvidenceCase, isNew: boolean) => Promise<EvidenceCase>
  onLoadReviewers: (value: EvidenceCase) => Promise<EvidenceReviewer[]>
  onSubmit: (value: EvidenceCase, reviewerId: string) => Promise<EvidenceCase>
  onReview: (value: EvidenceCase, decision: 'approve' | 'reject', reason: string) => Promise<EvidenceCase>
}

/** Render an editable case with maker-checker actions and bounded Context evidence. */
export default function EvidenceCaseDrawer({
  open, value, currentUserId, currentDepartmentId,
  isOrganizationAdmin = false, readOnly = false,
  loading, onClose, onSave, onLoadReviewers, onSubmit, onReview,
}: EvidenceCaseDrawerProps) {
  const [form] = Form.useForm<CaseFormValues>()
  const [localValue, setLocalValue] = useState<EvidenceCase | null>(value)
  const [reviewOpen, setReviewOpen] = useState(false)
  const [decision, setDecision] = useState<'approve' | 'reject'>('approve')
  const [reviewReason, setReviewReason] = useState('')
  const [submitOpen, setSubmitOpen] = useState(false)
  const [reviewers, setReviewers] = useState<EvidenceReviewer[]>([])
  const [reviewerId, setReviewerId] = useState<string>()
  const [reviewersLoading, setReviewersLoading] = useState(false)
  const isNew = Boolean(localValue && localValue.case_revision === 0)

  useEffect(() => {
    setLocalValue(value)
    if (!value) return
    form.setFieldsValue({
      id: value.id,
      department_id: value.department_id || currentDepartmentId || '',
      question: value.question,
      reference_answer: value.reference_answer || '',
      category: value.category,
      expected_response_status: value.expected_response_status,
      expected_evidence_states: value.expected_evidence_states,
      evidence_context_ids: value.expected_evidence_context_ids.join('\n'),
      citation_context_ids: value.expected_citation_context_ids.join('\n'),
      reason_codes: value.expected_reason_codes,
      source_document_ids: value.expected_source_document_ids.join(', '),
      missing_information_fields: value.expected_missing_information_fields.join(', '),
    })
  }, [currentDepartmentId, form, value])

  const save = async () => {
    if (!localValue) return
    const fields = await form.validateFields()
    const next: EvidenceCase = {
      ...localValue,
      id: fields.id.trim(),
      department_id: (fields.department_id ?? localValue.department_id ?? currentDepartmentId ?? '').trim(),
      question: fields.question.trim(),
      reference_answer: fields.reference_answer.trim(),
      category: fields.category,
      expected_response_status: fields.expected_response_status,
      expected_evidence_states: fields.expected_evidence_states,
      expected_evidence_context_ids: splitIds(fields.evidence_context_ids),
      expected_citation_context_ids: splitIds(fields.citation_context_ids),
      expected_reason_codes: Array.from(new Set(fields.reason_codes ?? [])),
      expected_source_document_ids: splitIds(fields.source_document_ids),
      expected_missing_information_fields: splitIds(fields.missing_information_fields),
    }
    const saved = await onSave(next, isNew)
    setLocalValue(saved)
  }

  const openSubmit = async () => {
    if (!localValue) return
    setSubmitOpen(true)
    setReviewersLoading(true)
    setReviewerId(undefined)
    try {
      setReviewers(await onLoadReviewers(localValue))
    } finally {
      setReviewersLoading(false)
    }
  }

  const submit = async () => {
    if (!localValue || !reviewerId) return
    const submitted = await onSubmit(localValue, reviewerId)
    setLocalValue(submitted)
    setSubmitOpen(false)
  }

  const review = async () => {
    if (!localValue || !reviewReason.trim()) return
    const reviewed = await onReview(localValue, decision, reviewReason.trim())
    setLocalValue(reviewed)
    setReviewOpen(false)
    setReviewReason('')
  }

  const makerCannotApprove = localValue?.last_editor_id === currentUserId
  const isAssignedReviewer = Boolean(
    currentUserId && localValue?.reviewer_id === currentUserId,
  )
  const canReview = isOrganizationAdmin && isAssignedReviewer
  const isAuthoring = !readOnly && localValue?.review_status === 'draft'

  return (
    <>
      <Drawer
        className="evidence-case-drawer"
        width={860}
        open={open}
        onClose={onClose}
        destroyOnHidden
        title={
          <Space>
            <span>{isNew ? '新增候选样本' : localValue?.id}</span>
            {localValue && statusTag(localValue.review_status)}
          </Space>
        }
        extra={
          <Space>
            {localValue?.review_status === 'draft' && !isNew && (
              <Button onClick={() => void openSubmit()} loading={loading}>提交复核</Button>
            )}
            {localValue?.review_status === 'pending_review' && canReview && (
              <>
                <Button
                  danger
                  disabled={makerCannotApprove}
                  onClick={() => { setDecision('reject'); setReviewOpen(true) }}
                >拒绝并退回</Button>
                <Button
                  type="primary"
                  disabled={makerCannotApprove}
                  onClick={() => { setDecision('approve'); setReviewOpen(true) }}
                >通过审核</Button>
              </>
            )}
            {(localValue?.review_status === 'draft' || isNew) && (
              <Button type="primary" onClick={() => void save()} loading={loading}>保存</Button>
            )}
          </Space>
        }
      >
        {makerCannotApprove && canReview && localValue?.review_status === 'pending_review' && (
          <Alert
            type="warning"
            showIcon
            message="Maker-checker 已生效"
            description="你是该样本最后一位制作者，不能批准自己的修改。请由其他公司管理员复核。"
            style={{ marginBottom: 20 }}
          />
        )}
        {!canReview && localValue?.review_status === 'pending_review' && (
          <Alert
            type="info"
            showIcon
            message="等待公司管理员复核"
            description={`该样本已指派给审核人 ${localValue.reviewer_display_name || localValue.reviewer_id || '—'}，只有被指派的公司管理员可以作出决定。`}
            style={{ marginBottom: 20 }}
          />
        )}
        {isAuthoring && <Form form={form} layout="vertical" requiredMark="optional">
          <Form.Item name="id" label="样本 ID" rules={[{ required: true }, { pattern: /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/ }]}>
            <Input disabled={!isNew} placeholder="例如 evidence-case-038" />
          </Form.Item>
          {isNew && (
            <Form.Item name="department_id" label="所属部门 ID" rules={[{ required: true, message: '请填写样本所属部门' }, { max: 128 }]}>
              <Input placeholder="例如 财务部（finance）/ 人力资源部（human_resources）/ 行政管理部（administration）" />
            </Form.Item>
          )}
          <Form.Item name="question" label="问题" rules={[{ required: true }]}>
            <Input.TextArea autoSize={{ minRows: 3, maxRows: 8 }} placeholder="输入用于评测的业务问题" />
          </Form.Item>
          <Form.Item
            name="reference_answer"
            label="审核参考答案"
            rules={[{ max: 8000 }]}
            extra="请独立撰写答案；Context 摘录只用于证据与检索评分。"
          >
            <Input.TextArea autoSize={{ minRows: 4, maxRows: 12 }} placeholder="写出希望系统回答的结论，不要复制整段 Context" />
          </Form.Item>
          <Space size={16} align="start" style={{ display: 'flex' }}>
            <Form.Item name="category" label="证据类别" rules={[{ required: true }]} style={{ flex: 1 }}>
              <Select options={EVIDENCE_CATEGORY_OPTIONS} showSearch optionFilterProp="label" />
            </Form.Item>
            <Form.Item name="expected_response_status" label="预期响应状态" rules={[{ required: true }]} style={{ flex: 1 }}>
              <Select options={responseStatuses.map((item) => ({ value: item, label: item }))} />
            </Form.Item>
          </Space>
          <Form.Item name="expected_evidence_states" label="预期证据状态" rules={[{ required: true }]}>
            <Select mode="multiple" options={evidenceStates.map((item) => ({ value: item, label: item }))} />
          </Form.Item>
          <Form.Item name="evidence_context_ids" label="证据 Context ID（每行一个）">
            <Input.TextArea autoSize={{ minRows: 3, maxRows: 8 }} placeholder="document-id#chunk-1" />
          </Form.Item>
          <Form.Item name="citation_context_ids" label="引用 Context ID（每行一个）">
            <Input.TextArea autoSize={{ minRows: 2, maxRows: 6 }} />
          </Form.Item>
          <Form.Item name="reason_codes" label="预期 Reason Code">
            <Select mode="multiple" options={EVIDENCE_REASON_CODE_OPTIONS} showSearch optionFilterProp="label" />
          </Form.Item>
          <Form.Item name="source_document_ids" label="预期来源文档 ID（逗号分隔）"><Input /></Form.Item>
          <Form.Item name="missing_information_fields" label="预期缺失字段（逗号分隔）"><Input /></Form.Item>
        </Form>}

        {!isAuthoring && localValue && (
          <Space direction="vertical" size={20} style={{ width: '100%' }}>
            <Card title="复核内容" bordered={false} style={{ background: '#f8fafc' }}>
              <Descriptions column={1} bordered size="middle">
                <Descriptions.Item label="问题">
                  <Paragraph style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>{localValue.question}</Paragraph>
                </Descriptions.Item>
                <Descriptions.Item label="审核参考答案">
                  <Paragraph style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>
                    {localValue.reference_answer || '—'}
                  </Paragraph>
                </Descriptions.Item>
                <Descriptions.Item label="证据类别">
                  <Tag color="blue">{evidenceCategoryLabel(localValue.category)}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="预期响应状态">
                  <Tag color="geekblue">{localValue.expected_response_status}</Tag>
                </Descriptions.Item>
              </Descriptions>
            </Card>

            <Card title="审核记录" size="small">
              <Descriptions size="small" column={{ xs: 1, sm: 2 }}>
                <Descriptions.Item label="所属部门">{departmentDisplayName(localValue.department_id)}</Descriptions.Item>
                <Descriptions.Item label="录入人">{editorLabel(localValue)}</Descriptions.Item>
                <Descriptions.Item label="提交时间">{reviewDate(localValue.submitted_at)}</Descriptions.Item>
                <Descriptions.Item label="审核人">{reviewerLabel(localValue)}</Descriptions.Item>
                <Descriptions.Item label="审核时间">{reviewDate(localValue.reviewed_at)}</Descriptions.Item>
                <Descriptions.Item label="审核意见" span={2}>
                  {localValue.review_reason || localValue.rejection_reason || '—'}
                </Descriptions.Item>
              </Descriptions>
            </Card>

            <div>
              <Divider orientation="left">对应 Context</Divider>
              {localValue.context_summaries.length === 0 ? (
                <Alert type="info" showIcon message="当前样本未绑定可展示的 Context" />
              ) : localValue.context_summaries.map((item, index) => (
                <Card
                  key={item.context_id}
                  size="small"
                  title={<Space wrap><Tag color="blue">Context {index + 1}</Tag><Text strong>{item.title || item.source_document_id || '未命名来源'}</Text></Space>}
                  extra={<Text code copyable>{item.context_id}</Text>}
                  style={{ marginBottom: 12 }}
                >
                  <Space wrap size={[8, 8]} style={{ marginBottom: 12 }}>
                    {item.department && <Tag>{departmentDisplayName(item.department)}</Tag>}
                    {item.chunk_index !== null && item.chunk_index !== undefined && <Tag>Chunk {item.chunk_index}</Tag>}
                    {item.source_document_id && <Text type="secondary">文档：{item.source_document_id}</Text>}
                  </Space>
                  <div className="evidence-context-markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.content_excerpt}</ReactMarkdown>
                  </div>
                </Card>
              ))}
            </div>
          </Space>
        )}

        {isAuthoring && !isNew && localValue && (
          <>
            <Divider orientation="left">审核记录</Divider>
            <Descriptions size="small" column={2}>
              <Descriptions.Item label="Case revision">{localValue.case_revision}</Descriptions.Item>
              <Descriptions.Item label="录入人">{editorLabel(localValue)}</Descriptions.Item>
              <Descriptions.Item label="所属部门">{departmentDisplayName(localValue.department_id)}</Descriptions.Item>
              <Descriptions.Item label="审核人">{reviewerLabel(localValue)}</Descriptions.Item>
              <Descriptions.Item label="复核理由">{localValue.review_reason || localValue.rejection_reason || '—'}</Descriptions.Item>
            </Descriptions>
            <Divider orientation="left">已绑定 Context</Divider>
            {localValue.context_summaries.length === 0 ? (
              <Alert type="info" showIcon message="当前样本未绑定可展示的 Context" />
            ) : (
              <Space direction="vertical" style={{ width: '100%' }}>
                {localValue.context_summaries.map((item) => (
                  <Card key={item.context_id} size="small" title={<Text code>{item.context_id}</Text>}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.content_excerpt}</ReactMarkdown>
                  </Card>
                ))}
              </Space>
            )}
          </>
        )}
      </Drawer>
      <Modal
        title={decision === 'approve' ? '通过审核' : '拒绝并退回重新审核'}
        open={reviewOpen}
        onCancel={() => setReviewOpen(false)}
        onOk={() => void review()}
        confirmLoading={loading}
        okText={decision === 'approve' ? '确认批准' : '确认退回'}
        okButtonProps={{ danger: decision === 'reject', disabled: !reviewReason.trim() }}
      >
        <Space direction="vertical" style={{ width: '100%' }} size={16}>
          <Input.TextArea
            value={reviewReason}
            onChange={(event) => setReviewReason(event.target.value)}
            rows={4}
            maxLength={1000}
            showCount
            placeholder={decision === 'approve' ? '填写通过依据；审核身份与时间由服务端记录' : '说明需要重新审核和修改的内容'}
          />
        </Space>
      </Modal>
      <Modal
        title="提交复核"
        open={submitOpen}
        onCancel={() => setSubmitOpen(false)}
        onOk={() => void submit()}
        okText="确认提交"
        confirmLoading={loading}
        okButtonProps={{ disabled: !reviewerId }}
      >
        <Alert
          type="info"
          showIcon
          message={`所属部门：${departmentDisplayName(localValue?.department_id) || '—'}`}
          description="请选择本部门审核人。提交后样本进入只读复核视图，仅被指派的审核人可以通过或拒绝。"
          style={{ marginBottom: 16 }}
        />
        <Select
          aria-label="审核人"
          value={reviewerId}
          onChange={setReviewerId}
          loading={reviewersLoading}
          placeholder={reviewers.length ? '选择审核人' : '暂无可用审核人'}
          options={reviewers.map((item) => ({
            value: item.user_id,
            label: `${item.display_name || item.username}（${item.username}）`,
          }))}
          style={{ width: '100%' }}
          showSearch
          optionFilterProp="label"
        />
      </Modal>
    </>
  )
}
