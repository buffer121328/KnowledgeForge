import dayjs from 'dayjs'
import { useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  Input,
  Modal,
  Popconfirm,
  Progress,
  Row,
  Select,
  Skeleton,
  Space,
  Statistic,
  Table,
  Tag,
  Tooltip,
  Typography,
} from 'antd'
import {
  CheckCircleOutlined,
  DeleteOutlined,
  DatabaseOutlined,
  FileProtectOutlined,
  PlusOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import { useAuthStore } from '@/stores/auth'
import { useEvidenceDatasets } from '@/pages/useEvidenceDatasets'
import type { EvidenceBulkSelectionMode, EvidenceCase, EvidenceReviewStatus } from '@/types'
import { evidenceCategoryLabel } from '@/utils/evidenceCategories'
import { evaluationDatasetLabel, evaluationVersionLabel } from '@/utils/evaluationDatasetLabels'
import EvidenceCaseDrawer from './EvidenceCaseDrawer'

const { Paragraph, Text, Title } = Typography
const REBINDABLE_SOURCE_DATASETS = new Set(['evidence-gates-v1', 'evidence-gates-v1-routine'])
const emptyCase: EvidenceCase = {
  id: '', question: '', reference_answer: '', category: 'fully_answerable', expected_response_status: 'answered',
  expected_evidence_states: ['direct_evidence'], expected_reason_codes: [],
  expected_source_document_ids: [], expected_evidence_context_ids: [],
  expected_evidence_sections: [], expected_citation_context_ids: [],
  expected_missing_information_fields: [], required_fixture: 'standard',
  review_status: 'draft', case_revision: 0, department_id: '', context_summaries: [],
}

function reviewTag(status: string) {
  if (status === 'approved') return <Tag color="success">已批准</Tag>
  if (status === 'pending_review') return <Tag color="processing">待复核</Tag>
  return <Tag>草稿</Tag>
}

interface BulkAction {
  action: 'submit' | 'approve'
  selectionMode: EvidenceBulkSelectionMode
}

/** Render the enterprise evidence-dataset authoring, maker-checker review, and freeze workbench. */
export default function EvidenceDatasetWorkbench() {
  const user = useAuthStore((state) => state.user)
  const state = useEvidenceDatasets()
  const [selected, setSelected] = useState<EvidenceCase | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [freezeOpen, setFreezeOpen] = useState(false)
  const [draftOpen, setDraftOpen] = useState(false)
  const [draftVersion, setDraftVersion] = useState<string>()
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([])
  const [bulkAction, setBulkAction] = useState<BulkAction | null>(null)
  const canAdministerDataset = user?.role === 'organization_admin'
  const canBulkReview = canAdministerDataset
  const canDeleteCases = canAdministerDataset
  const isFrozen = state.dataset?.status === 'frozen'
  const approvalPercent = state.dataset?.case_count
    ? Math.round(((state.counts.approved ?? 0) / state.dataset.case_count) * 100)
    : 0
  const freezeReady = Boolean(
    state.dataset
    && state.dataset.case_count > 0
    && state.counts.approved === state.dataset.case_count
    && state.dataset.covered_category_count === state.dataset.required_category_count
  )
  const blockers = useMemo(() => {
    if (!state.dataset) return []
    const values: string[] = []
    if (state.counts.approved !== state.dataset.case_count) values.push('仍有样本未批准')
    if (state.dataset.covered_category_count !== state.dataset.required_category_count) values.push('11 类证据场景未覆盖完整')
    return values
  }, [state.counts.approved, state.dataset])

  const openCase = (value: EvidenceCase) => {
    setSelected(value)
    setDrawerOpen(true)
  }

  const saveCase = async (value: EvidenceCase, isNew: boolean) => {
    const saved = await state.saveCase(value, isNew)
    setSelected(saved)
    return saved
  }

  const deleteCase = async (value: EvidenceCase) => {
    await state.deleteCase(value)
    setDrawerOpen(false)
    setSelected(null)
  }

  const submitCase = async (value: EvidenceCase, reviewerId: string) => {
    const next = await state.submitCase(value, reviewerId)
    setSelected(next)
    return next
  }

  const reviewCase = async (value: EvidenceCase, decision: 'approve' | 'reject', reason: string) => {
    const next = await state.reviewCase(value, decision, reason)
    setSelected(next)
    return next
  }

  const runBulkAction = async () => {
    if (!bulkAction) return
    const caseIds = bulkAction.selectionMode === 'selected' ? selectedRowKeys : []
    try {
      if (bulkAction.action === 'submit') {
        await state.bulkSubmit(bulkAction.selectionMode, caseIds)
      } else {
        await state.bulkApprove(bulkAction.selectionMode, caseIds)
      }
      setSelectedRowKeys([])
      setBulkAction(null)
    } catch {
      setBulkAction(null)
    }
  }

  if (state.loading && !state.dataset) {
    return (
      <div className="governance-view" data-testid="dataset-loading">
        <Row gutter={[16, 16]}>{[1, 2, 3, 4].map((item) => <Col xs={24} md={12} xl={6} key={item}><Card><Skeleton active paragraph={{ rows: 2 }} /></Card></Col>)}</Row>
        <Card style={{ marginTop: 16 }}><Skeleton active paragraph={{ rows: 8 }} /></Card>
      </div>
    )
  }

  if (state.error) {
    return <Alert type="error" showIcon message="无法加载数据集工作台" description={state.error} action={<Button onClick={() => void state.load()}>重试</Button>} />
  }

  if (!state.dataset) return <Empty description="当前组织暂无可治理的数据集" />

  const columns = [
    {
      title: '样本', dataIndex: 'id', key: 'id', width: 220,
      render: (id: string, record: EvidenceCase) => (
        <Space direction="vertical" size={2}>
          <Button type="link" onClick={() => openCase(record)} style={{ padding: 0, height: 'auto' }}>{id}</Button>
          <Text type="secondary" ellipsis style={{ maxWidth: 280 }}>{record.question}</Text>
        </Space>
      ),
    },
    { title: '类别', dataIndex: 'category', key: 'category', width: 190, render: (value: string) => <Tag color="blue">{evidenceCategoryLabel(value)}</Tag> },
    { title: '响应状态', dataIndex: 'expected_response_status', key: 'expected_response_status', width: 180 },
    { title: 'Context', key: 'contexts', width: 100, render: (_: unknown, record: EvidenceCase) => record.context_summaries.length },
    { title: '审核状态', dataIndex: 'review_status', key: 'review_status', width: 110, render: reviewTag },
    { title: '最后编辑人', dataIndex: 'last_editor_id', key: 'last_editor_id', width: 140, render: (_value: string | null, record: EvidenceCase) => record.last_editor_display_name || '系统操作' },
    { title: '样本版本', dataIndex: 'case_revision', key: 'case_revision', width: 90, render: (value: number) => `第 ${value} 版` },
    ...(canDeleteCases && !isFrozen ? [{
      title: '操作', key: 'actions', width: 100,
      render: (_: unknown, record: EvidenceCase) => (
        <Popconfirm
          title="确认删除该评测样本？"
          description="删除会使当前工作区回到编辑状态；历史冻结版本不受影响。"
          okText="确认删除"
          cancelText="取消"
          onConfirm={() => void deleteCase(record)}
        >
          <Button
            danger
            size="small"
            icon={<DeleteOutlined />}
            aria-label="删除样本"
            disabled={state.mutating}
          >
            删除样本
          </Button>
        </Popconfirm>
      ),
    }] : []),
  ]

  return (
    <div className="governance-view">
      {state.conflict && (
        <Alert
          type="warning"
          showIcon
          closable
          message="检测到版本冲突"
          description="其他员工或部门负责人已更新工作区。请刷新数据后再继续编辑，避免覆盖对方的审核结果。"
          action={<Button onClick={() => void state.reloadCases(state.page)}>刷新</Button>}
          style={{ marginBottom: 16 }}
        />
      )}
      <Row gutter={[16, 16]}>
        <Col xs={24} md={12} xl={6}>
          <Card className="governance-stat-card">
            <Statistic title="候选样本" value={state.dataset.case_count} prefix={<DatabaseOutlined />} suffix="条" />
            <Text type="secondary">
              {state.dataset.source_type === 'current_corpus'
                ? `冒烟测试绑定语料：${state.dataset.source_document_count ?? 0} 份文档`
                : `证据 Context：${state.dataset.context_count} 条`}
            </Text>
          </Card>
        </Col>
        <Col xs={24} md={12} xl={6}>
          <Card className="governance-stat-card">
            <Statistic title="待复核" value={state.counts.pending_review ?? 0} prefix={<SafetyCertificateOutlined />} />
            <Text type="secondary">草稿：{state.counts.draft ?? 0}</Text>
          </Card>
        </Col>
        <Col xs={24} md={12} xl={6}>
          <Card className="governance-stat-card">
            <Statistic title="已批准" value={state.counts.approved ?? 0} prefix={<CheckCircleOutlined />} />
            <Progress percent={approvalPercent} size="small" showInfo={false} strokeColor="#17b26a" />
          </Card>
        </Col>
        <Col xs={24} md={12} xl={6}>
          <Card className="governance-stat-card">
            <Statistic title="冻结准备" value={freezeReady ? '就绪' : '未就绪'} prefix={<FileProtectOutlined />} />
            <Text type={freezeReady ? 'success' : 'secondary'}>
              类别 {state.dataset.covered_category_count}/{state.dataset.required_category_count}
            </Text>
          </Card>
        </Col>
      </Row>

      <Card className="governance-panel" style={{ marginTop: 16 }}>
        <div className="governance-toolbar">
          <div>
            <Title level={4} style={{ margin: 0 }}>样本治理台</Title>
            <Text type="secondary">
              工作区第 {state.dataset.revision} 版 · 更新时间 {state.dataset.updated_at ? dayjs(state.dataset.updated_at).format('YYYY-MM-DD HH:mm') : '—'}
            </Text>
          </div>
          <Select
            aria-label="数据集工作台数据集"
            showSearch
            optionFilterProp="label"
            value={state.dataset.dataset_id}
            loading={state.loading && !state.cases.length}
            onChange={(value) => void state.selectDataset(value)}
            options={state.datasets.map((item) => ({
              value: item.dataset_id,
              label: evaluationDatasetLabel(item),
            }))}
            style={{ minWidth: 300 }}
          />
        </div>

        <div className="governance-filterbar" data-testid="dataset-filterbar">
          <div className="governance-filter-controls" data-testid="dataset-filter-controls">
            <Input
              allowClear
              prefix={<SearchOutlined />}
              placeholder="搜索样本 ID 或问题"
              value={state.query}
              onChange={(event) => state.setQuery(event.target.value)}
              onPressEnter={() => void state.reloadCases(1)}
              style={{ width: 280 }}
            />
            <Select
              allowClear placeholder="证据类别" value={state.category || undefined}
              onChange={(value) => state.setCategory(value ?? '')}
              options={Object.keys(state.dataset.category_counts).map((item) => ({ value: item, label: evidenceCategoryLabel(item) }))}
              style={{ width: 220 }}
            />
            <Select
              allowClear placeholder="审核状态" value={state.reviewStatus || undefined}
              onChange={(value) => state.setReviewStatus((value ?? '') as EvidenceReviewStatus | '')}
              options={[{ value: 'draft', label: '草稿' }, { value: 'pending_review', label: '待复核' }, { value: 'approved', label: '已批准' }]}
              style={{ width: 150 }}
            />
            <Button type="primary" onClick={() => void state.reloadCases(1)}>筛选</Button>
            <Button icon={<ReloadOutlined />} onClick={() => void state.load()}>刷新</Button>
          </div>
          <div className="governance-management-actions" data-testid="dataset-management-actions">
            <Space wrap>
              {canAdministerDataset && (
                <Tooltip title={isFrozen ? '已冻结；需先基于冻结版本创建新草稿' : blockers.join('；')}>
                  <Button icon={<FileProtectOutlined />} disabled={!freezeReady || isFrozen || state.mutating} onClick={() => setFreezeOpen(true)}>冻结版本</Button>
                </Tooltip>
              )}
              {canAdministerDataset && isFrozen && (
                <Button onClick={() => { setDraftVersion(state.dataset?.last_frozen_version || state.frozenVersions[0]?.version); setDraftOpen(true) }} disabled={!state.frozenVersions.length || state.mutating}>基于冻结版本创建新草稿</Button>
              )}
              {canAdministerDataset && REBINDABLE_SOURCE_DATASETS.has(state.dataset?.dataset_id ?? '') && !isFrozen && (
                <Button onClick={() => void state.rebindToCurrentCorpus()} disabled={state.mutating}>按当前语料重绑定</Button>
              )}
              {canAdministerDataset && state.dataset.source_type === 'current_corpus' && !isFrozen && (
                <Tooltip title="仅从冻结文档内容生成候选答案；生成后必须提交并通过人工复核才能冻结。">
                  <Button onClick={() => void state.populateReferenceAnswerCandidates()} disabled={state.mutating}>生成候选参考答案</Button>
                </Tooltip>
              )}
              <Button type="primary" icon={<PlusOutlined />} disabled={isFrozen || state.mutating} onClick={() => openCase({ ...emptyCase })}>新增样本</Button>
            </Space>
          </div>
        </div>

        {canBulkReview && (
          <div className="governance-bulk-actions" data-testid="dataset-bulk-actions">
            <Button
              disabled={isFrozen || selectedRowKeys.length === 0 || state.mutating}
              onClick={() => setBulkAction({ action: 'submit', selectionMode: 'selected' })}
            >批量提交复核</Button>
            <Button
              disabled={isFrozen || selectedRowKeys.length === 0 || state.mutating}
              onClick={() => setBulkAction({ action: 'approve', selectionMode: 'selected' })}
            >批量通过</Button>
            <Button
              disabled={isFrozen || state.total === 0 || state.mutating}
              onClick={() => setBulkAction({ action: 'submit', selectionMode: 'filtered' })}
            >一键提交复核</Button>
            <Button
              type="primary"
              disabled={isFrozen || state.total === 0 || state.mutating}
              onClick={() => setBulkAction({ action: 'approve', selectionMode: 'filtered' })}
            >一键通过</Button>
          </div>
        )}

        {state.dataset.source_type === 'current_corpus' && !isFrozen && state.counts.draft > 0 && (
          <Alert
            type="info"
            showIcon
            message="当前语料候选待人工复核"
            description="参考答案只会从冻结文档生成候选文本；请核对题干、Context 与事实后，提交复核并批准，再创建新的不可变版本。"
            style={{ marginBottom: 16 }}
          />
        )}

        <Table<EvidenceCase>
          rowKey="id"
          rowSelection={isFrozen ? undefined : {
            selectedRowKeys,
            preserveSelectedRowKeys: true,
            onChange: (keys) => setSelectedRowKeys(keys.map(String)),
          }}
          columns={columns}
          dataSource={state.cases}
          loading={state.loading}
          scroll={{ x: 1120 }}
          pagination={{
            current: state.page,
            pageSize: 20,
            total: state.total,
            showSizeChanger: false,
            showTotal: (value) => `共 ${value} 条`,
            onChange: (target) => void state.reloadCases(target),
          }}
          locale={{ emptyText: <Empty description="没有符合筛选条件的样本" /> }}
          onRow={(record) => ({ onDoubleClick: () => openCase(record) })}
        />
      </Card>

      {(state.dataset.last_frozen_version || state.lastFreeze) && (
        <Card size="small" style={{ marginTop: 16 }} title="冻结版本与不可变 Manifest" data-testid="frozen-version-card">
          <Space direction="vertical" size={8} style={{ width: '100%' }}>
            <Text>当前冻结版本：<Text strong>{evaluationVersionLabel(state.dataset.last_frozen_version || state.lastFreeze?.version)}</Text> <Text code>{state.dataset.last_frozen_version || state.lastFreeze?.version}</Text>{isFrozen && <Tag color="gold" style={{ marginLeft: 8 }}>已冻结，只读</Tag>}</Text>
            <Text>Manifest SHA-256：<Text code copyable>{state.dataset.last_frozen_manifest_sha256 || state.lastFreeze?.manifest_sha256}</Text></Text>
            <Text type="secondary">冻结时间：{state.dataset.last_frozen_at ? dayjs(state.dataset.last_frozen_at).format('YYYY-MM-DD HH:mm:ss') : '—'} · 制作者：{state.dataset.last_frozen_by_display_name || '系统管理员'}</Text>
            {state.frozenVersions.length > 1 && <Text type="secondary">历史版本：{state.frozenVersions.map((item) => evaluationVersionLabel(item.version)).join('；')}</Text>}
            {isFrozen && <Alert type="info" showIcon message="编辑已锁定" description="冻结材料是不可变发布输入。若要修订样本、审核记录或 Fixture，请先基于指定冻结版本创建新草稿；不会改写历史版本。" />}
          </Space>
        </Card>
      )}

      {state.lastFreeze && (
        <Alert
          type="success"
          showIcon
          style={{ marginTop: 16 }}
          message={`已生成冻结版本 ${state.lastFreeze.version}`}
          description={<span>Manifest SHA-256：<Text code copyable>{state.lastFreeze.manifest_sha256}</Text>。冻结仅完成数据集封存；baseline/calibration 由独立离线流程执行，生产 Gate 晋级需单独审批。</span>}
        />
      )}

      <EvidenceCaseDrawer
        open={drawerOpen}
        value={selected}
        currentUserId={user?.user_id}
        currentDepartmentId={user?.department_id}
        isDepartmentManager={user?.role === 'organization_admin' || user?.role === 'admin' || user?.is_department_manager}
        isOrganizationAdmin={user?.role === 'organization_admin'}
        readOnly={isFrozen}
        loading={state.mutating}
        onClose={() => setDrawerOpen(false)}
        onSave={saveCase}
        onLoadReviewers={state.loadReviewers}
        onSubmit={submitCase}
        onReview={reviewCase}
      />

      <Modal
        title={bulkAction?.action === 'approve' ? '批量通过样本' : '批量提交复核'}
        open={Boolean(bulkAction)}
        onCancel={() => setBulkAction(null)}
        onOk={() => void runBulkAction()}
        okText={bulkAction?.action === 'approve' ? '确认批量通过' : '确认提交复核'}
        confirmLoading={state.mutating}
        okButtonProps={{
          disabled: Boolean(
            bulkAction?.selectionMode === 'selected' && selectedRowKeys.length === 0,
          ),
        }}
      >
        <Paragraph>
          {bulkAction?.selectionMode === 'selected'
            ? `将${bulkAction.action === 'approve' ? '通过' : '提交'}所选 ${selectedRowKeys.length} 条样本${bulkAction.action === 'approve' ? '' : '进入复核'}`
            : '将处理当前筛选范围的全部分页样本'}
        </Paragraph>
        <Alert
          type="warning"
          showIcon
          message="Maker-checker 与负责人指派规则仍会逐条校验"
          description="不符合当前状态、部门范围、指派关系或制作者隔离要求的样本会保持不变，并在结果中计为跳过。"
        />
      </Modal>


      <Modal
        title="基于冻结版本创建新草稿"
        open={draftOpen}
        onCancel={() => setDraftOpen(false)}
        onOk={async () => { if (draftVersion) { await state.createDraftFromVersion(draftVersion); setDraftOpen(false) } }}
        okText="创建新草稿"
        confirmLoading={state.mutating}
        okButtonProps={{ disabled: !draftVersion }}
      >
        <Alert
          type="warning"
          showIcon
          message="不会修改冻结版本"
          description="系统会从所选不可变版本复制样本、Context 目录和 Fixture 配置，清除审批中的临时状态，并创建可编辑草稿。历史 Manifest 与冻结材料保持不变。"
          style={{ marginBottom: 16 }}
        />
        <Select
          aria-label="冻结版本"
          value={draftVersion}
          onChange={setDraftVersion}
          style={{ width: '100%' }}
          options={state.frozenVersions.map((item) => ({ value: item.version, label: `${item.version} · ${item.manifest_sha256.slice(0, 12)}…` }))}
        />
      </Modal>
      <Modal
        title="冻结数据集版本"
        open={freezeOpen}
        onCancel={() => setFreezeOpen(false)}
        onOk={async () => { await state.freezeDataset(); setFreezeOpen(false) }}
        okText="确认冻结"
        confirmLoading={state.mutating}
      >
        <Alert
          type="info" showIcon
          message="职责边界"
          description="前端仅用于制作、复核和冻结数据集。真实 baseline/calibration 仍由独立离线流程执行；生产 Gate 晋级必须单独审批。"
        />
      </Modal>
    </div>
  )
}
