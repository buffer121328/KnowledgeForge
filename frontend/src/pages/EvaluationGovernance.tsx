import { useState } from 'react'
import { Alert, Segmented, Space, Typography } from 'antd'
import { DatabaseOutlined, LineChartOutlined, SafetyCertificateOutlined } from '@ant-design/icons'
import EvidenceDatasetWorkbench from '@/components/evaluation/EvidenceDatasetWorkbench'
import EvaluationRuns from './EvaluationRuns'
import ReleaseEvaluationPanel from '@/components/evaluation/ReleaseEvaluationPanel'

const { Paragraph, Title } = Typography

/** Render the combined evidence authoring and offline run governance center. */
export default function EvaluationGovernance() {
  const [view, setView] = useState<'datasets' | 'release' | 'runs'>('datasets')
  const [releaseRunId, setReleaseRunId] = useState<string | undefined>()

  return (
    <div className="enterprise-page evaluation-governance-page">
      <div className="enterprise-page-header">
        <div>
          <Space align="center">
            <div className="page-title-icon"><DatabaseOutlined /></div>
            <div>
              <Title level={2} style={{ margin: 0 }}>评测治理中心</Title>
              <Paragraph type="secondary" style={{ margin: '4px 0 0' }}>
                统一制作证据门样本、执行 maker-checker 审核、冻结不可变数据版本并查看离线评测结果。
              </Paragraph>
            </div>
          </Space>
        </div>
        <Segmented
          size="large"
          value={view}
          onChange={(value) => setView(value as 'datasets' | 'release' | 'runs')}
          options={[
            { value: 'datasets', label: '数据集工作台', icon: <DatabaseOutlined /> },
            { value: 'release', label: '发布评测', icon: <SafetyCertificateOutlined /> },
            { value: 'runs', label: '运行结果', icon: <LineChartOutlined /> },
          ]}
        />
      </div>
      <Alert
        className="governance-boundary-alert"
        type="info"
        showIcon
        message="评测治理职责与发布范围"
        description="前端用于制作、复核、冻结并发起/查看发布评测；题目集只保留冒烟测试 12 条、标准评测 50 条和全量评测 100 条。真实 baseline、shadow、RAGAS 与 calibration 由后台 Worker 执行，不会自动改变生产 Gate。"
      />
      <div key={view} className="governance-view-switch">
        {view === 'datasets' ? <EvidenceDatasetWorkbench /> : view === 'release' ? <ReleaseEvaluationPanel onViewRuns={(runId) => { setReleaseRunId(runId); setView('runs') }} /> : <EvaluationRuns embedded initialRunId={releaseRunId} />}
      </div>
    </div>
  )
}
