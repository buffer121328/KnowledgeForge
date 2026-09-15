import { useCallback, useEffect, useMemo, useState } from 'react'
import { Alert, Button, Card, Collapse, Descriptions, Empty, Input, Select, Space, Steps, Tag, Timeline, Typography } from 'antd'
import { PlayCircleOutlined, ReloadOutlined } from '@ant-design/icons'
import { evaluationApi } from '@/api/evaluation'
import { getApiErrorInfo } from '@/api/client'
import type { EvidenceDatasetSummary, EvidenceDiagnosticSummary, EvidenceFrozenVersion, EvidenceReleaseAttempt, EvidenceReleaseWorkflow, EvidenceGateConfiguration } from '@/types'
import { formatChinaStandardTime } from '@/utils/time'
import { evaluationLabel, evaluationReasonLabel, evaluationStatusLabel } from '@/utils/evaluationDisplay'
import { evaluationSuiteLabel, evaluationVersionCompactLabel, evaluationVersionLabel } from '@/utils/evaluationDatasetLabels'

const { Text } = Typography
const stages = ['preflight', 'baseline', 'shadow', 'calibration']

/** Keep historical API mocks and older servers readable while Gate controls roll out. */
function loadEvidenceGate(): Promise<EvidenceGateConfiguration | null> {
  const api = evaluationApi as typeof evaluationApi & { getEvidenceGate?: () => Promise<EvidenceGateConfiguration> }
  return typeof api.getEvidenceGate === 'function' ? api.getEvidenceGate().catch(() => null) : Promise.resolve(null)
}

const visibleMetricKeys = new Set([
  'gate_mode', 'run_id', 'acceptance_version', 'calibration_version', 'manifest_sha256',
  'baseline_template_sha256', 'fixture_profile_sha256', 'quality_report_sha256',
  'calibration_record_sha256', 'evaluation_dataset_id', 'evaluation_version',
  'evaluation_manifest_sha256', 'evaluation_case_count', 'ragas_outcomes_sha256',
  'ragas_summary_sha256', 'ragas_judge_source', 'ragas_judge_model', 'p50', 'p95',
])

const lowerIsBetterMetrics = new Set([
  'noise_sensitivity',
  'unsupported_claim_rate', 'unsupported_resolution_rate', 'unsupported_answer_rate',
  'hallucination_rate', 'unauthorized_citation_rate', 'cross_scope_leakage_rate',
  'instruction_data_leakage_rate',
])

const fivePointMetrics = new Set([
  'rubrics_score_with_reference', 'rubrics_score_without_reference',
])

function metricInterpretation(metric: string, direction?: unknown, hardGate?: unknown) {
  const lowerIsBetter = direction === 'maximum' || (direction == null && lowerIsBetterMetrics.has(metric))
  const scale = fivePointMetrics.has(metric) ? '1–5' : '0–1'
  const gate = hardGate === true ? '；安全硬门' : ''
  return `${lowerIsBetter ? '越低越好' : '越高越好'}，${scale}${gate}`
}

function boundedMetricEntries(metrics: Record<string, unknown>) {
  return Object.entries(metrics)
    .filter(([key, value]) => visibleMetricKeys.has(key) && (typeof value === 'string' || typeof value === 'number'))
    .map(([key, value]) => [key, typeof value === 'string' && key.endsWith('_sha256') ? `${value.slice(0, 12)}…` : String(value)] as const)
}

function ragasReasonSummary(value: unknown) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return ''
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, count]) => typeof count === 'number' && Number.isFinite(count) && count > 0)
    .map(([code, count]) => `${evaluationReasonLabel(code)} ${count}`)
  return entries.length > 0 ? `（${entries.join('；')}）` : ''
}

function hasMissingRagasReference(metrics: Record<string, unknown>) {
  const coverage = metrics.ragas_coverage
  if (!coverage || typeof coverage !== 'object' || Array.isArray(coverage)) return false
  return Object.values(coverage as Record<string, unknown>).some((value) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return false
    const reasons = (value as Record<string, unknown>).failure_reasons
    return Boolean(reasons && typeof reasons === 'object' && !Array.isArray(reasons)
      && Number((reasons as Record<string, unknown>).missing_reference ?? 0) > 0)
  })
}

function calibrationSourceEntries(metrics: Record<string, unknown>) {
  const sources = metrics.source_identities
  if (!sources || typeof sources !== 'object' || Array.isArray(sources)) return []
  return Object.entries(sources as Record<string, unknown>).flatMap(([stage, value]) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const item = value as Record<string, unknown>
    const hash = typeof item.quality_report_sha256 === 'string' ? `${item.quality_report_sha256.slice(0, 12)}…` : '—'
    const runId = typeof item.run_id === 'string' ? item.run_id : '—'
    return [[stage, `${runId} · 报告 ${hash}`] as const]
  })
}

function calibrationDeltaEntries(metrics: Record<string, unknown>) {
  const deltas = metrics.metric_deltas
  if (!deltas || typeof deltas !== 'object' || Array.isArray(deltas)) return []
  return Object.entries(deltas as Record<string, unknown>).flatMap(([metric, value]) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const delta = (value as Record<string, unknown>).delta
    return typeof delta === 'number' && Number.isFinite(delta)
      ? [[metric, `${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`] as const]
      : []
  })
}

function ragasCoverageEntries(metrics: Record<string, unknown>) {
  const coverage = metrics.ragas_coverage
  if (!coverage || typeof coverage !== 'object' || Array.isArray(coverage)) return []
  return Object.entries(coverage as Record<string, unknown>).flatMap(([metric, value]) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const item = value as Record<string, unknown>
    const total = Number(item.total ?? 0)
    const scored = Number(item.scored ?? 0)
    const failedCount = Number(item.failed ?? 0)
    const applicable = scored + failedCount
    const notApplicable = `不适用 ${item.not_applicable ?? 0}${ragasReasonSummary(item.not_applicable_reasons)}`
    const skipped = Number(item.skipped ?? 0) > 0
      ? `；历史跳过 ${item.skipped}${ragasReasonSummary(item.skip_reasons)}`
      : ''
    const failed = `评分失败 ${failedCount}${ragasReasonSummary(item.failure_reasons)}`
    const exceptions = ragasReasonSummary(item.failure_exceptions)
    const finishes = ragasReasonSummary(item.failure_finish_reasons)
    const batchFallbacks = Number(item.batch_fallbacks ?? 0)
    const diagnostics = [
      exceptions ? `异常类型${exceptions}` : '',
      finishes ? `完成原因${finishes}` : '',
      batchFallbacks > 0 ? `批量回退 ${batchFallbacks}` : '',
    ].filter(Boolean).join('；')
    return [[
      metric,
      `适用 ${applicable}/${total}（已评分 ${scored}；${failed}）；${notApplicable}${skipped}${diagnostics ? `；诊断：${diagnostics}` : ''}；适用项均值 ${typeof item.mean === 'number' ? item.mean.toFixed(4) : '—'}（${metricInterpretation(metric)}）`,
    ] as const]
  })
}

function ragasCoverageSummary(metrics: Record<string, unknown>) {
  const coverage = metrics.ragas_coverage
  if (!coverage || typeof coverage !== 'object' || Array.isArray(coverage)) return null
  const totals = Object.values(coverage as Record<string, unknown>).reduce<{
    total: number
    scored: number
    notApplicable: number
    failed: number
  }>(
    (summary, value) => {
      if (!value || typeof value !== 'object' || Array.isArray(value)) return summary
      const item = value as Record<string, unknown>
      summary.total += Number(item.total ?? 0)
      summary.scored += Number(item.scored ?? 0)
      summary.notApplicable += Number(item.not_applicable ?? 0)
      summary.failed += Number(item.failed ?? 0)
      return summary
    },
    { total: 0, scored: 0, notApplicable: 0, failed: 0 },
  )
  if (totals.total <= 0) return null
  const evaluationCoverage = metrics.evaluation_coverage
  const planned = evaluationCoverage && typeof evaluationCoverage === 'object' && !Array.isArray(evaluationCoverage)
    ? Number((evaluationCoverage as Record<string, unknown>).ragas_planned_outcomes ?? 0)
    : 0
  const plannedText = planned > 0 ? `；类别策略应执行 ${planned} 项` : ''
  return `RAGAS 策略结果 ${totals.scored + totals.failed}/${totals.total}${plannedText}；按类别规则不执行 ${totals.notApplicable}；评分失败 ${totals.failed}。这里统计的是“样本 × 指标”组合；按规则不执行不代表样本漏评，也不会按 0 分计入均值。`
}

function evidenceGateContractCoverage(metrics: Record<string, unknown>) {
  const value = metrics.evidence_gate_contract_coverage
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const item = value as Record<string, unknown>
  const total = Number(item.total ?? 0)
  const scored = Number(item.scored ?? 0)
  const failedCount = Number(item.failed ?? 0)
  const notApplicable = Number(item.not_applicable ?? 0)
  const failed = `评分失败 ${failedCount}${ragasReasonSummary(item.failure_reasons)}`
  const mismatches = item.mismatch_by_category
  const mismatchEntries = mismatches && typeof mismatches === 'object' && !Array.isArray(mismatches)
    ? Object.entries(mismatches as Record<string, unknown>)
      .filter(([, count]) => typeof count === 'number' && Number.isFinite(count) && count > 0)
      .map(([category, count]) => `${evaluationLabel(category)} ${count}`)
    : []
  const mismatchSummary = mismatchEntries.length > 0 ? `；契约不匹配（${mismatchEntries.join('；')}）` : ''
  return `适用 ${scored + failedCount}/${total}（已评分 ${scored}；${failed}）；不适用 ${notApplicable}${mismatchSummary}；适用项均值 ${typeof item.mean === 'number' ? item.mean.toFixed(4) : '—'}`
}

function categoryMetricEntries(metrics: Record<string, unknown>) {
  const value = metrics.category_metrics
  if (!value || typeof value !== 'object' || Array.isArray(value)) return []
  const item = value as Record<string, unknown>
  const metricMap = item.metrics
  if (!metricMap || typeof metricMap !== 'object' || Array.isArray(metricMap)) return []
  return Object.entries(metricMap as Record<string, unknown>).flatMap(([metric, raw]) => {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return []
    const entry = raw as Record<string, unknown>
    const mean = typeof entry.mean === 'number' ? entry.mean.toFixed(4) : '—'
    const total = Number(entry.total ?? 0)
    const scored = Number(entry.scored ?? 0)
    const notApplicable = Number(entry.not_applicable ?? 0)
    const notApplicableText = notApplicable > 0
      ? `；不适用 ${notApplicable}${ragasReasonSummary(entry.not_applicable_reasons)}`
      : ''
    return [[metric, `已评分 ${scored}/${total}${notApplicableText}；均值 ${mean}（${metricInterpretation(metric, entry.direction, entry.hard_gate)}）`] as const]
  })
}

function categoryMetricCoverageSummary(metrics: Record<string, unknown>) {
  const value = metrics.category_metrics
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const item = value as Record<string, unknown>
  const covered = Number(item.case_count ?? 0)
  const expected = Number(metrics.evaluation_case_count ?? covered)
  if (covered <= 0 || expected <= 0) return null
  const categoryMetrics = metrics.category_metrics
  const outcomeCount = categoryMetrics && typeof categoryMetrics === 'object' && !Array.isArray(categoryMetrics)
    ? Number((categoryMetrics as Record<string, unknown>).outcome_count ?? 0)
    : 0
  const evaluationCoverage = metrics.evaluation_coverage
  const planned = evaluationCoverage && typeof evaluationCoverage === 'object' && !Array.isArray(evaluationCoverage)
    ? Number((evaluationCoverage as Record<string, unknown>).custom_planned_outcomes ?? 0)
    : 0
  const outcomeText = outcomeCount > 0 && planned > 0 ? `；类别专项结果 ${outcomeCount}/${planned}` : ''
  return `${covered}/${expected} 条样本已有类别专项评测${outcomeText}；普通答案、拒答、权限、故障和安全题型分别使用适合自身语义的指标。`
}

function diagnosticEntries(value: Record<string, Record<string, number | string | null>>) {
  return Object.entries(value).map(([name, metrics]) => [
    name,
    Object.entries(metrics).map(([key, item]) => `${evaluationLabel(key)} ${item ?? '—'}`).join('；'),
  ] as const)
}

function DiagnosticSummary({ summary }: { summary?: EvidenceDiagnosticSummary }) {
  if (!summary) return null
  if (summary.availability === 'not_available') return <Text type="secondary">本次运行无该诊断</Text>
  if (summary.availability === 'unsupported_schema') return <Text type="secondary">本次诊断版本暂不支持展示</Text>
  const root = summary.root_cause
  return <Space direction="vertical" size={2} style={{ width: '100%' }}>
    <Text strong>诊断四层结果</Text>
    {diagnosticEntries(summary.categories).map(([name, value]) => <Text key={`category-${name}`}>类别 · {evaluationLabel(name)}：{value}</Text>)}
    {diagnosticEntries(summary.stages).map(([name, value]) => <Text key={`stage-${name}`}>阶段 · {evaluationLabel(name)}：{value}</Text>)}
    {diagnosticEntries(summary.metric_coverage).map(([name, value]) => <Text key={`metric-${name}`}>指标适用性 · {evaluationLabel(name)}：{value}</Text>)}
    {Object.entries(summary.route_transitions).map(([route, count]) => <Text key={`route-${route}`}>路由迁移 · {route}：{count}</Text>)}
    {root && <Text>根因：{evaluationLabel(root.decision)} · {evaluationReasonLabel(root.reason_code)} · 配对 {root.paired_count ?? '—'}</Text>}
    {summary.hard_gates.length > 0 && <Text type="danger">Hard gate：{summary.hard_gates.map(evaluationReasonLabel).join('；')}</Text>}
  </Space>
}

function ragasMean(metrics: Record<string, unknown>, metric: string) {
  const coverage = metrics.ragas_coverage
  if (!coverage || typeof coverage !== 'object' || Array.isArray(coverage)) return null
  const value = (coverage as Record<string, unknown>)[metric]
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const mean = (value as Record<string, unknown>).mean
  return typeof mean === 'number' && Number.isFinite(mean) ? mean : null
}

function categoryMetricMean(metrics: Record<string, unknown>, metric: string) {
  const categoryMetrics = metrics.category_metrics
  if (!categoryMetrics || typeof categoryMetrics !== 'object' || Array.isArray(categoryMetrics)) return null
  const metricMap = (categoryMetrics as Record<string, unknown>).metrics
  if (!metricMap || typeof metricMap !== 'object' || Array.isArray(metricMap)) return null
  const value = (metricMap as Record<string, unknown>)[metric]
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const mean = (value as Record<string, unknown>).mean
  return typeof mean === 'number' && Number.isFinite(mean) ? mean : null
}

function evidenceGateContractMean(metrics: Record<string, unknown>) {
  const contract = metrics.evidence_gate_contract_coverage
  if (!contract || typeof contract !== 'object' || Array.isArray(contract)) return null
  const mean = (contract as Record<string, unknown>).mean
  return typeof mean === 'number' && Number.isFinite(mean) ? mean : null
}

function coverageHeadline(metrics: Record<string, unknown>) {
  const categoryMetrics = metrics.category_metrics
  const evaluationCoverage = metrics.evaluation_coverage
  const caseCount = categoryMetrics && typeof categoryMetrics === 'object' && !Array.isArray(categoryMetrics)
    ? Number((categoryMetrics as Record<string, unknown>).case_count ?? 0)
    : 0
  const expectedCases = Number(metrics.evaluation_case_count ?? caseCount)
  const customScored = categoryMetrics && typeof categoryMetrics === 'object' && !Array.isArray(categoryMetrics)
    ? Number((categoryMetrics as Record<string, unknown>).outcome_count ?? 0)
    : 0
  const customPlanned = evaluationCoverage && typeof evaluationCoverage === 'object' && !Array.isArray(evaluationCoverage)
    ? Number((evaluationCoverage as Record<string, unknown>).custom_planned_outcomes ?? 0)
    : 0
  const ragasCoverage = metrics.ragas_coverage
  const ragasTotals = ragasCoverage && typeof ragasCoverage === 'object' && !Array.isArray(ragasCoverage)
    ? Object.values(ragasCoverage as Record<string, unknown>).reduce<{ scored: number; failed: number }>((summary, value) => {
      if (!value || typeof value !== 'object' || Array.isArray(value)) return summary
      const item = value as Record<string, unknown>
      summary.scored += Number(item.scored ?? 0)
      summary.failed += Number(item.failed ?? 0)
      return summary
    }, { scored: 0, failed: 0 })
    : { scored: 0, failed: 0 }
  const ragasPlanned = evaluationCoverage && typeof evaluationCoverage === 'object' && !Array.isArray(evaluationCoverage)
    ? Number((evaluationCoverage as Record<string, unknown>).ragas_planned_outcomes ?? 0)
    : 0
  if (caseCount <= 0 && ragasPlanned <= 0 && customPlanned <= 0) return null
  return {
    caseCount,
    expectedCases,
    customScored,
    customPlanned,
    ragasCompleted: ragasTotals.scored + ragasTotals.failed,
    ragasFailed: ragasTotals.failed,
    ragasPlanned,
  }
}

function keyMetricEntries(metrics: Record<string, unknown>) {
  const metricIds = ['faithfulness', 'factual_correctness', 'context_recall', 'answer_relevancy']
  const entries = metricIds.flatMap((metric) => {
    const mean = ragasMean(metrics, metric)
    return mean === null ? [] : [[metric, mean] as const]
  })
  const contract = metrics.evidence_gate_contract_coverage
  if (contract && typeof contract === 'object' && !Array.isArray(contract)) {
    const mean = (contract as Record<string, unknown>).mean
    if (typeof mean === 'number' && Number.isFinite(mean)) entries.push(['evidence_gate_contract', mean])
  }
  return entries
}

function attemptRiskMessages(metrics: Record<string, unknown>, summary?: EvidenceDiagnosticSummary) {
  const risks: string[] = []
  const factualCorrectness = ragasMean(metrics, 'factual_correctness')
  const contextRecall = ragasMean(metrics, 'context_recall')
  const exactEvidenceRetrieval = categoryMetricMean(metrics, 'exact_evidence_retrieval')
  const behaviorContract = evidenceGateContractMean(metrics)
  const routeMetrics = [
    'response_route_correctness', 'clarification_route', 'missing_record_route',
    'authorization_route', 'zero_results_route',
  ].flatMap((metric) => {
    const mean = categoryMetricMean(metrics, metric)
    return mean === null ? [] : [[metric, mean] as const]
  })
  if (factualCorrectness !== null && factualCorrectness < 0.5) risks.push(`事实正确性仅 ${factualCorrectness.toFixed(4)}，回答内容与参考事实存在明显差距。`)
  if (contextRecall !== null && contextRecall < 0.7) risks.push(`上下文召回率为 ${contextRecall.toFixed(4)}，仍有应命中的证据没有进入最终上下文。`)
  if (exactEvidenceRetrieval !== null && exactEvidenceRetrieval < 0.6) risks.push(`精确证据召回为 ${exactEvidenceRetrieval.toFixed(4)}，需要检查 Chunk 身份映射和检索排序。`)
  const lowRouteMetrics = routeMetrics.filter(([, mean]) => mean < 0.5)
  if (lowRouteMetrics.length > 0) {
    if (behaviorContract !== null && behaviorContract >= 1) {
      risks.push(`类别专项中的${lowRouteMetrics.map(([metric]) => evaluationLabel(metric)).join('、')}偏低；但正式行为契约已通过，请结合阶段详情核查缺失字段或证据覆盖。`)
    } else {
      risks.push('部分拒答、澄清、权限或缺记录题型的路由正确率偏低。')
    }
  }
  if (summary?.hard_gates.length) risks.push(`存在安全硬门：${summary.hard_gates.map(evaluationReasonLabel).join('；')}。`)
  return risks.slice(0, 3)
}

function AttemptTimelineContent({ item, onViewRuns }: { item: EvidenceReleaseAttempt; onViewRuns?: (runId: string) => void }) {
  const allMetrics = item.metrics || {}
  const metrics = boundedMetricEntries(allMetrics)
  const ragasCoverage = ragasCoverageEntries(allMetrics)
  const ragasSummary = ragasCoverageSummary(allMetrics)
  const contractCoverage = evidenceGateContractCoverage(allMetrics)
  const categoryMetrics = categoryMetricEntries(allMetrics)
  const categoryCoverage = categoryMetricCoverageSummary(allMetrics)
  const headline = coverageHeadline(allMetrics)
  const keyMetrics = keyMetricEntries(allMetrics)
  const risks = attemptRiskMessages(allMetrics, item.diagnostic_summary)
  const isGateOffBaseline = item.stage === 'baseline' && item.metrics?.gate_mode === 'off'
  const runId = typeof item.metrics?.run_id === 'string' ? item.metrics.run_id : null
  const sources = calibrationSourceEntries(allMetrics)
  const deltas = calibrationDeltaEntries(allMetrics)
  const detailsAvailable = metrics.length > 0 || ragasCoverage.length > 0 || categoryMetrics.length > 0
    || Boolean(contractCoverage) || Boolean(item.diagnostic_summary) || sources.length > 0 || deltas.length > 0
  const stageLabel = item.stage === 'calibration' && allMetrics.acceptance_version === 'smoke-acceptance-v1'
    ? '冒烟验收'
    : evaluationLabel(item.stage)
  const reasonLabel = item.reason_code === 'ragas_required_score_failed' && hasMissingRagasReference(allMetrics)
    ? evaluationReasonLabel('ragas_reference_missing')
    : evaluationReasonLabel(item.reason_code)

  return <Space direction="vertical" size={8} style={{ width: '100%' }}>
    <span><Text strong>{stageLabel}</Text> · 第 {item.attempt_number} 次 · {evaluationStatusLabel(item.status)}</span>
    {item.reason_code && <Text type={item.status === 'failed' ? 'danger' : 'secondary'}>原因：{reasonLabel}</Text>}
    <Text type="secondary">开始：{formatChinaStandardTime(item.started_at)} · 结束：{formatChinaStandardTime(item.finished_at)}</Text>
    {headline && <Alert
      type={headline.ragasFailed > 0 || item.status === 'failed' ? 'warning' : 'success'}
      showIcon
      message={headline.ragasFailed > 0 ? '评测已产出结果，但仍有评分失败' : '评测覆盖完整'}
      description={<Space wrap size={[8, 4]}>
        {headline.expectedCases > 0 && <Tag color={headline.caseCount === headline.expectedCases ? 'success' : 'warning'}>样本 {headline.caseCount}/{headline.expectedCases}</Tag>}
        {headline.ragasPlanned > 0 && <Tag color={headline.ragasCompleted === headline.ragasPlanned ? 'success' : 'warning'}>RAGAS 计划 {headline.ragasCompleted}/{headline.ragasPlanned}</Tag>}
        {headline.customPlanned > 0 && <Tag color={headline.customScored === headline.customPlanned ? 'success' : 'warning'}>类别专项 {headline.customScored}/{headline.customPlanned}</Tag>}
        <Tag color={headline.ragasFailed > 0 ? 'error' : 'success'}>评分失败 {headline.ragasFailed}</Tag>
      </Space>}
    />}
    {keyMetrics.length > 0 && <Space wrap size={[4, 4]}>{keyMetrics.map(([metric, mean]) => <Tag key={metric}>{evaluationLabel(metric)}：{mean.toFixed(4)}</Tag>)}</Space>}
    {risks.length > 0 && <Alert type="warning" showIcon message="重点关注" description={<Space direction="vertical" size={0}>{risks.map((risk) => <Text key={risk}>{risk}</Text>)}</Space>} />}
    {isGateOffBaseline && <Alert type="info" showIcon message="本次为证据门关闭的对照基线；路由、拒答、权限和澄清类结果应与影子评测配对比较，不能当作启用证据门后的最终能力。" />}
    {detailsAvailable && <Collapse
      size="small"
      items={[{
        key: 'details',
        label: '查看详细评测数据与审计记录',
        children: <Space direction="vertical" size={10} style={{ width: '100%' }}>
          {metrics.length > 0 && <Space wrap size={4}>{metrics.map(([key, value]) => <Tag key={key}>{evaluationLabel(key)}：{value}</Tag>)}</Space>}
          {categoryCoverage && <Text>{categoryCoverage}</Text>}
          {ragasCoverage.length > 0 && <Space direction="vertical" size={0}><Text strong>RAGAS 适用性覆盖</Text>{ragasSummary && <Text>{ragasSummary}</Text>}{ragasCoverage.map(([metric, value]) => <Text key={metric}>{evaluationLabel(metric)}：{value}</Text>)}</Space>}
          {categoryMetrics.length > 0 && <Space direction="vertical" size={0}><Text strong>类别专项指标（不与 RAGAS 均值合并）</Text>{categoryMetrics.map(([metric, value]) => <Text key={metric}>{evaluationLabel(metric)}：{value}</Text>)}</Space>}
          {contractCoverage && <Space direction="vertical" size={0}><Text strong>正式行为契约覆盖</Text><Text>{contractCoverage}</Text></Space>}
          <DiagnosticSummary summary={item.diagnostic_summary} />
          {sources.length > 0 && <Space direction="vertical" size={0}><Text strong>校准来源</Text>{sources.map(([stage, value]) => <Text key={stage}>{evaluationLabel(stage)}：{value}</Text>)}</Space>}
          {deltas.length > 0 && <Space direction="vertical" size={0}><Text strong>基线→影子指标差异</Text>{deltas.map(([metric, value]) => <Text key={metric}>{evaluationLabel(metric)}：{value}</Text>)}</Space>}
        </Space>,
      }]}
    />}
    {runId && onViewRuns && <Button type="link" size="small" style={{ padding: 0 }} onClick={() => onViewRuns(runId)}>查看运行结果 {runId}</Button>}
  </Space>
}

export interface ReleaseEvaluationPanelProps {
  onViewRuns?: (runId: string) => void
}

/** Company-admin view for launching a release from a frozen corpus and a 12/50/100 question suite. */
export default function ReleaseEvaluationPanel({ onViewRuns }: ReleaseEvaluationPanelProps) {
  const [datasets, setDatasets] = useState<EvidenceDatasetSummary[]>([])
  const [selectedTargetDatasetId, setSelectedTargetDatasetId] = useState<string>()
  const [targetVersions, setTargetVersions] = useState<EvidenceFrozenVersion[]>([])
  const [selectedTargetVersion, setSelectedTargetVersion] = useState<string>()
  const [selectedFormalDatasetId, setSelectedFormalDatasetId] = useState<string>()
  const [formalVersions, setFormalVersions] = useState<EvidenceFrozenVersion[]>([])
  const [selectedFormalVersion, setSelectedFormalVersion] = useState<string>()
  const [workflows, setWorkflows] = useState<EvidenceReleaseWorkflow[]>([])
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>()
  const [workflow, setWorkflow] = useState<EvidenceReleaseWorkflow | null>(null)
  const [attempts, setAttempts] = useState<EvidenceReleaseAttempt[]>([])
  const [gate, setGate] = useState<EvidenceGateConfiguration | null>(null)
  const [reviewReason, setReviewReason] = useState('')
  const [targetMode, setTargetMode] = useState<'shadow' | 'enforce'>('shadow')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refreshSourceDataset = useMemo(
    () => datasets
      .filter((item) => item.active !== false && item.source_type === 'current_corpus')
      .sort((a, b) => String(b.updated_at || '').localeCompare(String(a.updated_at || '')))[0] ?? null,
    [datasets],
  )
  const corpusRefreshSourceId = refreshSourceDataset?.dataset_id
    ?? datasets.find((item) => item.dataset_id === 'evidence-gates-v1')?.dataset_id
    ?? 'evidence-gates-v1'

  const refreshCorpus = async () => {
    try {
      setLoading(true); setError(null)
      await evaluationApi.refreshCurrentCorpus(corpusRefreshSourceId)
      await refresh()
    } catch (caught) {
      const detail = getApiErrorInfo(caught)
      setError(detail?.message ? `刷新失败：${detail.message}` : '刷新失败：请确认当前目录已有 41 份已入库文档。')
    } finally { setLoading(false) }
  }

  const targetCorpusDatasets = useMemo(
    () => datasets
      .filter((item) => item.status === 'frozen' && item.active !== false && item.source_type === 'current_corpus')
      .sort((a, b) => String(b.updated_at || '').localeCompare(String(a.updated_at || '')))
      .slice(0, 1),
    [datasets],
  )
  const formalDatasets = useMemo(
    () => datasets
      .filter((item) => (item.status === 'frozen' || Boolean(item.last_frozen_version))
        && item.active !== false
        && [12, 50, 100].includes(item.case_count)
        && (item.dataset_id === selectedTargetDatasetId || ['evidence-gates-v1', 'evidence-gates-v1-routine'].includes(item.dataset_id)))
      .sort((left, right) => [100, 50, 12].indexOf(left.case_count) - [100, 50, 12].indexOf(right.case_count)),
    [datasets, selectedTargetDatasetId],
  )
  const selectedTargetDataset = useMemo(
    () => targetCorpusDatasets.find((item) => item.dataset_id === selectedTargetDatasetId) ?? null,
    [selectedTargetDatasetId, targetCorpusDatasets],
  )
  const selectedFormalSuite = useMemo(
    () => formalDatasets.find((item) => item.dataset_id === selectedFormalDatasetId) ?? null,
    [formalDatasets, selectedFormalDatasetId],
  )

  const loadTargetVersionHistory = () => {
    if (!selectedTargetDatasetId || targetVersions.length > 0) return
    void evaluationApi.listFrozenVersions(selectedTargetDatasetId).then(setTargetVersions).catch(() => undefined)
  }
  const loadFormalVersionHistory = () => {
    if (!selectedFormalDatasetId || formalVersions.length > 0) return
    void evaluationApi.listFrozenVersions(selectedFormalDatasetId).then(setFormalVersions).catch(() => undefined)
  }

  const refresh = useCallback(async (workflowId?: string) => {
    setLoading(true); setError(null)
    try {
      const [nextWorkflows, datasetResponse, nextGate] = await Promise.all([
        evaluationApi.listReleaseWorkflows(),
        evaluationApi.listDatasets(),
        loadEvidenceGate(),
      ])
      setDatasets(datasetResponse.datasets)
      setWorkflows(nextWorkflows)
      if (nextGate) setGate(nextGate)
      setWorkflow((current) => {
        const summary = nextWorkflows.find((item) => item.workflow_id === current?.workflow_id) ?? nextWorkflows[0]
        return summary ? { ...current, ...summary } : null
      })
      setSelectedWorkflowId((current) => nextWorkflows.some((item) => item.workflow_id === current) ? current : nextWorkflows[0]?.workflow_id)
      setSelectedTargetDatasetId((current) => {
        if (datasetResponse.datasets.some((item) => item.dataset_id === current && item.status === 'frozen' && item.active !== false && item.source_type === 'current_corpus')) return current
        return datasetResponse.datasets.filter((item) => item.status === 'frozen' && item.active !== false && item.source_type === 'current_corpus').sort((a, b) => String(b.updated_at || '').localeCompare(String(a.updated_at || '')))[0]?.dataset_id
      })
      setSelectedFormalDatasetId((current) => {
        if (datasetResponse.datasets.some((item) => item.dataset_id === current && (item.status === 'frozen' || Boolean(item.last_frozen_version)) && item.active !== false && (item.source_type === 'current_corpus' || ['evidence-gates-v1', 'evidence-gates-v1-routine'].includes(item.dataset_id)))) return current
        return datasetResponse.datasets.find((item) => item.dataset_id === 'evidence-gates-v1-routine' && (item.status === 'frozen' || Boolean(item.last_frozen_version)))?.dataset_id
          ?? datasetResponse.datasets.find((item) => item.dataset_id === 'evidence-gates-v1' && (item.status === 'frozen' || Boolean(item.last_frozen_version)))?.dataset_id
      })

      // A user-triggered refresh must also re-read the selected workflow's
      // detail and attempts.  The bootstrap/polling path intentionally stays
      // lightweight, while the button path gets a fresh server snapshot.
      if (workflowId) {
        try {
          const [detail, nextAttempts] = await Promise.all([
            evaluationApi.getReleaseWorkflow(workflowId),
            evaluationApi.listReleaseAttempts(workflowId),
          ])
          setWorkflow(detail)
          setAttempts(nextAttempts)
        } catch {
          // Keep the last known detail/attempts visible when only the deep
          // refresh fails; the list refresh above is still useful.
          setError('无法读取当前工作流详情，请稍后重试。')
        }
      }
    } catch {
      setError('无法读取发布评测状态，请检查公司管理员权限、数据库迁移和 Worker 服务。')
    } finally { setLoading(false) }
  }, [])

  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => {
    if (!selectedWorkflowId) {
      setWorkflow(null); setAttempts([])
      return
    }
    const summary = workflows.find((item) => item.workflow_id === selectedWorkflowId)
    if (summary) setWorkflow((current) => current?.workflow_id === selectedWorkflowId ? { ...current, ...summary } : summary)
    let current = true
    void evaluationApi.listReleaseAttempts(selectedWorkflowId).then((nextAttempts) => {
      if (!current) return
      setAttempts(nextAttempts)
    }).catch(() => { if (current) setAttempts([]) })
    return () => { current = false }
  }, [selectedWorkflowId, workflows])

  const review = async (decision: 'approve' | 'reject') => {
    if (!workflow || !reviewReason.trim()) { setError('请填写审批或驳回原因。'); return }
    try {
      setLoading(true); setError('')
      const next = await evaluationApi.reviewReleaseWorkflow(workflow.workflow_id, decision, workflow.revision, reviewReason.trim(), targetMode)
      setWorkflow(next); setAttempts(await evaluationApi.listReleaseAttempts(next.workflow_id)); setReviewReason('')
    } catch { setError('审批操作失败，请刷新状态后重试。') } finally { setLoading(false) }
  }

  const promote = async () => {
    if (!workflow) return
    try { setLoading(true); setError(''); setWorkflow(await evaluationApi.promoteReleaseWorkflow(workflow.workflow_id, workflow.revision)); setGate(await evaluationApi.getEvidenceGate()) }
    catch { setError('Gate 晋级失败，请刷新状态后重试。') } finally { setLoading(false) }
  }

  const rollback = async () => {
    if (!gate || !reviewReason.trim()) { setError('请填写 Gate 回滚原因。'); return }
    try { setLoading(true); setError(''); setGate(await evaluationApi.rollbackEvidenceGate(gate.revision, reviewReason.trim())); setReviewReason('') }
    catch { setError('Gate 回滚失败，请刷新状态后重试。') } finally { setLoading(false) }
  }

  useEffect(() => {
    if (!selectedTargetDatasetId) {
      setTargetVersions([]); setSelectedTargetVersion(undefined)
      return
    }
    const latestVersion = targetCorpusDatasets.find((item) => item.dataset_id === selectedTargetDatasetId)?.last_frozen_version
    if (latestVersion) setSelectedTargetVersion((value) => value ?? latestVersion)
    let current = true
    void evaluationApi.listFrozenVersions(selectedTargetDatasetId, 1)
      .then((next) => {
        if (!current) return
        setTargetVersions(next)
        setSelectedTargetVersion((value) => next.some((item) => item.version === value) ? value : next[0]?.version)
      })
      .catch(() => { if (current) { setTargetVersions([]); setSelectedTargetVersion(undefined) } })
    return () => { current = false }
  }, [selectedTargetDatasetId, targetCorpusDatasets])
  useEffect(() => {
    if (!selectedFormalDatasetId) {
      setFormalVersions([]); setSelectedFormalVersion(undefined)
      return
    }
    const latestVersion = formalDatasets.find((item) => item.dataset_id === selectedFormalDatasetId)?.last_frozen_version
    if (latestVersion) setSelectedFormalVersion((value) => value ?? latestVersion)
    let current = true
    void evaluationApi.listFrozenVersions(selectedFormalDatasetId, 1)
      .then((next) => {
        if (!current) return
        setFormalVersions(next)
        setSelectedFormalVersion((value) => next.some((item) => item.version === value) ? value : next[0]?.version)
      })
      .catch(() => { if (current) { setFormalVersions([]); setSelectedFormalVersion(undefined) } })
    return () => { current = false }
  }, [formalDatasets, selectedFormalDatasetId])
  useEffect(() => {
    if (!workflow || !['queued', 'running'].includes(workflow.status)) return
    const timer = window.setInterval(() => { void refresh() }, 5000)
    return () => window.clearInterval(timer)
  }, [refresh, workflow])

  const start = async () => {
    if (!selectedTargetDatasetId || !selectedTargetVersion || !selectedFormalDatasetId || !selectedFormalVersion) return
    setLoading(true); setError(null)
    try {
      const started = await evaluationApi.startReleaseWorkflow(
        selectedTargetDatasetId,
        selectedTargetVersion,
        selectedFormalDatasetId,
        selectedFormalVersion,
      )
      setWorkflows((current) => [started, ...current.filter((item) => item.workflow_id !== started.workflow_id)])
      setSelectedWorkflowId(started.workflow_id)
      setWorkflow(started)
      setAttempts(await evaluationApi.listReleaseAttempts(started.workflow_id))
    } catch {
      setError('无法发起流程：请确认已选择冻结语料版本，以及已审核冻结的冒烟测试 12 条、标准评测 50 条或全量评测 100 条。')
    } finally { setLoading(false) }
  }
  const retry = async () => {
    if (!workflow) return
    setLoading(true); setError(null)
    try {
      const queued = await evaluationApi.retryReleaseWorkflow(workflow.workflow_id, workflow.revision)
      setWorkflows((current) => [queued, ...current.filter((item) => item.workflow_id !== queued.workflow_id)])
      setWorkflow(queued)
      setAttempts(await evaluationApi.listReleaseAttempts(queued.workflow_id))
    } catch { setError('重试被拒绝：流程仍在运行、revision 已变更或冻结版本已不可用。') }
    finally { setLoading(false) }
  }
  const selectedSuiteCount = selectedFormalSuite?.case_count
  const launchReady = Boolean(selectedTargetDatasetId && selectedTargetVersion && selectedFormalDatasetId && selectedFormalVersion && [12, 50, 100].includes(selectedSuiteCount || 0))
  const launchReadinessMessage = !selectedTargetDatasetId
    ? '暂无已冻结的当前语料版本，请先完成审核并冻结版本。'
    : !selectedTargetVersion
      ? '请选择要评测的冻结语料版本。'
      : !selectedFormalDatasetId || !selectedFormalVersion
        ? '请选择已审核冻结的冒烟测试 12 条、标准评测 50 条或全量评测 100 条题目集版本。'
        : ![12, 50, 100].includes(selectedSuiteCount || 0)
          ? '评测套件必须是已审核冻结的冒烟测试 12 条、标准评测 50 条或全量评测 100 条。'
          : null
  const currentIndex = workflow ? Math.max(0, stages.indexOf(workflow.stage)) : 0

  return <Space direction="vertical" size={16} style={{ width: '100%' }}>
    {error && <Alert type="error" showIcon message={error} action={<Button size="small" onClick={() => void refresh(workflow?.workflow_id)}>刷新</Button>} />}
    <Card title="历史发布工作流" size="small">
      <Select aria-label="发布评测工作流历史" placeholder="选择历史发布工作流" value={selectedWorkflowId} onChange={setSelectedWorkflowId} style={{ width: '100%' }} options={workflows.map((item) => ({ value: item.workflow_id, label: `发布评测记录 · ${evaluationStatusLabel(item.status)} · ${formatChinaStandardTime(item.updated_at)}` }))} notFoundContent="暂无发布工作流" />
    </Card>
    <Card title="1. 选择评测内容" extra={<Button onClick={() => void refreshCorpus()} disabled={loading}>生成/刷新 41 份真实语料草稿</Button>}>
      {!targetCorpusDatasets.length ? <Empty description="暂无已冻结的当前语料版本，请先完成审核并冻结版本。" /> : <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Text strong>被测语料：当前真实入库文档（{selectedTargetDataset?.source_document_count || selectedTargetDataset?.context_count || 0} 份）</Text>
        <Select aria-label="被测语料冻结版本" placeholder="选择被测语料冻结版本" value={selectedTargetVersion} onChange={setSelectedTargetVersion} onOpenChange={(open) => { if (open) loadTargetVersionHistory() }} showSearch optionFilterProp="label" popupMatchSelectWidth={false} style={{ width: '100%', maxWidth: 520 }} disabled={!selectedTargetDatasetId} options={targetVersions.map((item) => ({ value: item.version, label: evaluationVersionCompactLabel(item.version) }))} />
        <Text strong>评测题目</Text>
        <Space wrap>
          <Select aria-label="评测题目集" placeholder="选择全量 100 条、标准 50 条或冒烟 12 条" value={selectedFormalDatasetId} onChange={setSelectedFormalDatasetId} showSearch optionFilterProp="label" popupMatchSelectWidth={false} style={{ minWidth: 300 }} options={formalDatasets.map((item) => ({ value: item.dataset_id, label: evaluationSuiteLabel(item) }))} />
          <Select aria-label="评测题目集冻结版本" placeholder="选择题目版本" value={selectedFormalVersion} onChange={setSelectedFormalVersion} onOpenChange={(open) => { if (open) loadFormalVersionHistory() }} showSearch optionFilterProp="label" popupMatchSelectWidth={false} style={{ minWidth: 320 }} disabled={!selectedFormalDatasetId} options={formalVersions.map((item) => ({ value: item.version, label: evaluationVersionCompactLabel(item.version) }))} />
          <Button type="primary" icon={<PlayCircleOutlined />} loading={loading} disabled={!launchReady || Boolean(workflow && ['queued', 'running'].includes(workflow.status))} onClick={() => void start()}>一键运行至完成</Button>
        </Space>
        {launchReadinessMessage && <Alert type="info" showIcon message="暂不能开始评测" description={launchReadinessMessage} />}
      </Space>}
    </Card>
    {workflow && <Card title="发布评测记录" extra={<Button icon={<ReloadOutlined />} onClick={() => void refresh(workflow.workflow_id)} loading={loading}>刷新</Button>}>
      <Steps current={currentIndex} status={workflow.status === 'failed' || workflow.status === 'rejected' ? 'error' : ['pending_approval', 'approved', 'promoted', 'completed'].includes(workflow.status) ? 'finish' : 'process'} items={stages.map((stage) => ({ title: stage === 'calibration' && workflow.evaluation_case_count === 12 ? '冒烟验收' : evaluationLabel(stage) }))} />
      <Descriptions column={{ xs: 1, md: 2 }} size="small" style={{ marginTop: 18 }}>
        {workflow.current_corpus_draft_id && <Descriptions.Item label="历史语料草稿">{workflow.current_corpus_draft_id}</Descriptions.Item>}
        <Descriptions.Item label="冻结语料版本">{workflow.version}</Descriptions.Item><Descriptions.Item label="语料 Manifest"><Text code>{workflow.manifest_sha256.slice(0, 16)}…</Text></Descriptions.Item>
        <Descriptions.Item label="评测题目集">{workflow.evaluation_case_count ? `${workflow.evaluation_case_count === 12 ? '冒烟测试' : workflow.evaluation_case_count === 50 ? '标准评测' : '全量评测'} · ${workflow.evaluation_case_count} 条题目` : '历史工作流未记录'}</Descriptions.Item><Descriptions.Item label="评测题目版本">{evaluationVersionLabel(workflow.evaluation_version)}</Descriptions.Item>
        <Descriptions.Item label="状态"><Tag color={workflow.status === 'failed' || workflow.status === 'rejected' ? 'error' : ['approved', 'promoted', 'completed'].includes(workflow.status) ? 'green' : 'blue'}>{evaluationStatusLabel(workflow.status)}</Tag></Descriptions.Item><Descriptions.Item label="失败原因">{evaluationReasonLabel(workflow.reason_code)}</Descriptions.Item>
        <Descriptions.Item label="校准版本">{workflow.calibration_version || '—'}</Descriptions.Item><Descriptions.Item label="发起人">{workflow.initiated_by_display_name || '历史用户'}</Descriptions.Item>
        <Descriptions.Item label="发起时间">{formatChinaStandardTime(workflow.created_at)}</Descriptions.Item><Descriptions.Item label="最近更新">{formatChinaStandardTime(workflow.updated_at)}</Descriptions.Item>
      </Descriptions>
      <Timeline style={{ marginTop: 18 }} items={attempts.map((item) => {
        return {
          color: item.status === 'failed' ? 'red' : item.status === 'succeeded' ? 'green' : 'blue',
          children: <AttemptTimelineContent item={item} onViewRuns={onViewRuns} />,
        }
      })} />
      <Space direction="vertical" style={{ width: '100%', marginTop: 12 }}>
        <Input.TextArea aria-label="审批或回滚原因" value={reviewReason} onChange={(event) => setReviewReason(event.target.value)} maxLength={1000} placeholder="填写审批、驳回或回滚原因（必填）" autoSize={{ minRows: 2, maxRows: 4 }} />
        <Space wrap>
          {workflow.status === 'pending_approval' && <><Select aria-label="目标 Gate 模式" value={targetMode} onChange={setTargetMode} style={{ width: 200 }} options={[{ value: 'shadow', label: '影子模式（只观测）' }, { value: 'enforce', label: '强制执行模式（拦截）' }]} /><Text type="secondary">Gate 模式决定评测通过后的生产策略：影子模式只记录，强制执行模式才会影响请求。</Text><Button type="primary" onClick={() => void review('approve')} loading={loading}>独立审批</Button><Button danger onClick={() => void review('reject')} loading={loading}>驳回</Button></>}
          {workflow.status === 'approved' && <Button type="primary" onClick={() => void promote()} loading={loading}>晋级至 {workflow.target_mode === 'enforce' ? '强制执行' : '影子模式'}</Button>}
          {gate && gate.mode !== 'off' && <Button danger onClick={() => void rollback()} loading={loading}>回滚 Gate（当前：{evaluationLabel(gate.mode)}）</Button>}
          <Button onClick={() => void retry()} disabled={!['failed', 'running'].includes(workflow.status) || loading}>
            {workflow.status === 'running' ? '检查并恢复中断任务' : '重试（新增尝试）'}
          </Button>
        </Space>
      </Space>
    </Card>}
  </Space>
}
