import { useCallback, useEffect, useState } from 'react'
import { App } from 'antd'
import { evaluationApi } from '@/api/evaluation'
import type {
  EvaluationRecordItem,
  EvaluationRunDetail,
  EvaluationRunSummary,
  EvaluationRunReportResponse,
  EvaluationSpotCheckResponse,
  SpotCheckVerdict,
} from '@/types'

const PAGE_SIZE = 20

/** Manage evaluation runs list, run detail and per-question records state. */
export function useEvaluationRuns() {
  const { message } = App.useApp()

  const [runs, setRuns] = useState<EvaluationRunSummary[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [runsError, setRunsError] = useState<string | null>(null)

  const [selectedRun, setSelectedRun] = useState<EvaluationRunDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  const [records, setRecords] = useState<EvaluationRecordItem[]>([])
  const [recordsLoading, setRecordsLoading] = useState(false)
  const [recordsTotal, setRecordsTotal] = useState(0)
  const [page, setPage] = useState(1)

  const [anomalies, setAnomalies] = useState<EvaluationSpotCheckResponse | null>(null)
  const [anomaliesLoading, setAnomaliesLoading] = useState(false)
  const [submittingSpotCheck, setSubmittingSpotCheck] = useState(false)

  const [runReport, setRunReport] = useState<EvaluationRunReportResponse | null>(null)
  const [reportLoading, setReportLoading] = useState(false)

  /** Fetch the evaluation run list. */
  const fetchRuns = useCallback(async () => {
    setRunsLoading(true)
    setRunsError(null)
    try {
      const data = await evaluationApi.listRuns()
      setRuns(data.runs ?? [])
    } catch {
      setRunsError('加载评测结果失败')
      message.error('加载评测结果失败')
    } finally {
      setRunsLoading(false)
    }
  }, [message])

  /** Fetch the records of the given run for the target page. */
  const loadRecords = useCallback(
    async (runId: string, targetPage: number) => {
      setRecordsLoading(true)
      try {
        const data = await evaluationApi.getRecords(runId, targetPage, PAGE_SIZE)
        setRecords(data.records ?? [])
        setRecordsTotal(data.total ?? 0)
        setPage(data.page ?? targetPage)
      } catch {
        message.error('加载评测明细失败')
      } finally {
        setRecordsLoading(false)
      }
    },
    [message],
  )

  /** Open a run detail and load the first page of its records. */
  const openRun = useCallback(
    async (runId: string) => {
      setDetailLoading(true)
      try {
        const [detail, firstPage] = await Promise.all([
          evaluationApi.getRun(runId),
          evaluationApi.getRecords(runId, 1, PAGE_SIZE),
        ])
        setSelectedRun(detail)
        setRecords(firstPage.records ?? [])
        setRecordsTotal(firstPage.total ?? 0)
        setPage(1)
      } catch {
        message.error('加载评测详情失败')
      } finally {
        setDetailLoading(false)
      }
    },
    [message],
  )

  /** Fetch the synthesized evaluation report for the given run. */
  const loadReport = useCallback(
    async (runId: string) => {
      setReportLoading(true)
      try {
        setRunReport(await evaluationApi.getRunReport(runId))
      } catch {
        message.error('加载评测报告失败')
      } finally {
        setReportLoading(false)
      }
    },
    [message],
  )

  /** Fetch the anomaly/spot-check list for the given run. */
  const loadAnomalies = useCallback(
    async (runId: string) => {
      setAnomaliesLoading(true)
      try {
        setAnomalies(await evaluationApi.getAnomalies(runId))
      } catch {
        message.error('加载异常清单失败')
      } finally {
        setAnomaliesLoading(false)
      }
    },
    [message],
  )

  /** Submit a human spot-check verdict and refresh the anomaly list. */
  const submitSpotCheck = useCallback(
    async (runId: string, benchmarkId: string, verdict: SpotCheckVerdict, note: string) => {
      setSubmittingSpotCheck(true)
      try {
        setAnomalies(await evaluationApi.submitSpotCheck(runId, benchmarkId, verdict, note))
        message.success('抽检结论已保存')
      } catch {
        message.error('保存抽检结论失败')
      } finally {
        setSubmittingSpotCheck(false)
      }
    },
    [message],
  )

  /** Return to the run list view. */
  const backToList = useCallback(() => {
    setSelectedRun(null)
    setRecords([])
    setRecordsTotal(0)
    setPage(1)
    setAnomalies(null)
    setRunReport(null)
  }, [])

  /** Switch to the requested page of records for the current run. */
  const changePage = useCallback(
    async (targetPage: number) => {
      if (!selectedRun) return
      setPage(targetPage)
      await loadRecords(selectedRun.run_id, targetPage)
    },
    [selectedRun, loadRecords],
  )

  useEffect(() => {
    void fetchRuns()
  }, [fetchRuns])

  return {
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
  }
}
