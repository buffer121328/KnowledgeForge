// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react'
import { App } from 'antd'
import type { PropsWithChildren } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { EvidenceBulkActionResponse, EvidenceCasePage, EvidenceDatasetList, EvidenceDatasetSummary } from '@/types'

const evaluationApi = vi.hoisted(() => ({
  bulkApproveCases: vi.fn(),
  bulkSubmitCases: vi.fn(),
  getCases: vi.fn(),
  getDataset: vi.fn(),
  listDatasets: vi.fn(),
  listFrozenVersions: vi.fn(),
  populateReferenceAnswerCandidates: vi.fn(),
}))

vi.mock('@/api/evaluation', () => ({ evaluationApi }))

import { selectWorkbenchDatasets, useEvidenceDatasets } from './useEvidenceDatasets'

const dataset: EvidenceDatasetSummary = {
  dataset_id: 'evidence-gates-v1',
  title: '证据门候选集',
  revision: 1,
  status: 'authoring',
  case_count: 40,
  context_count: 0,
  counts: {},
  category_counts: {},
  required_category_count: 11,
  covered_category_count: 0,
  fixture_status: 'ready',
  last_frozen_version: null,
  updated_at: null,
}

function Wrapper({ children }: PropsWithChildren) {
  return <App>{children}</App>
}

afterEach(() => {
  vi.useRealTimers()
})

beforeEach(() => {
  vi.clearAllMocks()
  evaluationApi.listDatasets.mockResolvedValue({ datasets: [dataset] } satisfies EvidenceDatasetList)
  evaluationApi.getDataset.mockResolvedValue(dataset)
  evaluationApi.listFrozenVersions.mockResolvedValue([])
  evaluationApi.getCases.mockImplementation((datasetId: string, filters: { page?: number; pageSize?: number }) =>
    Promise.resolve({
      dataset_id: datasetId,
      revision: 1,
      page: filters.page ?? 1,
      page_size: filters.pageSize ?? 20,
      total: 40,
      cases: [],
    } satisfies EvidenceCasePage),
  )
})

describe('useEvidenceDatasets pagination', () => {
  it('keeps only full, standard, and the active 12-case smoke suite in business order', () => {
    const selected = selectWorkbenchDatasets([
      { ...dataset, dataset_id: 'current-corpus-old', source_type: 'current_corpus', active: false, case_count: 12, source_document_count: 40 },
      { ...dataset, dataset_id: 'evidence-gates-v1-routine', case_count: 50 },
      { ...dataset, dataset_id: 'future-suite', case_count: 25 },
      { ...dataset, dataset_id: 'current-corpus-current', source_type: 'current_corpus', active: true, case_count: 12, source_document_count: 41 },
      { ...dataset, dataset_id: 'evidence-gates-v1', case_count: 100 },
    ])

    expect(selected.map((item) => item.dataset_id)).toEqual([
      'evidence-gates-v1',
      'evidence-gates-v1-routine',
      'current-corpus-current',
    ])
  })

  it('keeps the selected page without rerunning the initial homepage request', async () => {
    const { result } = renderHook(() => useEvidenceDatasets(), { wrapper: Wrapper })
    await waitFor(() => expect(evaluationApi.getCases).toHaveBeenCalledTimes(1))

    await act(async () => { await result.current.reloadCases(2) })

    await waitFor(() => {
      expect(result.current.page).toBe(2)
      expect(evaluationApi.getCases).toHaveBeenCalledTimes(2)
    })
    expect(evaluationApi.getCases).toHaveBeenNthCalledWith(
      1,
      'evidence-gates-v1',
      expect.objectContaining({ page: 1, pageSize: 20 }),
    )
    expect(evaluationApi.getCases).toHaveBeenNthCalledWith(
      2,
      'evidence-gates-v1',
      expect.objectContaining({ page: 2, pageSize: 20 }),
    )
  })

  it('sends filtered bulk scope and refreshes the authoritative page and summary', async () => {
    const outcome: EvidenceBulkActionResponse = {
      revision: 2,
      matched_count: 40,
      processed_count: 38,
      skipped_count: 2,
      processed_case_ids: [],
      skipped_items: [
        { case_id: 'case-1', code: 'maker_checker_violation' },
        { case_id: 'case-2', code: 'assigned_reviewer_required' },
      ],
    }
    evaluationApi.bulkSubmitCases.mockResolvedValue(outcome)
    const { result } = renderHook(() => useEvidenceDatasets(), { wrapper: Wrapper })
    await waitFor(() => expect(evaluationApi.getCases).toHaveBeenCalledTimes(1))

    await act(async () => { await result.current.bulkSubmit('filtered', []) })

    expect(evaluationApi.bulkSubmitCases).toHaveBeenCalledWith(
      'evidence-gates-v1',
      {
        expected_revision: 1,
        selection_mode: 'filtered',
        case_ids: [],
        category: undefined,
        review_status: undefined,
        query: undefined,
      },
    )
    expect(evaluationApi.getCases).toHaveBeenCalledTimes(2)
    expect(evaluationApi.getDataset).toHaveBeenCalledWith('evidence-gates-v1')
  })

  it('uses only explicit IDs for selected bulk approval and exposes revision conflicts', async () => {
    evaluationApi.bulkApproveCases.mockRejectedValue({
      isAxiosError: true,
      response: { data: { detail: { code: 'revision_conflict' } } },
    })
    const { result } = renderHook(() => useEvidenceDatasets(), { wrapper: Wrapper })
    await waitFor(() => expect(evaluationApi.getCases).toHaveBeenCalledTimes(1))

    let caught: unknown
    await act(async () => {
      try {
        await result.current.bulkApprove('selected', ['case-1'])
      } catch (error) {
        caught = error
      }
    })

    expect(caught).toBeTruthy()
    expect(evaluationApi.bulkApproveCases).toHaveBeenCalledWith(
      'evidence-gates-v1',
      { expected_revision: 1, selection_mode: 'selected', case_ids: ['case-1'] },
    )
    await waitFor(() => expect(result.current.conflict).toBe(true))
  })

  it('fills document-grounded candidates then reloads the editable workspace', async () => {
    evaluationApi.populateReferenceAnswerCandidates.mockResolvedValue({
      dataset_id: 'evidence-gates-v1',
      revision: 2,
      populated_case_ids: ['eg-company-demo-01'],
      skipped: [],
    })
    const { result } = renderHook(() => useEvidenceDatasets(), { wrapper: Wrapper })
    await waitFor(() => expect(evaluationApi.getCases).toHaveBeenCalledTimes(1))

    await act(async () => { await result.current.populateReferenceAnswerCandidates() })

    expect(evaluationApi.populateReferenceAnswerCandidates).toHaveBeenCalledWith('evidence-gates-v1', 1)
    expect(evaluationApi.getCases).toHaveBeenCalledTimes(2)
    expect(evaluationApi.getDataset).toHaveBeenCalledWith('evidence-gates-v1')
  })
})
