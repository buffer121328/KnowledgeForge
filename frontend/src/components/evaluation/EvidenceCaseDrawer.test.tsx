// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import EvidenceCaseDrawer from './EvidenceCaseDrawer'
import type { EvidenceCase } from '@/types'

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false, media: query, onchange: null, addListener: vi.fn(), removeListener: vi.fn(),
      addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn(),
    })),
  })
  class ResizeObserverMock { observe() {} unobserve() {} disconnect() {} }
  Object.defineProperty(window, 'ResizeObserver', { value: ResizeObserverMock })
})

afterEach(cleanup)

const sample: EvidenceCase = {
  id: 'case-001',
  question: '公司差旅标准是什么？',
  reference_answer: '差旅费用应按照公司制度报销。',
  category: 'fully_answerable',
  expected_response_status: 'answered',
  expected_evidence_states: ['direct_evidence'],
  expected_reason_codes: ['direct_support'],
  expected_source_document_ids: ['travel-policy'],
  expected_evidence_context_ids: ['travel-policy#chunk-1'],
  expected_evidence_sections: [],
  expected_citation_context_ids: ['travel-policy#chunk-1'],
  expected_missing_information_fields: [],
  required_fixture: 'standard',
  review_status: 'draft',
  case_revision: 2,
  last_editor_id: 'maker-a',
  department_id: 'finance',
  context_summaries: [],
}

function renderDrawer(onSave = vi.fn().mockResolvedValue(sample)) {
  render(
    <EvidenceCaseDrawer
      open
      value={sample}
      currentUserId="maker-a"
      currentDepartmentId="finance"
      isDepartmentManager
      loading={false}
      onClose={vi.fn()}
      onSave={onSave}
      onLoadReviewers={vi.fn().mockResolvedValue([])}
      onSubmit={vi.fn()}
      onReview={vi.fn()}
    />,
  )
  return onSave
}

describe('EvidenceCaseDrawer', () => {
  it('shows persisted Reason Codes as selected dropdown values and saves an array', async () => {
    const onSave = renderDrawer()

    expect(await screen.findByText('直接证据支持')).toBeTruthy()
    const reasonCodeSelect = screen.getByRole('combobox', { name: /预期 Reason Code/ })
    fireEvent.mouseDown(reasonCodeSelect)
    fireEvent.click(await screen.findByText('部分证据支持'))
    fireEvent.click(screen.getByRole('button', { name: /保\s*存/ }))

    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1))
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        expected_reason_codes: ['direct_support', 'partial_support'],
      }),
      false,
    )
  })

  it('selects an assigned reviewer in a dedicated submit dialog', async () => {
    const onSubmit = vi.fn().mockResolvedValue({ ...sample, review_status: 'pending_review', reviewer_id: 'manager-b' })
    render(
      <EvidenceCaseDrawer
        open value={sample} currentUserId="maker-a" currentDepartmentId="finance"
        loading={false} onClose={vi.fn()} onSave={vi.fn()}
        onLoadReviewers={vi.fn().mockResolvedValue([
          { user_id: 'manager-b', username: 'manager-b', display_name: '财务审核人' },
        ])}
        onSubmit={onSubmit} onReview={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText('提交复核'))
    fireEvent.mouseDown(await screen.findByRole('combobox', { name: '审核人' }))
    fireEvent.click(await screen.findByText('财务审核人（manager-b）'))
    fireEvent.click(screen.getByText('确认提交'))

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith(sample, 'manager-b'))
  })

  it('renders a read-only review summary with markdown contexts and separate decisions', async () => {
    render(
      <EvidenceCaseDrawer
        open
        value={{
          ...sample,
          review_status: 'pending_review',
          last_editor_id: 'system-import',
          reviewer_id: 'manager-b',
          reviewer_display_name: '组织审核人',
          submitted_at: '2026-08-09T02:31:57.960725+00:00',
          reviewed_at: '2026-08-14T10:34:55.330308+00:00',
          context_summaries: [{
            context_id: 'travel-policy#chunk-1',
            source_document_id: 'travel-policy',
            title: '差旅制度',
            department: 'finance',
            chunk_index: 1,
            content_excerpt: '**报销标准**\n\n- 凭票报销',
          }],
        }}
        currentUserId="manager-b" currentDepartmentId="finance" isOrganizationAdmin
        loading={false} onClose={vi.fn()} onSave={vi.fn()}
        onLoadReviewers={vi.fn()} onSubmit={vi.fn()} onReview={vi.fn()}
      />,
    )

    expect(await screen.findByText('复核内容')).toBeTruthy()
    expect(screen.queryByLabelText('问题')).toBeNull()
    expect(screen.getByText('报销标准').tagName).toBe('STRONG')
    expect(screen.getByText('系统导入')).toBeTruthy()
    expect(screen.getByText('组织审核人')).toBeTruthy()
    expect(screen.getByText('2026年08月09日 10:31')).toBeTruthy()
    expect(screen.getByText('2026年08月14日 18:34')).toBeTruthy()
    expect(screen.getByText('通过审核')).toBeTruthy()
    expect(screen.getByText('拒绝并退回')).toBeTruthy()
    expect(screen.queryByText('复核部门')).toBeNull()
  })
})
