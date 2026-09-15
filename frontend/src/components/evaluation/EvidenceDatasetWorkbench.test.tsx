// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { App } from 'antd'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { useEvidenceDatasets } from '@/pages/useEvidenceDatasets'
import EvidenceDatasetWorkbench from './EvidenceDatasetWorkbench'
import type { EvidenceCase } from '@/types'
import { readFileSync } from 'node:fs'

const auth = vi.hoisted(() => ({ user: { user_id: 'manager-b', username: 'manager-b', role: 'viewer', org_id: 'org-a', permissions: ['doc:read'], department_id: 'finance', is_department_manager: true } }))
vi.mock('@/stores/auth', () => ({ useAuthStore: (selector: (state: typeof auth) => unknown) => selector(auth) }))
vi.mock('@/pages/useEvidenceDatasets', () => ({ useEvidenceDatasets: vi.fn() }))
const mockedHook = vi.mocked(useEvidenceDatasets)

const sample: EvidenceCase = {
  id: 'case-001',
  question: '公司差旅标准是什么？',
  reference_answer: '差旅费用应按照公司制度报销。',
  category: 'fully_answerable',
  expected_response_status: 'answered',
  expected_evidence_states: ['direct_evidence'],
  expected_reason_codes: [],
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
  context_summaries: [{ context_id: 'travel-policy#chunk-1', content_excerpt: '差旅费用应按照公司制度报销。' }],
}

function state(overrides: Record<string, unknown> = {}) {
  const dataset = {
    dataset_id: 'evidence-gates-v1', title: 'Evidence gates', revision: 3,
    status: 'authoring', case_count: 1, context_count: 638,
    counts: { draft: 1, pending_review: 0, approved: 0 },
    category_counts: { fully_answerable: 1 }, required_category_count: 11,
    covered_category_count: 1, fixture_status: 'pending_operator_configuration',
  }
  return {
    bulkApprove: vi.fn(), bulkSubmit: vi.fn(), cases: [sample], category: '', conflict: false,
    counts: { draft: 1, pending_review: 0, approved: 0 },
    dataset,
    createDraftFromVersion: vi.fn(), datasets: [dataset], deleteCase: vi.fn().mockResolvedValue(undefined), error: null,
    freezeDataset: vi.fn(), frozenVersions: [], lastFreeze: null, load: vi.fn(), loadReviewers: vi.fn().mockResolvedValue([{ user_id: 'manager-b', username: 'manager-b', display_name: '财务审核人' }]), loading: false, mutating: false, page: 1, query: '',
    populateReferenceAnswerCandidates: vi.fn(), rebindToCurrentCorpus: vi.fn(), reloadCases: vi.fn(), reviewCase: vi.fn().mockResolvedValue(sample), reviewStatus: '',
    saveCase: vi.fn().mockResolvedValue(sample), setCategory: vi.fn(), setPage: vi.fn(),
    selectDataset: vi.fn(), setQuery: vi.fn(), setReviewStatus: vi.fn(), submitCase: vi.fn().mockResolvedValue({ ...sample, review_status: 'pending_review' }),
    total: 1,
    ...overrides,
  } as ReturnType<typeof useEvidenceDatasets>
}

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

beforeEach(() => { auth.user = { user_id: 'org-admin-b', username: 'org-admin-b', role: 'organization_admin', org_id: 'org-a', permissions: ['admin:manage'], department_id: 'finance', is_department_manager: true } })

afterEach(() => { cleanup(); vi.clearAllMocks() })

function renderWorkbench() {
  return render(<App><EvidenceDatasetWorkbench /></App>)
}

describe('EvidenceDatasetWorkbench', () => {
  it('shows current corpus documents instead of exposing an internal Context count', () => {
    mockedHook.mockReturnValue(state({
      dataset: {
        ...state().dataset,
        source_type: 'current_corpus',
        source_document_count: 41,
        context_count: 707,
      },
    }))
    renderWorkbench()

    expect(screen.getByText('冒烟测试绑定语料：41 份文档')).toBeTruthy()
    expect(screen.queryByText(/Context catalog/)).toBeNull()
  })

  it('renders loading and error states', () => {
    mockedHook.mockReturnValue(state({ dataset: null, loading: true }))
    const view = renderWorkbench()
    expect(screen.getByTestId('dataset-loading')).toBeTruthy()
    view.unmount()

    mockedHook.mockReturnValue(state({ dataset: null, loading: false, error: '服务不可用' }))
    renderWorkbench()
    expect(screen.getByText('无法加载数据集工作台')).toBeTruthy()
  })

  it('filters, opens a draft case and submits it for review', async () => {
    const current = state()
    mockedHook.mockReturnValue(current)
    renderWorkbench()

    fireEvent.change(screen.getByPlaceholderText('搜索样本 ID 或问题'), { target: { value: '差旅' } })
    expect(current.setQuery).toHaveBeenCalledWith('差旅')
    fireEvent.click(screen.getByText('case-001'))
    expect(await screen.findByText('已绑定 Context')).toBeTruthy()
    fireEvent.click(screen.getByText('提交复核'))
    fireEvent.mouseDown(await screen.findByRole('combobox', { name: '审核人' }))
    fireEvent.click(await screen.findByText('财务审核人（manager-b）'))
    fireEvent.click(screen.getByText('确认提交'))
    await waitFor(() => expect(current.submitCase).toHaveBeenCalledWith(sample, 'manager-b'))
  })

  it('allows switching from the formal suite to the 50-case routine suite', async () => {
    const current = state({
      datasets: [
        state().dataset,
        { ...state().dataset, dataset_id: 'evidence-gates-v1-routine', title: '日常基准', case_count: 50 },
      ],
    })
    mockedHook.mockReturnValue(current)
    renderWorkbench()

    fireEvent.mouseDown(screen.getByRole('combobox', { name: '数据集工作台数据集' }))
    fireEvent.click(await screen.findByText(/标准评测 · 50 条/))
    expect(current.selectDataset).toHaveBeenCalledWith('evidence-gates-v1-routine')
  })

  it.each(['evidence-gates-v1', 'evidence-gates-v1-routine'])(
    'offers current-corpus rebind for supported suite %s',
    async (datasetId) => {
      const rebindToCurrentCorpus = vi.fn().mockResolvedValue(undefined)
      const current = state({
        dataset: { ...state().dataset!, dataset_id: datasetId },
        rebindToCurrentCorpus,
      })
      mockedHook.mockReturnValue(current)
      renderWorkbench()

      fireEvent.click(screen.getByRole('button', { name: '按当前语料重绑定' }))
      await waitFor(() => expect(rebindToCurrentCorpus).toHaveBeenCalled())
    },
  )

  it('hides current-corpus rebind for derived or unknown datasets', () => {
    mockedHook.mockReturnValue(state({
      dataset: { ...state().dataset!, dataset_id: 'current-corpus-authoring-dataset' },
    }))
    renderWorkbench()

    expect(screen.queryByRole('button', { name: '按当前语料重绑定' })).toBeNull()
  })

  it('offers document-grounded answer candidates only for an editable current-corpus draft', async () => {
    const populateReferenceAnswerCandidates = vi.fn().mockResolvedValue(undefined)
    mockedHook.mockReturnValue(state({
      dataset: {
        ...state().dataset!,
        dataset_id: 'current-corpus-authoring-dataset',
        source_type: 'current_corpus',
      },
      populateReferenceAnswerCandidates,
    }))
    renderWorkbench()

    expect(screen.getByText('当前语料候选待人工复核')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '生成候选参考答案' }))
    await waitFor(() => expect(populateReferenceAnswerCandidates).toHaveBeenCalled())
  })

  it('disables self approval and displays revision conflict guidance', async () => {
    auth.user = { user_id: 'maker-a', username: 'maker-a', role: 'organization_admin', org_id: 'org-a', permissions: ['admin:manage'], department_id: 'finance', is_department_manager: true }
    const pending = { ...sample, review_status: 'pending_review' as const, reviewer_id: 'maker-a' }
    mockedHook.mockReturnValue(state({ cases: [pending], conflict: true, counts: { draft: 0, pending_review: 1, approved: 0 } }))
    renderWorkbench()

    expect(screen.getByText('检测到版本冲突')).toBeTruthy()
    fireEvent.click(screen.getByText('case-001'))
    expect(await screen.findByText('Maker-checker 已生效')).toBeTruthy()
    expect(screen.getByText(/其他公司管理员/)).toBeTruthy()
    expect((screen.getByText('通过审核').closest('button') as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByText('拒绝并退回').closest('button') as HTMLButtonElement).disabled).toBe(true)
  })

  it('shows review actions only to the matching department manager', async () => {
    auth.user = { user_id: 'employee-b', username: 'employee-b', role: 'viewer', org_id: 'org-a', permissions: ['doc:read'], department_id: 'finance', is_department_manager: false }
    const pending = { ...sample, review_status: 'pending_review' as const, reviewer_id: 'manager-b' }
    mockedHook.mockReturnValue(state({ cases: [pending], counts: { draft: 0, pending_review: 1, approved: 0 } }))
    renderWorkbench()

    fireEvent.click(screen.getByText('case-001'))
    expect(await screen.findByText('等待公司管理员复核')).toBeTruthy()
    expect(screen.queryByText('通过审核')).toBeNull()
  })

  it('approves as a different reviewer and freezes only after all blockers clear', async () => {
    auth.user = { user_id: 'org-admin-b', username: 'org-admin-b', role: 'organization_admin', org_id: 'org-a', permissions: ['admin:manage'], department_id: 'finance', is_department_manager: true }
    const pending = { ...sample, review_status: 'pending_review' as const, reviewer_id: 'org-admin-b', reviewer_display_name: '组织审核人' }
    const reviewCase = vi.fn().mockResolvedValue({ ...pending, review_status: 'approved', reviewer_id: 'reviewer-b' })
    const freezeDataset = vi.fn().mockResolvedValue(undefined)
    const current = state({
      cases: [pending], reviewCase, freezeDataset,
      counts: { draft: 0, pending_review: 0, approved: 1 },
      dataset: {
        ...state().dataset!, case_count: 1, counts: { draft: 0, pending_review: 0, approved: 1 },
        covered_category_count: 11, fixture_status: 'ready',
      },
    })
    mockedHook.mockReturnValue(current)
    renderWorkbench()

    fireEvent.click(screen.getByText('case-001'))
    fireEvent.click(await screen.findByText('通过审核'))
    fireEvent.change(screen.getByPlaceholderText(/填写通过依据/), { target: { value: 'Context 与预期一致' } })
    fireEvent.click(screen.getByText('确认批准'))
    await waitFor(() => expect(reviewCase).toHaveBeenCalledWith(pending, 'approve', 'Context 与预期一致'))

    fireEvent.click(screen.getByText('冻结版本'))
    expect(await screen.findByText('职责边界')).toBeTruthy()
    expect(screen.getByText(/前端仅用于制作、复核和冻结数据集。真实 baseline\/calibration 仍由独立离线流程执行；生产 Gate 晋级必须单独审批。/)).toBeTruthy()
    fireEvent.click(screen.getByText('确认冻结'))
    await waitFor(() => expect(freezeDataset).toHaveBeenCalled())
  })

  it('keeps Fixture execution out of the dataset workbench', () => {
    mockedHook.mockReturnValue(state({
      frozenVersions: [{ version: '20260814T010203Z-r42', manifest_sha256: 'a'.repeat(64), source_workspace_revision: 42 }],
    }))
    renderWorkbench()

    expect(screen.queryByRole('button', { name: '展开 Fixture 验证详情' })).toBeNull()
    expect(screen.queryByRole('button', { name: '运行全部验证' })).toBeNull()
    expect(screen.queryByRole('button', { name: '重试全部验证' })).toBeNull()
    expect(screen.queryByTestId('fixture-validation-live-status')).toBeNull()
  })

  it('does not make Fixture validation a prerequisite for freezing an approved dataset', () => {
    const approved = state({
      counts: { draft: 0, pending_review: 0, approved: 1 },
      dataset: {
        ...state().dataset!,
        case_count: 1,
        covered_category_count: 11,
        required_category_count: 11,
        counts: { draft: 0, pending_review: 0, approved: 1 },
      },
    })
    mockedHook.mockReturnValue(approved)
    renderWorkbench()

    expect((screen.getByText('冻结版本').closest('button') as HTMLButtonElement).disabled).toBe(false)
  })

  it('shows an organization-admin-only confirmed delete action for each sample', async () => {
    auth.user = { user_id: 'org-admin-a', username: 'org-admin-a', role: 'organization_admin', org_id: 'org-a', permissions: ['admin:manage'], department_id: 'human_resources', is_department_manager: false }
    const deleteCase = vi.fn().mockResolvedValue(undefined)
    mockedHook.mockReturnValue(state({ deleteCase }))
    renderWorkbench()

    fireEvent.click(screen.getByRole('button', { name: '删除样本' }))
    expect(await screen.findByText('确认删除该评测样本？')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '确认删除' }))
    await waitFor(() => expect(deleteCase).toHaveBeenCalledWith(sample))
  })

  it('does not expose deletion to a non-organization admin', () => {
    auth.user = { user_id: 'department-admin', username: 'department-admin', role: 'admin', org_id: 'org-a', permissions: ['admin:manage'], department_id: 'finance', is_department_manager: true }
    mockedHook.mockReturnValue(state())
    renderWorkbench()

    expect(screen.queryByRole('button', { name: '删除样本' })).toBeNull()
  })


  it('keeps selected batch actions disabled until a row is checked', async () => {
    const current = state()
    mockedHook.mockReturnValue(current)
    renderWorkbench()

    expect((screen.getByText('批量提交复核').closest('button') as HTMLButtonElement).disabled).toBe(true)
    const checkboxes = screen.getAllByRole('checkbox')
    const rowCheckbox = checkboxes[checkboxes.length - 1]
    fireEvent.click(rowCheckbox)
    expect((screen.getByRole('button', { name: '批量提交复核' }) as HTMLButtonElement).disabled).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: '批量提交复核' }))
    expect(await screen.findByText('将提交所选 1 条样本进入复核')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '确认提交复核' }))
    await waitFor(() => expect(current.bulkSubmit).toHaveBeenCalledWith('selected', ['case-001']))
  })

  it('runs one-click approval against all pages in the current filters', async () => {
    const bulkApprove = vi.fn().mockResolvedValue({
      revision: 4,
      matched_count: 3,
      processed_count: 2,
      skipped_count: 1,
      processed_case_ids: ['case-001', 'case-002'],
      skipped_items: [{ case_id: 'case-003', code: 'assigned_reviewer_required' }],
    })
    const current = state({
      bulkApprove,
      category: 'fully_answerable',
      query: '差旅',
      reviewStatus: 'pending_review',
      total: 3,
    })
    mockedHook.mockReturnValue(current)
    renderWorkbench()

    fireEvent.click(screen.getByRole('button', { name: '一键通过' }))
    expect(await screen.findByText('将处理当前筛选范围的全部分页样本')).toBeTruthy()
    expect(screen.getByText(/Maker-checker/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '确认批量通过' }))
    await waitFor(() => expect(bulkApprove).toHaveBeenCalledWith('filtered', []))
  })

  it('disables every bulk action while a mutation is running', () => {
    mockedHook.mockReturnValue(state({ mutating: true }))
    renderWorkbench()

    expect((screen.getByRole('button', { name: '批量提交复核' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: '批量通过' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: '一键提交复核' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByText('一键通过').closest('button') as HTMLButtonElement).disabled).toBe(true)
  })

  it('groups filters, dataset actions and bulk review actions into separate responsive areas', () => {
    auth.user = { user_id: 'org-admin-b', username: 'org-admin-b', role: 'organization_admin', org_id: 'org-a', permissions: ['admin:manage'], department_id: 'finance', is_department_manager: true }
    mockedHook.mockReturnValue(state())
    renderWorkbench()

    const filterbar = screen.getByTestId('dataset-filterbar')
    const filters = screen.getByTestId('dataset-filter-controls')
    const datasetActions = screen.getByTestId('dataset-management-actions')
    const bulkActions = screen.getByTestId('dataset-bulk-actions')

    expect(filterbar.contains(filters)).toBe(true)
    expect(filterbar.contains(datasetActions)).toBe(true)
    expect(screen.queryByRole('button', { name: '配置 Fixture' })).toBeNull()
    expect(datasetActions.contains(screen.getByRole('button', { name: /冻结版本/ }))).toBe(true)
    expect(datasetActions.contains(screen.getByRole('button', { name: /新增样本/ }))).toBe(true)
    expect(filterbar.contains(bulkActions)).toBe(false)
    expect(bulkActions.contains(screen.getByRole('button', { name: '批量提交复核' }))).toBe(true)
    expect(bulkActions.contains(screen.getByRole('button', { name: '一键通过' }))).toBe(true)
  })


  it('keeps frozen material visible while disabling every authoring control', async () => {
    const createDraftFromVersion = vi.fn().mockResolvedValue(undefined)
    mockedHook.mockReturnValue(state({
      createDraftFromVersion,
      dataset: {
        ...state().dataset,
        status: 'frozen',
        last_frozen_version: '20260814T010203Z-r42',
        last_frozen_manifest_sha256: 'a'.repeat(64),
        last_frozen_at: '2026-08-14T01:02:03+00:00',
        last_frozen_by: 'company-admin-a',
      },
      frozenVersions: [{ version: '20260814T010203Z-r42', manifest_sha256: 'a'.repeat(64), frozen_at: '2026-08-14T01:02:03+00:00', frozen_by: 'company-admin-a', source_workspace_revision: 42 }],
    }))
    renderWorkbench()

    expect(screen.getByTestId('frozen-version-card')).toBeTruthy()
    expect(screen.getAllByText('20260814T010203Z-r42').length).toBeGreaterThan(0)
    expect((screen.getByText('新增样本').closest('button') as HTMLButtonElement).disabled).toBe(true)
    expect(screen.queryByRole('button', { name: '删除' })).toBeNull()
    expect((screen.getByRole('button', { name: '批量提交复核' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: '一键通过' }) as HTMLButtonElement).disabled).toBe(true)

    fireEvent.click(screen.getByText('基于冻结版本创建新草稿').closest('button') as HTMLButtonElement)
    expect(await screen.findByText('不会修改冻结版本')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '创建新草稿' }))
    await waitFor(() => expect(createDraftFromVersion).toHaveBeenCalledWith('20260814T010203Z-r42'))
  })

  it('contains a reduced-motion fallback for transitions', () => {
    const css = readFileSync('src/index.css', 'utf8')
    expect(css).toContain('@media (prefers-reduced-motion: reduce)')
    expect(css).toContain('governance-enter')
  })
})
