// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeAll, describe, expect, it, vi } from 'vitest'
import EvaluationGovernance from './EvaluationGovernance'

vi.mock('@/components/evaluation/EvidenceDatasetWorkbench', () => ({ default: () => <div>数据集内容</div> }))
vi.mock('@/components/evaluation/ReleaseEvaluationPanel', () => ({ default: () => <div>发布评测内容</div> }))
vi.mock('./EvaluationRuns', () => ({ default: () => <div>运行结果内容</div> }))

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(() => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() })),
  })
  class ResizeObserverMock { observe() {} unobserve() {} disconnect() {} }
  Object.defineProperty(window, 'ResizeObserver', { value: ResizeObserverMock })
})

describe('EvaluationGovernance', () => {
  it('does not expose the retired current-corpus-draft page or navigation entry', () => {
    render(<EvaluationGovernance />)

    expect(screen.getByText('数据集内容')).toBeTruthy()
    expect(screen.queryByText('当前语料草稿')).toBeNull()
    expect(screen.queryByText('当前语料草稿内容')).toBeNull()

    fireEvent.click(screen.getByText('发布评测'))
    expect(screen.getByText('发布评测内容')).toBeTruthy()
  })
})
