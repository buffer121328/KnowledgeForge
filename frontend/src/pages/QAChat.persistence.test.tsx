// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

const actions = vi.hoisted(() => ({
  clear: vi.fn(),
  createConversation: vi.fn(),
  confirmSemantic: vi.fn(),
  deleteConversation: vi.fn(),
  handleSend: vi.fn(),
  loadConversations: vi.fn(),
  rejectSemantic: vi.fn(),
  selectConversation: vi.fn(),
  setInput: vi.fn(),
  submitFeedback: vi.fn(),
}))

const hookState = vi.hoisted(() => ({ value: {} as Record<string, unknown> }))
vi.mock('./useQAChat', () => ({ useQAChat: () => hookState.value }))

import QAChat from './QAChat'

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(() => ({
      matches: false,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  })
})

beforeEach(() => {
  vi.clearAllMocks()
  hookState.value = {
    ...actions,
    activeConversationId: undefined,
    conversations: [],
    currentUser: 'alice',
    historyError: undefined,
    historyLoading: false,
    input: '',
    loading: false,
    messages: [],
    scrollRef: { current: null },
    semanticConfirmation: undefined,
  }
})

afterEach(cleanup)

describe('QAChat persistent history and semantic UX', () => {
  it('renders history error/empty retry states without indefinite loading', () => {
    hookState.value.historyError = '历史会话加载失败，请重试。'
    render(<QAChat />)

    expect(screen.getByText('历史会话加载失败，请重试。')).toBeTruthy()
    expect(screen.getByText('暂无历史会话')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /重试/ }))
    expect(actions.loadConversations).toHaveBeenCalledOnce()
  })


  it('maps both new-conversation buttons to the durable creation handler', () => {
    render(<QAChat />)

    const buttons = screen.getAllByRole('button', { name: /新会话/ })
    expect(buttons).toHaveLength(2)
    fireEvent.click(buttons[0])
    fireEvent.click(buttons[1])
    expect(actions.createConversation).toHaveBeenCalledTimes(2)
    expect(actions.clear).not.toHaveBeenCalled()
  })

  it('shows only candidate metadata and maps confirm/reject controls', () => {
    hookState.value.semanticConfirmation = {
      status: 'semantic_confirmation_required',
      question: '怎样部署？',
      similar_question: '如何部署？',
      similarity: 0.96,
      cached_at: '2026-08-01T00:00:00Z',
      confirmation_token: 'opaque',
    }
    render(<QAChat />)

    expect(screen.getByText('如何部署？')).toBeTruthy()
    expect(screen.queryByText('受保护的缓存答案')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: '使用历史答案' }))
    expect(actions.confirmSemantic).toHaveBeenCalledOnce()
    fireEvent.click(screen.getByRole('button', { name: '重新检索' }))
    expect(actions.rejectSemantic).toHaveBeenCalledOnce()
  })
})
