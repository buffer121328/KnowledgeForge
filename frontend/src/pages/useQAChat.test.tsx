// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const qaApi = vi.hoisted(() => ({
  ask: vi.fn(),
  confirm: vi.fn(),
  createConversation: vi.fn(),
  deleteConversation: vi.fn(),
  feedback: vi.fn(),
  getConversation: vi.fn(),
  listConversations: vi.fn(),
  reject: vi.fn(),
}))

const chatStore = vi.hoisted(() => ({
  activeConversationId: 'conv-old' as string | undefined,
  conversations: [] as Array<Record<string, unknown>>,
  draft: '',
  historyError: undefined as string | undefined,
  historyLoading: false,
  loading: false,
  messages: [] as Array<Record<string, unknown>>,
  semanticConfirmation: undefined as Record<string, unknown> | undefined,
  addMessage: vi.fn(),
  clear: vi.fn(),
  setActiveConversationId: vi.fn(),
  setConversations: vi.fn(),
  setDraft: vi.fn(),
  setHistoryError: vi.fn(),
  setHistoryLoading: vi.fn(),
  setLoading: vi.fn(),
  setMessages: vi.fn(),
  setSemanticConfirmation: vi.fn(),
}))

vi.mock('@/api/qa', () => ({ qaApi }))
vi.mock('@/stores/auth', () => ({ useAuthStore: () => ({ user: { username: 'alice' } }) }))
vi.mock('@/stores/chat', () => ({ useChatStore: () => chatStore }))

import { useQAChat } from './useQAChat'

const existingMessage = {
  id: 'm-1', role: 'assistant', content: 'existing answer', timestamp: 1,
}

beforeEach(() => {
  vi.clearAllMocks()
  Object.assign(chatStore, {
    activeConversationId: 'conv-old',
    conversations: [],
    draft: '',
    historyError: undefined,
    historyLoading: false,
    loading: false,
    messages: [existingMessage],
    semanticConfirmation: undefined,
  })
  chatStore.setActiveConversationId.mockImplementation((value?: string) => { chatStore.activeConversationId = value })
  chatStore.setConversations.mockImplementation((value: Array<Record<string, unknown>>) => { chatStore.conversations = value })
  chatStore.setHistoryError.mockImplementation((value?: string) => { chatStore.historyError = value })
  chatStore.setHistoryLoading.mockImplementation((value: boolean) => { chatStore.historyLoading = value })
  chatStore.setMessages.mockImplementation((value: Array<Record<string, unknown>>) => { chatStore.messages = value })
  chatStore.setSemanticConfirmation.mockImplementation((value?: Record<string, unknown>) => { chatStore.semanticConfirmation = value })
  qaApi.listConversations.mockResolvedValue({ items: [], next_cursor: undefined })
})

describe('useQAChat durable new conversation', () => {
  it('creates, activates and refreshes a server conversation before clearing the view', async () => {
    qaApi.createConversation.mockResolvedValue({ id: 'conv-new', title: '' })
    qaApi.listConversations
      .mockResolvedValueOnce({ items: [], next_cursor: undefined })
      .mockResolvedValueOnce({ items: [{ id: 'conv-new', title: '' }], next_cursor: undefined })

    const { result } = renderHook(() => useQAChat())
    await waitFor(() => expect(qaApi.listConversations).toHaveBeenCalledOnce())

    await act(async () => { await result.current.createConversation() })

    expect(qaApi.createConversation).toHaveBeenCalledOnce()
    expect(qaApi.listConversations).toHaveBeenCalledTimes(2)
    expect(chatStore.activeConversationId).toBe('conv-new')
    expect(chatStore.messages).toEqual([])
  })


  it('deduplicates concurrent new-conversation requests', async () => {
    let resolveCreation: ((value: { id: string; title: string }) => void) | undefined
    qaApi.createConversation.mockImplementation(() => new Promise((resolve) => { resolveCreation = resolve }))
    const { result } = renderHook(() => useQAChat())
    await waitFor(() => expect(qaApi.listConversations).toHaveBeenCalledOnce())

    let firstRequest: Promise<void> | undefined
    act(() => {
      firstRequest = result.current.createConversation()
      void result.current.createConversation()
    })

    expect(qaApi.createConversation).toHaveBeenCalledOnce()
    await act(async () => {
      resolveCreation?.({ id: 'conv-new', title: '' })
      await firstRequest
    })
  })

  it('preserves the active chat and reports a retryable error when creation fails', async () => {
    qaApi.createConversation.mockRejectedValue(new Error('offline'))
    const { result } = renderHook(() => useQAChat())
    await waitFor(() => expect(qaApi.listConversations).toHaveBeenCalledOnce())

    await act(async () => { await result.current.createConversation() })

    expect(chatStore.activeConversationId).toBe('conv-old')
    expect(chatStore.messages).toEqual([existingMessage])
    expect(chatStore.historyError).toBe('新会话创建失败，请重试。')
  })
})
