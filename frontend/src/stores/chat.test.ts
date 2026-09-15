// @vitest-environment jsdom
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

const storageValues = new Map<string, string>()
const storage = {
  clear: () => storageValues.clear(),
  getItem: (key: string) => storageValues.get(key) ?? null,
  key: (index: number) => [...storageValues.keys()][index] ?? null,
  get length() { return storageValues.size },
  removeItem: (key: string) => storageValues.delete(key),
  setItem: (key: string, value: string) => storageValues.set(key, value),
}
vi.stubGlobal('localStorage', storage)

let useChatStore: typeof import('./chat').useChatStore

beforeAll(async () => {
  useChatStore = (await import('./chat')).useChatStore
})

beforeEach(() => {
  storage.clear()
  useChatStore.setState({
    messages: [],
    conversations: [],
    activeConversationId: undefined,
    draft: '',
    loading: false,
    historyLoading: false,
    historyError: undefined,
    semanticConfirmation: undefined,
  })
})

describe('chat store persistence boundary', () => {
  it('keeps server history transient and persists only bounded UI state', () => {
    const store = useChatStore.getState()
    store.addMessage({ id: 'm1', role: 'user', content: 'protected question', timestamp: 1 })
    store.setConversations([{ id: 'c1', title: 'history', status: 'active', created_at: '', updated_at: '' }])
    store.setActiveConversationId('c1')
    store.setDraft('draft')

    const partialize = useChatStore.persist.getOptions().partialize
    const persisted = partialize ? partialize(useChatStore.getState()) : useChatStore.getState()
    expect(persisted).toEqual({ activeConversationId: 'c1', draft: 'draft' })
    expect(JSON.stringify(persisted)).not.toContain('protected question')
    expect(JSON.stringify(persisted)).not.toContain('history')
  })

  it('bounds locally persisted drafts', () => {
    useChatStore.getState().setDraft('x'.repeat(20_000))
    expect(useChatStore.getState().draft).toHaveLength(10_000)
  })
})
