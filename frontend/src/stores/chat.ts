import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type {
  ChatMessage,
  QAConversation,
  SemanticConfirmationResponse,
} from '@/types'

interface PendingSemanticConfirmation extends SemanticConfirmationResponse {
  question: string
}

interface ChatState {
  messages: ChatMessage[]
  conversations: QAConversation[]
  activeConversationId?: string
  draft: string
  loading: boolean
  historyLoading: boolean
  historyError?: string
  semanticConfirmation?: PendingSemanticConfirmation

  addMessage: (msg: ChatMessage) => void
  setMessages: (messages: ChatMessage[]) => void
  setConversations: (conversations: QAConversation[]) => void
  setActiveConversationId: (conversationId?: string) => void
  setDraft: (draft: string) => void
  setSemanticConfirmation: (value?: PendingSemanticConfirmation) => void
  clear: () => void
  setLoading: (loading: boolean) => void
  setHistoryLoading: (loading: boolean) => void
  setHistoryError: (error?: string) => void
}

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      messages: [],
      conversations: [],
      draft: '',
      loading: false,
      historyLoading: false,

      addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
      setMessages: (messages) => set({ messages }),
      setConversations: (conversations) => set({ conversations }),
      setActiveConversationId: (activeConversationId) => set({ activeConversationId }),
      setDraft: (draft) => set({ draft: draft.slice(0, 10_000) }),
      setSemanticConfirmation: (semanticConfirmation) => set({ semanticConfirmation }),
      clear: () => set({ messages: [], activeConversationId: undefined }),
      setLoading: (loading) => set({ loading }),
      setHistoryLoading: (historyLoading) => set({ historyLoading }),
      setHistoryError: (historyError) => set({ historyError }),
    }),
    {
      name: 'chat-storage',
      // PostgreSQL is authoritative for messages/history. Persist only bounded UI state.
      partialize: (state) => ({
        activeConversationId: state.activeConversationId,
        draft: state.draft,
      }),
    },
  ),
)
