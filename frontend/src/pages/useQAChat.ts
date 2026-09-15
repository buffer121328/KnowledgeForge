import { useCallback, useEffect, useRef } from 'react'
import { qaApi } from '@/api/qa'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'
import type { ChatMessage, QAAskResponse, QAConversationDetail } from '@/types'

function messagesFromDetail(detail: QAConversationDetail): ChatMessage[] {
  let assistantIndex = 0
  return detail.messages.map((message) => {
    const run = message.role === 'assistant' ? detail.runs[assistantIndex++] : undefined
    const sources = Array.isArray(run?.sources)
      ? run.sources.map((source: Record<string, unknown>) => ({
          content: String(source.display_summary ?? ''),
          source: String(source.doc_id ?? ''),
          score: Number(source.score ?? 0),
          type: String(source.retrieval_type ?? 'vector') as 'vector' | 'graph' | 'hybrid',
        }))
      : undefined
    return {
      id: message.id,
      role: message.role,
      content: message.content,
      timestamp: Date.parse(message.created_at),
      sources,
      confidence: run ? Number(run.confidence ?? 0) : undefined,
      intent: run ? String(run.intent ?? '') : undefined,
      degradation_code: run?.degradation_code as ChatMessage['degradation_code'],
      qa_run_id: run?.id,
    }
  })
}

/** Manage server-authoritative QA history, question submission and confirmation. */
export function useQAChat() {
  const store = useChatStore()
  const { user } = useAuthStore()
  const scrollRef = useRef<HTMLDivElement>(null)
  const creatingConversationRef = useRef(false)

  const loadConversations = useCallback(async () => {
    store.setHistoryLoading(true)
    store.setHistoryError(undefined)
    try {
      const page = await qaApi.listConversations()
      store.setConversations(page.items)
    } catch {
      store.setHistoryError('历史会话加载失败，请重试。')
    } finally {
      store.setHistoryLoading(false)
    }
  }, [store.setConversations, store.setHistoryError, store.setHistoryLoading])

  const selectConversation = useCallback(async (conversationId?: string) => {
    store.setActiveConversationId(conversationId)
    store.setSemanticConfirmation(undefined)
    if (!conversationId) {
      store.setMessages([])
      return
    }
    store.setHistoryLoading(true)
    store.setHistoryError(undefined)
    try {
      const detail = await qaApi.getConversation(conversationId)
      store.setMessages(messagesFromDetail(detail))
    } catch {
      store.setHistoryError('会话内容加载失败，请重试。')
      store.setMessages([])
    } finally {
      store.setHistoryLoading(false)
    }
  }, [store.setActiveConversationId, store.setHistoryError, store.setHistoryLoading, store.setMessages, store.setSemanticConfirmation])

  useEffect(() => {
    void loadConversations()
  }, [loadConversations])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [store.messages])

  const applyAnswer = useCallback(async (question: string, result: QAAskResponse) => {
    if (result.status === 'semantic_confirmation_required') {
      store.setSemanticConfirmation({ ...result, question })
      return
    }
    store.addMessage({
      id: result.qa_run_id ?? `a-${Date.now()}`,
      role: 'assistant',
      content: result.answer,
      timestamp: Date.now(),
      sources: result.sources,
      confidence: result.confidence,
      intent: result.intent,
      reasoning_steps: result.reasoning_steps,
      degradation_code: result.degradation_code,
      qa_run_id: result.qa_run_id,
      response_status: result.response_status ?? undefined,
      evidence_state: result.evidence_state ?? undefined,
      evidence_reason_codes: result.evidence_reason_codes,
      claims: result.claims,
      citations: result.citations,
      missing_information: result.missing_information,
      grounding_result: result.grounding_result ?? undefined,
      policy_version: result.policy_version ?? undefined,
    })
    if (result.conversation_id) store.setActiveConversationId(result.conversation_id)
    await loadConversations()
  }, [loadConversations, store.addMessage, store.setActiveConversationId, store.setSemanticConfirmation])

  const handleSend = async () => {
    const question = store.draft.trim()
    if (!question || store.loading) return
    store.addMessage({ id: `u-${Date.now()}`, role: 'user', content: question, timestamp: Date.now() })
    store.setDraft('')
    store.setLoading(true)
    try {
      await applyAnswer(
        question,
        await qaApi.ask({ question, conversation_id: store.activeConversationId }),
      )
    } catch {
      store.addMessage({
        id: `a-${Date.now()}`,
        role: 'assistant',
        content: '抱歉，问答服务暂不可用，请稍后重试。',
        timestamp: Date.now(),
      })
    } finally {
      store.setLoading(false)
    }
  }

  const confirmSemantic = async () => {
    const pending = store.semanticConfirmation
    if (!pending || store.loading) return
    store.setLoading(true)
    store.setSemanticConfirmation(undefined)
    try {
      await applyAnswer(
        pending.question,
        await qaApi.confirm({
          question: pending.question,
          confirmation_token: pending.confirmation_token,
          conversation_id: store.activeConversationId,
        }),
      )
    } finally {
      store.setLoading(false)
    }
  }

  const rejectSemantic = async () => {
    const pending = store.semanticConfirmation
    if (!pending || store.loading) return
    store.setLoading(true)
    store.setSemanticConfirmation(undefined)
    try {
      const rejected = await qaApi.reject({
        question: pending.question,
        confirmation_token: pending.confirmation_token,
        conversation_id: store.activeConversationId,
      })
      const result = rejected.semantic_bypass_token
        ? await qaApi.ask({
            question: pending.question,
            conversation_id: store.activeConversationId,
            semantic_bypass_token: rejected.semantic_bypass_token,
          })
        : await qaApi.confirm({
            question: pending.question,
            confirmation_token: pending.confirmation_token,
            conversation_id: store.activeConversationId,
          })
      await applyAnswer(pending.question, result)
    } finally {
      store.setLoading(false)
    }
  }

  /** Create and activate one durable empty conversation without discarding state on failure. */
  const createConversation = async () => {
    if (creatingConversationRef.current || store.loading || store.historyLoading) return
    creatingConversationRef.current = true
    store.setHistoryLoading(true)
    store.setHistoryError(undefined)
    try {
      const conversation = await qaApi.createConversation()
      store.setActiveConversationId(conversation.id)
      store.setMessages([])
      store.setSemanticConfirmation(undefined)
      try {
        const page = await qaApi.listConversations()
        store.setConversations(page.items)
      } catch {
        store.setHistoryError('历史会话加载失败，请重试。')
      }
    } catch {
      store.setHistoryError('新会话创建失败，请重试。')
    } finally {
      creatingConversationRef.current = false
      store.setHistoryLoading(false)
    }
  }

  const deleteConversation = async (conversationId: string) => {
    await qaApi.deleteConversation(conversationId)
    if (store.activeConversationId === conversationId) store.clear()
    await loadConversations()
  }

  const submitFeedback = async (runId: string, rating: 'up' | 'down' | 'issue') => {
    await qaApi.feedback(runId, rating)
  }

  return {
    ...store,
    confirmSemantic,
    createConversation,
    currentUser: user?.username ?? '',
    deleteConversation,
    handleSend,
    input: store.draft,
    loadConversations,
    rejectSemantic,
    scrollRef,
    selectConversation,
    setInput: store.setDraft,
    submitFeedback,
  }
}
