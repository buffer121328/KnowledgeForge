import client from './client'
import type {
  QAAskResponse,
  QAConversation,
  QAConversationDetail,
  QAConversationPage,
  QuestionRequest,
} from '@/types'

export const qaApi = {
  /** Submit a question to the question-answering API. */
  ask: (request: string | QuestionRequest) =>
    client
      .post<QAAskResponse>(
        '/qa/ask',
        typeof request === 'string' ? { question: request } : request,
        { timeout: 120000 },
      )
      .then((r) => r.data),

  /** Fetch a cursor page from the server-authoritative conversation history. */
  listConversations: (cursor?: string, limit = 20) =>
    client
      .get<QAConversationPage>('/qa/conversations', { params: { cursor, limit } })
      .then((r) => r.data),

  /** Create an empty durable conversation. */
  createConversation: (title = '') =>
    client.post<QAConversation>('/qa/conversations', { title }).then((r) => r.data),

  /** Load one owned conversation and its ordered messages/runs. */
  getConversation: (conversationId: string) =>
    client
      .get<QAConversationDetail>(`/qa/conversations/${conversationId}`)
      .then((r) => r.data),

  /** Delete one owned conversation and best-effort invalidate associated cache entries. */
  deleteConversation: (conversationId: string) =>
    client.delete(`/qa/conversations/${conversationId}`).then((r) => r.data),

  /** Submit bounded feedback for one owned run. */
  feedback: (runId: string, rating: 'up' | 'down' | 'issue', note = '') =>
    client.post(`/qa/runs/${runId}/feedback`, { rating, note }).then((r) => r.data),

  /** Confirm a semantic candidate under the original request boundary. */
  confirm: (request: {
    question: string
    confirmation_token: string
    conversation_id?: string
    retrieval_mode?: 'vector' | 'hybrid'
  }) => client.post<QAAskResponse>('/qa/cache/confirm', request, { timeout: 120000 }).then((r) => r.data),

  /** Reject or close a semantic candidate and obtain one full-RAG bypass. */
  reject: (request: {
    question: string
    confirmation_token: string
    conversation_id?: string
    retrieval_mode?: 'vector' | 'hybrid'
  }) =>
    client
      .post<{ status: 'semantic_rejected'; semantic_bypass_token?: string }>(
        '/qa/cache/reject',
        request,
      )
      .then((r) => r.data),
}
