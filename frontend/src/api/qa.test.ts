import { beforeEach, describe, expect, it, vi } from 'vitest'

const client = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  delete: vi.fn(),
}))

vi.mock('./client', () => ({ default: client }))

import { qaApi } from './qa'

beforeEach(() => vi.clearAllMocks())

describe('qaApi durable history and semantic confirmation', () => {
  it('sends conversation scope when asking and preserves the discriminated response', async () => {
    const confirmation = {
      status: 'semantic_confirmation_required' as const,
      similar_question: '如何部署？',
      similarity: 0.96,
      cached_at: '2026-08-01T00:00:00Z',
      confirmation_token: 'opaque',
    }
    client.post.mockResolvedValue({ data: confirmation })

    const result = await qaApi.ask({ question: '怎样部署？', conversation_id: 'conv-1' })

    expect(result).toEqual(confirmation)
    expect(client.post).toHaveBeenCalledWith(
      '/qa/ask',
      { question: '怎样部署？', conversation_id: 'conv-1' },
      { timeout: 120000 },
    )
    expect(result).not.toHaveProperty('answer')
  })

  it('uses durable history, feedback, confirmation and rejection endpoints', async () => {
    client.get.mockResolvedValue({ data: { items: [] } })
    client.post.mockResolvedValue({ data: {} })
    client.delete.mockResolvedValue({ data: {} })

    await qaApi.listConversations()
    await qaApi.getConversation('conv-1')
    await qaApi.feedback('run-1', 'up')
    await qaApi.confirm({ question: 'q', confirmation_token: 'token' })
    await qaApi.reject({ question: 'q', confirmation_token: 'token' })
    await qaApi.deleteConversation('conv-1')

    expect(client.get).toHaveBeenCalledWith('/qa/conversations', { params: { cursor: undefined, limit: 20 } })
    expect(client.get).toHaveBeenCalledWith('/qa/conversations/conv-1')
    expect(client.post).toHaveBeenCalledWith('/qa/runs/run-1/feedback', { rating: 'up', note: '' })
    expect(client.post).toHaveBeenCalledWith('/qa/cache/confirm', expect.any(Object), { timeout: 120000 })
    expect(client.post).toHaveBeenCalledWith('/qa/cache/reject', expect.any(Object))
    expect(client.delete).toHaveBeenCalledWith('/qa/conversations/conv-1')
  })
})
