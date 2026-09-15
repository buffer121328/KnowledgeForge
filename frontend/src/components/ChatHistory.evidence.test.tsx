// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ChatHistory } from './ChatHistory'
import type { ChatMessage, QAResponseStatus } from '@/types'

const mocks = vi.hoisted(() => ({ chunks: vi.fn() }))
vi.mock('@/api/docs', () => ({ docApi: { chunks: mocks.chunks } }))

afterEach(cleanup)

function renderMessage(message: Partial<ChatMessage>) {
  render(
    <ChatHistory
      currentUser="tester"
      loading={false}
      messages={[
        {
          id: 'answer-1',
          role: 'assistant',
          content: '受控响应内容',
          timestamp: 1,
          ...message,
        },
      ]}
      scrollRef={{ current: null }}
    />,
  )
}

describe('ChatHistory evidence response states', () => {
  it.each<[QAResponseStatus, string]>([
    ['answered', '已回答'],
    ['partially_answered', '部分回答'],
    ['insufficient_evidence', '证据不足'],
    ['needs_clarification', '需要补充信息'],
    ['conflicting_evidence', '证据冲突'],
    ['human_review_required', '需要人工审核'],
    ['source_unavailable', '来源暂不可用'],
  ])('renders %s as %s', (responseStatus, label) => {
    renderMessage({ response_status: responseStatus })

    expect(screen.getAllByText(label).length).toBeGreaterThan(0)
  })

  it('renders partial supported claims, authorized citations and missing information', () => {
    renderMessage({
      content: '培训预算为100万元。',
      response_status: 'partially_answered',
      claims: [
        {
          claim_id: 'claim-1',
          text: '培训预算为100万元。',
          citation_ids: ['cite-current'],
          material: true,
        },
      ],
      citations: [
        {
          citation_id: 'cite-current',
          source: 'policy.md',
          content: '培训预算为100万元。',
          document_id: 'doc-policy',
          chunk_id: 'chunk-3',
          chunk_index: 3,
        },
      ],
      missing_information: [
        { field: 'effective_date', description: '缺少生效日期。' },
      ],
    })

    expect(screen.getByText('部分回答')).toBeTruthy()
    expect(screen.getByText('结论与引用')).toBeTruthy()
    expect(screen.getByText('policy.md · 分块 #3')).toBeTruthy()
    expect(screen.getByText('缺少生效日期。')).toBeTruthy()
  })

  it('opens the cited document chunk and highlights relevant text without retrieval labels', async () => {
    mocks.chunks.mockResolvedValueOnce([{ chunk_id: 'chunk-3', chunk_index: 3, doc_id: 'doc-policy', doc_type: 'text', source: 'policy.md', content: '前文。培训预算为100万元。后文。' }])
    renderMessage({
      response_status: 'answered',
      claims: [{ claim_id: 'claim-1', text: '培训预算为100万元。', citation_ids: ['cite-current'], material: true }],
      citations: [{ citation_id: 'cite-current', source: 'policy.md', content: '培训预算为100万元。', document_id: 'doc-policy', chunk_id: 'chunk-3', chunk_index: 3 }],
    })

    fireEvent.click(screen.getByRole('button', { name: /policy.md · 分块 #3/ }))

    await waitFor(() => expect(mocks.chunks).toHaveBeenCalledWith('doc-policy'))
    expect(await screen.findByText('培训预算为100万元。', { selector: 'mark' })).toBeTruthy()
    expect(screen.queryByText('vector')).toBeNull()
    expect(screen.queryByText('bm25')).toBeNull()
  })

  it('does not fabricate a citation when a claim references an absent citation ID', () => {
    renderMessage({
      response_status: 'answered',
      claims: [
        {
          claim_id: 'claim-1',
          text: '受支持结论。',
          citation_ids: ['cite-missing'],
          material: true,
        },
      ],
      citations: [],
    })

    expect(screen.getByText('受支持结论。')).toBeTruthy()
    expect(screen.queryByText('cite-missing')).toBeNull()
  })
})
