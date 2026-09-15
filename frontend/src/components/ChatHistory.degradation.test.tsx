// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { ChatHistory } from './ChatHistory'

afterEach(cleanup)

describe('ChatHistory degradation status', () => {
  it('renders a human-readable warning for a successful degraded QA answer', () => {
    render(
      <ChatHistory
        currentUser="tester"
        loading={false}
        messages={[
          {
            id: 'answer-1',
            role: 'assistant',
            content: '这是基于向量证据的回答。',
            timestamp: 1,
            degradation_code: 'graph_retrieval_unavailable',
          },
        ]}
        scrollRef={{ current: null }}
      />,
    )

    expect(screen.getByText('图谱增强暂不可用，已基于文档证据回答。')).toBeTruthy()
  })
})
