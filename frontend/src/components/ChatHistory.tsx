import { DislikeOutlined, LikeOutlined, RobotOutlined, UserOutlined, WarningOutlined } from '@ant-design/icons'
import { Alert, Button, Card, Collapse, Drawer, Empty, Progress, Space, Spin, Tag, Tooltip, Typography } from 'antd'
import { useState } from 'react'
import type { RefObject, ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { docApi } from '@/api/docs'
import type { ChatMessage, ChunkItem, QAAnswerCitation, QAResponseStatus } from '@/types'

const { Paragraph, Text } = Typography

const intentColors: Record<string, string> = {
  factoid: 'blue',
  analytical: 'purple',
  comparative: 'cyan',
  procedural: 'orange',
  exploratory: 'green',
}

const degradationLabels: Partial<Record<string, string>> = {
  graph_retrieval_unavailable: '图谱增强暂不可用，已基于文档证据回答。',
  vector_retrieval_unavailable: '向量检索暂不可用，已基于可验证图谱证据回答。',
  insufficient_verified_evidence: '可验证证据不足，未生成推测性回答。',
  structured_output_invalid: '结构化答案校验失败，未返回未验证内容。',
  grounding_failed: '答案未通过引用校验，已停止自动回答。',
}

const responseStatusMeta: Record<QAResponseStatus, { label: string; color: string }> = {
  answered: { label: '已回答', color: 'success' },
  partially_answered: { label: '部分回答', color: 'warning' },
  insufficient_evidence: { label: '证据不足', color: 'orange' },
  needs_clarification: { label: '需要补充信息', color: 'processing' },
  conflicting_evidence: { label: '证据冲突', color: 'error' },
  human_review_required: { label: '需要人工审核', color: 'purple' },
  source_unavailable: { label: '来源暂不可用', color: 'default' },
}

const controlledResponseStatuses = new Set<QAResponseStatus>([
  'insufficient_evidence',
  'needs_clarification',
  'conflicting_evidence',
  'human_review_required',
  'source_unavailable',
])

interface ChatHistoryProps {
  currentUser: string
  loading: boolean
  messages: ChatMessage[]
  scrollRef: RefObject<HTMLDivElement | null>
  onFeedback?: (runId: string, rating: 'up' | 'down' | 'issue') => void
}

/** Render the chat history component. */
export function ChatHistory({ loading, messages, scrollRef, onFeedback }: ChatHistoryProps) {
  const [selectedCitation, setSelectedCitation] = useState<QAAnswerCitation>()
  const [selectedChunk, setSelectedChunk] = useState<ChunkItem>()
  const [chunkLoading, setChunkLoading] = useState(false)
  const [chunkError, setChunkError] = useState('')

  const openCitation = async (citation: QAAnswerCitation) => {
    setSelectedCitation(citation)
    setSelectedChunk(undefined)
    setChunkError('')
    if (!citation.document_id) return
    setChunkLoading(true)
    try {
      const chunks = await docApi.chunks(citation.document_id)
      const chunk = chunks.find((item) => item.chunk_id === citation.chunk_id)
        ?? chunks.find((item) => item.chunk_index === citation.chunk_index)
      if (!chunk) setChunkError('未找到对应分块，可能已重新入库。')
      setSelectedChunk(chunk)
    } catch {
      setChunkError('分块加载失败，请确认当前账号仍有该文档的读取权限。')
    } finally {
      setChunkLoading(false)
    }
  }

  return (
    <>
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflow: 'auto',
          padding: 24,
          background: '#fafafa',
        }}
      >
        {messages.length === 0 && (
          <Empty description="开始你的第一个问题" style={{ marginTop: 80 }} />
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} onCitationOpen={openCitation} onFeedback={onFeedback} />
        ))}
        {loading && (
          <div style={{ textAlign: 'center', padding: 16 }}>
            <Spin tip="思考中..." />
          </div>
        )}
      </div>
      <Drawer
        open={Boolean(selectedCitation)}
        width={760}
        onClose={() => setSelectedCitation(undefined)}
        title={selectedCitation ? citationLabel(selectedCitation) : '引用分块'}
      >
        {chunkLoading ? <Spin /> : chunkError ? <Alert type="warning" showIcon message={chunkError} /> : selectedChunk ? (
          <Card size="small" title={`分块 #${selectedChunk.chunk_index}`}>
            <Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
              {highlightText(selectedChunk.content, selectedCitation?.highlight || selectedCitation?.content || '')}
            </Paragraph>
          </Card>
        ) : <Empty description="该引用未包含可打开的文档分块标识" />}
      </Drawer>
    </>
  )
}

function citationLabel(citation: QAAnswerCitation): string {
  const documentLabel = citation.source || citation.document_id || '授权文档'
  const chunkLabel = citation.chunk_index !== undefined && citation.chunk_index !== null
    ? `分块 #${citation.chunk_index}`
    : citation.chunk_id ? `分块 ${citation.chunk_id}` : '分块'
  return `${documentLabel} · ${chunkLabel}`
}

function highlightText(content: string, relevant: string): ReactNode {
  const needle = relevant.trim()
  if (!needle) return content
  const index = content.indexOf(needle)
  if (index < 0) return content
  return <>{content.slice(0, index)}<mark style={{ background: '#fff1b8', color: '#cf1322', fontWeight: 700 }}>{needle}</mark>{content.slice(index + needle.length)}</>
}

/** Render the message bubble component. */
function MessageBubble({
  message,
  onCitationOpen,
  onFeedback,
}: {
  message: ChatMessage
  onCitationOpen: (citation: QAAnswerCitation) => void
  onFeedback?: ChatHistoryProps['onFeedback']
}) {
  const isUser = message.role === 'user'
  const confidencePct = message.confidence ? Math.round(message.confidence * 100) : null
  const responseMeta = message.response_status
    ? responseStatusMeta[message.response_status]
    : undefined
  const isControlledResponse = Boolean(
    message.response_status && controlledResponseStatuses.has(message.response_status),
  )
  const citationsById = new Map(
    (message.citations ?? []).map((citation) => [citation.citation_id, citation]),
  )

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: isUser ? 'row-reverse' : 'row',
        gap: 12,
        marginBottom: 16,
      }}
    >
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: '50%',
          background: isUser ? '#1677ff' : '#722ed1',
          color: '#fff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        {isUser ? <UserOutlined /> : <RobotOutlined />}
      </div>
      <div style={{ maxWidth: '70%' }}>
        <div
          style={{
            background: isUser ? '#1677ff' : '#fff',
            color: isUser ? '#fff' : '#000',
            padding: '10px 16px',
            borderRadius: 8,
            border: isUser ? 'none' : '1px solid #f0f0f0',
            boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
          }}
        >
          {isUser ? (
            <Paragraph style={{ margin: 0, color: 'inherit' }}>{message.content}</Paragraph>
          ) : isControlledResponse ? (
            <Alert
              showIcon
              type={message.response_status === 'conflicting_evidence' ? 'error' : 'warning'}
              message={responseMeta?.label}
              description={message.content}
            />
          ) : (
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {!isUser && (message.intent || message.degradation_code || responseMeta) && (
          <Space size={4} style={{ marginTop: 8, flexWrap: 'wrap' }}>
            {responseMeta && <Tag color={responseMeta.color}>{responseMeta.label}</Tag>}
            {message.intent && <Tag color={intentColors[message.intent] ?? 'default'}>{message.intent}</Tag>}
            {confidencePct !== null && (
              <Tooltip title={`置信度: ${confidencePct}%`}>
                <Progress
                  type="circle"
                  size={28}
                  percent={confidencePct}
                  strokeColor={
                    confidencePct >= 80 ? '#52c41a' : confidencePct >= 50 ? '#faad14' : '#ff4d4f'
                  }
                />
              </Tooltip>
            )}
            {message.sources && message.sources.length > 0 && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                来源: {message.sources.length} 条
              </Text>
            )}
            {message.degradation_code && degradationLabels[message.degradation_code] && (
              <Tag color="warning">{degradationLabels[message.degradation_code]}</Tag>
            )}
          </Space>
        )}

        {!isUser && message.claims && message.claims.length > 0 && (
          <div style={{ marginTop: 8, padding: 12, background: '#f6ffed', borderRadius: 6 }}>
            <Text strong>结论与引用</Text>
            {message.claims.map((claim) => {
              const citations = claim.citation_ids
                .map((citationId) => citationsById.get(citationId))
                .filter((citation) => citation !== undefined)
              return (
                <div key={claim.claim_id} style={{ marginTop: 8 }}>
                  <Paragraph style={{ marginBottom: 4 }}>{claim.text}</Paragraph>
                  {citations.map((citation) => (
                    <div key={citation.citation_id} style={{ fontSize: 12 }}>
                      <Button
                        type="link"
                        size="small"
                        disabled={!citation.document_id}
                        onClick={() => onCitationOpen(citation)}
                      >
                        {citationLabel(citation)}
                      </Button>
                      <Text type="secondary" ellipsis style={{ maxWidth: 420 }}>{citation.content}</Text>
                    </div>
                  ))}
                </div>
              )
            })}
          </div>
        )}

        {!isUser && message.missing_information && message.missing_information.length > 0 && (
          <Alert
            style={{ marginTop: 8 }}
            type="info"
            showIcon
            message="仍需补充"
            description={
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {message.missing_information.map((item) => (
                  <li key={item.field}>{item.description}</li>
                ))}
              </ul>
            }
          />
        )}

        {!isUser && message.reasoning_steps && message.reasoning_steps.length > 0 && (
          <Collapse
            size="small"
            style={{ marginTop: 8 }}
            items={[
              {
                key: 'reasoning',
                label: '推理步骤',
                children: (
                  <ol style={{ margin: 0, paddingLeft: 20 }}>
                    {message.reasoning_steps.map((step, index) => (
                      <li key={index}>{step}</li>
                    ))}
                  </ol>
                ),
              },
            ]}
          />
        )}

        {!isUser && (!message.citations || message.citations.length === 0) && message.sources && message.sources.length > 0 && (
          <Collapse
            size="small"
            style={{ marginTop: 8 }}
            items={[
              {
                key: 'sources',
                label: `引用来源 (${message.sources.length})`,
                children: message.sources.map((source, index) => (
                  <div
                    key={index}
                    style={{
                      padding: 8,
                      marginBottom: 4,
                      background: '#fafafa',
                      borderRadius: 4,
                      fontSize: 12,
                    }}
                  >
                    <Space size={4}>
                      <Text strong>{source.source || source.document_id || '授权文档'}</Text>
                      {(source.chunk_index !== undefined && source.chunk_index !== null) && <Tag>分块 #{source.chunk_index}</Tag>}
                    </Space>
                    <Paragraph
                      style={{ margin: '4px 0 0', color: '#666', fontSize: 12 }}
                      ellipsis={{ rows: 2 }}
                    >
                      {source.content}
                    </Paragraph>
                  </div>
                )),
              },
            ]}
          />
        )}

        {!isUser && message.qa_run_id && onFeedback && (
          <Space size={4} style={{ marginTop: 8 }}>
            <Button aria-label="有帮助" size="small" type="text" icon={<LikeOutlined />} onClick={() => onFeedback(message.qa_run_id!, 'up')} />
            <Button aria-label="没帮助" size="small" type="text" icon={<DislikeOutlined />} onClick={() => onFeedback(message.qa_run_id!, 'down')} />
            <Button aria-label="报告问题" size="small" type="text" icon={<WarningOutlined />} onClick={() => onFeedback(message.qa_run_id!, 'issue')} />
          </Space>
        )}
      </div>
    </div>
  )
}
