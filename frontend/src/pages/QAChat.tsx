import { ClearOutlined, DeleteOutlined, PlusOutlined, ReloadOutlined } from '@ant-design/icons'
import { Alert, Button, Card, Empty, Flex, List, Modal, Popconfirm, Spin, Typography } from 'antd'
import { ChatHistory } from '@/components/ChatHistory'
import { QuestionComposer } from '@/components/QuestionComposer'
import { useQAChat } from './useQAChat'

/** Render the question-answering chat page. */
export default function QAChat() {
  const {
    activeConversationId,
    createConversation,
    confirmSemantic,
    conversations,
    currentUser,
    deleteConversation,
    handleSend,
    historyError,
    historyLoading,
    input,
    loadConversations,
    loading,
    messages,
    rejectSemantic,
    scrollRef,
    selectConversation,
    semanticConfirmation,
    setInput,
    submitFeedback,
  } = useQAChat()

  return (
    <Flex gap={16} style={{ height: 'calc(100vh - 160px)' }}>
      <Card
        title="历史会话"
        extra={<Button aria-label="新会话" icon={<PlusOutlined />} size="small" loading={historyLoading} onClick={() => void createConversation()} />}
        style={{ width: 280, flexShrink: 0, overflow: 'auto' }}
      >
        {historyError && (
          <Alert
            type="error"
            message={historyError}
            action={<Button icon={<ReloadOutlined />} size="small" onClick={() => void loadConversations()}>重试</Button>}
          />
        )}
        {historyLoading && conversations.length === 0 ? (
          <Spin />
        ) : conversations.length === 0 ? (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无历史会话" />
        ) : (
          <List
            dataSource={conversations}
            renderItem={(conversation) => (
              <List.Item
                style={{ cursor: 'pointer', background: conversation.id === activeConversationId ? '#e6f4ff' : undefined, paddingInline: 8 }}
                onClick={() => void selectConversation(conversation.id)}
                actions={[
                  <Popconfirm key="delete" title="删除此会话？" onConfirm={() => void deleteConversation(conversation.id)}>
                    <Button aria-label="删除会话" type="text" danger icon={<DeleteOutlined />} onClick={(event) => event.stopPropagation()} />
                  </Popconfirm>,
                ]}
              >
                <Typography.Text ellipsis>{conversation.title || '未命名会话'}</Typography.Text>
              </List.Item>
            )}
          />
        )}
      </Card>

      <Card
        title="智能问答"
        extra={<Button icon={<ClearOutlined />} loading={historyLoading} onClick={() => void createConversation()} size="small">新会话</Button>}
        style={{ flex: 1 }}
        styles={{ body: { padding: 0, height: 'calc(100vh - 220px)', display: 'flex', flexDirection: 'column' } }}
      >
        <ChatHistory
          currentUser={currentUser}
          loading={loading || historyLoading}
          messages={messages}
          scrollRef={scrollRef}
          onFeedback={(runId, rating) => void submitFeedback(runId, rating)}
        />
        <QuestionComposer input={input} loading={loading} onInputChange={setInput} onSend={handleSend} />
      </Card>

      <Modal
        open={Boolean(semanticConfirmation)}
        title="发现相似历史问题"
        okText="使用历史答案"
        cancelText="重新检索"
        confirmLoading={loading}
        onOk={() => void confirmSemantic()}
        onCancel={() => void rejectSemantic()}
        closable
        maskClosable={false}
      >
        {semanticConfirmation && (
          <>
            <Typography.Paragraph>{semanticConfirmation.similar_question}</Typography.Paragraph>
            <Typography.Text type="secondary">
              相似度 {Math.round(semanticConfirmation.similarity * 100)}% · 缓存于 {semanticConfirmation.cached_at}
            </Typography.Text>
          </>
        )}
      </Modal>
    </Flex>
  )
}
