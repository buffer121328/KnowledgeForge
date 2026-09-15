import { SendOutlined } from '@ant-design/icons'
import { Button, Input, Space } from 'antd'

const { TextArea } = Input

interface QuestionComposerProps {
  input: string
  loading: boolean
  onInputChange: (input: string) => void
  onSend: () => Promise<void>
}

/** Render the question composer component. */
export function QuestionComposer({
  input,
  loading,
  onInputChange,
  onSend,
}: QuestionComposerProps) {
  return (
    <div style={{ padding: 16, borderTop: '1px solid #f0f0f0', background: '#fff' }}>
      <Space.Compact style={{ width: '100%' }}>
        <TextArea
          value={input}
          onChange={(event) => onInputChange(event.target.value)}
          placeholder="输入问题，Enter 发送，Shift+Enter 换行"
          autoSize={{ minRows: 1, maxRows: 4 }}
          onPressEnter={(event) => {
            if (!event.shiftKey) {
              event.preventDefault()
              onSend()
            }
          }}
          style={{ borderRadius: '6px 0 0 6px' }}
        />
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={onSend}
          loading={loading}
          style={{ height: 'auto', borderRadius: '0 6px 6px 0' }}
        >
          发送
        </Button>
      </Space.Compact>
    </div>
  )
}
