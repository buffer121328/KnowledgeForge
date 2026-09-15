import { ApiOutlined, PlusOutlined, ReloadOutlined } from '@ant-design/icons'
import { Button, Card, Space, Typography } from 'antd'
import { WebhookForm } from '@/components/WebhookForm'
import { WebhookTable } from '@/components/WebhookTable'
import { useWebhooks } from './useWebhooks'

const { Paragraph } = Typography

/** Render the webhooks page. */
export default function Webhooks() {
  const {
    closeCreateModal,
    eventOptions,
    fetchWebhooks,
    form,
    handleCreate,
    handleDelete,
    handleTest,
    loading,
    modalOpen,
    openCreateModal,
    submitting,
    webhooks,
  } = useWebhooks()

  return (
    <Card
      title={
        <Space>
          <ApiOutlined />
          Webhook 管理
        </Space>
      }
      extra={
        <Space>
          <Button icon={<ReloadOutlined />} onClick={fetchWebhooks}>
            刷新
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={openCreateModal}>
            新建 Webhook
          </Button>
        </Space>
      }
    >
      <Paragraph type="secondary" style={{ marginBottom: 12 }}>
        订阅业务事件后，系统会向回调 URL 发起真实 HTTP POST（含签名头）。测试按钮会立即请求该地址。
      </Paragraph>

      <WebhookTable
        loading={loading}
        onDelete={handleDelete}
        onTest={handleTest}
        webhooks={webhooks}
      />
      <WebhookForm
        eventOptions={eventOptions}
        form={form}
        onCancel={closeCreateModal}
        onSubmit={handleCreate}
        open={modalOpen}
        submitting={submitting}
      />
    </Card>
  )
}
