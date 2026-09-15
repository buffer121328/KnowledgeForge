import { Form, Input, Modal, Select, Switch } from 'antd'
import type { FormInstance } from 'antd'
import type { WebhookEventOption, WebhookFormValues } from '@/pages/useWebhooks'
import { labelOf, webhookEventLabels } from '@/utils/labels'

interface WebhookFormProps {
  eventOptions: WebhookEventOption[]
  form: FormInstance<WebhookFormValues>
  onCancel: () => void
  onSubmit: () => Promise<void>
  open: boolean
  submitting: boolean
}

/** Render the webhook form component. */
export function WebhookForm({
  eventOptions,
  form,
  onCancel,
  onSubmit,
  open,
  submitting,
}: WebhookFormProps) {
  return (
    <Modal
      title="新建 Webhook"
      open={open}
      onOk={onSubmit}
      onCancel={onCancel}
      confirmLoading={submitting}
      destroyOnClose
      width={520}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{ events: ['doc.ingested'], is_active: true }}
      >
        <Form.Item
          name="url"
          label="回调 URL"
          rules={[
            { required: true, message: '请输入回调 URL' },
            { type: 'url', message: 'URL 格式不正确' },
          ]}
        >
          <Input placeholder="https://example.com/webhook" />
        </Form.Item>
        <Form.Item
          name="events"
          label="订阅事件"
          rules={[{ required: true, message: '请至少选择一个事件' }]}
        >
          <Select
            mode="multiple"
            placeholder="选择要订阅的事件"
            options={eventOptions.map((event) => ({
              value: event.value,
              label: event.label || labelOf(webhookEventLabels, event.value),
            }))}
          />
        </Form.Item>
        <Form.Item
          name="secret"
          label="签名密钥（可选）"
          tooltip="用于校验请求来源，将放在 X-Webhook-Signature 头中；留空则由服务端自动生成"
        >
          <Input.Password placeholder="留空则自动生成" />
        </Form.Item>
        <Form.Item name="is_active" label="启用" valuePropName="checked">
          <Switch />
        </Form.Item>
      </Form>
    </Modal>
  )
}
