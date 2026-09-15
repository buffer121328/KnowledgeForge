import { useEffect, useState } from 'react'
import { App, Form } from 'antd'
import { webhookApi } from '@/api/admin'
import type { Webhook } from '@/types'

export interface WebhookFormValues {
  events: string[]
  is_active?: boolean
  secret?: string
  url: string
}

export interface WebhookEventOption {
  label: string
  value: string
}

/** 与后端 WebhookEvent 对齐的兜底列表。 */
export const FALLBACK_EVENTS: WebhookEventOption[] = [
  { value: 'doc.ingested', label: '文档入库' },
  { value: 'doc.updated', label: '文档更新' },
  { value: 'doc.deleted', label: '文档删除' },
  { value: 'qa.completed', label: '问答完成' },
  { value: 'qa.feedback', label: '问答反馈' },
  { value: 'system.error', label: '系统错误' },
]

/** Extract a user-facing message from an API failure. */
function apiErrorMessage(error: unknown, fallback: string): string {
  const detail = (error as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
  return typeof detail === 'string' ? detail : fallback
}

/** Manage webhooks state and related actions. */
export function useWebhooks() {
  const { message } = App.useApp()
  const [webhooks, setWebhooks] = useState<Webhook[]>([])
  const [loading, setLoading] = useState(false)
  const [modalOpen, setModalOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [eventOptions, setEventOptions] = useState<WebhookEventOption[]>(FALLBACK_EVENTS)
  const [form] = Form.useForm<WebhookFormValues>()

  /** Fetch webhooks and normalize the response for the table. */
  const fetchWebhooks = async () => {
    setLoading(true)
    try {
      const data = await webhookApi.list()
      setWebhooks(data)
    } catch (error) {
      message.error(apiErrorMessage(error, '加载 Webhook 列表失败'))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void fetchWebhooks()
    void webhookApi
      .events()
      .then((events) => {
        if (events?.length) setEventOptions(events)
      })
      .catch(() => undefined)
  }, [])

  /** Open the form for creating a record. */
  const openCreateModal = () => {
    form.resetFields()
    setModalOpen(true)
  }

  /** Close the webhook creation form. */
  const closeCreateModal = () => {
    setModalOpen(false)
  }

  /** Create a webhook from the submitted form values. */
  const handleCreate = async () => {
    try {
      const values = await form.validateFields()
      setSubmitting(true)
      await webhookApi.create({
        url: values.url,
        events: values.events,
        secret: values.secret || undefined,
        is_active: values.is_active !== false,
      })
      message.success('Webhook 创建成功')
      setModalOpen(false)
      form.resetFields()
      void fetchWebhooks()
    } catch (error) {
      if ((error as { errorFields?: unknown })?.errorFields) return
      message.error(apiErrorMessage(error, '创建失败'))
    } finally {
      setSubmitting(false)
    }
  }

  /** Delete the selected record after confirmation. */
  const handleDelete = async (id: string) => {
    try {
      await webhookApi.delete(id)
      message.success('Webhook 已删除')
      void fetchWebhooks()
    } catch (error) {
      message.error(apiErrorMessage(error, '删除失败'))
    }
  }

  /** Send a test event to the selected webhook. */
  const handleTest = async (id: string) => {
    try {
      const result = await webhookApi.test(id)
      if (result.success) {
        message.success(result.message || '测试回调成功')
      } else {
        message.error(result.error || '测试回调失败')
      }
      void fetchWebhooks()
    } catch (error) {
      message.error(apiErrorMessage(error, '测试发送失败'))
    }
  }

  return {
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
  }
}
