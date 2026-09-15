import { beforeEach, describe, expect, it, vi } from 'vitest'

const client = vi.hoisted(() => ({
  defaults: { baseURL: '/api/v1' },
  get: vi.fn(),
}))

vi.mock('./client', () => ({ default: client }))

import { adminApi } from './admin'

beforeEach(() => {
  vi.clearAllMocks()
  client.defaults.baseURL = '/api/v1'
  client.get.mockResolvedValue({ data: {} })
})

describe('adminApi route and audit filter contracts', () => {
  it('uses the lightweight request trend endpoint with an explicit window', async () => {
    await adminApi.requestTrends('24h')

    expect(client.get).toHaveBeenCalledWith('/admin/request-trends', { params: { window: '24h' } })
  })

  it('uses the unversioned health base while accepting ready and not-ready reports', async () => {
    await adminApi.readiness()

    expect(client.get).toHaveBeenCalledWith(
      '/health/ready',
      expect.objectContaining({ baseURL: '/api', validateStatus: expect.any(Function) }),
    )
    const config = client.get.mock.calls[0][1]
    expect(config.validateStatus(200)).toBe(true)
    expect(config.validateStatus(503)).toBe(true)
    expect(config.validateStatus(500)).toBe(false)
  })

  it('forwards identical active filters to list and export requests', async () => {
    const filters = {
      user_id: 'user-1',
      action: 'document.read',
      start: '2026-08-01T00:00:00.000Z',
      end: '2026-08-02T00:00:00.000Z',
    }

    await adminApi.auditLogs({ ...filters, limit: 500 })
    await adminApi.exportAuditLogs({ ...filters, limit: 10000 })

    expect(client.get).toHaveBeenNthCalledWith(1, '/audit/logs', {
      params: { ...filters, limit: 500 },
    })
    expect(client.get).toHaveBeenNthCalledWith(2, '/audit/logs/export', {
      params: { ...filters, limit: 10000 },
      responseType: 'blob',
    })
  })
})
