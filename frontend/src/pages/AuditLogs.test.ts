import dayjs from 'dayjs'
import { describe, expect, it } from 'vitest'
import { buildAuditLogFilters } from './AuditLogs'

describe('AuditLogs filter normalization', () => {
  it('trims user IDs and serializes the selected date range for shared list/export filters', () => {
    expect(buildAuditLogFilters(
      '  user-1  ',
      'document.read',
      [dayjs('2026-08-01T08:00:00+08:00'), dayjs('2026-08-02T08:00:00+08:00')],
    )).toEqual({
      user_id: 'user-1',
      action: 'document.read',
      start: '2026-08-01T00:00:00.000Z',
      end: '2026-08-02T00:00:00.000Z',
    })
  })

  it('omits empty controls', () => {
    expect(buildAuditLogFilters('   ', undefined, null)).toEqual({})
  })
})
