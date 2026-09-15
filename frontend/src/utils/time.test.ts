import { describe, expect, it } from 'vitest'
import { formatChinaStandardTime } from './time'

describe('formatChinaStandardTime', () => {
  it('renders UTC timestamps in the fixed Asia/Shanghai zone without a suffix', () => {
    expect(formatChinaStandardTime('2026-08-01T10:00:00Z')).toBe('2026-08-01 18:00:00')
  })

  it('uses a bounded placeholder for missing or malformed values', () => {
    expect(formatChinaStandardTime(undefined)).toBe('—')
    expect(formatChinaStandardTime('not-a-time')).toBe('—')
  })
})
