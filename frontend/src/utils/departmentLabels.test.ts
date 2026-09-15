import { describe, expect, it } from 'vitest'
import { departmentDisplayName } from './departmentLabels'

describe('department labels', () => {
  it('formats all company-demo departments consistently', () => {
    expect(departmentDisplayName('human_resources')).toBe('人力资源部')
    expect(departmentDisplayName('finance')).toBe('财务部')
    expect(departmentDisplayName('procurement_warehouse')).toBe('采购/仓储部')
    expect(departmentDisplayName('administration')).toBe('行政管理部')
  })

  it('preserves explicit custom department names and falls back to ids', () => {
    expect(departmentDisplayName('research', '研究院')).toBe('研究院')
    expect(departmentDisplayName('research')).toBe('research')
  })
})
