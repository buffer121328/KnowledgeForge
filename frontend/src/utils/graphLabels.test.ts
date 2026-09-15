import { describe, expect, it } from 'vitest'
import { formatGraphNodeId, formatGraphRelation } from './graphLabels'

describe('graph label localization', () => {
  it('maps internal department relations regardless of storage casing', () => {
    expect(formatGraphRelation('reports_to')).toBe('汇报给')
    expect(formatGraphRelation('collaborates-with')).toBe('协作')
    expect(formatGraphRelation('work at')).toBe('任职于')
    expect(formatGraphRelation('WORKS_AT')).toBe('任职于')
    expect(formatGraphRelation('part of')).toBe('隶属于')
    expect(formatGraphRelation('is-part-of')).toBe('隶属于')
    expect(formatGraphRelation('employed by')).toBe('受雇于')
    expect(formatGraphRelation('located in')).toBe('位于')
  })

  it('maps known department identifiers to Chinese labels', () => {
    expect(formatGraphNodeId('department:human_resources')).toBe('人力资源部（部门）')
    expect(formatGraphNodeId('department:finance')).toBe('财务部（部门）')
    expect(formatGraphNodeId('department:procurement_warehouse')).toBe('采购/仓储部（部门）')
    expect(formatGraphNodeId('department:administration')).toBe('行政管理部（部门）')
  })
})
