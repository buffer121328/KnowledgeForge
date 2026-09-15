import type { DepartmentItem } from '@/types'

export const DEPARTMENT_LABELS: Readonly<Record<string, string>> = {
  administration: '行政管理部',
  finance: '财务部',
  general: '默认部门',
  hr: '人力资源部',
  human_resources: '人力资源部',
  it: '信息技术部',
  legal: '法务部',
  marketing: '市场部',
  operations: '运营部',
  procurement: '采购部',
  procurement_warehouse: '采购/仓储部',
  sales: '销售部',
  technology: '技术部',
}

/** Format a known department consistently while preserving explicit custom names. */
export function departmentDisplayName(
  departmentId?: string | null,
  explicitName?: string | null,
): string {
  const id = departmentId?.trim() ?? ''
  const name = explicitName?.trim() ?? ''
  return DEPARTMENT_LABELS[id.toLowerCase()] ?? (name || id)
}

export function departmentOptionLabel(department: DepartmentItem): string {
  return departmentDisplayName(department.department_id, department.name)
}
