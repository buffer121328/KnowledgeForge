/** User-facing names for system permission roles in the user-management UI. */
export function userRoleLabel(role?: string | null): string {
  if (role === 'organization_admin') return '公司管理员'
  if (role === 'admin') return '部门负责人'
  if (role === 'api_user') return '接口账号'
  return '员工'
}

/** Selectable system roles exposed to administrators when managing users. */
export const USER_MANAGEMENT_ROLE_OPTIONS = [
  { value: 'viewer', label: '员工' },
  { value: 'admin', label: '部门负责人' },
  { value: 'organization_admin', label: '公司管理员' },
]

/** Filter values used by the user-management page. */
export const USER_ROLE_FILTER_OPTIONS = [
  { value: 'employee', label: '员工' },
  { value: 'admin', label: '部门负责人' },
  { value: 'organization_admin', label: '公司管理员' },
]
