import type { User } from '@/types'

export type WorkspaceKind = 'employee' | 'department_manager' | 'organization_admin'

/** Map persisted system roles to the three user-facing workspaces. */
export function workspaceKindFor(user: User | null): WorkspaceKind {
  if (user?.role === 'organization_admin') return 'organization_admin'
  if (user?.role === 'admin') return 'department_manager'
  return 'employee'
}

/** Resolve permissions exactly as issued by the server. */
export function userHasPermission(user: User | null, permission: string): boolean {
  return Boolean(user?.permissions.includes(permission))
}

export function isOrganizationAdmin(user: User | null): boolean {
  return user?.role === 'organization_admin'
}
