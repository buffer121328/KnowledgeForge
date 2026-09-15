import { describe, expect, it } from 'vitest'
import type { User } from '@/types'
import { isOrganizationAdmin, userHasPermission, workspaceKindFor } from './rolePolicy'

function user(role: User['role'], permissions: string[] = []): User {
  return { user_id: role, username: role, role, org_id: 'org-1', permissions }
}

describe('role policy', () => {
  it('maps legacy roles into the three business workspaces', () => {
    expect(workspaceKindFor(user('viewer'))).toBe('employee')
    expect(workspaceKindFor(user('editor'))).toBe('employee')
    expect(workspaceKindFor(user('admin'))).toBe('department_manager')
    expect(workspaceKindFor(user('organization_admin'))).toBe('organization_admin')
  })

  it('never grants implicit permissions from the legacy admin role name', () => {
    expect(userHasPermission(user('admin'), 'admin:manage')).toBe(false)
    expect(userHasPermission(user('organization_admin', ['admin:manage']), 'admin:manage')).toBe(true)
    expect(isOrganizationAdmin(user('admin'))).toBe(false)
  })
})
