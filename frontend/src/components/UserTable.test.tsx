// @vitest-environment jsdom
import { render, screen } from '@testing-library/react'
import { beforeAll, describe, expect, it, vi } from 'vitest'
import type { UserListItem } from '@/types'
import { UserTable } from './UserTable'

const users: UserListItem[] = [
  {
    user_id: 'organization-admin-1',
    username: 'governor',
    display_name: '公司管理员',
    email: 'governor@example.com',
    role: 'organization_admin',
    org_id: 'tenant-a',
    department_id: 'legacy-department',
    is_department_manager: false,
    is_active: true,
  },
  {
    user_id: 'manager-1',
    username: 'manager',
    display_name: '张经理',
    email: 'manager@example.com',
    role: 'admin',
    org_id: 'tenant-a',
    department_id: 'finance',
    is_department_manager: false,
    is_active: true,
  },
  {
    user_id: 'employee-1',
    username: 'employee',
    display_name: '李员工',
    email: 'employee@example.com',
    role: 'viewer',
    org_id: 'tenant-a',
    department_id: null,
    is_department_manager: true,
    is_active: true,
  },
]

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(() => ({
      addEventListener: vi.fn(),
      addListener: vi.fn(),
      dispatchEvent: vi.fn(),
      matches: false,
      media: '',
      onchange: null,
      removeEventListener: vi.fn(),
      removeListener: vi.fn(),
    })),
  })
})

describe('UserTable role and scope labels', () => {
  it('uses roles as the sole management identity and renders role-aware scope', () => {
    render(
      <UserTable
        loading={false}
        onDelete={vi.fn()}
        onEdit={vi.fn()}
        onResetPassword={vi.fn()}
        onToggleActive={vi.fn()}
        users={users}
      />,
    )

    expect(screen.getAllByText('公司管理员')).toHaveLength(2)
    expect(screen.getAllByText('部门负责人')).toHaveLength(1)
    expect(screen.getAllByText('员工')).toHaveLength(1)
    expect(screen.queryByRole('columnheader', { name: '负责人' })).toBeNull()
    expect(screen.getByText('全组织')).not.toBeNull()
    expect(screen.getByText('财务部')).not.toBeNull()
    expect(screen.getByText('未分配')).not.toBeNull()
    expect(screen.queryByText('legacy-department')).toBeNull()
    expect(screen.queryByText('系统权限')).toBeNull()
    expect(screen.queryByText('部门职责')).toBeNull()
    expect(screen.queryByText('admin')).toBeNull()
    expect(screen.queryByText('viewer')).toBeNull()
  })
})
