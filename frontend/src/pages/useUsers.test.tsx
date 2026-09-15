// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react'
import { App } from 'antd'
import type { PropsWithChildren } from 'react'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import type { UserListItem } from '@/types'

const userApi = vi.hoisted(() => ({
  create: vi.fn(),
  delete: vi.fn(),
  list: vi.fn(),
  resetPassword: vi.fn(),
  toggleActive: vi.fn(),
  update: vi.fn(),
}))

vi.mock('@/api/users', () => ({ userApi }))

import { useUsers } from './useUsers'

const adminUser: UserListItem = {
  user_id: 'admin-1', username: 'admin', display_name: 'Admin', email: 'admin@example.com',
  role: 'admin', org_id: 'org-1', is_active: true,
}
const employeeUser: UserListItem = {
  user_id: 'viewer-1', username: 'alice', display_name: 'Alice', email: 'alice@example.com',
  role: 'viewer', org_id: 'org-1', is_active: true,
}
const organizationAdminUser: UserListItem = {
  user_id: 'org-admin-1', username: 'org-admin', display_name: 'Org Admin', email: 'owner@example.com',
  role: 'organization_admin', org_id: 'org-1', is_active: true,
}

function Wrapper({ children }: PropsWithChildren) {
  return <App>{children}</App>
}

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(() => ({
      matches: false,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  })
})

beforeEach(() => {
  vi.clearAllMocks()
  userApi.list.mockResolvedValue([organizationAdminUser, adminUser, employeeUser])
})

describe('useUsers server-backed filters', () => {
  it('loads a bounded user page and sends search on an explicit query', async () => {
    const { result } = renderHook(() => useUsers(), { wrapper: Wrapper })
    await waitFor(() => expect(userApi.list).toHaveBeenCalledWith({ page_size: 100 }))

    act(() => result.current.setSearch('alice'))
    await act(async () => { await result.current.fetchUsers() })

    expect(userApi.list).toHaveBeenLastCalledWith({ page_size: 100, search: 'alice' })
  })

  it('queries exact admin roles on the server', async () => {
    const { result } = renderHook(() => useUsers(), { wrapper: Wrapper })
    await waitFor(() => expect(userApi.list).toHaveBeenCalledOnce())

    await act(async () => { await result.current.handleRoleFilterChange('admin') })

    expect(userApi.list).toHaveBeenLastCalledWith({ page_size: 100, role: 'admin' })
  })

  it('queries organization administrators separately', async () => {
    const { result } = renderHook(() => useUsers(), { wrapper: Wrapper })
    await waitFor(() => expect(userApi.list).toHaveBeenCalledOnce())

    await act(async () => { await result.current.handleRoleFilterChange('organization_admin') })

    expect(userApi.list).toHaveBeenLastCalledWith({ page_size: 100, role: 'organization_admin' })
  })

  it('refreshes before applying the aggregate employee filter locally', async () => {
    const { result } = renderHook(() => useUsers(), { wrapper: Wrapper })
    await waitFor(() => expect(userApi.list).toHaveBeenCalledOnce())

    await act(async () => { await result.current.handleRoleFilterChange('employee') })

    expect(userApi.list).toHaveBeenLastCalledWith({ page_size: 100 })
    expect(result.current.users).toEqual([employeeUser])
  })
})


describe('useUsers mutation submission', () => {
  it('updates the selected user without submitting an organization override', async () => {
    const { result } = renderHook(() => useUsers(), { wrapper: Wrapper })
    await waitFor(() => expect(userApi.list).toHaveBeenCalledOnce())

    act(() => result.current.openEditModal(employeeUser))
    await act(async () => {
      await result.current.handleSubmit({
        username: 'alice',
        display_name: 'Alice Updated',
        email: 'alice@example.com',
        role: 'viewer',
        department_id: 'finance',
      })
    })

    expect(userApi.update).toHaveBeenCalledWith('viewer-1', {
      display_name: 'Alice Updated',
      email: 'alice@example.com',
      role: 'viewer',
      department_id: 'finance',
      is_department_manager: false,
    })
    expect(result.current.modalOpen).toBe(false)
    expect(result.current.submitting).toBe(false)
  })

  it('keeps the edit dialog retryable after an API failure', async () => {
    userApi.update.mockRejectedValueOnce(new Error('update failed'))
    const { result } = renderHook(() => useUsers(), { wrapper: Wrapper })
    await waitFor(() => expect(userApi.list).toHaveBeenCalledOnce())

    act(() => result.current.openEditModal(employeeUser))
    await act(async () => {
      await result.current.handleSubmit({
        username: 'alice',
        display_name: 'Alice Updated',
        email: 'alice@example.com',
        role: 'viewer',
        department_id: 'finance',
      })
    })

    expect(result.current.modalOpen).toBe(true)
    expect(result.current.submitting).toBe(false)
  })
})
