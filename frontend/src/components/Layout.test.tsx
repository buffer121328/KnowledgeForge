// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter, Outlet } from 'react-router-dom'
import type { User } from '@/types'

const authState = vi.hoisted(() => ({
  user: null as User | null,
  logout: vi.fn().mockResolvedValue(undefined),
  hasPermission: vi.fn(() => true),
}))

vi.mock('@/stores/auth', () => ({ useAuthStore: () => authState }))
vi.mock('@/navigation', () => ({
  getVisibleNavigation: () => [{ key: '/dashboard', label: '系统监控', group: 'operations' }],
}))

import Layout from './Layout'

beforeEach(() => {
  authState.user = {
    user_id: 'admin-1',
    username: 'admin.admin',
    display_name: '部门负责人',
    role: 'admin',
    org_id: 'org_123',
    permissions: [],
  }
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(() => ({
      addEventListener: vi.fn(),
      addListener: vi.fn(),
      dispatchEvent: vi.fn(),
      matches: true,
      media: '',
      onchange: null,
      removeEventListener: vi.fn(),
      removeListener: vi.fn(),
    })),
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('authenticated header identity', () => {
  it('shows the user display name and localized role instead of the internal username and role value', () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Layout />
        <Outlet />
      </MemoryRouter>,
    )

    expect(screen.getByText('部门负责人')).toBeTruthy()
    expect(screen.getByText('部门负责人 · org_123')).toBeTruthy()
    expect(screen.queryByText('admin.admin')).toBeNull()
  })
})
