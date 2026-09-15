// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Outlet } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { User } from '@/types'

const authState = vi.hoisted(() => ({
  value: {} as {
    fetchUser: ReturnType<typeof vi.fn>
    hasPermission: (permission: string) => boolean
    token: string | null
    user: User | null
  },
}))

const dashboardModuleControl = vi.hoisted(() => {
  let resolvePending: (() => void) | undefined
  let deferred = false
  const loads = vi.fn()
  const module = { default: () => <div>dashboard-page</div> }
  return {
    defer: () => { deferred = true },
    loads,
    resolve: () => resolvePending?.(),
    load: () => {
      loads()
      if (!deferred) return Promise.resolve(module)
      return new Promise((resolve) => {
        resolvePending = () => {
          deferred = false
          resolve(module)
        }
      })
    },
  }
})

const evaluationModuleLoads = vi.hoisted(() => vi.fn())

vi.mock('@/stores/auth', () => {
  const useAuthStore = Object.assign(
    () => authState.value,
    { getState: () => authState.value },
  )
  return { useAuthStore }
})

vi.mock('@/components/Layout', () => ({ default: () => <Outlet /> }))
vi.mock('@/pages/Login', () => ({ default: () => <div>login-page</div> }))
vi.mock('@/pages/Dashboard', () => dashboardModuleControl.load())
vi.mock('@/pages/DocList', () => ({ default: () => <div>docs-page</div> }))
vi.mock('@/pages/QAChat', () => ({ default: () => <div>qa-page</div> }))
vi.mock('@/pages/GraphView', () => ({ default: () => <div>graph-page</div> }))
vi.mock('@/pages/UserList', () => ({ default: () => <div>users-page</div> }))
vi.mock('@/pages/AuditLogs', () => ({ default: () => <div>audit-page</div> }))
vi.mock('@/pages/Webhooks', () => ({ default: () => <div>webhooks-page</div> }))
vi.mock('@/pages/EvaluationGovernance', () => {
  evaluationModuleLoads()
  return { default: () => <div>evaluation-page</div> }
})
vi.mock('@/pages/RoleWorkspace', () => ({ default: () => <div>workspace-page</div> }))

import App, { RouteLoadErrorBoundary } from './App'

const viewer: User = {
  user_id: 'viewer-1',
  username: 'viewer',
  role: 'viewer',
  org_id: 'org-1',
  permissions: ['doc:read'],
}

const organizationAdmin: User = {
  user_id: 'org-admin-1',
  username: 'org-admin',
  role: 'organization_admin',
  org_id: 'org-1',
  permissions: ['admin:manage'],
}

const departmentAdmin: User = {
  user_id: 'department-admin-1',
  username: 'department-admin',
  role: 'admin',
  org_id: 'org-1',
  permissions: ['admin:manage'],
  department_id: 'finance',
  is_department_manager: true,
}

beforeEach(() => {
  authState.value = {
    fetchUser: vi.fn(),
    hasPermission: (permission) => viewer.permissions.includes(permission),
    token: 'token',
    user: viewer,
  }
})

afterEach(() => {
  cleanup()
})

describe('permission-aware application routes', () => {
  it('redirects a non-administrator root route to the role workspace', async () => {
    render(<MemoryRouter initialEntries={['/']}><App /></MemoryRouter>)

    expect(await screen.findByText('workspace-page')).toBeTruthy()
  })

  it('does not render dashboard for a direct non-administrator visit', async () => {
    render(<MemoryRouter initialEntries={['/dashboard']}><App /></MemoryRouter>)

    expect(await screen.findByText('workspace-page')).toBeTruthy()
    expect(screen.queryByText('dashboard-page')).toBeNull()
  })

  it('renders an authorized page through the lazy loading fallback', async () => {
    dashboardModuleControl.loads.mockClear()
    dashboardModuleControl.defer()
    authState.value = {
      ...authState.value,
      hasPermission: () => true,
      user: organizationAdmin,
    }

    render(<MemoryRouter initialEntries={['/dashboard']}><App /></MemoryRouter>)

    expect(await screen.findByRole('status', { name: '正在加载页面' })).toBeTruthy()
    expect(dashboardModuleControl.loads).toHaveBeenCalledTimes(1)

    dashboardModuleControl.resolve()
    expect(await screen.findByText('dashboard-page')).toBeTruthy()
  })

  it('offers a safe retry state when a route module fails', () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const BrokenRoute = () => {
      throw new Error('chunk retrieval failed')
    }

    render(<RouteLoadErrorBoundary><BrokenRoute /></RouteLoadErrorBoundary>)

    expect(screen.getByText('页面暂时无法加载')).toBeTruthy()
    expect(screen.getByRole('button', { name: '重新加载' })).toBeTruthy()
    consoleError.mockRestore()
  })

  it('redirects a department administrator away from the organization-only evaluation route', async () => {
    authState.value = {
      ...authState.value,
      hasPermission: () => true,
      user: departmentAdmin,
    }

    render(<MemoryRouter initialEntries={['/evaluation']}><App /></MemoryRouter>)

    expect(await screen.findByText('workspace-page')).toBeTruthy()
    expect(screen.queryByText('evaluation-page')).toBeNull()
    expect(evaluationModuleLoads).not.toHaveBeenCalled()
  })

  it('redirects an API identity away from the organization-only evaluation route', async () => {
    authState.value = {
      ...authState.value,
      hasPermission: () => true,
      user: {
        user_id: 'api-user-1',
        username: 'api-user',
        role: 'api_user',
        org_id: 'org-1',
        permissions: [],
      },
    }

    render(<MemoryRouter initialEntries={['/evaluation']}><App /></MemoryRouter>)

    expect(await screen.findByText('workspace-page')).toBeTruthy()
    expect(screen.queryByText('evaluation-page')).toBeNull()
  })

  it('keeps dashboard available only to users issued the governance permission', async () => {
    authState.value = {
      ...authState.value,
      hasPermission: () => true,
      user: organizationAdmin,
    }

    render(<MemoryRouter initialEntries={['/dashboard']}><App /></MemoryRouter>)

    expect(await screen.findByText('dashboard-page')).toBeTruthy()
  })
})
