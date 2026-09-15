import { describe, expect, it } from 'vitest'
import { getDefaultAuthenticatedPath, getVisibleNavigation } from './navigation'

describe('permission-aware application navigation', () => {
  it('uses the role workspace as the administrator default', () => {
    const hasPermission = () => true

    expect(getDefaultAuthenticatedPath(hasPermission)).toBe('/workspace')
    expect(getVisibleNavigation(hasPermission).some((item) => item.key === '/dashboard')).toBe(true)
  })

  it('hides evaluation governance from a department administrator even with admin:manage', () => {
    const hasPermission = (permission: string) => permission === 'admin:manage'

    expect(getVisibleNavigation(hasPermission, false).some((item) => item.key === '/evaluation')).toBe(false)
    expect(getVisibleNavigation(hasPermission, true).some((item) => item.key === '/evaluation')).toBe(true)
  })

  it('hides dashboard while retaining the useful workspace without admin manage', () => {
    const hasPermission = (permission: string) => permission === 'doc:read'

    expect(getDefaultAuthenticatedPath(hasPermission)).toBe('/workspace')
    expect(getVisibleNavigation(hasPermission).some((item) => item.key === '/dashboard')).toBe(false)
  })
})
