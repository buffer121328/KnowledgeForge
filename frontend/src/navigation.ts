export type NavigationGroup = 'knowledge' | 'governance' | 'operations'

export interface NavigationItem {
  key: string
  label: string
  permission?: string
  organizationAdminOnly?: boolean
  group: NavigationGroup
}

/** Shared route metadata used by menus and permission-aware redirects. */
export const APP_NAVIGATION: NavigationItem[] = [
  { key: '/workspace', label: '工作台', group: 'knowledge' },
  { key: '/docs', label: '知识文档', permission: 'doc:read', group: 'knowledge' },
  { key: '/qa', label: '智能问答', permission: 'qa:query', group: 'knowledge' },
  { key: '/graph', label: '知识图谱', permission: 'graph:read', group: 'knowledge' },
  { key: '/evaluation', label: '评测治理', organizationAdminOnly: true, group: 'governance' },
  { key: '/users', label: '用户与权限', permission: 'admin:user', group: 'governance' },
  { key: '/audit', label: '审计日志', permission: 'admin:audit', group: 'governance' },
  { key: '/dashboard', label: '系统监控', permission: 'admin:manage', group: 'operations' },
  { key: '/webhooks', label: 'Webhook', permission: 'admin:manage', group: 'operations' },
]

/** Return navigation items authorized by the current permission resolver. */
export function getVisibleNavigation(
  hasPermission: (permission: string) => boolean,
  isOrganizationAdmin = false,
) {
  return APP_NAVIGATION.filter(
    (item) =>
      (!item.permission || hasPermission(item.permission))
      && (!item.organizationAdminOnly || isOrganizationAdmin),
  )
}

/** Use the role-aware workspace as every interactive user's stable landing page. */
export function getDefaultAuthenticatedPath(
  hasPermission: (permission: string) => boolean,
  isOrganizationAdmin = false,
): string | undefined {
  return getVisibleNavigation(hasPermission, isOrganizationAdmin)[0]?.key
}
