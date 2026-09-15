import client from './client'
import type { UserListItem, UserCreateRequest, Role } from '@/types'

export const userApi = {
  /** Fetch user records. */
  list: (params?: { search?: string; role?: string; page?: number; page_size?: number }) =>
    client.get<UserListItem[]>('/users', { params }).then((r) => r.data),

  /** Fetch user details. */
  get: (userId: string) =>
    client.get<UserListItem>(`/users/${userId}`).then((r) => r.data),

  /** Create a user. */
  create: (data: UserCreateRequest) =>
    client.post<UserListItem>('/users', data).then((r) => r.data),

  /** Update a user. */
  update: (userId: string, data: Partial<UserCreateRequest>) =>
    client.patch<UserListItem>(`/users/${userId}`, data).then((r) => r.data),

  /** Delete a user. */
  delete: (userId: string) => client.delete(`/users/${userId}`),

  /** Reset the selected user password. */
  resetPassword: (userId: string, newPassword: string) =>
    client.post(`/users/${userId}/reset-password`, { new_password: newPassword }),

  /** Enable or disable the selected user account. */
  toggleActive: (userId: string) =>
    client.post(`/users/${userId}/toggle-active`),

  /** Change the current user password. */
  changeOwnPassword: (oldPassword: string, newPassword: string) =>
    client.post('/users/me/password', { old_password: oldPassword, new_password: newPassword }),

  /** Assign a role to multiple users. */
  batchUpdateRole: (userIds: string[], role: string) =>
    client.post('/users/batch/role', { user_ids: userIds, role }),
}

export const roleApi = {
  /** Fetch role records. */
  list: () => client.get<Role[]>('/roles').then((r) => r.data),

  /** Fetch role details. */
  get: (roleName: string) => client.get<Role>(`/roles/${roleName}`).then((r) => r.data),

  /** Create a role. */
  create: (data: Partial<Role>) => client.post<Role>('/roles', data).then((r) => r.data),

  /** Update a role. */
  update: (roleName: string, data: Partial<Role>) =>
    client.patch<Role>(`/roles/${roleName}`, data).then((r) => r.data),

  /** Delete a role. */
  delete: (roleName: string) => client.delete(`/roles/${roleName}`),

  /** Fetch permissions assigned to a role. */
  getPermissions: (roleName: string) =>
    client.get<string[]>(`/roles/${roleName}/permissions`).then((r) => r.data),

  /** Replace permissions assigned to a role. */
  updatePermissions: (roleName: string, permissions: string[]) =>
    client.put(`/roles/${roleName}/permissions`, { permissions }),

  /** Fetch all permissions available for role assignment. */
  listAllPermissions: () =>
    client.get('/roles/meta/permissions').then((r) => r.data),
}
