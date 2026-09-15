import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { User } from '@/types'
import { authApi } from '@/api/auth'
import { isOrganizationAdmin, userHasPermission } from '@/rolePolicy'

interface AuthState {
  user: User | null
  token: string | null
  refreshToken: string | null
  loading: boolean

  login: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
  fetchUser: () => Promise<void>
  hasPermission: (permission: string) => boolean
  isAdmin: () => boolean
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      refreshToken: null,
      loading: false,

      /** Authenticate the user and persist the returned session tokens. */
      login: async (username: string, password: string) => {
        set({ loading: true })
        try {
          const data = await authApi.login({ username, password })
          localStorage.setItem('access_token', data.access_token)
          localStorage.setItem('refresh_token', data.refresh_token)
          set({ token: data.access_token, refreshToken: data.refresh_token })
          const user = await authApi.me()
          set({ user, loading: false })
        } catch (e) {
          set({ loading: false })
          throw e
        }
      },

      /** Invalidate the session and clear locally persisted credentials. */
      logout: async () => {
        try {
          await authApi.logout()
        } catch {
          // ignore
        }
        localStorage.removeItem('access_token')
        localStorage.removeItem('refresh_token')
        set({ user: null, token: null, refreshToken: null })
      },

      /** Fetch the currently authenticated user. */
      fetchUser: async () => {
        const token = localStorage.getItem('access_token')
        if (!token) return
        try {
          const user = await authApi.me()
          set({ user, token })
        } catch {
          localStorage.removeItem('access_token')
          localStorage.removeItem('refresh_token')
          set({ user: null, token: null })
        }
      },

      /** Report whether the current user has a permission. */
      hasPermission: (permission: string) => {
        const { user } = get()
        return userHasPermission(user, permission)
      },

      /** Report whether the current user has an administrator role. */
      isAdmin: () => isOrganizationAdmin(get().user),
    }),
    {
      name: 'auth-storage',
      /** Select the authentication state persisted between sessions. */
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        refreshToken: state.refreshToken,
      }),
    },
  ),
)
