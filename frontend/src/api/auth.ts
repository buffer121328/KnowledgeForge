import client from './client'
import type { LoginRequest, TokenResponse, User } from '@/types'

export const authApi = {
  /** Authenticate with a username and password. */
  login: (data: LoginRequest) =>
    client.post<TokenResponse>('/auth/login', data).then((r) => r.data),

  /** Register a new user account. */
  register: (data: LoginRequest & { org_id?: string; role?: string }) =>
    client.post<TokenResponse>('/auth/register', data).then((r) => r.data),

  /** Exchange a refresh token for a new access token. */
  refresh: (refreshToken: string) =>
    client.post<TokenResponse>('/auth/refresh', { refresh_token: refreshToken }).then((r) => r.data),

  /** Invalidate the current authenticated session. */
  logout: () => client.post('/auth/logout'),

  /** Fetch the current authenticated user. */
  me: () => client.get<User>('/auth/me').then((r) => r.data),

  /** Create an API key for the current user. */
  createApiKey: (data: { name: string; permissions?: string[]; expires_days?: number }) =>
    client.post('/auth/apikey/create', data).then((r) => r.data),

  /** Fetch API keys owned by the current user. */
  listApiKeys: () => client.get('/auth/apikey/list').then((r) => r.data),

  /** Delete an API key owned by the current user. */
  deleteApiKey: (keyId: string) => client.delete(`/auth/apikey/${keyId}`),
}
