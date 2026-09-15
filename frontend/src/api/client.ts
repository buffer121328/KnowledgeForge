import axios, { AxiosError, type AxiosInstance, type InternalAxiosRequestConfig } from 'axios'
import { message } from 'antd'

type ApiErrorDetail = string | { code?: string; message?: string; request_id?: string }

export interface ApiErrorInfo {
  code: string
  message: string
  requestId: string
  status?: number
}

/** Read one structured API error without coupling page hooks to Axios internals. */
export function getApiErrorInfo(error: unknown): ApiErrorInfo | null {
  if (!axios.isAxiosError<{ detail?: ApiErrorDetail }>(error)) return null
  const status = error.response?.status
  if (status === 413) {
    return { code: 'request_entity_too_large', message: '上传内容超过网关限制，请减少文件大小或联系管理员调整上传限制', requestId: '', status }
  }
  const detail = error.response?.data?.detail
  if (typeof detail === 'string') {
    return detail ? { code: '', message: detail, requestId: '', status } : null
  }
  if (!detail || typeof detail !== 'object') return null
  const code = typeof detail.code === 'string' ? detail.code : ''
  const message = typeof detail.message === 'string' ? detail.message : ''
  const requestId = typeof detail.request_id === 'string' ? detail.request_id : ''
  return code || message || requestId ? { code, message, requestId, status } : null
}

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api/v1'

/** Convert an API error payload into a user-facing message. */
function formatErrorDetail(detail: ApiErrorDetail | undefined, fallback: string): string {
  if (typeof detail === 'string' && detail) return detail
  if (detail && typeof detail === 'object' && typeof detail.message === 'string' && detail.message) {
    return detail.message
  }
  return fallback
}

const client: AxiosInstance = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器：注入 JWT Token；FormData 去掉默认 JSON Content-Type
client.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('access_token')
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`
    }
    if (config.data instanceof FormData && config.headers) {
      // 交给浏览器自动带 boundary，避免 Missing boundary / 解析失败
      const headers = config.headers as { delete?: (name: string) => void } & Record<string, unknown>
      if (typeof headers.delete === 'function') {
        headers.delete('Content-Type')
      } else {
        delete headers['Content-Type']
      }
    }
    return config
  },
  (error) => Promise.reject(error),
)

// 响应拦截器：统一错误处理 + Token 刷新
let isRefreshing = false
let failedQueue: Array<(token: string | null) => void> = []

/** Resume queued API requests after token refresh completes. */
const processQueue = (token: string | null) => {
  failedQueue.forEach((cb) => cb(token))
  failedQueue = []
}

client.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<{ detail?: ApiErrorDetail }>) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean }
    const status = error.response?.status

    // 401 且非重试且非登录接口 → 尝试刷新 Token
    if (
      status === 401 &&
      !originalRequest._retry &&
      !originalRequest.url?.includes('/auth/login') &&
      !originalRequest.url?.includes('/auth/refresh')
    ) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push((token) => {
            if (!token) {
              reject(error)
              return
            }
            originalRequest.headers!.Authorization = `Bearer ${token}`
            resolve(client(originalRequest))
          })
        })
      }

      originalRequest._retry = true
      isRefreshing = true

      try {
        const refreshToken = localStorage.getItem('refresh_token')
        if (!refreshToken) {
          throw new Error('No refresh token')
        }
        const { data } = await axios.post(`${API_BASE}/auth/refresh`, {
          refresh_token: refreshToken,
        })
        const newToken = data.access_token
        localStorage.setItem('access_token', newToken)
        localStorage.setItem('refresh_token', data.refresh_token)
        processQueue(newToken)
        originalRequest.headers!.Authorization = `Bearer ${newToken}`
        return client(originalRequest)
      } catch (refreshError) {
        processQueue(null)
        localStorage.removeItem('access_token')
        localStorage.removeItem('refresh_token')
        message.error('登录已过期，请重新登录')
        window.location.href = '/login'
        return Promise.reject(refreshError)
      } finally {
        isRefreshing = false
      }
    }

    // 其他错误
    const detail = formatErrorDetail(error.response?.data?.detail, error.message || '请求失败')
    if (status === 403) {
      message.error(`权限不足: ${detail}`)
    } else if (status && status >= 500) {
      message.error(`服务器错误: ${detail}`)
    } else if (status === 429) {
      message.warning('请求过于频繁，请稍后再试')
    }
    return Promise.reject(error)
  },
)

export default client
export { API_BASE }
