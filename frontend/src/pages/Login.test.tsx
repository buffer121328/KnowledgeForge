// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { App } from 'antd'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import Login from './Login'

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  })
})

afterEach(cleanup)

describe('Login production credentials', () => {
  it('renders empty credential fields without local default-account guidance', () => {
    render(
      <MemoryRouter>
        <App>
          <Login />
        </App>
      </MemoryRouter>,
    )

    expect((screen.getByPlaceholderText('用户名') as HTMLInputElement).value).toBe('')
    expect((screen.getByPlaceholderText('密码') as HTMLInputElement).value).toBe('')
    expect(screen.queryByText(/admin\s*\/\s*admin123/i)).toBeNull()
  })
})
