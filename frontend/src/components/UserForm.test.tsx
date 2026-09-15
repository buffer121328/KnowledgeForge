// @vitest-environment jsdom
import { Form } from 'antd'
import { useEffect } from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import type { UserFormValues } from '@/pages/useUsers'
import { UserForm } from './UserForm'

afterEach(cleanup)

beforeAll(() => {  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(() => ({
      addEventListener: vi.fn(),
      addListener: vi.fn(),
      dispatchEvent: vi.fn(),
      matches: false,
      media: '',
      onchange: null,
      removeEventListener: vi.fn(),
      removeListener: vi.fn(),
    })),
  })
})

function UserFormHarness() {
  const [form] = Form.useForm<UserFormValues>()
  return (
    <UserForm
      editingUser={null}
      form={form}
      onCancel={vi.fn()}
      onSubmit={vi.fn()}
      open
    />
  )
}

describe('UserForm system role selector labels', () => {
  it('shows renamed system permission roles only', () => {
    render(<UserFormHarness />)

    expect(screen.queryByText('系统权限角色')).toBeNull()
    expect(screen.getByText('角色')).toBeTruthy()
    expect(screen.getByText('员工')).toBeTruthy()
    expect(screen.queryByText(/admin|editor|viewer|api_user/)).toBeNull()
  })
})


function EditingUserFormHarness({ onSubmit }: { onSubmit: (values: UserFormValues) => Promise<void> }) {
  const [form] = Form.useForm<UserFormValues>()
  useEffect(() => {
    form.setFieldsValue({
      username: 'cheng',
      display_name: 'cheng',
      email: 'gc0507@163.com',
      role: 'viewer',
      department_id: 'finance',
    })
  }, [form])
  return (
    <UserForm
      editingUser={{
        user_id: 'user-cheng',
        username: 'cheng',
        display_name: 'cheng',
        email: 'gc0507@163.com',
        role: 'viewer',
        org_id: 'company-a',
        department_id: 'finance',
        is_active: true,
      }}
      form={form}
      onCancel={vi.fn()}
      onSubmit={onSubmit}
      open
      organizationId="company-a"
    />
  )
}

describe('UserForm confirmation', () => {
  it('submits valid edit values and keeps organization ownership read-only', async () => {
    const onSubmit = vi.fn<(values: UserFormValues) => Promise<void>>().mockResolvedValue(undefined)
    render(<EditingUserFormHarness onSubmit={onSubmit} />)

    expect((screen.getByDisplayValue('company-a') as HTMLInputElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: /确\s*定/ }))

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      username: 'cheng',
      department_id: 'finance',
    })))
    expect(onSubmit).not.toHaveBeenCalledWith(expect.objectContaining({ org_id: 'company-a' }))
  })
})
