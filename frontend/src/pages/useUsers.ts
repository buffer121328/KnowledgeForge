import { createElement, useEffect, useState } from 'react'
import { App, Form, Input as AntInput, Modal } from 'antd'
import { userApi } from '@/api/users'
import { docApi } from '@/api/docs'
import { getApiErrorInfo } from '@/api/client'
import { useAuthStore } from '@/stores/auth'
import type { DepartmentItem, UserCreateRequest, UserListItem } from '@/types'

export type UserFormValues = Omit<UserCreateRequest, 'password' | 'org_id'> & { password?: string }
export type UserRoleFilter = 'organization_admin' | 'admin' | 'employee' | undefined

interface UserQueryOverrides {
  search?: string
  roleFilter?: UserRoleFilter
}

/** Manage users state and related actions. */
export function useUsers() {
  const { message } = App.useApp()
  const { user: currentUser } = useAuthStore()
  const [users, setUsers] = useState<UserListItem[]>([])
  const [loading, setLoading] = useState(false)
  const [modalOpen, setModalOpen] = useState(false)
  const [editingUser, setEditingUser] = useState<UserListItem | null>(null)
  const [form] = Form.useForm<UserFormValues>()
  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<UserRoleFilter>()
  const [submitting, setSubmitting] = useState(false)
  const [departments, setDepartments] = useState<DepartmentItem[]>([])

  /** Fetch the user list using current controls or an immediate control override. */
  const fetchUsers = async (overrides: UserQueryOverrides = {}) => {
    const activeSearch = overrides.search ?? search
    const activeRole = Object.prototype.hasOwnProperty.call(overrides, 'roleFilter')
      ? overrides.roleFilter
      : roleFilter
    const normalizedSearch = activeSearch.trim()
    setLoading(true)
    try {
      const data = await userApi.list({
        page_size: 100,
        ...(normalizedSearch ? { search: normalizedSearch } : {}),
        ...(activeRole === 'admin' || activeRole === 'organization_admin' ? { role: activeRole } : {}),
      })
      setUsers(activeRole === 'employee'
        ? data.filter((item) => !['admin', 'organization_admin', 'api_user'].includes(item.role))
        : data)
    } catch {
      message.error('加载用户列表失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void fetchUsers({ search: '', roleFilter: undefined })
    void docApi.departments().then(setDepartments).catch(() => setDepartments([]))
  }, [])

  /** Apply a role control change using a fresh server result. */
  const handleRoleFilterChange = async (value?: string) => {
    const nextRole = value === 'organization_admin' || value === 'admin' || value === 'employee' ? value : undefined
    setRoleFilter(nextRole)
    await fetchUsers({ roleFilter: nextRole })
  }

  /** Open the form for creating a record. */
  const openCreateModal = () => {
    setEditingUser(null)
    form.resetFields()
    setModalOpen(true)
  }

  /** Open the form for editing the selected user. */
  const openEditModal = (user: UserListItem) => {
    setEditingUser(user)
    form.setFieldsValue({
      username: user.username,
      display_name: user.display_name,
      email: user.email,
      role: ['organization_admin', 'admin'].includes(user.role) ? user.role : 'viewer',
      department_id: user.department_id,
    })
    setModalOpen(true)
  }

  /** Close the active form and clear its selection. */
  const closeModal = () => {
    setModalOpen(false)
  }

  /** Create or update the user represented by the form values. */
  const handleSubmit = async (submittedValues?: UserFormValues) => {
    setSubmitting(true)
    try {
      const values = submittedValues ?? await form.validateFields()
      if (editingUser) {
        const payload: Partial<UserCreateRequest> = {
          display_name: values.display_name,
          email: values.email,
          role: values.role,
          department_id: values.role === 'organization_admin' ? null : values.department_id,
          is_department_manager: values.role === 'admin',
        }
        await userApi.update(editingUser.user_id, payload)
        message.success('用户更新成功')
      } else {
        await userApi.create({
          ...values,
          department_id: values.role === 'organization_admin' ? null : values.department_id,
          is_department_manager: values.role === 'admin',
        } as UserCreateRequest)
        message.success('用户创建成功')
      }
      setModalOpen(false)
      await fetchUsers()
    } catch (error) {
      if (error && typeof error === 'object' && 'errorFields' in error) return
      const detail = getApiErrorInfo(error)
      message.error(detail?.message || '保存用户失败，请稍后重试')
    } finally {
      setSubmitting(false)
    }
  }

  /** Delete the selected record after confirmation. */
  const handleDelete = async (user: UserListItem) => {
    try {
      await userApi.delete(user.user_id)
      message.success('用户已删除')
      void fetchUsers()
    } catch {
      message.error('删除失败')
    }
  }

  /** Enable or disable the selected user account. */
  const handleToggleActive = async (user: UserListItem) => {
    try {
      await userApi.toggleActive(user.user_id)
      message.success(`用户已${user.is_active ? '禁用' : '启用'}`)
      void fetchUsers()
    } catch {
      message.error('操作失败')
    }
  }

  /** Reset the selected user password after confirmation. */
  const handleResetPassword = async (user: UserListItem) => {
    Modal.confirm({
      title: '重置密码',
      content: createElement(
        'div',
        { style: { marginTop: 16 } },
        createElement(AntInput.Password, { placeholder: '输入新密码', id: 'new-pwd-input' }),
      ),
      /** Validate and apply the password entered in the confirmation modal. */
      onOk: async () => {
        const input = document.getElementById('new-pwd-input') as HTMLInputElement
        const newPassword = input.value
        if (!newPassword || newPassword.length < 8) {
          message.error('密码至少 8 位')
          return Promise.reject()
        }
        await userApi.resetPassword(user.user_id, newPassword)
        message.success('密码已重置')
      },
    })
  }

  return {
    closeModal,
    currentUser,
    departments,
    editingUser,
    fetchUsers,
    form,
    handleDelete,
    handleResetPassword,
    handleRoleFilterChange,
    handleSubmit,
    handleToggleActive,
    loading,
    modalOpen,
    openCreateModal,
    openEditModal,
    roleFilter,
    submitting,
    search,
    setSearch,
    users,
  }
}
