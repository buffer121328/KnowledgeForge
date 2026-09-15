import { Form, Input, Modal, Select } from 'antd'
import type { FormInstance } from 'antd'
import type { UserFormValues } from '@/pages/useUsers'
import { USER_MANAGEMENT_ROLE_OPTIONS } from '@/utils/userRoleLabels'
import type { UserListItem } from '@/types'
import type { DepartmentItem } from '@/types'
import { departmentOptionLabel } from '@/utils/departmentLabels'

interface UserFormProps {
  editingUser: UserListItem | null
  form: FormInstance<UserFormValues>
  onCancel: () => void
  onSubmit: (values: UserFormValues) => Promise<void>
  open: boolean
  organizationId?: string
  departments?: DepartmentItem[]
  submitting?: boolean
}

/** Render the user form component. */
export function UserForm({
  editingUser,
  form,
  onCancel,
  onSubmit,
  open,
  organizationId,
  departments = [],
  submitting = false,
}: UserFormProps) {
  const selectedRole = Form.useWatch('role', form)
  const needsDepartment = selectedRole !== 'organization_admin'

  return (
    <Modal
      title={editingUser ? '编辑用户' : '新建用户'}
      open={open}
      onOk={() => form.submit()}
      onCancel={onCancel}
      confirmLoading={submitting}
      okText="确定"
      cancelText="取消"
      destroyOnHidden
      width={480}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{ role: 'viewer', is_department_manager: false }}
        onFinish={(values) => { void onSubmit(values) }}
      >
        <Form.Item
          name="username"
          label="用户名"
          rules={[
            { required: true, message: '请输入用户名' },
            { min: 3, message: '至少 3 个字符' },
          ]}
        >
          <Input disabled={Boolean(editingUser)} />
        </Form.Item>
        {!editingUser && (
          <Form.Item
            name="password"
            label="密码"
            rules={[
              { required: true, message: '请输入密码' },
              { min: 8, message: '密码至少 8 位' },
            ]}
          >
            <Input.Password />
          </Form.Item>
        )}
        <Form.Item name="display_name" label="显示名">
          <Input />
        </Form.Item>
        <Form.Item
          name="email"
          label="邮箱"
          rules={[{ type: 'email', message: '邮箱格式不正确' }]}
        >
          <Input />
        </Form.Item>
        <Form.Item name="role" label="角色" rules={[{ required: true }]}>
          <Select options={USER_MANAGEMENT_ROLE_OPTIONS} />
        </Form.Item>
        {needsDepartment && (
          <Form.Item
            name="department_id"
            label="所属部门"
            rules={[{ required: true, message: selectedRole === 'admin' ? '请选择负责人所属部门' : '请选择员工所属部门' }]}
          >
            <Select
              showSearch
              optionFilterProp="label"
              placeholder="选择部门"
              options={departments.map((department) => ({
                label: departmentOptionLabel(department),
                value: department.department_id,
              }))}
            />
          </Form.Item>
        )}
        <Form.Item label="组织 ID（默认）">
          <Input value={editingUser?.org_id || organizationId || '默认组织'} disabled />
        </Form.Item>
      </Form>
    </Modal>
  )
}
