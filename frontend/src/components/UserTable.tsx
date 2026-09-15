import { DeleteOutlined, EditOutlined, KeyOutlined } from '@ant-design/icons'
import { Button, Popconfirm, Space, Switch, Table, Tag } from 'antd'
import type { TableColumnsType } from 'antd'
import type { DepartmentItem, UserListItem } from '@/types'
import { userRoleLabel } from '@/utils/userRoleLabels'
import { departmentDisplayName } from '@/utils/departmentLabels'

interface UserTableProps {
  currentUserId?: string
  departments?: DepartmentItem[]
  loading: boolean
  onDelete: (user: UserListItem) => Promise<void>
  onEdit: (user: UserListItem) => void
  onResetPassword: (user: UserListItem) => void
  onToggleActive: (user: UserListItem) => Promise<void>
  users: UserListItem[]
}

/** Render the user table component. */
export function UserTable({
  currentUserId,
  departments = [],
  loading,
  onDelete,
  onEdit,
  onResetPassword,
  onToggleActive,
  users,
}: UserTableProps) {
  const departmentNames = new Map(
    departments.map((department) => [department.department_id, department.name]),
  )
  const columns: TableColumnsType<UserListItem> = [
    { title: '用户名', dataIndex: 'username', width: 120 },
    { title: '显示名', dataIndex: 'display_name', width: 140 },
    { title: '邮箱', dataIndex: 'email', ellipsis: true },
    {
      title: '角色',
      dataIndex: 'role',
      width: 120,
      /** Render the system permission role with the product-facing name. */
      render: (role: string) => (
        <Tag color={role === 'organization_admin' ? 'purple' : role === 'admin' ? 'gold' : 'blue'}>{userRoleLabel(role)}</Tag>
      ),
    },
    {
      title: '部门',
      dataIndex: 'department_id',
      width: 120,
      render: (value: string | null | undefined, record) => (
        record.role === 'organization_admin'
          ? '全组织'
          : departmentDisplayName(value, value ? departmentNames.get(value) : undefined) || '未分配'
      ),
    },
    { title: '组织', dataIndex: 'org_id', width: 120 },
    {
      title: '状态',
      dataIndex: 'is_active',
      width: 80,
      /** Render the table cell for this column. */
      render: (active: boolean, record: UserListItem) => (
        <Switch
          checked={active}
          size="small"
          onChange={() => onToggleActive(record)}
          disabled={record.user_id === currentUserId}
        />
      ),
    },
    {
      title: '操作',
      width: 200,
      /** Render the table cell for this column. */
      render: (_: unknown, record: UserListItem) => (
        <Space size={4}>
          <Button size="small" icon={<EditOutlined />} onClick={() => onEdit(record)}>
            编辑
          </Button>
          <Button size="small" icon={<KeyOutlined />} onClick={() => onResetPassword(record)}>
            重置密码
          </Button>
          <Popconfirm
            title="确定删除该用户？"
            onConfirm={() => onDelete(record)}
            disabled={record.user_id === currentUserId}
          >
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <Table
      columns={columns}
      dataSource={users}
      rowKey="user_id"
      loading={loading}
      pagination={{ pageSize: 10, showSizeChanger: true }}
      size="middle"
    />
  )
}
