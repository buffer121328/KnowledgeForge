import { PlusOutlined, ReloadOutlined } from '@ant-design/icons'
import { Button, Card, Input, Select, Space } from 'antd'
import { UserForm } from '@/components/UserForm'
import { UserTable } from '@/components/UserTable'
import { USER_ROLE_FILTER_OPTIONS } from '@/utils/userRoleLabels'
import { useUsers } from './useUsers'

/** Render the user list page. */
export default function UserList() {
  const {
    closeModal,
    currentUser,
    departments,
    submitting,
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
    search,
    setSearch,
    users,
  } = useUsers()

  return (
    <Card
      title="用户管理"
      extra={
        <Space>
          <Input
            placeholder="搜索用户名"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            onPressEnter={() => void fetchUsers()}
            style={{ width: 200 }}
          />
          <Select
            allowClear
            placeholder="角色筛选"
            value={roleFilter}
            onChange={(value) => void handleRoleFilterChange(value)}
            style={{ width: 140 }}
            options={USER_ROLE_FILTER_OPTIONS}
          />
          <Button icon={<ReloadOutlined />} onClick={() => void fetchUsers()}>
            刷新
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={openCreateModal}>
            新建用户
          </Button>
        </Space>
      }
    >
      <UserTable
        currentUserId={currentUser?.user_id}
        departments={departments}
        loading={loading}
        onDelete={handleDelete}
        onEdit={openEditModal}
        onResetPassword={handleResetPassword}
        onToggleActive={handleToggleActive}
        users={users}
      />
      <UserForm
        editingUser={editingUser}
        form={form}
        onCancel={closeModal}
        onSubmit={handleSubmit}
        open={modalOpen}
        organizationId={currentUser?.org_id}
        departments={departments}
        submitting={submitting}
      />
    </Card>
  )
}
