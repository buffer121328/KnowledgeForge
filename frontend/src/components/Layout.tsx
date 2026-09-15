import { useEffect, useMemo, useState } from 'react'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import {
  Avatar,
  Breadcrumb,
  Button,
  Drawer,
  Dropdown,
  Grid,
  Layout as AntLayout,
  Menu,
  Space,
  Typography,
  type MenuProps,
} from 'antd'
import {
  ApartmentOutlined,
  ApiOutlined,
  AuditOutlined,
  DashboardOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  MessageOutlined,
  TeamOutlined,
  UserOutlined,
  HomeOutlined,
} from '@ant-design/icons'
import { useAuthStore } from '@/stores/auth'
import { getVisibleNavigation } from '@/navigation'
import { userRoleLabel } from '@/utils/userRoleLabels'
import { departmentDisplayName } from '@/utils/departmentLabels'

const { Header, Sider, Content } = AntLayout
const { Text } = Typography

const navigationIcons: Record<string, React.ReactNode> = {
  '/workspace': <HomeOutlined />,
  '/docs': <FileTextOutlined />,
  '/qa': <MessageOutlined />,
  '/graph': <ApartmentOutlined />,
  '/evaluation': <ExperimentOutlined />,
  '/users': <TeamOutlined />,
  '/audit': <AuditOutlined />,
  '/dashboard': <DashboardOutlined />,
  '/webhooks': <ApiOutlined />,
}

const groupLabels = { knowledge: '知识应用', governance: '治理与合规', operations: '平台运维' }

/** Render the responsive enterprise administration shell. */
export default function Layout() {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const screens = Grid.useBreakpoint()
  const { user, logout, hasPermission } = useAuthStore()
  const mobile = !screens.lg

  useEffect(() => {
    if (!user) navigate('/login', { replace: true })
  }, [user, navigate])

  const visibleNavigation = useMemo(
    () => getVisibleNavigation(hasPermission, user?.role === 'organization_admin'),
    [hasPermission, user?.role],
  )
  const menuItems = useMemo<MenuProps['items']>(() =>
    (Object.keys(groupLabels) as Array<keyof typeof groupLabels>).map((group) => ({
      type: 'group',
      label: groupLabels[group],
      key: group,
      children: visibleNavigation.filter((item) => item.group === group).map((item) => ({
        key: item.key,
        icon: navigationIcons[item.key],
        label: item.label,
      })),
    })), [visibleNavigation])

  const activeItem = visibleNavigation.find((item) => location.pathname.startsWith(item.key))
  const selectedKey = activeItem?.key ?? visibleNavigation[0]?.key

  const handleMenu: MenuProps['onClick'] = ({ key }) => {
    navigate(key)
    setMobileOpen(false)
  }

  const handleLogout = async () => {
    await logout()
    navigate('/login', { replace: true })
  }

  const navigationMenu = (
    <Menu
      className="enterprise-menu"
      theme="dark"
      mode="inline"
      selectedKeys={selectedKey ? [selectedKey] : []}
      items={menuItems}
      onClick={handleMenu}
      inlineCollapsed={!mobile && collapsed}
    />
  )

  return (
    <AntLayout className="enterprise-shell">
      {!mobile && (
        <Sider
          width={248}
          collapsedWidth={80}
          collapsed={collapsed}
          trigger={null}
          theme="dark"
          className="enterprise-sider"
        >
          <div className="enterprise-brand">
            <div className="enterprise-brand-mark">KF</div>
            {!collapsed && <div><strong>KnowledgeForge</strong><span>Enterprise AI</span></div>}
          </div>
          <div className="enterprise-menu-scroll">{navigationMenu}</div>
          <div className="enterprise-sider-footer">
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed((value) => !value)}
            >{collapsed ? null : '收起导航'}</Button>
          </div>
        </Sider>
      )}

      <Drawer
        className="mobile-navigation-drawer"
        placement="left"
        width={280}
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        styles={{ body: { padding: 0, background: '#101828' }, header: { background: '#101828', borderBottomColor: '#344054' } }}
        title={<div className="enterprise-brand"><div className="enterprise-brand-mark">KF</div><strong>KnowledgeForge</strong></div>}
      >{navigationMenu}</Drawer>

      <AntLayout className="enterprise-main">
        <Header className="enterprise-header">
          <Space size={14}>
            {mobile && <Button type="text" icon={<MenuUnfoldOutlined />} onClick={() => setMobileOpen(true)} />}
            <div>
              <Breadcrumb items={[{ title: '管理控制台' }, { title: activeItem?.label ?? '首页' }]} />
              <Text strong className="enterprise-header-title">{activeItem?.label ?? '管理控制台'}</Text>
            </div>
          </Space>
          <Dropdown
            placement="bottomRight"
            menu={{ items: [{ key: 'logout', icon: <LogoutOutlined />, label: '退出登录', onClick: handleLogout }] }}
          >
            <Space className="enterprise-user-menu">
              <Avatar icon={<UserOutlined />} />
              {!mobile && <div className="enterprise-user-copy"><Text strong>{user?.display_name || user?.username || '未登录'}</Text><Text type="secondary">{userRoleLabel(user?.role)}{user?.department_id ? ` · ${departmentDisplayName(user.department_id)}` : ''} · {user?.org_id}</Text></div>}
            </Space>
          </Dropdown>
        </Header>
        <Content className="enterprise-content">
          <div className="enterprise-content-inner"><Outlet /></div>
        </Content>
      </AntLayout>
    </AntLayout>
  )
}
