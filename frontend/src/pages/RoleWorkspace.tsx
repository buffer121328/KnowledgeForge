import {
  ApartmentOutlined,
  AuditOutlined,
  FileSearchOutlined,
  FileTextOutlined,
  SafetyCertificateOutlined,
  TeamOutlined,
} from '@ant-design/icons'
import { Button, Card, Col, Row, Space, Tag, Typography } from 'antd'
import { useNavigate } from 'react-router-dom'
import { workspaceKindFor } from '@/rolePolicy'
import { useAuthStore } from '@/stores/auth'
import { departmentDisplayName } from '@/utils/departmentLabels'

const { Paragraph, Text, Title } = Typography

interface WorkspaceAction {
  description: string
  icon: React.ReactNode
  label: string
  path: string
}

const employeeActions: WorkspaceAction[] = [
  { label: '查阅知识文档', description: '浏览组织授权给你的文档与原文分块。', icon: <FileTextOutlined />, path: '/docs' },
  { label: '向知识库提问', description: '基于企业知识获取带证据的回答。', icon: <FileSearchOutlined />, path: '/qa' },
  { label: '探索知识关联', description: '从图谱查看实体、关系和来源证据。', icon: <ApartmentOutlined />, path: '/graph' },
]

const managerActions: WorkspaceAction[] = [
  { label: '维护本部门文档', description: '上传、更新并治理本部门知识资产。', icon: <FileTextOutlined />, path: '/docs' },
  { label: '核查部门知识', description: '用智能问答检验知识覆盖和可用性。', icon: <FileSearchOutlined />, path: '/qa' },
  { label: '校正知识图谱', description: '查看并校正部门范围内的实体与关系。', icon: <ApartmentOutlined />, path: '/graph' },
]

const governanceActions: WorkspaceAction[] = [
  { label: '评测治理', description: '维护评测数据集并查看质量运行结果。', icon: <SafetyCertificateOutlined />, path: '/evaluation' },
  { label: '用户与权限', description: '管理组织成员、角色和访问边界。', icon: <TeamOutlined />, path: '/users' },
  { label: '审计与监控', description: '追踪治理操作和系统 API 运行状态。', icon: <AuditOutlined />, path: '/dashboard' },
]

/** Render a useful landing workspace tailored to the authenticated business role. */
export default function RoleWorkspace() {
  const navigate = useNavigate()
  const user = useAuthStore((state) => state.user)
  const kind = workspaceKindFor(user)
  const isEmployee = kind === 'employee'
  const isManager = kind === 'department_manager'
  const actions = isEmployee ? employeeActions : isManager ? managerActions : governanceActions
  const title = isEmployee ? '我的知识工作台' : isManager ? '部门知识运营台' : '组织治理中心'
  const subtitle = isEmployee
    ? '安全查阅、提问和探索已授权的企业知识。你的账号为只读模式。'
    : isManager
      ? `聚焦${user?.department_id ? ` ${departmentDisplayName(user.department_id)} ` : '本部门'}知识的入库、质量核查与持续维护。`
      : '统一管理组织评测、身份权限、审计、集成与系统运行状态。'

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      <Card className={`role-workspace-hero role-workspace-${kind}`}>
        <Tag color={isEmployee ? 'blue' : isManager ? 'gold' : 'purple'}>
          {isEmployee ? '员工 · 只读' : isManager ? '部门负责人' : '公司管理员'}
        </Tag>
        <Title level={2}>{title}</Title>
        <Paragraph>{subtitle}</Paragraph>
        <Text type="secondary">{user?.display_name || user?.username} · {user?.org_id}</Text>
      </Card>
      <Row gutter={[16, 16]}>
        {actions.map((action) => (
          <Col xs={24} md={8} key={action.path}>
            <Card className="role-workspace-action" hoverable onClick={() => navigate(action.path)}>
              <div className="role-workspace-action-icon">{action.icon}</div>
              <Title level={4}>{action.label}</Title>
              <Paragraph type="secondary">{action.description}</Paragraph>
              <Button type="link" style={{ padding: 0 }}>进入</Button>
            </Card>
          </Col>
        ))}
      </Row>
      {isEmployee && (
        <Card size="small" title="权限说明">
          你可以查阅文档、使用智能问答和浏览知识图谱；上传、删除、编辑及组织治理功能不会显示，也无法通过 API 执行。
        </Card>
      )}
    </Space>
  )
}
