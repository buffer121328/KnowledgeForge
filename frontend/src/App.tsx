import { Component, lazy, Suspense, useEffect, type ReactNode } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { Button, Result, Spin } from 'antd'
import { useAuthStore } from '@/stores/auth'
import { getDefaultAuthenticatedPath } from '@/navigation'
import Layout from '@/components/Layout'

const Login = lazy(() => import('@/pages/Login'))
const Dashboard = lazy(() => import('@/pages/Dashboard'))
const DocList = lazy(() => import('@/pages/DocList'))
const QAChat = lazy(() => import('@/pages/QAChat'))
const GraphView = lazy(() => import('@/pages/GraphView'))
const UserList = lazy(() => import('@/pages/UserList'))
const AuditLogs = lazy(() => import('@/pages/AuditLogs'))
const Webhooks = lazy(() => import('@/pages/Webhooks'))
const EvaluationGovernance = lazy(() => import('@/pages/EvaluationGovernance'))
const RoleWorkspace = lazy(() => import('@/pages/RoleWorkspace'))

type RouteLoadBoundaryProps = { children: ReactNode }
type RouteLoadBoundaryState = { failed: boolean }

/** Catch route-chunk failures without exposing transport or module details to users. */
export class RouteLoadErrorBoundary extends Component<RouteLoadBoundaryProps, RouteLoadBoundaryState> {
  state: RouteLoadBoundaryState = { failed: false }

  static getDerivedStateFromError(): RouteLoadBoundaryState {
    return { failed: true }
  }

  private retry = () => {
    window.location.reload()
  }

  render() {
    if (this.state.failed) {
      return (
        <Result
          extra={<Button onClick={this.retry} type="primary">重新加载</Button>}
          status="error"
          subTitle="请检查网络连接后重试。"
          title="页面暂时无法加载"
        />
      )
    }
    return this.props.children
  }
}

/** Keep page transitions responsive while the selected route module is downloaded. */
export function RouteLoadingFallback() {
  return (
    <div aria-label="正在加载页面" role="status" style={{ alignItems: 'center', display: 'flex', gap: 12, justifyContent: 'center', minHeight: 240 }}>
      <Spin />
      <span>正在加载页面...</span>
    </div>
  )
}

/** Reset route-load failures after navigation while keeping guards outside lazy page modules. */
function RoutePage({ children }: { children: ReactNode }) {
  const { pathname } = useLocation()
  return (
    <RouteLoadErrorBoundary key={pathname}>
      <Suspense fallback={<RouteLoadingFallback />}>{children}</Suspense>
    </RouteLoadErrorBoundary>
  )
}

/** Render a route that redirects unauthenticated users to the login page. */
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { token, user } = useAuthStore()
  if (!token) return <Navigate to="/login" replace />
  if (!user) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Spin size="large" />
      </div>
    )
  }
  return <>{children}</>
}

/** Redirect authenticated users to their first accessible application page. */
function DefaultAuthenticatedRoute() {
  const { hasPermission, user } = useAuthStore()
  const path = getDefaultAuthenticatedPath(
    hasPermission,
    user?.role === 'organization_admin',
  )
  return path
    ? <Navigate to={path} replace />
    : <Result status="403" title="暂无可访问页面" subTitle="请联系管理员分配应用权限。" />
}

/** Keep a page route aligned with the permission required by its backend API. */
function PermissionRoute({ children, permission }: { children: React.ReactNode; permission: string }) {
  const { hasPermission } = useAuthStore()
  return hasPermission(permission) ? <>{children}</> : <Navigate to="/" replace />
}

/** Evidence governance is intentionally an organization-administrator-only surface. */
function OrganizationAdminRoute({ children }: { children: React.ReactNode }) {
  const { user } = useAuthStore()
  return user?.role === 'organization_admin' ? <>{children}</> : <Navigate to="/" replace />
}

/** Render the application router and top-level providers. */
export default function App() {
  const { token, fetchUser } = useAuthStore()

  useEffect(() => {
    if (token) {
      void fetchUser()
    }
  }, [token, fetchUser])

  return (
    <Routes>
      <Route path="/login" element={<RoutePage><Login /></RoutePage>} />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<DefaultAuthenticatedRoute />} />
        <Route path="workspace" element={<RoutePage><RoleWorkspace /></RoutePage>} />
        <Route
          path="dashboard"
          element={<PermissionRoute permission="admin:manage"><RoutePage><Dashboard /></RoutePage></PermissionRoute>}
        />
        <Route path="docs" element={<PermissionRoute permission="doc:read"><RoutePage><DocList /></RoutePage></PermissionRoute>} />
        <Route path="qa" element={<PermissionRoute permission="qa:query"><RoutePage><QAChat /></RoutePage></PermissionRoute>} />
        <Route path="graph" element={<PermissionRoute permission="graph:read"><RoutePage><GraphView /></RoutePage></PermissionRoute>} />
        <Route path="users" element={<PermissionRoute permission="admin:user"><RoutePage><UserList /></RoutePage></PermissionRoute>} />
        <Route path="audit" element={<PermissionRoute permission="admin:audit"><RoutePage><AuditLogs /></RoutePage></PermissionRoute>} />
        <Route path="webhooks" element={<PermissionRoute permission="admin:manage"><RoutePage><Webhooks /></RoutePage></PermissionRoute>} />
        <Route path="evaluation" element={<OrganizationAdminRoute><RoutePage><EvaluationGovernance /></RoutePage></OrganizationAdminRoute>} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
