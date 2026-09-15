# AgentKnowledgeHub Frontend

企业级多 Agent 知识管理系统的 React 前端。

## 技术栈

| 类别 | 技术选型 |
|------|----------|
| 框架 | React 19 + TypeScript |
| UI 组件库 | Ant Design 5 + Ant Design Pro Components |
| 样式 | TailwindCSS 4 |
| 状态管理 | Zustand |
| 路由 | React Router DOM 7 |
| HTTP 客户端 | Axios |
| 图表 | Recharts |
| 构建工具 | Vite 8 |

## 项目结构

```
src/
├── api/                    # API 接口
│   ├── auth.ts             # 认证接口 (登录/刷新/ApiKey)
│   ├── docs.ts             # 文档管理接口
│   ├── qa.ts               # 问答接口
│   ├── graph.ts            # 知识图谱接口
│   ├── admin.ts            # 管理接口 (统计/指标/任务)
│   ├── users.ts            # 用户管理接口
│   └── client.ts           # Axios 封装
├── pages/                  # 页面组件
│   ├── Login.tsx           # 登录页
│   ├── Dashboard.tsx       # 数据统计面板
│   ├── DocList.tsx         # 文档列表与管理
│   ├── QAChat.tsx          # 智能问答聊天
│   ├── GraphView.tsx       # 知识图谱可视化
│   ├── Webhooks.tsx        # Webhook 管理
│   ├── UserList.tsx        # 用户与角色管理
│   └── AuditLogs.tsx       # 审计日志查看
├── stores/                 # Zustand 状态管理
│   ├── auth.ts             # 认证状态
│   └── chat.ts             # 聊天状态
├── components/             # 公共组件
│   └── Layout.tsx          # 布局组件
├── types/                  # TypeScript 类型定义
│   └── index.ts
└── utils/                  # 工具函数
    └── labels.ts
```

## 功能页面

| 页面 | 路由 | 功能描述 |
|------|------|----------|
| 登录 | `/login` | 用户名密码 JWT 登录（API Key 仅作服务端 Header，页面不提供 Key 登录表单） |
| Dashboard | `/` | 向量/图谱统计、请求趋势图表 |
| 文档管理 | `/docs` | 文档列表、上传、删除、分块查看 |
| 智能问答 | `/qa` | 问答交互、上下文记忆、反馈 |
| 知识图谱 | `/graph` | 图谱可视化、实体邻域、子图检索 |
| Webhook | `/webhooks` | Webhook 配置、事件类型、回调日志 |
| 用户管理 | `/users` | 用户 CRUD、角色分配、权限配置 |
| 审计日志 | `/audit` | 操作日志查询、筛选、导出 |

## 快速开始

### 安装依赖

```bash
npm install
```

### 开发模式

```bash
npm run dev
```

访问 [http://localhost:5173](http://localhost:5173)

### 生产构建

```bash
npm run build
```

构建产物输出到 `dist/` 目录。

### 类型检查

```bash
npm run type-check
```

## API 对接

前端默认 `baseURL` 为 `/api/v1`（经 Vite 代理到后端）。可用环境变量覆盖：

```env
VITE_API_BASE=/api/v1
```

### 认证流程

1. 登录获取 Access Token + Refresh Token
2. 请求自动带 `Authorization: Bearer ...`
3. 401 时尝试 refresh；失败则回登录页
4. 服务间调用可另用 Header `X-API-Key`（管理台登录页本身不用 Key）

### 主要 API 端点（相对 `VITE_API_BASE`，默认 `/api/v1`）

| 模块 | 路径 | 方法 |
|------|------|------|
| 认证 | `/auth/login` | POST |
| 认证 | `/auth/refresh` | POST |
| 认证 | `/auth/apikey/create` `/list` `/ {id}` | POST/GET/DELETE |
| 文档 | `/ingest/upload` `/docs` `/docs/{id}/chunks` | POST/GET |
| 问答 | `/qa/ask` | POST |
| 图谱 | `/graph/subgraph` `/graph/types` | GET |
| 统计 | `/admin/stats` | GET |
| Webhook | `/webhooks` `/webhooks/{id}/test` | GET/POST/DELETE |
| 审计 | `/audit/logs` | GET |

## 设计规范

### 组件规范

- 使用 Ant Design 组件库保持 UI 一致性
- 页面级组件放在 `pages/` 目录
- 公共组件放在 `components/` 目录
- 组件内部使用 TailwindCSS 样式

### 状态管理

- 认证状态（用户信息、Token） → `stores/auth.ts`
- 聊天状态（问答历史） → `stores/chat.ts`
- 其他状态使用组件内部 useState

### 类型定义

所有 API 响应类型和业务实体类型定义在 `types/index.ts`。

## 预览

### Dashboard

![Dashboard](https://via.placeholder.com/800x400?text=Dashboard)

展示系统统计信息：文档数量、实体数量、关系数量、请求趋势图表。

### 智能问答

![QA Chat](https://via.placeholder.com/800x400?text=QA+Chat)

对话式问答界面，支持上下文记忆、来源引用展示。

### 知识图谱

![Graph View](https://via.placeholder.com/800x400?text=Graph+View)

基于 D3/SVG 的简易图谱可视化（非 React Flow），支持搜索与类型筛选。

## 许可证

MIT
