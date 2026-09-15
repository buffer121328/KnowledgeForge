/** 状态码 → 中文标签映射，集中管理避免散落各页面 */

export const taskStatusLabels: Record<string, string> = {
  SUCCESS: '成功',
  FAILURE: '失败',
  PENDING: '排队中',
  STARTED: '运行中',
  RETRY: '重试中',
  REVOKED: '已取消',
  RECEIVED: '已接收',
}

export const docStatusLabels: Record<string, string> = {
  success: '成功',
  processing: '处理中',
  failed: '失败',
  pending: '排队中',
}

export const auditResultLabels: Record<string, string> = {
  success: '成功',
  failure: '失败',
  denied: '拒绝',
}

export const auditActionLabels: Record<string, string> = {
  'doc.upload': '文档上传',
  'doc.update': '文档更新',
  'doc.delete': '文档删除',
  'user.create': '用户创建',
  'user.update': '用户更新',
  'user.delete': '用户删除',
  'role.update': '角色变更',
  'auth.login': '登录',
  'auth.logout': '注销',
}

export const webhookEventLabels: Record<string, string> = {
  'doc.ingested': '文档入库',
  'doc.updated': '文档更新',
  'doc.deleted': '文档删除',
  'qa.completed': '问答完成',
  'qa.feedback': '问答反馈',
  'system.error': '系统错误',
  // 旧别名（兼容历史数据展示）
  'document.created': '文档创建',
  'document.updated': '文档更新',
  'document.deleted': '文档删除',
  'task.completed': '任务完成',
  'task.failed': '任务失败',
  'user.created': '用户创建',
  'user.updated': '用户更新',
  'knowledge.extracted': '知识抽取',
}

/** 知识图谱实体类型 → 中文 */
export const entityTypeLabels: Record<string, string> = {
  Person: '人物',
  Organization: '组织',
  Technology: '技术',
  Product: '产品',
  Concept: '概念',
  Location: '地点',
  Event: '事件',
  Document: '文档',
  Company: '公司',
  Project: '项目',
  Skill: '技能',
  Tool: '工具',
  System: '系统',
  Process: '流程',
  Metric: '指标',
  Role: '角色',
  Department: '部门',
  RelationClaim: '关系主张',
  Date: '日期',
  Time: '时间',
  Money: '金额',
  Law: '法规',
  Policy: '政策',
  Disease: '疾病',
  Drug: '药品',
  Material: '材料',
  Device: '设备',
  Service: '服务',
  Feature: '功能',
  Module: '模块',
  API: '接口',
  Database: '数据库',
  Framework: '开发库',
  Language: '语言',
  Protocol: '协议',
  Standard: '标准',
  Other: '其他',
}

/** Return the localized label for a known status or role value. */
export function labelOf(map: Record<string, string>, value: string | undefined | null): string {
  if (!value) return '-'
  return map[value] ?? value
}
