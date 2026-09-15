import type { GraphNode } from '@/api/graph'
import { entityTypeLabels, labelOf } from '@/utils/labels'

const RELATION_LABELS: Record<string, string> = {
  APPROVES: '审批',
  APPROVED_BY: '由其审批',
  AFFILIATED_WITH: '隶属或关联于',
  APPLIES_TO: '适用于',
  ASSIGNED_TO: '分配给',
  ASSOCIATED_WITH: '关联于',
  AUDITS: '审计',
  BASED_IN: '设于',
  BELONGS_TO: '属于',
  COLLABORATES_WITH: '协作',
  CONTAINS: '包含',
  COMPLIES_WITH: '遵循',
  CONSISTS_OF: '由…组成',
  CONTRIBUTES_TO: '为…做出贡献',
  COORDINATES_WITH: '协调',
  CROSS_DEPARTMENT: '跨部门多跳关联',
  DELIVERS_TO: '交付给',
  DEPENDS_ON: '依赖',
  DEPARTMENT_OF: '隶属于',
  DESCRIBES: '描述',
  DEVELOPED_BY: '由…开发',
  DIRECTOR_OF: '负责',
  DIVISION_OF: '隶属于',
  EMPLOYS: '雇佣',
  EMPLOYED_BY: '受雇于',
  ENABLES: '支持或使能',
  FROM: '来源于',
  FOUNDED_BY: '由…创立',
  FOUNDER_OF: '创立',
  GOVERNED_BY: '受…治理',
  HAS_DEPARTMENT: '包含部门',
  HAS_ENTITY: '包含实体',
  HAS_DOCUMENT: '包含文档',
  HAS_MEMBER: '拥有成员',
  HAS_PART: '包含组成部分',
  HAS_PROCESS: '包含流程',
  HAS_ROLE: '具有角色',
  HEAD_OF: '负责',
  HEADS: '负责',
  IMPLEMENTS: '实现',
  INCLUDES: '包括',
  INVOLVED_IN: '参与',
  IS_PART_OF: '隶属于',
  LEADS: '领导',
  LED_BY: '由其领导',
  LOCATED_AT: '位于',
  LOCATED_IN: '位于',
  MAINTAINED_BY: '由…维护',
  MANAGES: '管理',
  MEMBER_OF: '是…成员',
  MENTIONS: '提及',
  NOTIFIES: '通知',
  OWNS: '拥有',
  OWNED_BY: '归…所有',
  OVERSEES: '监督',
  OPERATES_IN: '运营于',
  PARTICIPATES_IN: '参与',
  PART_OF: '隶属于',
  PRODUCES: '产出',
  PROVIDED_BY: '由…提供',
  PROVIDES_TO: '提供给',
  RECEIVES_FROM: '接收自',
  RELATED_TO: '相关',
  REFERENCES: '引用',
  REPORTS_TO: '汇报给',
  REQUIRES: '要求',
  RESPONSIBLE_FOR: '负责',
  REVIEWS: '复核',
  SHARES_WITH: '共享给',
  SERVES_AS: '担任',
  SUBSIDIARY_OF: '是…的子公司',
  SUPPORTS: '支持',
  SUPPORTED_BY: '由…支持',
  SUPERVISES: '监督',
  TARGETS: '面向',
  TEAM_OF: '隶属于',
  UNIT_OF: '隶属于',
  USES: '使用',
  USED_BY: '被…使用',
  WORK_AT: '任职于',
  WORK_FOR: '任职于',
  WORK_IN: '任职于',
  WORK_ON: '参与',
  WORKS_AT: '任职于',
  WORKS_FOR: '任职于',
  WORKS_IN: '任职于',
  WORKS_ON: '参与',
  WORKS_WITH: '协同工作',
}

const RELATION_ALIASES: Record<string, string> = {
  A_PART_OF: 'PART_OF',
  COMPOSED_OF: 'CONSISTS_OF',
  EMPLOYED_AT: 'WORK_AT',
  IS_A_PART_OF: 'PART_OF',
  IS_MEMBER_OF: 'MEMBER_OF',
  PARTNERED_WITH: 'COLLABORATES_WITH',
  PARTNERS_WITH: 'COLLABORATES_WITH',
  WORKING_AT: 'WORK_AT',
  WORKING_FOR: 'WORK_FOR',
}

/** Return a human-friendly relation label while preserving unknown extracted labels. */
export function formatGraphRelation(label: string): string {
  const normalized = label
    .trim()
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[^\p{L}\p{N}]+/gu, '_')
    .replace(/^_+|_+$/g, '')
    .toUpperCase()
  const canonical = RELATION_ALIASES[normalized] ?? normalized
  return RELATION_LABELS[canonical] ?? label.replaceAll('_', ' ').toLowerCase()
}

/** Display graph IDs without leaking storage prefixes such as entity:. */
export function formatGraphNodeId(id: string): string {
  const [prefix, ...rest] = id.split(':')
  const value = rest.join(':')
  if (!value) return id
  if (prefix === 'entity') return value
  if (prefix === 'department') return `${departmentDisplayName(value)}（部门）`
  if (prefix === 'document') return `${value}（文档）`
  if (prefix === 'relationclaim') return `${value}（关系声明）`
  if (prefix === 'company') return `${value}（公司）`
  return id
}

/** Format a node reference for tables and path details. */
export function formatGraphNodeRef(node: GraphNode | undefined, fallbackId: string): string {
  if (!node) return formatGraphNodeId(fallbackId)
  return `${node.label}（${labelOf(entityTypeLabels, node.type)}）`
}
import { departmentDisplayName } from './departmentLabels'
