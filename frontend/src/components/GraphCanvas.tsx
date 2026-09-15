import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { Button, Card, Empty, Input, Modal, Space, Spin, Tag, Typography } from 'antd'
import type { GraphEdge, GraphNode } from '@/api/graph'
import {
  colorOf,
  NODE_SHAPE_DETAILS,
  shapeOf,
} from '@/utils/graphVisualization'
import { formatGraphRelation } from '@/utils/graphLabels'
import { entityTypeLabels, labelOf } from '@/utils/labels'

const { Paragraph, Text } = Typography

const KnowledgeGraph3D = lazy(async () => {
  const module = await import('./KnowledgeGraph3D')
  return { default: module.KnowledgeGraph3D }
})

interface GraphCanvasProps {
  colorMap: Record<string, string>
  edges: GraphEdge[]
  entityTypes: string[]
  loading: boolean
  nodes: GraphNode[]
  onDepartmentOpen?: (node: GraphNode) => void
  onEntityUpdate?: (node: GraphNode, values: { label?: string; type?: string; description?: string }) => Promise<void>
  onNodeSelect: (node: GraphNode) => void
  selected: GraphNode | null
  status: string
}

interface GraphLegendItemProps {
  color: string
  type: string
}

interface SelectedRelationSummary {
  key: string
  text: string
}

/** Build a stable lookup for relationship detail rendering. */
function buildNodeLookup(nodes: GraphNode[]): Map<string, GraphNode> {
  return new Map(nodes.map((node) => [node.id, node] as const))
}

/** Format the neighbor label and type used in the selected-entity relation summary. */
function formatNeighbor(node: GraphNode | undefined, fallbackId: string): string {
  if (!node) return fallbackId
  return `${node.label}（${labelOf(entityTypeLabels, node.type)}）`
}

/** Build concise relationship rows connected to the selected node. */
function selectedRelationSummaries(
  selected: GraphNode | null,
  nodes: GraphNode[],
  edges: GraphEdge[],
): SelectedRelationSummary[] {
  if (!selected) return []

  const nodeLookup = buildNodeLookup(nodes)
  return edges
    .filter((edge) => edge.source === selected.id || edge.target === selected.id)
    .slice(0, 8)
    .map((edge, index) => {
      if (edge.source === selected.id) {
        return {
          key: `${edge.source}-${edge.label}-${edge.target}-${index}`,
          text: `${selected.label} —${formatGraphRelation(edge.label)}→ ${formatNeighbor(
            nodeLookup.get(edge.target),
            edge.target,
          )}`,
        }
      }
      return {
        key: `${edge.source}-${edge.label}-${edge.target}-${index}`,
        text: `${formatNeighbor(nodeLookup.get(edge.source), edge.source)} —${formatGraphRelation(edge.label)}→ ${selected.label}`,
      }
    })
}

/** Render one legend item with redundant color, glyph, and shape-name semantics. */
function GraphLegendItem({ color, type }: GraphLegendItemProps) {
  const shape = shapeOf(type)
  const details = NODE_SHAPE_DETAILS[shape]

  return (
    <div
      data-testid={`graph-legend-${type}`}
      style={{
        alignItems: 'center',
        background: 'linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,250,252,0.9))',
        border: '1px solid rgba(148, 163, 184, 0.22)',
        borderRadius: 12,
        boxShadow: '0 6px 18px rgba(15, 23, 42, 0.05)',
        display: 'flex',
        gap: 10,
        minWidth: 158,
        padding: '9px 12px',
      }}
    >
      <span
        aria-hidden
        style={{
          alignItems: 'center',
          background: `${color}18`,
          border: `1px solid ${color}44`,
          borderRadius: 10,
          color,
          display: 'inline-flex',
          fontSize: 22,
          height: 36,
          justifyContent: 'center',
          lineHeight: 1,
          textShadow: `0 4px 12px ${color}55`,
          width: 36,
        }}
      >
        {details.glyph}
      </span>
      <span style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Text strong ellipsis={{ tooltip: labelOf(entityTypeLabels, type) }}>
          {labelOf(entityTypeLabels, type)}
        </Text>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {details.label}
        </Text>
      </span>
    </div>
  )
}

/** Render the graph canvas, visual legend, and selected-entity context. */
export function GraphCanvas({
  colorMap,
  edges,
  entityTypes,
  loading,
  nodes,
  onDepartmentOpen,
  onEntityUpdate,
  onNodeSelect,
  selected,
  status,
}: GraphCanvasProps) {
  const isCompanyOverview = nodes.some((node) => node.type === 'Company')
    && nodes.every((node) => node.type === 'Company' || node.type === 'Department')
  const displayEdges = edges
  const relationSummaries = useMemo(
    () => selectedRelationSummaries(selected, nodes, displayEdges),
    [displayEdges, nodes, selected],
  )
  const selectedRelationCount = useMemo(
    () =>
      selected
        ? displayEdges.filter((edge) => edge.source === selected.id || edge.target === selected.id).length
        : 0,
    [displayEdges, selected],
  )
  const [entityEditorOpen, setEntityEditorOpen] = useState(false)
  const [entitySaving, setEntitySaving] = useState(false)
  const [entityDraft, setEntityDraft] = useState({ description: '', label: '', type: '' })

  useEffect(() => {
    if (!selected) return
    setEntityDraft({
      description: selected.description ?? '',
      label: selected.label,
      type: selected.type,
    })
  }, [selected])

  const canEditSelectedEntity = Boolean(selected?.id.startsWith('entity:') && onEntityUpdate)
  const saveEntityDraft = async () => {
    if (!selected || !onEntityUpdate) return
    setEntitySaving(true)
    try {
      await onEntityUpdate(selected, entityDraft)
      setEntityEditorOpen(false)
    } finally {
      setEntitySaving(false)
    }
  }

  return (
    <>
      <div
        style={{
          background:
            'radial-gradient(circle at 14% 18%, rgba(14, 165, 233, 0.26), transparent 34%), radial-gradient(circle at 86% 14%, rgba(99, 102, 241, 0.23), transparent 31%), radial-gradient(circle at 50% 92%, rgba(20, 184, 166, 0.13), transparent 38%), linear-gradient(145deg, #020617 0%, #06172d 48%, #0b2544 100%)',
          border: '1px solid rgba(125, 211, 252, 0.26)',
          borderRadius: 18,
          boxShadow: '0 24px 70px rgba(2, 8, 23, 0.34), inset 0 1px 0 rgba(186, 230, 253, 0.12)',
          height: 'clamp(460px, 68vh, 620px)',
          minHeight: 460,
          overflow: 'hidden',
          position: 'relative',
          width: '100%',
        }}
      >
        <div
          aria-hidden
          style={{
            backgroundImage:
              'linear-gradient(rgba(125,211,252,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(125,211,252,0.07) 1px, transparent 1px)',
            backgroundSize: '42px 42px',
            inset: 0,
            maskImage: 'radial-gradient(ellipse at center, rgba(0,0,0,0.88), transparent 88%)',
            pointerEvents: 'none',
            position: 'absolute',
          }}
        />

        <div
          aria-hidden
          style={{
            background: 'radial-gradient(circle, rgba(56, 189, 248, 0.16), transparent 68%)',
            filter: 'blur(18px)',
            height: '72%',
            left: '14%',
            pointerEvents: 'none',
            position: 'absolute',
            top: '12%',
            width: '72%',
          }}
        />

        {nodes.length > 0 && (
          <div style={{ inset: 0, position: 'absolute', zIndex: 2 }}>
            <Suspense
              fallback={
                <div
                  style={{
                    left: '50%',
                    position: 'absolute',
                    top: '50%',
                    transform: 'translate(-50%, -50%)',
                  }}
                >
                  <Spin size="small" />
                </div>
              }
            >
              <KnowledgeGraph3D
                colorMap={colorMap}
                edges={displayEdges}
                nodes={nodes}
                onDepartmentOpen={onDepartmentOpen}
                onNodeSelect={onNodeSelect}
                selected={selected}
              />
            </Suspense>
          </div>
        )}

        {nodes.length > 0 && !loading && (
          <div
            style={{
              backdropFilter: 'blur(14px)',
              background: 'rgba(3, 12, 28, 0.72)',
              border: '1px solid rgba(125, 211, 252, 0.22)',
              borderRadius: 12,
              boxShadow: '0 12px 34px rgba(2, 8, 23, 0.28)',
              left: 14,
              padding: '9px 11px',
              pointerEvents: 'none',
              position: 'absolute',
              top: 14,
              zIndex: 4,
            }}
          >
            <Text strong style={{ color: '#e2e8f0', display: 'block', fontSize: 12, marginBottom: 5 }}>
              多维视觉编码
            </Text>
            <Space size={5} wrap>
              <Tag bordered={false} color="blue">
                形状：实体类型
              </Tag>
              <Tag bordered={false} color="purple">
                大小：可见关系数
              </Tag>
              <Tag bordered={false} color="cyan">
                流光：关系方向
              </Tag>
              <Tag bordered={false} color="processing">
                慢速环绕：已启用
              </Tag>
              {isCompanyOverview && (
                <Tag bordered={false} color="geekblue">
                  层级：公司 → 部门
                </Tag>
              )}
            </Space>
          </div>
        )}

        {loading && (
          <div
            style={{
              backdropFilter: 'blur(10px)',
              background: 'rgba(255, 255, 255, 0.78)',
              border: '1px solid rgba(148, 163, 184, 0.22)',
              borderRadius: 14,
              boxShadow: '0 14px 36px rgba(15, 23, 42, 0.12)',
              left: '50%',
              padding: '18px 24px',
              position: 'absolute',
              top: '50%',
              transform: 'translate(-50%, -50%)',
              zIndex: 10,
            }}
          >
            <Space direction="vertical" align="center" size={8}>
              <Spin size="large" />
              <Text type="secondary">查询中...</Text>
            </Space>
          </div>
        )}

        {nodes.length === 0 && !loading && (
          <div
            style={{
              backdropFilter: 'blur(10px)',
              background: 'rgba(255, 255, 255, 0.72)',
              border: '1px solid rgba(148, 163, 184, 0.2)',
              borderRadius: 16,
              left: '50%',
              padding: '20px 28px',
              position: 'absolute',
              top: '50%',
              transform: 'translate(-50%, -50%)',
            }}
          >
            <Empty
              description={
                status === 'disconnected'
                  ? 'Neo4j 未连接，无法展示图谱'
                  : status === 'empty'
                    ? '文档管理中暂无已入库文档，知识图谱已清空'
                    : status === 'permission-denied'
                    ? '当前账号无权查看该图谱范围'
                    : status === 'timeout'
                      ? '图谱查询超时，请缩小范围后重试'
                      : '当前公司/部门暂无图谱数据，请先入库文档'
              }
            />
          </div>
        )}
      </div>

      {selected && (
        <Card
          type="inner"
          title={
            <Space size={10}>
              <span
                aria-hidden
                style={{
                  color: colorOf(selected.type, colorMap),
                  fontSize: 19,
                  textShadow: `0 4px 12px ${colorOf(selected.type, colorMap)}55`,
                }}
              >
                {NODE_SHAPE_DETAILS[shapeOf(selected.type)].glyph}
              </span>
              实体详情
            </Space>
          }
          extra={<Tag>{selectedRelationCount} 条直接关系</Tag>}
          style={{
            background: 'linear-gradient(135deg, #ffffff, #f8fbff)',
            borderColor: 'rgba(148, 163, 184, 0.24)',
            borderRadius: 14,
            boxShadow: '0 10px 28px rgba(15, 23, 42, 0.06)',
            marginTop: 16,
          }}
        >
          <Space direction="vertical" size={8} style={{ width: '100%' }}>
            <Text>
              <Text type="secondary">ID：</Text>
              {selected.id}
            </Text>
            <Text>
              <Text type="secondary">名称：</Text>
              {selected.label}
            </Text>
            <Text>
              <Text type="secondary">类型：</Text>
              <Tag color={colorOf(selected.type, colorMap)}>
                {labelOf(entityTypeLabels, selected.type)}
              </Tag>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {NODE_SHAPE_DETAILS[shapeOf(selected.type)].label}
              </Text>
            </Text>
            <Paragraph style={{ marginBottom: 0 }}>
              <Text type="secondary">描述：</Text>
              {selected.description || '无'}
            </Paragraph>
            {canEditSelectedEntity && (
              <Button onClick={() => setEntityEditorOpen(true)}>修改实体</Button>
            )}
            {selected.type === 'Department' && onDepartmentOpen && (
              <Button type="primary" onClick={() => onDepartmentOpen(selected)}>
                查看部门内部网络
              </Button>
            )}
            <div style={{ width: '100%' }}>
              <Text type="secondary">关联关系：</Text>
              {relationSummaries.length === 0 ? (
                <Text>当前可见子图中暂无直接关系</Text>
              ) : (
                <div
                  style={{
                    display: 'grid',
                    gap: 7,
                    marginTop: 8,
                  }}
                >
                  {relationSummaries.map((relation) => (
                    <div
                      key={relation.key}
                      style={{
                        alignItems: 'center',
                        background: 'rgba(241, 245, 249, 0.72)',
                        border: '1px solid rgba(148, 163, 184, 0.18)',
                        borderRadius: 9,
                        display: 'flex',
                        gap: 8,
                        padding: '7px 10px',
                      }}
                    >
                      <span aria-hidden style={{ color: '#1677ff', fontWeight: 700 }}>
                        ↗
                      </span>
                      <Text>{relation.text}</Text>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Space>
        </Card>
      )}

      <Card
        type="inner"
        title="图例与视觉编码"
        style={{
          borderColor: 'rgba(148, 163, 184, 0.24)',
          borderRadius: 14,
          boxShadow: '0 10px 28px rgba(15, 23, 42, 0.05)',
          marginTop: 16,
        }}
      >
        <div
          style={{
            display: 'grid',
            gap: 10,
            gridTemplateColumns: 'repeat(auto-fit, minmax(158px, 1fr))',
          }}
        >
          {entityTypes.map((type) => (
            <GraphLegendItem color={colorOf(type, colorMap)} key={type} type={type} />
          ))}
        </div>
        <div
          style={{
            background: '#f8fafc',
            border: '1px solid rgba(148, 163, 184, 0.16)',
            borderRadius: 10,
            marginTop: 12,
            padding: '9px 11px',
          }}
        >
          <Space size={[6, 6]} wrap>
            <Text style={{ fontSize: 12 }}>形状：实体类型</Text>
            <Text type="secondary">•</Text>
            <Text style={{ fontSize: 12 }}>大小：可见关系数</Text>
            <Text type="secondary">•</Text>
            <Text style={{ fontSize: 12 }}>流光：聚焦关系方向</Text>
            <Text type="secondary">•</Text>
            <Text style={{ fontSize: 12 }}>光环：当前焦点</Text>
            {isCompanyOverview && <>
              <Text type="secondary">•</Text>
              <Text style={{ fontSize: 12 }}>公司节点位于顶部</Text>
              <Text type="secondary">•</Text>
              <Text style={{ fontSize: 12 }}>实线：公司与部门组织关系</Text>
              <Text type="secondary">•</Text>
              <Text style={{ fontSize: 12 }}>细虚线：部门间接关系</Text>
            </>}
          </Space>
        </div>
        <Paragraph type="secondary" style={{ fontSize: 12, marginBottom: 0, marginTop: 9 }}>
          数据通过后端图谱接口加载。公司总览将公司节点置于顶部，部门沿层级向下排布；实线表示公司与部门之间的组织关系，细虚线表示由证据汇总出的部门间接/多跳关联，每条路径单独显示。
          公司总览与部门内部图谱均启用慢速自动环绕；可拖拽旋转、滚轮缩放、右键平移，交互后仍保留自动环绕。悬停节点查看信息，点击节点固定焦点并查看详情。
          后端仍独立执行鉴权、租户隔离与子图裁剪。
        </Paragraph>
      </Card>

      <Modal
        confirmLoading={entitySaving}
        okText="保存"
        onCancel={() => setEntityEditorOpen(false)}
        onOk={() => { void saveEntityDraft() }}
        open={entityEditorOpen}
        title="修改抽取实体"
      >
        <Space direction="vertical" size={10} style={{ width: '100%' }}>
          <Input
            aria-label="实体名称"
            onChange={(event) => setEntityDraft((draft) => ({ ...draft, label: event.target.value }))}
            placeholder="实体名称"
            value={entityDraft.label}
          />
          <Input
            aria-label="实体类型"
            onChange={(event) => setEntityDraft((draft) => ({ ...draft, type: event.target.value }))}
            placeholder="实体类型，如 Person / Concept / Policy"
            value={entityDraft.type}
          />
          <Input.TextArea
            aria-label="实体描述"
            autoSize={{ minRows: 3, maxRows: 6 }}
            onChange={(event) => setEntityDraft((draft) => ({ ...draft, description: event.target.value }))}
            placeholder="实体描述"
            value={entityDraft.description}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            实体和关系默认由入库流程自动抽取；这里保存的是人工校正，会写回图数据库。
          </Text>
        </Space>
      </Modal>
    </>
  )
}
