import { ApartmentOutlined, ReloadOutlined } from '@ant-design/icons'
import { useMemo, useState } from 'react'
import { Breadcrumb, Button, Card, Empty, Input, Modal, Space, Table, Tag, Typography } from 'antd'
import type { GraphEdge } from '@/api/graph'
import { GraphCanvas } from '@/components/GraphCanvas'
import { GraphFilters } from '@/components/GraphFilters'
import { formatGraphNodeRef, formatGraphRelation } from '@/utils/graphLabels'
import { useKnowledgeGraph } from './useKnowledgeGraph'
import { useAuthStore } from '@/stores/auth'

/** Render the interactive company and department relationship network. */
export default function GraphView() {
  const canEdit = useAuthStore((state) => state.hasPermission('graph:edit'))
  const {
    clearFilters,
    colorMap,
    companyName,
    departmentName,
    edges,
    enterDepartment,
    entityQuery,
    entityType,
    entityTypes,
    filterEntityTypes,
    handleReset,
    hasActiveFilters,
    loading,
    nodes,
    relationType,
    relationTypes,
    selected,
    setEntityQuery,
    setEntityType,
    setRelationType,
    setSelected,
    stats,
    updateEntity,
    updateRelation,
    status,
    scope,
    visibleEdges,
    visibleNodes,
  } = useKnowledgeGraph()

  const [editingRelation, setEditingRelation] = useState<GraphEdge | null>(null)
  const [relationDraft, setRelationDraft] = useState('')
  const nodeLookup = useMemo(
    () => new Map(visibleNodes.map((node) => [node.id, node] as const)),
    [visibleNodes],
  )
  const openRelationEditor = (edge: GraphEdge, relationType?: string, claimId?: string) => {
    setEditingRelation({ ...edge, claim_id: claimId ?? edge.claim_id, label: relationType ?? edge.label })
    setRelationDraft(relationType ?? edge.label)
  }
  const saveRelation = async () => {
    if (!editingRelation) return
    await updateRelation(editingRelation, relationDraft)
    setEditingRelation(null)
  }

  const edgeColumns = [
    {
      title: '来源',
      dataIndex: 'source',
      ellipsis: true,
      render: (value: string) => formatGraphNodeRef(nodeLookup.get(value), value),
    },
    {
      title: scope === 'company' ? '关系 / 关联路径' : '关系',
      dataIndex: 'label',
      width: 220,
      render: (value: string, edge: GraphEdge) => (
        <Space size={6} wrap>
          <span>{formatGraphRelation(value)}</span>
          {(value === 'CROSS_DEPARTMENT' || edge.relation_details?.some((detail) => detail.claim_id !== edge.claim_id)) && (
            <Tag color="orange">跨部门</Tag>
          )}
        </Space>
      ),
    },
    {
      title: '目标',
      dataIndex: 'target',
      ellipsis: true,
      render: (value: string) => formatGraphNodeRef(nodeLookup.get(value), value),
    },
    { title: '关联数', dataIndex: 'path_count', width: 90, render: (value?: number) => value ?? 1 },
    ...(canEdit ? [{
      title: '操作',
      width: 96,
      render: (_: unknown, edge: GraphEdge) => (
        <Button
          disabled={!edge.claim_id}
          onClick={() => openRelationEditor(edge)}
          size="small"
        >
          修改
        </Button>
      ),
    }] : []),
  ]

  return (
    <Card
      title={<Space><ApartmentOutlined />{scope === 'company' ? '公司组织架构 / 跨部门关联网络' : `${departmentName} · 部门内部网络`}</Space>}
      extra={(
        <Space>
          <Tag color="blue">节点 {stats.nodes}</Tag>
          <Tag color="purple">关系 {stats.edges}</Tag>
          {status !== 'ok' && <Tag color="orange">{status}</Tag>}
          <Button size="small" icon={<ReloadOutlined />} onClick={handleReset} loading={loading}>
            刷新图谱
          </Button>
        </Space>
      )}
    >
      {scope === 'department' && <Breadcrumb items={[
        { title: <Button type="link" size="small" onClick={handleReset}>{companyName || '公司总览'}</Button> },
        { title: departmentName },
      ]} style={{ marginBottom: 16 }} />}
      {scope === 'department' && (
        <GraphFilters
          active={hasActiveFilters}
          entityQuery={entityQuery}
          entityType={entityType}
          entityTypes={filterEntityTypes}
          onClear={clearFilters}
          onEntityQueryChange={setEntityQuery}
          onEntityTypeChange={setEntityType}
          onRelationTypeChange={setRelationType}
          relationType={relationType}
          relationTypes={relationTypes}
          totalEdges={edges.length}
          totalNodes={nodes.length}
          visibleEdges={visibleEdges.length}
          visibleNodes={visibleNodes.length}
        />
      )}
      <GraphCanvas
        colorMap={colorMap}
        edges={visibleEdges}
        entityTypes={entityTypes}
        loading={loading}
        nodes={visibleNodes}
        onDepartmentOpen={scope === 'company' ? (node) => void enterDepartment(node.id.replace(/^department:/, '')) : undefined}
        onEntityUpdate={canEdit ? updateEntity : undefined}
        onNodeSelect={setSelected}
        selected={selected}
        status={status}
      />

      <Card size="small" title={scope === 'company' ? '组织关系 / 跨部门关联' : '部门内部关系'} style={{ marginTop: 16 }}>
        <Table<GraphEdge>
          rowKey={(edge) => `${edge.source}-${edge.label}-${edge.target}`}
          columns={edgeColumns}
          dataSource={visibleEdges}
          size="small"
          pagination={{ pageSize: 8 }}
          expandable={{
            expandedRowRender: (edge) => {
              const details = edge.relation_details ?? []
              if (details.length === 0) return <Typography.Text type="secondary">暂无明细</Typography.Text>
              return (
                <Space direction="vertical" size={8} style={{ width: '100%' }}>
                  {details.map((detail) => (
                    <div key={detail.claim_id} style={{ alignItems: 'center', display: 'flex', gap: 8, justifyContent: 'space-between' }}>
                      <Space size={8} wrap>
                        <Typography.Text>
                          {detail.head_name} —{formatGraphRelation(detail.relation_type)}→ {detail.tail_name}
                        </Typography.Text>
                        <Tag>{detail.claim_id.slice(0, 8)}</Tag>
                      </Space>
                      {canEdit && <Button size="small" onClick={() => openRelationEditor(edge, detail.relation_type, detail.claim_id)}>
                        修改关系
                      </Button>}
                    </div>
                  ))}
                </Space>
              )
            },
            rowExpandable: (edge) => Boolean(edge.relation_details?.length),
          }}
          locale={{ emptyText: <Empty description={status === 'disconnected' ? 'Neo4j 未连接' : hasActiveFilters ? '没有符合筛选条件的关系' : '当前范围无关系'} /> }}
        />
      </Card>
      {canEdit && <Modal
        okText="保存"
        onCancel={() => setEditingRelation(null)}
        onOk={() => { void saveRelation() }}
        open={Boolean(editingRelation)}
        title="修改抽取关系"
      >
        <Space direction="vertical" size={10} style={{ width: '100%' }}>
          <Input
            aria-label="关系类型"
            onChange={(event) => setRelationDraft(event.target.value)}
            placeholder="关系类型，如 APPROVES / DEPENDS_ON"
            value={relationDraft}
          />
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            关系默认由入库流程自动抽取；这里修改的是当前关系声明的类型，保存后会写回图数据库。
          </Typography.Text>
        </Space>
      </Modal>}
    </Card>
  )
}
