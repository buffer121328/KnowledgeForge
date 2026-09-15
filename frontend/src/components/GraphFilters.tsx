import { ClearOutlined, SearchOutlined } from '@ant-design/icons'
import { Button, Card, Input, Select, Space, Tag } from 'antd'
import { formatGraphRelation } from '@/utils/graphLabels'
import { entityTypeLabels, labelOf } from '@/utils/labels'

interface GraphFiltersProps {
  active: boolean
  entityQuery: string
  entityType: string
  entityTypes: string[]
  onClear: () => void
  onEntityQueryChange: (query: string) => void
  onEntityTypeChange: (entityType: string) => void
  onRelationTypeChange: (relationType: string) => void
  relationType: string
  relationTypes: string[]
  totalEdges: number
  totalNodes: number
  visibleEdges: number
  visibleNodes: number
}

/** Render client-side filters for the currently loaded department graph. */
export function GraphFilters({
  active,
  entityQuery,
  entityType,
  entityTypes,
  onClear,
  onEntityQueryChange,
  onEntityTypeChange,
  onRelationTypeChange,
  relationType,
  relationTypes,
  totalEdges,
  totalNodes,
  visibleEdges,
  visibleNodes,
}: GraphFiltersProps) {
  return (
    <Card size="small" style={{ marginBottom: 16 }}>
      <Space wrap size={12}>
        <Input
          allowClear
          aria-label="筛选实体关键词"
          onChange={(event) => onEntityQueryChange(event.target.value)}
          placeholder="按实体名称或描述筛选"
          prefix={<SearchOutlined />}
          style={{ width: 260 }}
          value={entityQuery}
        />
        <Select
          aria-label="筛选实体类型"
          onChange={onEntityTypeChange}
          options={[
            { value: 'all', label: '全部实体类型' },
            ...entityTypes.map((type) => ({
              value: type,
              label: labelOf(entityTypeLabels, type),
            })),
          ]}
          style={{ width: 180 }}
          value={entityType}
        />
        <Select
          aria-label="筛选关系类型"
          onChange={onRelationTypeChange}
          options={[
            { value: 'all', label: '全部关系类型' },
            ...relationTypes.map((type) => ({ value: type, label: formatGraphRelation(type) })),
          ]}
          style={{ width: 190 }}
          value={relationType}
        />
        <Button
          disabled={!active}
          icon={<ClearOutlined />}
          onClick={onClear}
        >
          清除筛选
        </Button>
        <Tag color={active ? 'blue' : 'default'}>节点 {visibleNodes} / {totalNodes}</Tag>
        <Tag color={active ? 'purple' : 'default'}>关系 {visibleEdges} / {totalEdges}</Tag>
      </Space>
    </Card>
  )
}
