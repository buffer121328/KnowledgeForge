// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { GraphFilters } from './GraphFilters'

describe('GraphFilters', () => {
  it('renders department entity and relationship controls and clears active filters', () => {
    const onEntityQueryChange = vi.fn()
    const onClear = vi.fn()

    render(
      <GraphFilters
        active
        entityQuery="预算"
        entityType="Concept"
        entityTypes={['Concept', 'Person']}
        onClear={onClear}
        onEntityQueryChange={onEntityQueryChange}
        onEntityTypeChange={vi.fn()}
        onRelationTypeChange={vi.fn()}
        relationType="MANAGES"
        relationTypes={['MANAGES', 'USES']}
        totalEdges={3}
        totalNodes={4}
        visibleEdges={1}
        visibleNodes={2}
      />,
    )

    fireEvent.change(screen.getByLabelText('筛选实体关键词'), { target: { value: '负责人' } })
    expect(onEntityQueryChange).toHaveBeenCalledWith('负责人')
    expect(screen.getAllByLabelText('筛选实体类型')).toHaveLength(2)
    expect(screen.getAllByLabelText('筛选关系类型')).toHaveLength(2)
    expect(screen.getByText('节点 2 / 4')).toBeTruthy()
    expect(screen.getByText('关系 1 / 3')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /清除筛选/ }))
    expect(onClear).toHaveBeenCalledTimes(1)
  })
})
