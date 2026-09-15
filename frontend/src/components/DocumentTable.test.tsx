// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import type { DocListItem, User } from '@/types'

const authState = vi.hoisted(() => ({
  user: {
    user_id: 'manager-a',
    username: 'manager-a',
    role: 'admin',
    org_id: 'org-a',
    permissions: ['doc:read', 'doc:delete'],
    department_id: 'finance',
    is_department_manager: true,
  } as User,
}))

vi.mock('@/stores/auth', () => ({
  useAuthStore: (selector: (state: typeof authState) => unknown) => selector(authState),
}))

import { DocumentTable } from './DocumentTable'

afterEach(cleanup)

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(() => ({
      addEventListener: vi.fn(),
      addListener: vi.fn(),
      dispatchEvent: vi.fn(),
      matches: false,
      media: '',
      onchange: null,
      removeEventListener: vi.fn(),
      removeListener: vi.fn(),
    })),
  })
})

beforeEach(() => {
  authState.user = {
    user_id: 'manager-a',
    username: 'manager-a',
    role: 'admin',
    org_id: 'org-a',
    permissions: ['doc:read', 'doc:delete'],
    department_id: 'finance',
    is_department_manager: true,
  }
})

const documentRow: DocListItem = {
  doc_id: 'doc-a',
  file_name: '预算制度',
  source: '',
  file_reference: 'departments/finance/documents/doc-a/预算制度.txt',
  doc_type: 'text',
  chunks_count: 1,
  department_id: 'finance',
}

const commonProps = {
  departments: [{ department_id: 'finance', company_id: 'org-a', name: '财务部', normalized_key: 'finance', status: 'active' }],
  onDepartmentChange: vi.fn(),
}

describe('DocumentTable extraction health', () => {
  it('marks zero chunks, entities, and relations on an ingested document', () => {
    render(
      <DocumentTable
        activeDoc={null}
        chunks={[]}
        chunksLoading={false}
        docs={[{
          ...documentRow,
          chunks_count: 0,
          entities_count: 0,
          relations_count: 0,
          ingest_status: 'ingested',
        }]}
        {...commonProps}
        drawerOpen={false}
        loading={false}
        onCloseChunks={vi.fn()}
        onDelete={vi.fn()}
        onOpenDocument={vi.fn()}
        onOpenChunks={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )

    expect(screen.getByText(/抽取异常：/)).toBeTruthy()
    expect(screen.getByText(/分块、实体、关系为 0/)).toBeTruthy()
  })

  it('does not mark processing, failed, or legacy rows solely because counts are zero', () => {
    render(
      <DocumentTable
        activeDoc={null}
        chunks={[]}
        chunksLoading={false}
        docs={[
          { ...documentRow, doc_id: 'processing', ingest_status: 'processing', chunks_count: 0, entities_count: 0, relations_count: 0 },
          { ...documentRow, doc_id: 'failed', ingest_status: 'failed', chunks_count: 0, entities_count: 0, relations_count: 0 },
          { ...documentRow, doc_id: 'legacy', ingest_status: 'ingested', legacy: true, chunks_count: 0, entities_count: 0, relations_count: 0 },
        ]}
        {...commonProps}
        drawerOpen={false}
        loading={false}
        onCloseChunks={vi.fn()}
        onDelete={vi.fn()}
        onOpenDocument={vi.fn()}
        onOpenChunks={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )

    expect(screen.queryByText('分块、实体、关系为 0')).toBeNull()
  })

})

describe('DocumentTable source opening', () => {
  it('offers an open action only when the source reference is available', () => {
    const onOpenDocument = vi.fn().mockResolvedValue(undefined)
    render(
      <DocumentTable
        activeDoc={null}
        chunks={[]}
        chunksLoading={false}
        docs={[documentRow]}
        {...commonProps}
        drawerOpen={false}
        loading={false}
        onCloseChunks={vi.fn()}
        onDelete={vi.fn()}
        onOpenDocument={onOpenDocument}
        onOpenChunks={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: '打开文档' }))

    expect(onOpenDocument).toHaveBeenCalledWith(documentRow)
  })

  it('disables opening when the source file is unavailable', () => {
    render(
      <DocumentTable
        activeDoc={null}
        chunks={[]}
        chunksLoading={false}
        docs={[{ ...documentRow, file_reference: '' }]}
        {...commonProps}
        drawerOpen={false}
        loading={false}
        onCloseChunks={vi.fn()}
        onDelete={vi.fn()}
        onOpenDocument={vi.fn()}
        onOpenChunks={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )

    expect((screen.getByRole('button', { name: '打开文档' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('disables delete for a department manager viewing another department', () => {
    render(
      <DocumentTable
        {...commonProps}
        activeDoc={null}
        chunks={[]}
        chunksLoading={false}
        docs={[{ ...documentRow, department_id: 'hr' }]}
        drawerOpen={false}
        loading={false}
        onCloseChunks={vi.fn()}
        onDelete={vi.fn()}
        onOpenDocument={vi.fn()}
        onOpenChunks={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )

    expect((screen.getByRole('button', { name: /删除/ }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('offers an all-departments selector backed by authorized departments', () => {
    render(
      <DocumentTable
        {...commonProps}
        activeDoc={null}
        chunks={[]}
        chunksLoading={false}
        docs={[documentRow]}
        drawerOpen={false}
        loading={false}
        onCloseChunks={vi.fn()}
        onDelete={vi.fn()}
        onOpenDocument={vi.fn()}
        onOpenChunks={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )
    expect(screen.getByText('全部部门')).toBeTruthy()
  })

  it('does not render a write control for a read-only employee', () => {
    authState.user = {
      ...authState.user,
      role: 'viewer',
      permissions: ['doc:read'],
      is_department_manager: false,
    }
    render(
      <DocumentTable
        {...commonProps}
        activeDoc={null}
        chunks={[]}
        chunksLoading={false}
        docs={[documentRow]}
        drawerOpen={false}
        loading={false}
        onCloseChunks={vi.fn()}
        onDelete={vi.fn()}
        onOpenDocument={vi.fn()}
        onOpenChunks={vi.fn()}
        onRefresh={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: /删除/ })).toBeNull()
  })
})
