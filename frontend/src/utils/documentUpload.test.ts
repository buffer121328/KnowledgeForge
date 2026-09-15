import { describe, expect, it } from 'vitest'
import {
  MAX_DOCUMENT_UPLOAD_BATCH_FILES,
  MAX_DOCUMENT_UPLOAD_SIZE,
  folderPreviewStatusLabel,
  validateDocumentUploadBatch,
} from './documentUpload'

function file(name: string, size = 1): Pick<File, 'name' | 'size'> {
  return { name, size }
}

describe('validateDocumentUploadBatch', () => {
  it.each([1, MAX_DOCUMENT_UPLOAD_BATCH_FILES])(
    'accepts a batch containing %i supported files',
    (count) => {
      const files = Array.from({ length: count }, (_, index) => file(`doc-${index}.txt`))

      expect(validateDocumentUploadBatch(files)).toEqual({ valid: true })
    },
  )

  it('rejects 51 files before upload', () => {
    const files = Array.from({ length: 51 }, (_, index) => file(`doc-${index}.txt`))

    expect(validateDocumentUploadBatch(files)).toEqual({
      valid: false,
      reason: 'too-many-files',
      maxFiles: 50,
      fileCount: 51,
    })
  })

  it('preserves unsupported file validation inside an allowed-size batch', () => {
    expect(validateDocumentUploadBatch([file('safe.txt'), file('malware.exe')])).toEqual({
      valid: false,
      reason: 'unsupported',
      extension: '.exe',
      fileName: 'malware.exe',
    })
  })

  it('preserves per-file size validation inside an allowed-size batch', () => {
    expect(
      validateDocumentUploadBatch([
        file('safe.txt'),
        file('large.pdf', MAX_DOCUMENT_UPLOAD_SIZE + 1),
      ]),
    ).toEqual({
      valid: false,
      reason: 'oversized',
      extension: '.pdf',
      fileName: 'large.pdf',
    })
  })
})

import { buildFolderUploadPreview, createFolderUploadManifest } from './documentUpload'

/** Build a browser folder file with a stable webkitRelativePath. */
function folderFile(path: string, size = 1): File {
  const name = path.split('/').at(-1) ?? path
  const value = new File(['x'], name, { type: 'text/plain' })
  Object.defineProperty(value, 'size', { value: size })
  Object.defineProperty(value, 'webkitRelativePath', { value: path })
  return value
}

describe('buildFolderUploadPreview', () => {
  it('builds the company-demo 21/5/5/10 normalized department preview', () => {
    const files = [
      ...Array.from({ length: 21 }, (_, index) => folderFile(`company-demo/normalized/human_resources/hr-${index}.txt`)),
      ...Array.from({ length: 5 }, (_, index) => folderFile(`company-demo/normalized/finance/finance-${index}.txt`)),
      ...Array.from({ length: 5 }, (_, index) => folderFile(`company-demo/normalized/procurement_warehouse/procurement-${index}.txt`)),
      ...Array.from({ length: 10 }, (_, index) => folderFile(`company-demo/normalized/administration/admin-${index}.txt`)),
    ]

    const preview = buildFolderUploadPreview(files)

    expect(preview.readyCount).toBe(41)
    expect(preview.departmentCounts).toEqual({ human_resources: 21, finance: 5, procurement_warehouse: 5, administration: 10 })
    expect(preview.files[0].relativePath).toBe('human_resources/hr-0.txt')
  })

  it('accepts a directly selected department folder and preserves its department segment', () => {
    const preview = buildFolderUploadPreview([
      folderFile('finance/expense.txt'),
      folderFile('finance/policies/travel.txt'),
    ])

    expect(preview.readyCount).toBe(2)
    expect(preview.issueCount).toBe(0)
    expect(preview.departmentCounts).toEqual({ finance: 2 })
    expect(preview.files.map((item) => item.relativePath)).toEqual([
      'finance/expense.txt',
      'finance/policies/travel.txt',
    ])
  })

  it('presents invalid paths with a bounded Chinese status label', () => {
    expect(folderPreviewStatusLabel('invalid-path')).toBe('路径无效')
  })

  it('keeps ignored, unsupported, and duplicate rows visible but out of the manifest', () => {
    const preview = buildFolderUploadPreview([
      folderFile('root/finance/a.txt'),
      folderFile('root/finance/A.txt'),
      folderFile('root/finance/.DS_Store'),
      folderFile('root/finance/malware.exe'),
    ])

    expect(preview.files.map((item) => item.status)).toEqual([
      'duplicate', 'duplicate', 'ignored', 'unsupported',
    ])
    const manifest = createFolderUploadManifest(preview, 'tenant-a', { finance: 'finance' })
    expect(manifest.files).toEqual([])
  })

  it('serializes only ready files after department mapping confirmation', () => {
    const preview = buildFolderUploadPreview([folderFile('root/finance/expense.txt')])
    const manifest = createFolderUploadManifest(preview, 'tenant-a', { finance: 'finance-dept' })

    expect(manifest.company_id).toBe('tenant-a')
    expect(manifest.department_mappings).toEqual({ finance: 'finance-dept' })
    expect(manifest.files[0]).toMatchObject({
      relative_path: 'finance/expense.txt',
      department_id: 'finance-dept',
      provenance_source_filename: 'expense.txt',
    })
  })
})
