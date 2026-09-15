import type { FolderPreviewFile, FolderUploadManifest, FolderUploadPreview } from '@/types'

/** 与后端上传白名单保持一致。 */
export const DOCUMENT_ALLOWED_EXTENSIONS = [
  '.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.docx', '.doc', '.xlsx', '.xls',
  '.csv', '.md', '.txt', '.html', '.htm',
] as const

const allowedExtensions = new Set<string>(DOCUMENT_ALLOWED_EXTENSIONS)
const ignoredNames = new Set(['.ds_store', 'thumbs.db', 'desktop.ini'])

export const FOLDER_PREVIEW_STATUS_LABELS: Record<FolderPreviewFile['status'], string> = {
  ready: '待入库',
  ignored: '已忽略',
  unsupported: '类型不支持',
  oversized: '文件过大',
  duplicate: '路径重复',
  'invalid-path': '路径无效',
}

export function folderPreviewStatusLabel(status: FolderPreviewFile['status']): string {
  return FOLDER_PREVIEW_STATUS_LABELS[status]
}

export const DOCUMENT_UPLOAD_ACCEPT = [
  ...DOCUMENT_ALLOWED_EXTENSIONS,
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
].join(',')
export const MAX_DOCUMENT_UPLOAD_BATCH_FILES = 50
export const MAX_DOCUMENT_UPLOAD_SIZE = 50 * 1024 * 1024

export type DocumentUploadValidationResult =
  | { valid: true }
  | { valid: false; reason: 'unsupported'; extension: string }
  | { valid: false; reason: 'oversized'; extension: string }

export type DocumentUploadBatchValidationResult =
  | { valid: true }
  | { valid: false; reason: 'empty' }
  | { valid: false; reason: 'too-many-files'; maxFiles: number; fileCount: number }
  | { valid: false; reason: 'unsupported'; extension: string; fileName: string }
  | { valid: false; reason: 'oversized'; extension: string; fileName: string }

/** Return the normalized extension of an uploaded document. */
export function getDocumentFileExtension(name: string): string {
  const index = name.lastIndexOf('.')
  return index >= 0 ? name.slice(index).toLowerCase() : ''
}

/** Validate an uploaded document before sending it to the API. */
export function validateDocumentUpload(file: Pick<File, 'name' | 'size'>): DocumentUploadValidationResult {
  const extension = getDocumentFileExtension(file.name)
  if (!extension || !allowedExtensions.has(extension)) {
    return { valid: false, reason: 'unsupported', extension }
  }
  if (file.size > MAX_DOCUMENT_UPLOAD_SIZE) {
    return { valid: false, reason: 'oversized', extension }
  }
  return { valid: true }
}

/** Validate a complete selection before sending one flat batch request. */
export function validateDocumentUploadBatch(
  files: Array<Pick<File, 'name' | 'size'>>,
): DocumentUploadBatchValidationResult {
  if (files.length === 0) return { valid: false, reason: 'empty' }
  if (files.length > MAX_DOCUMENT_UPLOAD_BATCH_FILES) {
    return { valid: false, reason: 'too-many-files', maxFiles: MAX_DOCUMENT_UPLOAD_BATCH_FILES, fileCount: files.length }
  }
  for (const file of files) {
    const validation = validateDocumentUpload(file)
    if (!validation.valid) return { ...validation, fileName: file.name }
  }
  return { valid: true }
}

/** Normalize an untrusted browser relative path without permitting traversal or absolute roots. */
export function normalizeBrowserRelativePath(value: string): string | null {
  const normalized = value.normalize('NFC').replaceAll('\\', '/').replace(/^\.\//, '')
  if (!normalized || normalized.startsWith('/') || /^[A-Za-z]:\//.test(normalized)) return null
  const parts = normalized.split('/')
  if (parts.some((part) => !part || part === '.' || part === '..' || part.includes('\0'))) return null
  return parts.join('/')
}

/** Return the browser-provided folder path, falling back to the selected basename. */
export function getBrowserRelativePath(file: File): string {
  const relative = (file as File & { webkitRelativePath?: string }).webkitRelativePath
  return relative || file.name
}

/** Build a safe folder preview, retaining ignored and rejected rows for explicit user review. */
export function buildFolderUploadPreview(files: File[]): FolderUploadPreview {
  const seen = new Map<string, number[]>()
  const selectedPaths = files
    .map((file) => normalizeBrowserRelativePath(getBrowserRelativePath(file))?.split('/') ?? [])
  const selectedRootIsDepartment = selectedPaths.some((parts) => parts.length === 2)
    && selectedPaths.filter((parts) => parts.length > 0).every((parts) => parts[0] === selectedPaths[0]?.[0])
  const previewFiles: FolderPreviewFile[] = files.map((file, index) => {
    const originalPath = getBrowserRelativePath(file)
    const normalized = normalizeBrowserRelativePath(originalPath)
    const rawParts = normalized?.split('/') ?? []
    const normalizedIndex = rawParts.findIndex((part) => part.toLocaleLowerCase() === 'normalized')
    const relativeParts = normalizedIndex >= 0
      ? rawParts.slice(normalizedIndex + 1)
      : selectedRootIsDepartment
        ? rawParts
        : rawParts.slice(rawParts.length > 1 ? 1 : 0)
    const relativePath = relativeParts.join('/')
    const folderKey = relativeParts.length > 1 ? relativeParts[0] : 'general'
    const lowerName = file.name.toLocaleLowerCase()
    let status: FolderPreviewFile['status'] = 'ready'
    let reason = ''
    if (ignoredNames.has(lowerName) || rawParts.some((part) => part.startsWith('.') && part !== file.name)) {
      status = 'ignored'; reason = '系统或隐藏文件'
    } else if (!normalized || !relativePath || relativeParts.length < 2) {
      status = 'invalid-path'; reason = '缺少安全的部门/文件相对路径'
    } else {
      const validation = validateDocumentUpload(file)
      if (!validation.valid) {
        status = validation.reason
        reason = validation.reason === 'unsupported' ? '不支持的文件类型' : '文件超过 50MB'
      }
    }
    const clientFileId = `folder-${index}-${relativePath || file.name}`
    if (status === 'ready') {
      const key = relativePath.normalize('NFC').toLocaleLowerCase()
      const indexes = seen.get(key) ?? []
      indexes.push(index); seen.set(key, indexes)
    }
    return { clientFileId, file, originalPath, relativePath, folderKey, status, reason }
  })

  for (const indexes of seen.values()) {
    if (indexes.length < 2) continue
    for (const index of indexes) {
      previewFiles[index] = { ...previewFiles[index], status: 'duplicate', reason: '规范化相对路径重复' }
    }
  }
  const departmentCounts: Record<string, number> = {}
  for (const item of previewFiles) {
    if (item.status === 'ready') departmentCounts[item.folderKey] = (departmentCounts[item.folderKey] ?? 0) + 1
  }
  const originalRoot = files[0]
    ? normalizeBrowserRelativePath(getBrowserRelativePath(files[0]))?.split('/')[0]
    : undefined
  return {
    rootFolderName: originalRoot || 'selected-folder',
    files: previewFiles,
    departmentCounts,
    readyCount: previewFiles.filter((item) => item.status === 'ready').length,
    issueCount: previewFiles.filter((item) => item.status !== 'ready' && item.status !== 'ignored').length,
  }
}

/** Serialize ready preview rows into the backend manifest after mapping confirmation. */
export function createFolderUploadManifest(
  preview: FolderUploadPreview,
  companyId: string,
  departmentMappings: Record<string, string>,
): FolderUploadManifest {
  const ready = preview.files.filter((item) => item.status === 'ready')
  return {
    root_folder_name: preview.rootFolderName,
    company_id: companyId,
    department_mappings: departmentMappings,
    files: ready.map((item) => ({
      client_file_id: item.clientFileId,
      relative_path: item.relativePath,
      department_id: departmentMappings[item.folderKey],
      display_name: item.file.name.replace(/\.[^.]+$/, ''),
      provenance_source_filename: item.file.name,
      metadata: {},
    })),
  }
}
