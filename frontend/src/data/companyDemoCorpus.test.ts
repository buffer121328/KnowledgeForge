// @vitest-environment node
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { COMPANY_DEMO_SOURCE_DOCUMENT_IDS, isCompanyDemoSourceDocumentId } from './companyDemoCorpus'

describe('company-demo corpus allowlist', () => {
  it('matches all and only the 41 stable IDs in the authoritative corpus manifest', () => {
    const manifestPath = new URL('../../../backend/evaluation/data/company-demo/corpus_manifest.json', import.meta.url)
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
      documents: Array<{ source_document_id: string }>
    }
    const expected = manifest.documents.map((document) => document.source_document_id).sort()

    expect(COMPANY_DEMO_SOURCE_DOCUMENT_IDS).toHaveLength(41)
    expect(new Set(COMPANY_DEMO_SOURCE_DOCUMENT_IDS)).toHaveLength(41)
    expect([...COMPANY_DEMO_SOURCE_DOCUMENT_IDS].sort()).toEqual(expected)
    expect(isCompanyDemoSourceDocumentId('not-a-company-demo-id')).toBe(false)
  })
})
