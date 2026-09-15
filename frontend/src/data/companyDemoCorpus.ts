/** Stable IDs of the 41 current-corpus company-demo documents.
 *
 * Keep this generated from `backend/evaluation/data/company-demo/corpus_manifest.json`.
 * The companion regression test prevents the UI allowlist from drifting from that manifest.
 */
export const COMPANY_DEMO_SOURCE_DOCUMENT_IDS = [
  'a23c6a1b-2e6f-54ca-b9bb-97359f53f349',
  '9b24f3ba-9fd6-5dab-8c5c-fab973bcbaa3',
  '46d29a92-e357-59a9-ac58-d257a885dace',
  'ce4a2c69-ee78-5804-a307-bd2cdc164f81',
  'b1e9f120-570f-52f8-b1f1-5a3608c98b4b',
  '6ddbf1c0-63e5-56f8-b6eb-b51b50dc8f04',
  'c4027e05-b985-5cc0-a3db-abde842dd4f8',
  '4cacc7a1-cd9f-5f41-908f-3ad6a543c0d3',
  'df459e31-94a8-5a5f-8757-6d0fdd1d93c9',
  '78c30b34-29ba-5a7f-b327-72af3fb55e4e',
  '440b4df6-9250-50d9-a725-fa2c00a2b50c',
  'bcab26a2-9c0b-59eb-9f29-64be26e7781d',
  'dd80ebcd-e420-593a-8ce7-4d901cc21404',
  '9a7f0075-ae5f-5dbe-9fd8-cab65f8ce117',
  '5d4d37c4-be81-5017-b362-bc5c70028f7a',
  '88a70e65-8669-5efd-b64f-50ae08f6b91d',
  'b33756a4-39f0-53a0-879a-5a655347ac4f',
  '5ba9a3c9-a35a-5070-ab11-3cc28cf9d7d9',
  '5d47c22c-d81b-5bd1-84b4-4474673cbf5e',
  '77f63f2d-91f6-51cf-8741-45d54002c2ae',
  '704c03fe-8b10-5261-9f18-bd99c9eb4901',
  'e054b187-672b-56ce-83fd-69dfdfeb29fb',
  'de9673c0-2caf-56bb-900d-cfe2cb5e6bc5',
  '5e005b4c-dd78-5d20-93a6-9b57af8c5820',
  '7efa9c4b-43b5-57ac-acf4-9fa25bb1b5da',
  '2bbf4d77-1a84-5f2e-ad28-0d52e408b856',
  '9e2ff871-9917-5122-beca-e31f3c6cac8d',
  '52290647-3d0e-5611-adb0-b555a7f57a6d',
  'e54ac156-48c5-5d38-b5e5-56652203491e',
  'c104acac-67fb-57e1-a4ff-233719942d5f',
  '9ae596b0-429a-584e-a9ba-2ed61b4bab15',
  '807dc11d-1c18-5817-95b2-97c45db0a977',
  'ce122b75-7e11-5a94-a4e9-b7c7da6dd0d5',
  '7fd36aac-af45-523b-9d96-4c4edbf1bbf4',
  '39ec5765-600e-5f82-9183-fe73177af9e3',
  '9d0e1b1e-52bc-5557-981c-c37de0b10158',
  '1b910ac7-2e26-5165-bd65-4e8ddee9bfad',
  '13125e7e-765d-5837-819b-9a4e2f617d1d',
  'a0a88361-c979-51b6-9edc-bf1929851728',
  '449df1d0-d246-5e29-a91c-b1219ef775fe',
  '2d666052-f382-53e1-849d-585f4134cd4d',
] as const

const companyDemoSourceDocumentIdSet = new Set<string>(COMPANY_DEMO_SOURCE_DOCUMENT_IDS)

/** Return whether a durable catalog ID belongs to the controlled company-demo corpus. */
export function isCompanyDemoSourceDocumentId(docId: string | null | undefined): boolean {
  return Boolean(docId && companyDemoSourceDocumentIdSet.has(docId))
}
