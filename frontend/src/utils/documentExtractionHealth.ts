import type { DocListItem } from '@/types'

export type ExtractionAnomalyMetric = '分块' | '实体' | '关系'

/** Return zero-count extraction categories only when an ingestion result is final and comparable. */
export function extractionAnomalyMetrics(document: DocListItem): ExtractionAnomalyMetric[] {
  if (document.legacy || document.ingest_status !== 'ingested') return []

  return [
    document.chunks_count === 0 ? '分块' : null,
    document.entities_count === 0 ? '实体' : null,
    document.relations_count === 0 ? '关系' : null,
  ].filter((metric): metric is ExtractionAnomalyMetric => metric !== null)
}

/** Format a safe, concise document-extraction anomaly description for the catalog. */
export function extractionAnomalyDescription(document: DocListItem): string | null {
  const metrics = extractionAnomalyMetrics(document)
  return metrics.length > 0 ? `${metrics.join('、')}为 0` : null
}
