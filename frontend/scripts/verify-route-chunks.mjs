import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const manifestPath = resolve('dist/.vite/manifest.json')
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
const entries = Object.entries(manifest)
const entry = entries.find(([, value]) => value.isEntry)

if (!entry) {
  throw new Error('Vite manifest does not contain an application entry chunk')
}

const visited = new Set()
const pending = [entry[0]]
while (pending.length > 0) {
  const key = pending.pop()
  if (!key || visited.has(key)) continue
  visited.add(key)
  for (const dependency of manifest[key]?.imports ?? []) {
    pending.push(dependency)
  }
}

const bundledPageSources = [...visited]
  .map((key) => manifest[key]?.src)
  .filter((source) => typeof source === 'string' && source.startsWith('src/pages/'))

if (bundledPageSources.length > 0) {
  throw new Error(`Initial entry chunk statically includes routed pages: ${bundledPageSources.join(', ')}`)
}

console.log(`Verified ${entry[0]} excludes routed page modules from its static chunk graph.`)
