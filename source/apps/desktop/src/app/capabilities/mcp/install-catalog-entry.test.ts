import { afterEach, expect, it, vi } from 'vitest'

import { installMcpCatalogEntry, type McpCatalogEntry } from '@/hermes'

import { installBundledEntry } from './install-catalog-entry'

vi.mock('@/hermes', () => ({ installMcpCatalogEntry: vi.fn(async () => ({ ok: true })), getActionStatus: vi.fn() }))

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

it('refuses MCP package bootstrapping in the browser but keeps config-only catalog entries', async () => {
  vi.stubGlobal('hermesDesktop', { browser: {} })
  const entry = { name: 'fixture', needs_install: true } as McpCatalogEntry

  await expect(installBundledEntry(entry, {}, 'work')).rejects.toThrow(/unavailable in the browser/i)
  expect(installMcpCatalogEntry).not.toHaveBeenCalled()

  await installBundledEntry({ ...entry, needs_install: false }, { API_KEY: 'fixture' }, 'work')
  expect(installMcpCatalogEntry).toHaveBeenCalledWith('fixture', { API_KEY: 'fixture' }, 'work')
})

it('preserves desktop package installation', async () => {
  vi.stubGlobal('hermesDesktop', {})
  await installBundledEntry({ name: 'fixture', needs_install: true } as McpCatalogEntry, {})
  expect(installMcpCatalogEntry).toHaveBeenCalledWith('fixture', {}, undefined)
})
