import { afterEach, expect, it, vi } from 'vitest'

import { createBrowserBridge } from '@/browser/bridge'

// Use the browser's actual, deliberately partial bridge: a full Electron mock
// hides unguarded native calls made by newly exported SDK modules at import time.
afterEach(() => vi.unstubAllGlobals())

it('loads the plugin SDK without Electron and preserves stock skill APIs', async () => {
  const fetch = vi.fn(async () => new Response(JSON.stringify([{ name: 'fixture-skill', enabled: true }])))
  vi.stubGlobal('fetch', fetch)
  vi.stubGlobal('hermesDesktop', createBrowserBridge({ token: 'fixture-token', authRequired: false }))

  const { installPluginSdk } = await import('./runtime')
  const { host, isBrowserClient } = await import('./index')
  installPluginSdk()
  expect(isBrowserClient()).toBe(true)

  await expect(host.skills.list('worker')).resolves.toEqual([{ name: 'fixture-skill', enabled: true }])
  const [url, options] = fetch.mock.calls.at(-1)! as unknown as [string, RequestInit]
  expect(new URL(url).pathname).toBe('/api/skills')
  expect(new URL(url).searchParams.get('profile')).toBe('worker')
  expect(new Headers(options.headers).get('X-Hermes-Session-Token')).toBe('fixture-token')

  const { discoverRuntimePlugins } = await import('@/contrib/runtime-loader')
  await expect(discoverRuntimePlugins()).resolves.toBeUndefined()
}, 60_000)
