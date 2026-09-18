import { resolveGatewayWsUrl } from '@hermes/shared'
import { afterEach, expect, it, vi } from 'vitest'

import { mediaExternalUrl } from '@/lib/media'
import { $connection } from '@/store/session'

import { createBrowserBridge } from './bridge'

afterEach(() => vi.unstubAllGlobals())

it('routes REST to the same backend, preserving explicit profile and rejecting other origins', async () => {
  const fetch = vi.fn().mockImplementation(async () => new Response('{"ok":true}'))
  vi.stubGlobal('fetch', fetch)
  const bridge = createBrowserBridge({ token: 'fixture-token', authRequired: false })

  await bridge.api({ path: '/api/config?profile=alpha', profile: 'beta', method: 'POST', body: { test: true } })
  const [url, init] = fetch.mock.calls[0]
  expect(new URL(url).origin).toBe(window.location.origin)
  expect(new URL(url).searchParams.get('profile')).toBe('alpha')
  expect(init.headers.get('X-Hermes-Session-Token')).toBe('fixture-token')
  expect(init.body).toBe('{"test":true}')
  await bridge.api({ path: '/api/config', profile: 'beta' })
  expect(new URL(fetch.mock.calls[1][0]).searchParams.get('profile')).toBe('beta')
  await expect(bridge.api({ path: 'https://elsewhere.invalid/api/config' })).rejects.toThrow('same-origin')
  await expect(bridge.api({ path: '/api/config', connectionId: 'other' })).rejects.toThrow('connection')
  expect(fetch).toHaveBeenCalledTimes(2)
})

it('mints a new cookie-authenticated ticket for each dial and surfaces rejection', async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(new Response('{"ticket":"first"}'))
    .mockResolvedValueOnce(new Response('{"ticket":"second"}'))
    .mockResolvedValueOnce(new Response('expired', { status: 401 }))

  vi.stubGlobal('fetch', fetch)
  const bridge = createBrowserBridge({ token: '', authRequired: true })

  expect((await bridge.getConnection('alpha')).sharedPrimary).toBe(true)
  expect(fetch).not.toHaveBeenCalled()
  const first = new URL(await bridge.getGatewayWsUrl('alpha'))
  const second = new URL(await bridge.getGatewayWsUrl('alpha'))
  expect(first.searchParams.get('ticket')).toBe('first')
  expect(second.searchParams.get('ticket')).toBe('second')
  expect(first.searchParams.get('profile')).toBe('alpha')
  expect(fetch.mock.calls[0][1].credentials).toBe('same-origin')
  const progress = vi.fn()
  const unsubscribe = bridge.onBootProgress(progress)
  await expect(bridge.getGatewayWsUrl()).rejects.toMatchObject({ needsOauthLogin: true })
  expect(progress).toHaveBeenCalledWith(
    expect.objectContaining({
      error: expect.stringContaining('Gateway sign-in required'),
      retryable: false
    })
  )
  unsubscribe()
})

it('refreshes a rotated loopback token before dialing without navigating or executing bootstrap scripts', async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(
      new Response(
        '<html><head><script>window.__HERMES_SESSION_TOKEN__="new-token";window.bootstrapExecuted=true;</script></head></html>'
      )
    )
    .mockResolvedValueOnce(new Response('{"ok":true}'))

  vi.stubGlobal('fetch', fetch)
  const bridge = createBrowserBridge({ token: 'old-token', authRequired: false })
  const conn = await bridge.getConnection('alpha')
  const url = new URL(await resolveGatewayWsUrl(bridge, conn))
  expect(url.searchParams.get('token')).toBe('new-token')
  $connection.set(conn)

  try {
    expect(new URL(mediaExternalUrl('/tmp/image.png')).searchParams.get('token')).toBe('new-token')
  } finally {
    $connection.set(null)
  }

  expect(url.searchParams.get('profile')).toBe('alpha')
  expect(fetch.mock.calls[0][1]).toMatchObject({ cache: 'no-store', credentials: 'same-origin', redirect: 'error' })
  expect((window as unknown as Record<string, unknown>).bootstrapExecuted).toBeUndefined()
  await bridge.api({ path: '/api/config' })
  expect(fetch.mock.calls[1][1].headers.get('X-Hermes-Session-Token')).toBe('new-token')
})

it.each([401, 403, 503])('only confirmed session rejection requests sign-in for HTTP %s', async status => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation(async () => new Response('rejected', { status }))
  )
  const bridge = createBrowserBridge({ token: '', authRequired: true })
  const progress = vi.fn()
  bridge.onBootProgress(progress)
  await expect(bridge.api({ path: '/api/config' })).rejects.toThrow()
  expect(progress).toHaveBeenCalledTimes(status === 401 ? 1 : 0)
  await expect(bridge.getGatewayWsUrl()).rejects.toThrow()
  expect((await bridge.getBootProgress()).retryable).toBe(status === 503)
})
