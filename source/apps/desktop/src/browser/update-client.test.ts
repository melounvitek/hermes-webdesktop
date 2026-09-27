/** @vitest-environment jsdom
 * @vitest-environment-options {"url":"https://browser.example/"}
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  applyBrowserUpdate,
  BrowserUpdateError,
  type BrowserUpdateOffer,
  type BrowserUpdatePhase,
  checkBrowserUpdateSession,
  prepareBrowserUpdate,
  readBrowserUpdateStatus
} from './update-client'

const offer: BrowserUpdateOffer = {
  id: 'a'.repeat(32),
  current_release: 'browser-1',
  target_release: 'browser-2',
  expires_at: 1, // Expiry is a UI apply guard, not a response parsing failure.
  warning: 'Restart disconnects browser users. Rollback may be required.',
  tested_backend: 'b'.repeat(40),
  compatibility: 'not-exercised'
}

const offered = { capabilities: ['install'], phase: 'offered', offer }
const identity = { provider: 'password', user_id: 'user', org_id: null, expires_at: 123 }
const fetchMock = vi.fn<typeof fetch>()

function json(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json; charset=utf-8' } })
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  window.history.replaceState(null, '', '/')
})

describe('browser update transport', () => {
  it('uses only the dedicated cookie endpoints and never invokes native/backend update APIs', async () => {
    const nativeCall = vi.fn(() => {
      throw new Error('Native API must not be used')
    })

    vi.stubGlobal('hermes', { api: nativeCall, update: nativeCall, installUpdate: nativeCall })
    fetchMock.mockImplementation(async () => json(offered))
    expect(await readBrowserUpdateStatus()).toEqual(offered)
    await prepareBrowserUpdate()
    await applyBrowserUpdate(offer.id)
    expect(fetchMock.mock.calls.map(([url]) => String(url))).toEqual([
      '/browser-updater/api/status',
      '/browser-updater/api/offer',
      '/browser-updater/api/apply'
    ])

    for (const [, options] of fetchMock.mock.calls) {
      expect(options).toMatchObject({
        credentials: 'same-origin',
        mode: 'same-origin',
        redirect: 'error',
        cache: 'no-store'
      })
      expect(options?.signal).toBeInstanceOf(AbortSignal)
      const headers = new Headers(options?.headers)
      expect(headers.has('Authorization')).toBe(false)
      expect(headers.has('X-Hermes-Session-Token')).toBe(false)
    }

    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: 'POST', body: '{}' })
    expect(fetchMock.mock.calls[1][1]).toMatchObject({ method: 'POST', body: '{}' })
    expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get('Content-Type')).toBe('application/json')
    expect(JSON.parse(fetchMock.mock.calls[2][1]?.body as string)).toEqual({
      offer_id: offer.id,
      confirm_restart_and_rollback: true
    })
    expect(nativeCall).not.toHaveBeenCalled()
  })

  it.each([
    [404, 'unsupported'],
    [401, 'forbidden'],
    [403, 'forbidden'],
    [409, 'conflict'],
    [503, 'network']
  ] as const)('classifies HTTP %i without parsing HTML or retrying mutations', async (status, kind) => {
    fetchMock.mockResolvedValue(new Response('<html>private proxy details</html>', { status }))
    const error = await prepareBrowserUpdate().catch(error => error)
    expect(error).toBeInstanceOf(BrowserUpdateError)
    expect(error.kind).toBe(kind)
    expect(error.message).not.toContain('private proxy details')
    expect(error.offer).toBeUndefined()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('does not retry or check the session after a network/redirect rejection', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(applyBrowserUpdate(offer.id)).rejects.toMatchObject({ kind: 'network' })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('refuses invalid offer IDs without sending a mutation', async () => {
    await expect(applyBrowserUpdate('../other')).rejects.toMatchObject({ kind: 'invalid' })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('rejects an apply response for a different offer', async () => {
    const other = { ...offer, id: 'c'.repeat(32) }
    fetchMock.mockResolvedValue(
      json({ ...offered, phase: 'stopping', offer: other, job: { id: other.id, phase: 'stopping' } })
    )
    await expect(applyBrowserUpdate(offer.id)).rejects.toMatchObject({ kind: 'invalid' })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('cancels an unread HTTP error body instead of downloading proxy markup', async () => {
    const cancel = vi.fn()
    fetchMock.mockResolvedValue(new Response(new ReadableStream({ cancel }), { status: 503 }))
    await expect(readBrowserUpdateStatus()).rejects.toMatchObject({ kind: 'network' })
    expect(cancel).toHaveBeenCalled()
  })

  it('refuses HTTP before any request', async () => {
    vi.stubGlobal('window', { location: new URL('http://browser.example/') })
    await expect(readBrowserUpdateStatus()).rejects.toMatchObject({ kind: 'insecure' })
    await expect(prepareBrowserUpdate()).rejects.toMatchObject({ kind: 'insecure' })
    await expect(applyBrowserUpdate(offer.id)).rejects.toMatchObject({ kind: 'insecure' })
    await expect(checkBrowserUpdateSession()).rejects.toMatchObject({ kind: 'insecure' })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses base-prefix deployments but permits hash routes at the root', async () => {
    window.history.replaceState(null, '', '/prefix/')
    await expect(readBrowserUpdateStatus()).rejects.toMatchObject({ kind: 'unsupported' })
    expect(fetchMock).not.toHaveBeenCalled()
    window.history.replaceState(null, '', '/#/settings')
    fetchMock.mockResolvedValue(json({ capabilities: [] }))
    await expect(readBrowserUpdateStatus()).resolves.toEqual({ capabilities: [] })
  })
})

describe('snapshot validation', () => {
  const confirmed: BrowserUpdatePhase[] = [
    'stopping',
    'switching',
    'starting',
    'rollback_stopping',
    'rollback_switching',
    'rollback_starting',
    'succeeded',
    'rolled_back',
    'failed',
    'recovery_required'
  ]

  const valid = [
    { capabilities: [] },
    ...['idle', 'preparing', 'failed', 'recovery_required'].map(phase => ({ capabilities: ['install'], phase })),
    offered,
    { ...offered, phase: 'current' },
    { ...offered, phase: 'failed', error: 'Preparation failed.' },
    { ...offered, phase: 'recovery_required' },
    ...confirmed.map(phase => ({ ...offered, phase, job: { id: offer.id, phase } }))
  ]

  it.each(valid)('accepts helper state %# including expired metadata', async value => {
    fetchMock.mockResolvedValue(json(value))
    await expect(readBrowserUpdateStatus()).resolves.toEqual(value)
  })

  const invalid = [
    null,
    [],
    {},
    { capabilities: ['unknown'] },
    { capabilities: ['install', 'install'], phase: 'idle' },
    { capabilities: [], phase: 'idle' },
    { capabilities: ['install'] },
    { ...offered, phase: 'unknown' },
    { ...offered, phase: undefined },
    { ...offered, phase: 'idle' },
    { ...offered, phase: 'preparing' },
    { ...offered, offer: null },
    { ...offered, offer: undefined },
    { ...offered, offer: { ...offer, id: 'short' } },
    { ...offered, offer: { ...offer, id: `${offer.id}\n` } },
    { ...offered, offer: { ...offer, tested_backend: `${offer.tested_backend}\n` } },
    { ...offered, offer: { ...offer, target_release: `${offer.target_release}\n` } },
    { ...offered, offer: { ...offer, current_release: '<b>version</b>' } },
    { ...offered, offer: { ...offer, target_release: '' } },
    { ...offered, offer: { ...offer, tested_backend: 'b'.repeat(39) } },
    { ...offered, offer: { ...offer, expires_at: null } },
    { ...offered, offer: { ...offer, compatibility: 'verified' } },
    { ...offered, offer: { ...offer, warning: '<html>Restart</html>' } },
    { ...offered, offer: { ...offer, warning: 'x'.repeat(4097) } },
    { ...offered, error: 'bad\u0000text' },
    { ...offered, error: '<script>bad()</script>' },
    { ...offered, error: 'x'.repeat(4097) },
    { ...offered, phase: 'stopping' },
    { ...offered, phase: 'stopping', job: { id: 'c'.repeat(32), phase: 'stopping' } },
    { ...offered, phase: 'stopping', job: { id: offer.id, phase: 'starting' } },
    { ...offered, job: { id: offer.id, phase: 'offered' } },
    { capabilities: ['install'], phase: 'failed', job: { id: offer.id, phase: 'failed' } }
  ]

  it.each(invalid)('rejects malformed/actionably inconsistent state %#', async value => {
    fetchMock.mockResolvedValue(json(value))
    const error = await readBrowserUpdateStatus().catch(error => error)
    expect(error).toBeInstanceOf(BrowserUpdateError)
    expect(error).toMatchObject({ kind: 'invalid' })
    expect(error).not.toHaveProperty('offer')
    expect(error).not.toHaveProperty('job')
  })

  it.each([
    new Response('<html>SPA fallback</html>', { headers: { 'Content-Type': 'text/html' } }),
    new Response('{', { headers: { 'Content-Type': 'application/json' } }),
    new Response(JSON.stringify(offered)),
    new Response(JSON.stringify(offered).replace('"expires_at":1', '"expires_at":1e999'), {
      headers: { 'Content-Type': 'application/json' }
    }),
    new Response(' '.repeat(32769), { headers: { 'Content-Type': 'application/json' } })
  ])('rejects non-JSON, invalid JSON, nonfinite expiry, and oversized bodies %#', async response => {
    fetchMock.mockResolvedValue(response)
    await expect(readBrowserUpdateStatus()).rejects.toMatchObject({ kind: 'invalid' })
  })

  it('bounds streamed bodies without relying on Content-Length', async () => {
    const cancel = vi.fn()

    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(32769))
      },
      cancel
    })

    fetchMock.mockResolvedValue(new Response(stream, { headers: { 'Content-Type': 'application/json' } }))
    await expect(readBrowserUpdateStatus()).rejects.toMatchObject({ kind: 'invalid' })
    expect(cancel).toHaveBeenCalled()
  })
})

describe('request lifetime', () => {
  it('times out before headers without retrying a mutation', async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation(() => new Promise(() => {}))
    const result = expect(prepareBrowserUpdate()).rejects.toMatchObject({ kind: 'network' })
    await vi.advanceTimersByTimeAsync(30_000)
    await result
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('times out while reading the body, aborts transport, and releases the stream', async () => {
    vi.useFakeTimers()
    const cancel = vi.fn()
    fetchMock.mockResolvedValue(
      new Response(new ReadableStream({ cancel }), { headers: { 'Content-Type': 'application/json' } })
    )
    const result = expect(readBrowserUpdateStatus()).rejects.toMatchObject({ kind: 'network' })
    await vi.advanceTimersByTimeAsync(30_000)
    await result
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true)
    expect(cancel).toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('cancels before fetch and while reading, without retaining timers', async () => {
    vi.useFakeTimers()
    const controller = new AbortController()
    controller.abort()
    await expect(prepareBrowserUpdate(controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
    expect(fetchMock).not.toHaveBeenCalled()
    const next = new AbortController()
    fetchMock.mockResolvedValue(new Response(new ReadableStream(), { headers: { 'Content-Type': 'application/json' } }))
    const result = expect(readBrowserUpdateStatus(next.signal)).rejects.toMatchObject({ name: 'AbortError' })
    await vi.advanceTimersByTimeAsync(0)
    next.abort()
    await result
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true)
    expect(vi.getTimerCount()).toBe(0)
  })
})

describe('explicit session check', () => {
  it.each([null, '', 'organization'])(
    'checks stock identity with org %j, not administrator role, without applying',
    async org_id => {
      fetchMock.mockResolvedValue(json({ ...identity, org_id }))
      await expect(checkBrowserUpdateSession()).resolves.toBeUndefined()
      expect(fetchMock).toHaveBeenCalledTimes(1)
      expect(fetchMock.mock.calls[0]).toEqual([
        '/api/auth/me',
        expect.objectContaining({
          method: 'GET',
          credentials: 'same-origin',
          mode: 'same-origin',
          redirect: 'error',
          cache: 'no-store'
        })
      ])
    }
  )
  it.each([
    {},
    { role: 'admin' },
    { ...identity, provider: '' },
    { ...identity, user_id: 7 },
    { ...identity, org_id: undefined }
  ])('rejects missing/malformed identity %#', async value => {
    fetchMock.mockResolvedValue(json(value))
    await expect(checkBrowserUpdateSession()).rejects.toMatchObject({ kind: 'invalid' })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
