import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as Attention from './attention-notifications'

// Each test is a new document: module state such as the sign-in pause lasts until reload.
let attention: typeof Attention
let released: boolean

beforeEach(async () => {
  vi.resetModules()
  localStorage.clear()
  released = false
  vi.stubGlobal('isSecureContext', true)
  vi.stubGlobal('Notification', Object.assign(vi.fn(), { permission: 'granted', requestPermission: vi.fn() }))
  window.hermesDesktop = { browser: {} } as Window['hermesDesktop']
  Object.defineProperty(navigator, 'locks', {
    configurable: true,
    value: {
      request: async (_name: string, _options: unknown, cb: (lock: object) => Promise<void>) => {
        await cb({})
        released = true
      }
    }
  })
  vi.spyOn(document, 'hidden', 'get').mockReturnValue(false)
  vi.spyOn(document, 'hasFocus').mockReturnValue(true)
  attention = await import('./attention-notifications')
})

afterEach(() => {
  window.dispatchEvent(new Event('pagehide'))
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it('releases the lock exactly at the memory bound even when the last request is focused or replayed', async () => {
  const { $browserAttentionStatus, enableBrowserAttention, notifyBrowserAttention, setBrowserAttentionConnected } =
    attention

  setBrowserAttentionConnected(true)
  await enableBrowserAttention()

  for (let i = 0; i < 1024; i++) {
    notifyBrowserAttention({
      id: String(i),
      method: 'clarify',
      sessionId: 's',
      owned: true,
      active: true,
      replayed: i === 1023
    })
  }

  await Promise.resolve()
  expect($browserAttentionStatus.get()).toBe('limit')
  await vi.waitFor(() => expect(released).toBe(true))
  notifyBrowserAttention({ id: '1023', method: 'clarify', sessionId: 's', owned: true, active: false })
  await enableBrowserAttention()
  expect($browserAttentionStatus.get()).toBe('limit')
  expect(Notification).not.toHaveBeenCalled()
})

it('pauses after a failed sign-in until the page reloads, keeping the opt-in', async () => {
  const { $browserAttentionStatus, $browserAttentionWanted, enableBrowserAttention, notifyBrowserAttention } =
    attention

  const { createBrowserBridge } = await import('./bridge')
  attention.setBrowserAttentionConnected(true)
  await enableBrowserAttention()
  vi.stubGlobal('fetch', vi.fn(async () => new Response('expired', { status: 401 })))
  await createBrowserBridge({ token: '', authRequired: true })
    .api({ path: '/api/config' })
    .catch(() => {})
  // The socket can stay open after a REST sign-in failure.
  attention.setBrowserAttentionConnected(true)
  notifyBrowserAttention({ id: 'after-auth', method: 'clarify', sessionId: 's', owned: true, active: false })
  await vi.waitFor(() => expect(released).toBe(true))
  expect(Notification).not.toHaveBeenCalled()
  expect($browserAttentionStatus.get()).toBe('signedOut')
  expect($browserAttentionWanted.get()).toBe(true)
})
