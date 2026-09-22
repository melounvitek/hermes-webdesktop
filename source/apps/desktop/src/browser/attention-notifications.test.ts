import { afterEach, expect, it, vi } from 'vitest'

import {
  $browserAttentionStatus,
  enableBrowserAttention,
  notifyBrowserAttention,
  setBrowserAttentionConnected
} from './attention-notifications'

afterEach(() => {
  window.dispatchEvent(new Event('pagehide'))
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it('releases the lock exactly at the memory bound even when the last request is focused or replayed', async () => {
  let released = false
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
