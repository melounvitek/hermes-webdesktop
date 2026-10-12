import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  handleServerRequest,
  type ServerRequestContext
} from '@/app/session/hooks/use-message-stream/gateway-event/server-requests'
import { startBrowserAttention } from '@/browser/attention-notifications'
import type { HermesGateway } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { sessionClarifyRequest } from '@/store/clarify'
import { ensureGatewayForProfile, reportPrimaryGatewayState, setPrimaryGateway } from '@/store/gateway'
import { dispatchNativeNotification, setNativeNotifyEnabled } from '@/store/native-notifications'
import { setActiveProfile } from '@/store/profile'
import { $activeSessionId, setSessions } from '@/store/session'
import { stampSecondaryProfileOwner } from '@/store/session-event-provenance'
import { clearAllSessionStates, publishSessionState, recordSessionEventScope } from '@/store/session-states'
import { makeSessionInfo } from '@/test/session-info'

import { NotificationsSettings } from './notifications-settings'

// Mock only browser/transport effects, not the settings or request dispatcher.
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/completion-sound', () => ({ COMPLETION_SOUND_VARIANTS: [], previewCompletionSound: vi.fn() }))

let permission: NotificationPermission
let requestPermission: ReturnType<typeof vi.fn<() => Promise<NotificationPermission>>>
let requestLock: ReturnType<typeof vi.fn>
let locked: boolean
let waiting: (() => void)[]
let failConstruction: boolean
let notices: FakeNotification[]

class FakeNotification {
  static get permission() {
    return permission
  }
  static requestPermission() {
    return requestPermission()
  }
  onclick: ((event: Event) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  close = vi.fn()
  constructor(
    readonly title: string,
    readonly options?: NotificationOptions
  ) {
    if (failConstruction) {
      throw new Error('blocked')
    }

    notices.push(this)
  }
}

const deps: ServerRequestContext['deps'] = {
  activeSessionIdRef: { current: 'active' },
  sessionInterrupted: () => false,
  sessionStateByRuntimeIdRef: { current: new Map() },
  updateSessionState: (_id, update) => update(createClientSessionState('stored')),
  upsertToolCall: () => undefined
}

function request(id: string, sessionId = 'owned-a', method = 'clarify', replayed = false) {
  const respond = vi.fn()
  handleServerRequest(
    {
      id,
      method,
      profile: 'unproven-primary',
      replayed,
      params: {
        session_id: sessionId,
        questions: [{ qid: 'q1', question: 'PRIVATE QUESTION' }],
        command: 'PRIVATE COMMAND',
        site: 'PRIVATE SITE'
      },
      respond,
      fail: vi.fn()
    },
    deps,
    $activeSessionId.get()
  )
  expect(respond).not.toHaveBeenCalled()
}

async function enable() {
  await act(async () =>
    fireEvent.click(screen.getByRole('button', { name: 'Enable attention notifications' }))
  )
}

// Another tab on this origin holding the alert lock until it closes.
function otherTab(): () => Promise<void> {
  let close!: () => void
  void navigator.locks.request('hermes:browser-attention', {}, () => new Promise<void>(resolve => (close = resolve)))

  return async () => act(async () => close())
}

function otherTabStores(on: boolean) {
  const key = 'hermes:browser:global:attentionNotifications'
  localStorage.setItem(key, String(on))
  window.dispatchEvent(new StorageEvent('storage', { key }))
}

function mount() {
  render(<NotificationsSettings />)
}

beforeEach(() => {
  localStorage.clear()
  notices = []
  permission = 'granted'
  locked = false
  failConstruction = false
  requestPermission = vi.fn(async () => permission)
  waiting = []
  // One origin-wide exclusive lock, queued like navigator.locks.
  requestLock = vi.fn(async (_name, options: LockOptions, callback) => {
    if (locked) {
      if (options.ifAvailable) {
        return callback(null)
      }

      await new Promise<void>((resolve, reject) => {
        waiting.push(resolve)
        options.signal?.addEventListener('abort', () => {
          waiting.splice(waiting.indexOf(resolve), 1)
          reject(options.signal?.reason)
        })
      })
    }

    locked = true

    try {
      return await callback({ name: 'test' })
    } finally {
      const next = waiting.shift()

      if (next) {
        next()
      } else {
        locked = false
      }
    }
  })
  vi.stubGlobal('Notification', FakeNotification)
  vi.stubGlobal('isSecureContext', true)
  vi.stubGlobal(
    'fetch',
    vi.fn(() => Promise.reject(new Error('No network in notification tests')))
  )
  Object.defineProperty(navigator, 'locks', { configurable: true, value: { request: requestLock } })
  window.hermesDesktop = { browser: {}, notify: vi.fn(async () => false) } as unknown as Window['hermesDesktop']
  vi.spyOn(document, 'hasFocus').mockReturnValue(true)
  vi.spyOn(document, 'hidden', 'get').mockReturnValue(false)
  vi.spyOn(window, 'focus').mockImplementation(() => {})
  setActiveProfile('default')
  reportPrimaryGatewayState('open')
  $activeSessionId.set('active')
  setNativeNotifyEnabled(true)

  for (const profile of ['a', 'b']) {
    recordSessionEventScope(
      stampSecondaryProfileOwner({ type: 'message.start', session_id: `owned-${profile}` }, profile)
    )
  }
})
afterEach(async () => {
  act(() => window.dispatchEvent(new Event('pagehide')))
  cleanup()
  await Promise.resolve()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('browser attention settings and request admission', () => {
  it.each([undefined, 'alerts'])('keeps browser attention opt-in and native kinds hidden on page %s', async subpage => {
    render(<NotificationsSettings subpage={subpage} />)
    // Seen requests survive disable/remount for the document's lifetime.
    request(`default-off-${subpage ?? 'all'}`)
    expect(notices).toHaveLength(0)
    expect(screen.getByRole('button', { name: 'Enable attention notifications' })).toBeTruthy()
    expect(screen.queryAllByRole('switch')).toHaveLength(0)
    expect(requestPermission).not.toHaveBeenCalled()
    await enable()
    request(`fresh-${subpage ?? 'all'}`)
    expect(notices.map(n => [n.title, n.options])).toEqual([
      ['Hermes', { body: 'A conversation needs your attention.' }]
    ])
    expect(window.hermesDesktop.notify).not.toHaveBeenCalled()
  })

  it('requests permission synchronously only from Enable, before requesting the lock', async () => {
    permission = 'default'
    mount()
    requestPermission.mockImplementation(() => {
      expect(requestLock).not.toHaveBeenCalled()
      permission = 'granted'

      return Promise.resolve(permission)
    })
    fireEvent.click(screen.getByRole('button', { name: 'Enable attention notifications' }))
    expect(requestPermission).toHaveBeenCalledOnce()
    await act(async () => {})
    request('permission-fresh')
    expect(notices).toHaveLength(1)
  })

  it.each(['default', 'denied'] as const)(
    'leaves %s permission off without event/reconnect prompting',
    async result => {
      permission = 'default'
      requestPermission.mockResolvedValue(result)
      mount()
      await enable()
      request('no-prompt')
      reportPrimaryGatewayState('open')
      expect(requestPermission).toHaveBeenCalledOnce()
      expect(notices).toHaveLength(0)
      expect(screen.getByRole('status').textContent).toMatch(result === 'denied' ? /blocked/i : /not granted/i)
    }
  )

  it('does not prompt again when denied', async () => {
    permission = 'denied'
    mount()
    await enable()
    expect(requestPermission).not.toHaveBeenCalled()
    expect(notices).toHaveLength(0)
    expect(screen.getByRole('status').textContent).toMatch(/site settings/i)
  })

  it.each(['insecure', 'notification', 'locks'])('reports unavailable %s without prompting', async reason => {
    if (reason === 'insecure') {
      vi.stubGlobal('isSecureContext', false)
    }

    if (reason === 'notification') {
      vi.stubGlobal('Notification', undefined)
    }

    if (reason === 'locks') {
      Object.defineProperty(navigator, 'locks', { configurable: true, value: undefined })
    }

    mount()
    await enable()
    expect(requestPermission).not.toHaveBeenCalled()
    expect(notices).toHaveLength(0)
    expect(screen.getByRole('status').textContent).toMatch(/HTTPS|unavailable/i)
  })

  it('records focused/replayed requests, preserves distinct fresh requests and proven owners A/B/A', async () => {
    mount()
    await enable()
    $activeSessionId.set('owned-a')
    request('focused')
    expect(notices).toHaveLength(0)
    $activeSessionId.set('active')
    request('focused')
    request('replayed', 'owned-a', 'clarify', true)
    request('replayed')
    request('unknown', 'unknown')
    request('unscoped', '')
    expect(notices).toHaveLength(0)
    request('a1')
    request('a2')
    request('b1', 'owned-b')
    request('a3')
    expect(notices).toHaveLength(4)
    vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 10_000)
    request('a1')
    expect(notices).toHaveLength(4)
    vi.mocked(document.hasFocus).mockReturnValue(false)
    $activeSessionId.set('owned-a')
    request('away-active')
    expect(notices).toHaveLength(5)
  })

  it.each(['approval', 'clarify', 'sudo', 'secret', 'vault.code', 'vault.save_login', 'vault.unlock_prompt'])(
    'keeps %s private and clicks focus-only',
    async method => {
      mount()
      await enable()
      request(`private-${method}`, 'owned-a', method)
      expect(notices).toHaveLength(1)
      expect(notices[0].options).toEqual({ body: 'A conversation needs your attention.' })
      const event = new Event('click', { cancelable: true })
      notices[0].onclick?.(event)
      expect(window.focus).toHaveBeenCalledOnce()
      expect(notices[0].close).toHaveBeenCalledOnce()
      expect($activeSessionId.get()).toBe('active')
    }
  )

  it('uses the same permission and lock gate for the generic test and reports only a request', async () => {
    mount()
    expect(screen.getByRole('button', { name: 'Send test notification' })).toHaveProperty('disabled', true)
    await enable()
    act(() => fireEvent.click(screen.getByRole('button', { name: 'Send test notification' })))
    expect(notices).toHaveLength(1)
    expect(screen.getByRole('status').textContent).toMatch(/requested/i)
  })

  it('waits behind the alerting tab and takes over when it closes', async () => {
    const close = otherTab()
    mount()
    await enable()
    expect(screen.getByRole('status').textContent).toMatch(/another tab/i)
    expect(screen.getByRole('button', { name: 'Enable attention notifications' })).toHaveProperty('disabled', true)
    request('while-waiting')
    expect(notices).toHaveLength(0)
    await close()
    expect(screen.getByRole('status').textContent).toMatch(/^On/)
    request('after-takeover')
    expect(notices).toHaveLength(1)
  })

  it('follows Enable and Disable from another tab', async () => {
    mount()
    startBrowserAttention()
    await act(async () => otherTabStores(true))
    expect(screen.getByRole('status').textContent).toMatch(/^On/)
    request('enabled-elsewhere')
    expect(notices).toHaveLength(1)
    await act(async () => otherTabStores(false))
    expect(locked).toBe(false)
    expect(screen.getByRole('status').textContent).toMatch(/^Off/)
    expect(screen.queryByRole('button', { name: 'Disable' })).toBeNull()
  })

  it('stops waiting when another tab disables alerts', async () => {
    const close = otherTab()
    mount()
    await enable()
    await act(async () => otherTabStores(false))
    await close()
    expect(locked).toBe(false)
    request('after-disable-elsewhere')
    expect(notices).toHaveLength(0)
    expect(screen.getByRole('status').textContent).toMatch(/^Off/)
  })

  it('pauses while disconnected and keeps alerting after a reconnect, a replaced socket or a profile switch', async () => {
    const gateway = { connectionState: 'open' } as HermesGateway

    try {
      mount()
      await enable()
      await act(async () => reportPrimaryGatewayState('closed'))
      expect(screen.getByRole('status').textContent).toMatch(/^Paused/)
      request('while-closed')
      expect(notices).toHaveLength(0)
      await act(async () => reportPrimaryGatewayState('open'))
      request('after-reconnect')
      await act(async () => setActiveProfile('b'))
      request('after-profile-switch')
      await act(async () => setPrimaryGateway(gateway, 'default'))
      await act(async () => ensureGatewayForProfile('default'))
      request('after-socket-replacement')
      expect(notices).toHaveLength(3)
      expect(screen.getByRole('status').textContent).toMatch(/^On/)
    } finally {
      setPrimaryGateway(null, 'default')
    }
  })

  it.each([
    ['pagehide', 'pageshow'],
    ['freeze', 'resume']
  ] as const)('releases on %s, ignores old clicks and resumes on %s', async (leave, back) => {
    const target = leave === 'freeze' ? document : window
    mount()
    await enable()
    request(`before-${leave}`)
    const click = notices[0].onclick
    await act(async () => target.dispatchEvent(new Event(leave)))
    expect(notices[0].close).toHaveBeenCalled()
    expect(locked).toBe(false)
    click?.(new Event('click'))
    expect(window.focus).not.toHaveBeenCalled()
    await act(async () => target.dispatchEvent(new Event(back)))
    request(`after-${back}`)
    expect(notices).toHaveLength(2)
  })

  it.each(['here', 'in another tab', 'in another tab, reported after the answer'])(
    'does not enable from a late permission answer after Disable %s',
    async where => {
      let grant!: (v: NotificationPermission) => void
      mount()
      await enable()
      // A remembered opt-in whose permission was reset offers Enable and Disable together.
      permission = 'default'
      await act(async () => window.dispatchEvent(new Event('focus')))
      requestPermission.mockReturnValue(
        new Promise<NotificationPermission>(r => {
          grant = r
        })
      )
      fireEvent.click(screen.getByRole('button', { name: 'Enable attention notifications' }))
      const key = 'hermes:browser:global:attentionNotifications'
      await act(async () => {
        if (where === 'here') {
          fireEvent.click(screen.getByRole('button', { name: 'Disable' }))
        } else if (where === 'in another tab') {
          otherTabStores(false)
        } else {
          localStorage.removeItem(key)
        }
      })
      await act(async () => {
        permission = 'granted'
        grant('granted')
      })
      // Storage events are asynchronous and can arrive after the permission answer.
      await act(async () => window.dispatchEvent(new StorageEvent('storage', { key })))
      request('late-permission')
      expect(notices).toHaveLength(0)
      expect(locked).toBe(false)
      expect(screen.getByRole('status').textContent).toMatch(/^Off/)
    }
  )

  it('keeps an Enable that is answered after a reconnect', async () => {
    let grant!: (v: NotificationPermission) => void
    mount()
    await enable()
    permission = 'default'
    await act(async () => window.dispatchEvent(new Event('focus')))
    requestPermission.mockReturnValue(
      new Promise<NotificationPermission>(r => {
        grant = r
      })
    )
    fireEvent.click(screen.getByRole('button', { name: 'Enable attention notifications' }))
    await act(async () => reportPrimaryGatewayState('closed'))
    await act(async () => reportPrimaryGatewayState('open'))
    await act(async () => {
      permission = 'granted'
      grant('granted')
    })
    expect(screen.getByRole('status').textContent).toMatch(/^On/)
    request('answered-after-reconnect')
    expect(notices).toHaveLength(1)
  })

  it('removes the stored opt-in on Disable', async () => {
    mount()
    await enable()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Disable' })))
    expect(localStorage.getItem('hermes:browser:global:attentionNotifications')).toBeNull()
  })

  it('keeps excluded native kinds inert and does not route connection.request through the attention door', async () => {
    mount()
    await enable()

    for (const kind of ['turnDone', 'turnError', 'backgroundDone', 'credits', 'plugin'] as const) {
      dispatchNativeNotification({ kind, global: true, title: 'PRIVATE', body: 'PRIVATE' })
    }

    expect(
      handleServerRequest(
        {
          id: 'excluded',
          method: 'connection.request',
          params: { session_id: 'owned-a' },
          profile: 'a',
          respond: vi.fn(),
          fail: vi.fn()
        },
        deps,
        'active'
      )
    ).toBe(false)
    expect(notices).toHaveLength(0)
  })

  it('rechecks permission on focus without prompting and refuses an unproven primary-only profile', async () => {
    recordSessionEventScope({ session_id: 'unproven', profile: 'a' })
    mount()
    await enable()
    request('unproven-owner', 'unproven')
    expect(notices).toHaveLength(0)
    await act(async () => {
      permission = 'denied'
      window.dispatchEvent(new Event('focus'))
    })
    expect(locked).toBe(false)
    expect(requestPermission).not.toHaveBeenCalled()
  })

  it('does not make an observed unknown-owner request fresh after its owner is learned', async () => {
    mount()
    await enable()
    request('learned-owner', 'learned-runtime')
    recordSessionEventScope(stampSecondaryProfileOwner({ type: 'message.start', session_id: 'learned-runtime' }, 'a'))
    request('learned-owner', 'learned-runtime')
    expect(notices).toHaveLength(0)
    request('new-known-owner', 'learned-runtime')
    expect(notices).toHaveLength(1)
  })

  it('retains seen requests over explicit disable/re-enable and leaves denied input actionable in-app', async () => {
    mount()
    await enable()
    request('before-disable')
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Disable' })))
    await enable()
    request('before-disable')
    request('after-enable')
    expect(notices).toHaveLength(2)
    permission = 'denied'
    act(() => request('denied-in-app'))
    expect(notices).toHaveLength(2)
    expect(sessionClarifyRequest('owned-a').get()).toMatchObject({
      requestId: 'denied-in-app',
      questions: [{ question: 'PRIVATE QUESTION' }]
    })
  })

  it('ignores a late lock callback after disable', async () => {
    let grant!: (lock: { name: string }) => Promise<void>
    requestLock.mockImplementation((_name, _options, callback) => {
      grant = callback

      return Promise.resolve()
    })
    mount()
    await enable()
    act(() => window.dispatchEvent(new Event('pagehide')))
    await act(async () => grant({ name: 'late' }))
    request('late-lock')
    expect(notices).toHaveLength(0)
  })

  it.each(['constructor', 'revoked', 'error'])('fails honestly on %s and releases ownership', async failure => {
    mount()
    await enable()

    if (failure === 'constructor') {
      failConstruction = true
    }

    if (failure === 'revoked') {
      permission = 'denied'
    }

    act(() => request(`failure-${failure}`))

    if (failure === 'error') {
      act(() => notices[0].onerror?.())
    }

    await act(async () => {})
    expect(locked).toBe(false)
    expect(screen.getByRole('status').textContent).toMatch(/blocked|failed/i)
    expect(screen.getByRole('button', { name: 'Enable attention notifications' })).toBeTruthy()
  })

  it('remembers Enable for the next document without prompting again', async () => {
    permission = 'default'
    requestPermission.mockImplementation(async () => (permission = 'granted'))
    mount()
    await enable()
    await act(async () => window.dispatchEvent(new Event('pagehide')))
    await act(async () => startBrowserAttention())
    expect(requestPermission).toHaveBeenCalledOnce()
    expect(screen.getByRole('status').textContent).toMatch(/^On/)
    request('next-document')
    expect(notices).toHaveLength(1)
  })

  it('forgets the opt-in after Disable', async () => {
    mount()
    await enable()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Disable' })))
    expect(screen.getByRole('status').textContent).toMatch(/^Off/)
    expect(screen.queryByRole('button', { name: 'Disable' })).toBeNull()
    await act(async () => window.dispatchEvent(new Event('pagehide')))
    await act(async () => startBrowserAttention())
    request('after-disable')
    expect(notices).toHaveLength(0)
  })

  it('offers Disable for a remembered opt-in whose permission was reset', async () => {
    mount()
    await enable()
    await act(async () => window.dispatchEvent(new Event('pagehide')))
    permission = 'default'
    await act(async () => startBrowserAttention())
    expect(requestPermission).not.toHaveBeenCalled()
    expect(screen.getByRole('status').textContent).toMatch(/not granted/i)
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Disable' })))
    expect(screen.getByRole('status').textContent).toMatch(/^Off/)
    permission = 'granted'
    await act(async () => startBrowserAttention())
    expect(screen.getByRole('status').textContent).toMatch(/^Off/)
  })

  it('fails closed on conflicting stored owners and row-order reversal, but admits a unique shared-primary owner', async () => {
    publishSessionState('collision-runtime', createClientSessionState('collision-stored'))
    const a = makeSessionInfo({ id: 'collision-stored', profile: 'a' })
    const b = makeSessionInfo({ id: 'collision-stored', profile: 'b' })

    try {
      mount()
      await enable()
      setSessions([a, b])
      request('collision-request', 'collision-runtime')
      setSessions([b, a])
      request('collision-request', 'collision-runtime')
      expect(notices).toHaveLength(0)
      setSessions([a])
      request('unique-owner', 'collision-runtime')
      expect(notices).toHaveLength(1)
    } finally {
      setSessions([])
      clearAllSessionStates()
    }
  })
})
