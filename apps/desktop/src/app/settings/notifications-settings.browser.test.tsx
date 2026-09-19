import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  handleServerRequest,
  type ServerRequestContext
} from '@/app/session/hooks/use-message-stream/gateway-event/server-requests'
import { createBrowserBridge } from '@/browser/bridge'
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
      params: { session_id: sessionId, question: 'PRIVATE QUESTION', command: 'PRIVATE COMMAND', site: 'PRIVATE SITE' },
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
    fireEvent.click(screen.getByRole('button', { name: 'Enable attention notifications in this tab' }))
  )
}

function mount() {
  render(<NotificationsSettings />)
}

beforeEach(() => {
  notices = []
  permission = 'granted'
  locked = false
  failConstruction = false
  requestPermission = vi.fn(async () => permission)
  requestLock = vi.fn(async (_name, options, callback) => {
    expect(options).toEqual({ mode: 'exclusive', ifAvailable: true })

    if (locked) {
      return callback(null)
    }

    locked = true

    try {
      return await callback({ name: 'test' })
    } finally {
      locked = false
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
  it('stays off with granted permission and legacy prefs; does not advertise other native kinds', async () => {
    mount()
    request('default-off')
    expect(notices).toHaveLength(0)
    expect(screen.getByRole('button', { name: 'Enable attention notifications in this tab' })).toBeTruthy()
    expect(screen.queryAllByRole('switch')).toHaveLength(0)
    expect(requestPermission).not.toHaveBeenCalled()
    await enable()
    request('fresh')
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
    fireEvent.click(screen.getByRole('button', { name: 'Enable attention notifications in this tab' }))
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

  it('refuses an occupied lock without queued takeover', async () => {
    locked = true
    mount()
    await enable()
    request('contended')
    expect(notices).toHaveLength(0)
    expect(screen.getByRole('status').textContent).toMatch(/another tab/i)
    locked = false
    request('no-takeover')
    expect(notices).toHaveLength(0)
  })

  it.each(['closed', 'rehome', 'pagehide', 'freeze', 'auth'])(
    'disables on %s, closes notices and ignores old clicks',
    async reason => {
      mount()
      await enable()
      request(`lifecycle-${reason}`)
      const click = notices[0].onclick
      await act(async () => {
        if (reason === 'closed') {
          reportPrimaryGatewayState('closed')
        } else if (reason === 'rehome') {
          setActiveProfile('b')
        } else if (reason === 'auth') {
          vi.mocked(fetch).mockResolvedValue(new Response('expired', { status: 401 }))
          await createBrowserBridge({ token: '', authRequired: true })
            .api({ path: '/api/config' })
            .catch(() => {})
        } else {
          ;(reason === 'freeze' ? document : window).dispatchEvent(new Event(reason))
        }
      })
      reportPrimaryGatewayState('open')
      request(`after-${reason}`)
      expect(notices).toHaveLength(1)
      expect(notices[0].close).toHaveBeenCalled()
      click?.(new Event('click'))
      expect(window.focus).not.toHaveBeenCalled()
      expect(locked).toBe(false)
      expect(screen.getByRole('button', { name: 'Enable attention notifications in this tab' })).toBeTruthy()
    }
  )

  it('does not enable from a late permission answer after lifecycle reset', async () => {
    let grant!: (v: NotificationPermission) => void
    permission = 'default'
    requestPermission.mockReturnValue(
      new Promise<NotificationPermission>(r => {
        grant = r
      })
    )
    mount()
    fireEvent.click(screen.getByRole('button', { name: 'Enable attention notifications in this tab' }))
    act(() => document.dispatchEvent(new Event('freeze')))
    await act(async () => {
      permission = 'granted'
      grant('granted')
    })
    request('late-permission')
    expect(notices).toHaveLength(0)
    expect(locked).toBe(false)
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
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Disable in this tab' })))
    await enable()
    request('before-disable')
    request('after-enable')
    expect(notices).toHaveLength(2)
    permission = 'denied'
    act(() => request('denied-in-app'))
    expect(notices).toHaveLength(2)
    expect(sessionClarifyRequest('owned-a').get()).toMatchObject({
      requestId: 'denied-in-app',
      question: 'PRIVATE QUESTION'
    })
  })

  it('keeps monitoring on same-route activation but disables when the primary socket is replaced', async () => {
    const gateway = { connectionState: 'open' } as HermesGateway

    try {
      setPrimaryGateway(gateway, 'default')
      await ensureGatewayForProfile('default')
      mount()
      await enable()
      await act(async () => ensureGatewayForProfile('default'))
      request('same-route')
      expect(notices).toHaveLength(1)
      await act(async () => setPrimaryGateway({ connectionState: 'open' } as HermesGateway, 'default'))
      request('replacement-route')
      expect(notices).toHaveLength(1)
      expect(locked).toBe(false)
    } finally {
      setPrimaryGateway(null, 'default')
    }
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
    expect(screen.getByRole('button', { name: 'Enable attention notifications in this tab' })).toBeTruthy()
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
