import { useStore } from '@nanostores/react'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { useSessionTileActions } from '@/app/chat/session-tile-actions'
import { resetLiveRuntimeTracking } from '@/app/contrib/hooks/use-background-sync'
import { useSessionTileDelegate } from '@/app/contrib/hooks/use-session-tile-delegate'
import { requestGatewayForAgent } from '@/store/gateway'
import { $profiles } from '@/store/profile'
import {
  $activeSessionId,
  $busy,
  $messages,
  _resetSessionOwnerHintsForTests,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSessions
} from '@/store/session'
import {
  $sessionStates,
  $sessionTiles,
  $workingSessionIds,
  clearAllSessionStates,
  sessionTileDelegate,
  setSessionTileDelegate
} from '@/store/session-states'

import { useMessageStream } from './use-message-stream'
import { usePromptActions } from './use-prompt-actions'
import {
  clearSingleFlightSessionResumeState,
  registerRecoveredRuntime
} from './use-prompt-actions/single-flight-resume'
import { waitForStoppedTurn } from './use-prompt-actions/utils'
import { resolveSessionOwner } from './use-session-actions/utils'
import { useSessionStateCache } from './use-session-state-cache'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  retainGatewayForSessionTurn: vi.fn(async () => () => {})
}))
vi.mock('@/app/session/hooks/use-session-actions/utils', async original => ({
  ...(await original<Record<string, unknown>>()),
  resolveSessionProfile: vi.fn(async () => 'default'),
  resolveSessionOwner: vi.fn(async () => undefined)
}))

const A = 'missing-stop-a'
const B = 'recovered-stop-b'
const C = 'unrelated-stop-c'
const STORED = 'stored-stop'
const owner = { connectionId: 'stop-source-a', profile: 'default', targetProfile: 'served-a' }

const partial = {
  id: 'partial',
  role: 'assistant' as const,
  parts: [{ type: 'text' as const, text: 'retained partial' }],
  pending: true
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(r => {
    resolve = r
  })

  return { promise, resolve }
}

function mount(surface: 'main' | 'tile') {
  setActiveSessionId(surface === 'main' ? A : C)
  $sessionTiles.set([{ runtimeId: A, storedSessionId: STORED, ownerRoute: owner }])
  const request = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({}))
  vi.mocked(requestGatewayForAgent).mockImplementation((connection, profile, method, params) => {
    expect([connection, profile]).toEqual([owner.connectionId, owner.profile])

    return request(method, params) as never
  })
  let cache!: ReturnType<typeof useSessionStateCache>
  let stream!: ReturnType<typeof useMessageStream>
  let actions!: Pick<ReturnType<typeof usePromptActions>, 'cancelRun' | 'submitText'>

  function Harness() {
    const active = useStore($activeSessionId)
    const tiles = useStore($sessionTiles)
    const busyRef = useRef(true)
    cache = useSessionStateCache({
      activeSessionId: active,
      busyRef,
      selectedStoredSessionId: active === C ? 'other-stored' : STORED,
      setAwaitingResponse,
      setBusy,
      setMessages
    })
    useSessionTileDelegate({
      archiveSession: async () => {},
      branchStoredSession: async () => {},
      executeSlashCommand: async () => {},
      removeSession: async () => {},
      requestGateway: request as never,
      runtimeIdByStoredSessionIdRef: cache.runtimeIdByStoredSessionIdRef,
      sessionStateByRuntimeIdRef: cache.sessionStateByRuntimeIdRef,
      updateSessionState: cache.updateSessionState
    })

    const common = {
      activeSessionId: active,
      activeSessionIdRef: cache.activeSessionIdRef,
      busyRef,
      selectedStoredSessionIdRef: cache.selectedStoredSessionIdRef,
      runtimeIdByStoredSessionIdRef: cache.runtimeIdByStoredSessionIdRef,
      updateSessionState: cache.updateSessionState,
      requestGateway: request as never,
      branchCurrentSession: async () => true,
      createBackendSessionForSend: async () => cache.activeSessionIdRef.current!,
      getRoutedStoredSessionId: () => cache.selectedStoredSessionIdRef.current,
      getRuntimeIdForStoredSession: (id: string) => cache.runtimeIdByStoredSessionIdRef.current.get(id) ?? null,
      getRouteToken: () => 'stable',
      handleSkinCommand: () => '',
      openMemoryGraph: () => {},
      refreshSessions: async () => {},
      resumeStoredSession: async () => {},
      startFreshSessionDraft: () => {},
      sttEnabled: false
    }

    const main = usePromptActions(common)

    const tile = useSessionTileActions({
      requestGateway: request as never,
      runtimeId: tiles[0]?.runtimeId ?? A,
      storedSessionId: tiles[0]?.storedSessionId ?? STORED,
      scope: MAIN_COMPOSER_SCOPE
    })

    actions = surface === 'main' ? main : tile
    stream = useMessageStream({
      activeSessionIdRef: cache.activeSessionIdRef,
      sessionStateByRuntimeIdRef: cache.sessionStateByRuntimeIdRef,
      updateSessionState: cache.updateSessionState,
      queryClient: new QueryClient(),
      hydrateFromStoredSession: async () => {},
      refreshHermesConfig: async () => {},
      refreshSessions: async () => {}
    })

    return null
  }

  render(<Harness />)
  act(() => {
    cache.updateSessionState(
      A,
      state => ({ ...state, busy: true, turnLive: true, streamId: partial.id, messages: [partial] }),
      STORED
    )
    cache.updateSessionState(
      C,
      state => ({ ...state, busy: true, messages: [{ ...partial, id: 'unrelated' }] }),
      'other-stored'
    )
  })

  return {
    request,
    get actions() {
      return actions
    },
    get cache() {
      return cache
    },
    event: (id: string, type: string, payload: Record<string, unknown>) =>
      act(() => stream.handleGatewayEvent({ session_id: id, type, payload } as never))
  }
}

beforeEach(() => {
  setSessions([])
  _resetSessionOwnerHintsForTests()
  clearAllSessionStates()
  resetLiveRuntimeTracking()
  clearSingleFlightSessionResumeState()
  vi.clearAllMocks()
  vi.mocked(resolveSessionOwner).mockResolvedValue(undefined)
})
afterEach(() => {
  cleanup()
  $profiles.set([])
  clearAllSessionStates()
  $sessionTiles.set([])
  setSessionTileDelegate(null as never)
  setActiveSessionId(null)
  $busy.set(false)
  $messages.set([])
})

it.each(['main', 'tile'] as const)(
  'preserves Stop ownership through missing-runtime recovery for %s',
  async surface => {
    const h = mount(surface)
    const resume = deferred<{ session_id: string }>()
    const interrupt = deferred<object>()
    h.request.mockImplementation(async (method, params) => {
      if (method === 'session.interrupt') {
        if (params?.session_id === A) {
          throw new Error('session not found')
        }

        return interrupt.promise
      }

      if (method === 'session.resume') {
        return resume.promise
      }

      if (method === 'session.active_list') {
        return { sessions: [{ id: B, status: 'streaming' }] }
      }

      return {}
    })
    let stopping!: Promise<void>
    act(() => {
      stopping = h.actions.cancelRun()
    })
    await waitFor(() =>
      expect(h.request).toHaveBeenCalledWith('session.resume', expect.objectContaining({ session_id: STORED }))
    )
    let released = false

    const waiting = waitForStoppedTurn(A).then(() => {
      released = true
    })

    await act(async () => {
      expect(await h.actions.submitText('queued intent', { fromQueue: true })).toBe(false)
    })
    expect(released).toBe(false)
    await act(async () => {
      resume.resolve({ session_id: B })
      await resume.promise
    })
    await waitFor(() => expect(h.request).toHaveBeenCalledWith('session.interrupt', { session_id: B }))
    expect($sessionStates.get()[B]).toMatchObject({
      storedSessionId: STORED,
      busy: true,
      interrupted: true,
      messages: [{ parts: [{ text: 'retained partial' }], pending: false }]
    })
    await act(async () => {
      interrupt.resolve({})
      await stopping
    })
    await waiting
    expect(released).toBe(true)
    expect($sessionStates.get()[A].busy).toBe(false)
    expect($workingSessionIds.get()).toContain(STORED)
    expect(h.cache.getRuntimeIdForStoredSession(STORED)).toBe(B)
    expect(await sessionTileDelegate()!.resumeTile(STORED)).toBe(B)
    expect(h.request.mock.calls.filter(([method]) => method === 'session.resume')).toHaveLength(1)
    h.event(B, 'message.complete', { status: 'interrupted', text: 'retained partial' })
    expect($sessionStates.get()[B].busy).toBe(true)
    await act(async () => {
      expect(await h.actions.submitText('queued intent', { fromQueue: true })).toBe(false)
    })
    h.event(B, 'session.info', { running: false })
    expect($workingSessionIds.get()).not.toContain(STORED)
    expect($workingSessionIds.get()).toContain('other-stored')
    await act(async () => {
      expect(await h.actions.submitText('queued intent', { fromQueue: true })).toBe(true)
    })
    expect(
      h.request.mock.calls.filter(([method]) => method === 'prompt.submit').map(([method, params]) => [method, params])
    ).toEqual([['prompt.submit', expect.objectContaining({ session_id: B, text: 'queued intent' })]])
  }
)

it.each(['main', 'tile'] as const)(
  'retires only proven absence on failed %s recovery and keeps retryable replacement work',
  async surface => {
    const h = mount(surface)
    h.request.mockRejectedValueOnce(new Error('transport failed'))
    await act(() => h.actions.cancelRun())
    expect($sessionStates.get()[A].busy).toBe(true)
    h.request.mockImplementation(async method => {
      if (method === 'session.interrupt') {
        throw new Error('session not found')
      }

      throw new Error('resume failed')
    })
    await act(() => h.actions.cancelRun())
    expect($sessionStates.get()[A]).toMatchObject({
      busy: false,
      messages: [{ parts: [{ text: 'retained partial' }] }]
    })
    expect($workingSessionIds.get()).not.toContain(STORED)
    expect($sessionStates.get()[C].busy).toBe(true)
  }
)

it.each(['main', 'tile'] as const)('advances past a missing cached replacement for %s', async surface => {
  const h = mount(surface)
  const fresh = 'fresh-stop-d'
  registerRecoveredRuntime(STORED, B)
  h.request.mockImplementation(async (method, params) => {
    if (method === 'session.interrupt' && (params?.session_id === A || params?.session_id === B)) {
      throw new Error('session not found')
    }

    if (method === 'session.resume') {
      return { session_id: fresh }
    }

    if (method === 'session.active_list') {
      return { sessions: [{ id: fresh, status: 'streaming' }] }
    }

    return {}
  })
  await act(() => h.actions.cancelRun())
  expect(h.cache.getRuntimeIdForStoredSession(STORED)).toBe(fresh)
  expect(surface === 'main' ? $activeSessionId.get() : $sessionTiles.get()[0].runtimeId).toBe(fresh)
  expect(await sessionTileDelegate()!.resumeTile(STORED)).toBe(fresh)
  expect(h.request.mock.calls.filter(([method]) => method === 'session.resume')).toHaveLength(1)
  expect($sessionStates.get()[A].busy).toBe(false)
  expect($sessionStates.get()[B].busy).toBe(false)
  expect($sessionStates.get()[fresh]).toMatchObject({ busy: true, interrupted: true })
  await act(async () => {
    expect(
      await h.actions.submitText('queued intent', { fromQueue: true, sessionId: A, storedSessionId: STORED })
    ).toBe(false)
  })
  h.event(fresh, 'session.info', { running: false })
  expect($workingSessionIds.get()).not.toContain(STORED)
})

it.each(['main', 'tile'] as const)('keeps a replacement blocked on retry failure for %s', async surface => {
  const h = mount(surface)
  h.request.mockImplementation(async (method, params) => {
    if (method === 'session.interrupt' && params?.session_id === A) {
      throw new Error('session not found')
    }

    if (method === 'session.resume') {
      return { session_id: B }
    }

    throw new Error('retry transport failed')
  })
  await act(() => h.actions.cancelRun())
  expect($sessionStates.get()[A].busy).toBe(false)
  expect($sessionStates.get()[B]).toMatchObject({ busy: true, interrupted: true })
  await act(async () => {
    expect(await h.actions.submitText('retained queue', { fromQueue: true })).toBe(false)
  })
  expect($workingSessionIds.get()).toContain(STORED)
  h.event(B, 'session.info', { running: false })
  expect($workingSessionIds.get()).not.toContain(STORED)
})

it.each(['main', 'tile'] as const)('ignores a held recovery idle response after newer work for %s', async surface => {
  const h = mount(surface)
  const status = deferred<{ sessions: Array<{ id: string; status: string }> }>()
  h.request.mockImplementation(async (method, params) => {
    if (method === 'session.interrupt' && params?.session_id === A) {
      throw new Error('session not found')
    }

    if (method === 'session.resume') {
      return { session_id: B }
    }

    if (method === 'session.active_list') {
      return status.promise
    }

    return {}
  })
  let stopping!: Promise<void>
  act(() => {
    stopping = h.actions.cancelRun()
  })
  await waitFor(() => expect(h.request).toHaveBeenCalledWith('session.active_list', {}))
  h.event(B, 'session.info', { running: false })
  await act(async () => {
    expect(await h.actions.submitText('next turn')).toBe(true)
  })
  const next = $sessionStates.get()[B]
  await act(async () => {
    status.resolve({ sessions: [{ id: B, status: 'idle' }] })
    await stopping
  })
  expect($sessionStates.get()[B]).toBe(next)
  expect(next.busy).toBe(true)
})

it.each(['main', 'tile'] as const)(
  'resolves an uncached owner and refuses a genuine owner miss for %s',
  async surface => {
    const h = mount(surface)
    act(() => {
      $sessionTiles.set([{ runtimeId: A, storedSessionId: STORED }])
      $profiles.set([{ name: 'default' }, { name: 'other' }] as never)
    })
    await act(() => h.actions.cancelRun())
    expect(h.request).not.toHaveBeenCalled()
    expect($sessionStates.get()[A].busy).toBe(true)
    vi.mocked(resolveSessionOwner).mockResolvedValue(owner)
    await act(() => h.actions.cancelRun())
    expect(resolveSessionOwner).toHaveBeenCalledWith(STORED)
    expect(requestGatewayForAgent).toHaveBeenCalledWith(owner.connectionId, owner.profile, 'session.interrupt', {
      session_id: A
    })
    expect($sessionStates.get()[A].busy).toBe(true)
  }
)

it.each(['main', 'tile'] as const)('does not steal a changed view or owner during %s recovery', async surface => {
  const h = mount(surface)
  const resume = deferred<{ session_id: string }>()
  h.request.mockImplementation(async (method, params) => {
    if (method === 'session.interrupt' && params?.session_id === A) {
      throw new Error('session not found')
    }

    if (method === 'session.resume') {
      return resume.promise
    }

    if (method === 'session.active_list') {
      return { sessions: [{ id: B, status: 'idle' }] }
    }

    return {}
  })
  let stopping!: Promise<void>
  act(() => {
    stopping = h.actions.cancelRun()
  })
  await waitFor(() => expect(h.request).toHaveBeenCalledWith('session.resume', expect.anything()))
  const unrelated = $sessionStates.get()[C]
  act(() => {
    setActiveSessionId(C)

    if (surface === 'tile') {
      $sessionTiles.set([
        {
          runtimeId: C,
          storedSessionId: 'other-stored',
          ownerRoute: { connectionId: 'stop-source-c', profile: 'default' }
        }
      ])
    }
  })
  await act(async () => {
    resume.resolve({ session_id: B })
    await stopping
  })
  expect($activeSessionId.get()).toBe(C)

  if (surface === 'tile') {
    expect($sessionTiles.get()[0].runtimeId).toBe(C)
  }

  expect($sessionStates.get()[C]).toBe(unrelated)
  expect($sessionStates.get()[A].busy).toBe(false)
  expect($sessionStates.get()[B].busy).toBe(false)
  expect($workingSessionIds.get()).not.toContain(STORED)
})
