import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import {
  reconcileActiveTranscript,
  reconcileTileTranscripts,
  rehydrateLiveSessionStatuses
} from '@/app/contrib/hooks/use-background-sync'
import type { ClientSessionState } from '@/app/types'
import { getLatestSessionMessages } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $busy, $messages, setActiveSessionId } from '@/store/session'
import {
  $sessionStates,
  clearAllSessionStates,
  publishSessionState,
  reconcileBusyStatesOnReconnect,
  setSessionTileDelegate
} from '@/store/session-states'

import { useMessageStream } from './use-message-stream'
import { usePromptActions } from './use-prompt-actions'
import { waitForStoppedTurn } from './use-prompt-actions/utils'

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))

const SID = 'stopping-runtime'
const STORED = 'stopping-stored'

const partial = {
  id: 'partial',
  role: 'assistant' as const,
  parts: [{ type: 'text' as const, text: 'partial output' }],
  pending: true
}

function mountStop() {
  const activeSessionIdRef = { current: SID as string | null }
  const storedRef = { current: STORED as string | null }
  const busyRef = { current: true }

  const states = new Map<string, ClientSessionState>([
    [
      SID,
      { ...createClientSessionState(STORED), busy: true, turnLive: true, streamId: partial.id, messages: [partial] }
    ]
  ])

  const request = vi.fn(
    async (_method: string, _params?: Record<string, unknown>, _timeout?: number) => ({ status: 'streaming' }) as never
  )

  const update = (id: string, change: (state: ClientSessionState) => ClientSessionState) => {
    const next = change(states.get(id) ?? createClientSessionState(STORED))
    states.set(id, next)
    publishSessionState(id, next)
    busyRef.current = next.busy
    $busy.set(next.busy)
    $messages.set(next.messages)

    return next
  }

  update(SID, state => state)
  setActiveSessionId(SID)
  let actions!: ReturnType<typeof usePromptActions>
  let stream!: ReturnType<typeof useMessageStream>

  function Harness() {
    actions = usePromptActions({
      activeSessionId: SID,
      activeSessionIdRef,
      busyRef,
      selectedStoredSessionIdRef: storedRef,
      runtimeIdByStoredSessionIdRef: { current: new Map([[STORED, SID]]) },
      updateSessionState: update,
      requestGateway: request,
      branchCurrentSession: async () => true,
      createBackendSessionForSend: async () => SID,
      getRoutedStoredSessionId: () => STORED,
      getRuntimeIdForStoredSession: () => SID,
      getRouteToken: () => 'stable-route',
      handleSkinCommand: () => '',
      openMemoryGraph: () => {},
      refreshSessions: async () => {},
      resumeStoredSession: async () => {},
      startFreshSessionDraft: () => {},
      sttEnabled: false
    })
    stream = useMessageStream({
      activeSessionIdRef,
      sessionStateByRuntimeIdRef: { current: states },
      updateSessionState: update,
      queryClient: new QueryClient(),
      hydrateFromStoredSession: vi.fn(async () => {}),
      refreshHermesConfig: async () => {},
      refreshSessions: async () => {}
    })

    return null
  }

  render(<Harness />)

  return {
    actions,
    stream,
    request,
    states,
    update,
    busyRef,
    activeSessionIdRef,
    storedRef,
    state: () => states.get(SID)!,
    event: (type: 'message.complete' | 'session.info', payload: Record<string, unknown>) =>
      act(() => stream.handleGatewayEvent({ type, session_id: SID, payload } as never))
  }
}

beforeEach(() => {
  clearAllSessionStates()
  clearNotifications()
  vi.clearAllMocks()
})
afterEach(() => {
  cleanup()
  clearAllSessionStates()
  setSessionTileDelegate(null as never)
  setActiveSessionId(null)
  $busy.set(false)
})

it.each([false, true])(
  'keeps partial and next-send ownership until idle, including missing completion=%s',
  async missingComplete => {
    const h = mountStop()
    await act(() => h.actions.cancelRun())
    expect(h.state()).toMatchObject({ busy: true, interrupted: true, streamId: null })
    expect(h.state().messages).toMatchObject([{ id: 'partial', pending: false, parts: [{ text: 'partial output' }] }])

    for (const fromQueue of [false, true, true]) {
      await act(async () => {
        expect(await h.actions.submitText('next prompt', { fromQueue })).toBe(false)
      })
    }

    await act(async () => {
      expect(
        await h.actions.submitText('next prompt', {
          fromQueue: true,
          sessionId: 'stale-runtime',
          storedSessionId: STORED
        })
      ).toBe(false)
      expect(await h.actions.redirectPrompt('next prompt')).toBe(false)
      expect(await h.actions.injectHiddenPrompt('hidden intent')).toBe(false)
    })
    expect(h.request.mock.calls.map(call => call[0])).toEqual(['session.interrupt'])

    if (!missingComplete) {
      h.event('message.complete', { text: 'partial output', status: 'interrupted' })
      expect(h.state().busy).toBe(true)
    }

    h.event('session.info', { running: true })
    expect(h.state().busy).toBe(true)
    h.event('session.info', { running: false })
    expect(h.state().busy).toBe(false)
    await act(async () => {
      expect(await h.actions.submitText('next prompt')).toBe(true)
    })
    // A late old idle report must not settle the newly armed, not-yet-live turn.
    h.event('session.info', { running: false })
    expect(h.state()).toMatchObject({ busy: true, awaitingResponse: true })
    expect(h.state().messages.filter(m => m.role === 'user')).toHaveLength(1)
    expect(h.request.mock.calls.filter(call => call[0] === 'prompt.submit')).toHaveLength(1)
  }
)

it.each(['active', 'tile'])(
  'does not read a pre-persistence Stop snapshot for the %s surface; later authority still wins',
  async surface => {
    const h = mountStop()
    await act(() => h.actions.cancelRun())

    if (surface === 'tile') {
      setActiveSessionId('another-runtime')
    }

    const read = () =>
      surface === 'active'
        ? reconcileActiveTranscript({
            activeSessionIdRef: h.activeSessionIdRef,
            busyRef: h.busyRef,
            selectedStoredSessionIdRef: h.storedRef,
            requestSequenceRef: { current: 0 },
            signatureRef: { current: new Map() },
            resolveSession: () => ({ id: STORED, profile: 'default' }) as never,
            updateSessionState: h.update
          })
        : reconcileTileTranscripts({
            tiles: [{ runtimeId: SID, storedSessionId: STORED }],
            requestSequenceRef: { current: 0 },
            signatureRef: { current: new Map() },
            updateSessionState: h.update
          })

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [] } as never)
    await act(read)
    expect(getLatestSessionMessages).not.toHaveBeenCalled()
    expect(h.state().messages).toHaveLength(1)
    h.event('session.info', { running: false })
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [{ id: 42, role: 'assistant', content: 'partial output' }]
    } as never)
    await act(read)
    expect(h.state().messages).toHaveLength(1)
    expect(h.state().messages[0].parts).toMatchObject([{ text: 'partial output' }])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [] } as never)
    await act(read)
    expect(h.state().messages).toEqual([])
  }
)

it('does not treat reconnect as Stop settlement; a current stock status read recovers missing idle', async () => {
  const h = mountStop()
  setSessionTileDelegate({
    updateSession: h.update,
    retireBusyClaim: () => true
  } as never)
  const stale = $sessionStates.get()
  await act(() => h.actions.cancelRun())
  act(() => reconcileBusyStatesOnReconnect())
  expect(h.state().busy).toBe(true)
  expect($busy.get()).toBe(true)
  act(() => rehydrateLiveSessionStatuses({ sessions: [{ id: SID, session_key: STORED, status: 'starting' }] }))
  expect(h.state().busy).toBe(true)
  const idle = { sessions: [{ id: SID, session_key: STORED, status: 'idle' as const }] }
  act(() => rehydrateLiveSessionStatuses(idle, Date.now(), 'default', stale))
  expect(h.state().busy).toBe(true)
  act(() => rehydrateLiveSessionStatuses(idle))
  expect(h.state().busy).toBe(false)
  expect($busy.get()).toBe(false)
  expect(h.state().messages).toHaveLength(1)
  h.update(SID, state => ({ ...state, busy: true, interrupted: true }))
  act(() => rehydrateLiveSessionStatuses({ sessions: [] }))
  expect(h.state().busy).toBe(false)
  expect($busy.get()).toBe(false)
})

it.each(['edit', 'restore'])(
  'keeps %s intent behind Stop and aborts if its session changes while waiting',
  async operation => {
    const h = mountStop()
    await act(() => h.actions.cancelRun())
    const before = h.state().messages
    let settled = false

    const pending = (
      operation === 'edit'
        ? h.actions.editMessage({ content: [{ type: 'text', text: 'edited intent' }] } as never)
        : h.actions.restoreToMessage('earlier-user')
    ).then(() => {
      settled = true
    })

    await Promise.resolve()
    expect(settled).toBe(false)
    expect(h.state().messages).toBe(before)
    h.activeSessionIdRef.current = 'other-runtime'
    h.event('session.info', { running: false })
    await pending
    expect(h.request.mock.calls.map(call => call[0])).toEqual(['session.interrupt'])
  }
)

it('waits before edit/restore ownership and times out without dropping the Stop claim', async () => {
  const h = mountStop()
  await act(() => h.actions.cancelRun())
  let settled = false

  const wait = waitForStoppedTurn(SID).then(() => {
    settled = true
  })

  await Promise.resolve()
  expect(settled).toBe(false)
  h.event('message.complete', { text: 'partial output', status: 'interrupted' })
  await Promise.resolve()
  expect(settled).toBe(false)
  h.event('session.info', { running: false })
  await wait
  expect(settled).toBe(true)
  h.update(SID, s => ({ ...s, busy: true, interrupted: true }))
  vi.useFakeTimers()

  try {
    const pending = expect(waitForStoppedTurn(SID)).rejects.toThrow('Retry Stop or reconnect')
    await vi.advanceTimersByTimeAsync(15_000)
    await pending
    expect(h.state().busy).toBe(true)
    expect(h.state().messages).toHaveLength(1)
  } finally {
    vi.useRealTimers()
  }
})

it('keeps Stop retryable and the partial intact after an interrupt error, without permitting a queued send', async () => {
  const h = mountStop()
  h.request.mockRejectedValueOnce(new Error('interrupt transport failed'))
  await act(() => h.actions.cancelRun())
  expect($notifications.get()).not.toHaveLength(0)
  expect(h.state().busy).toBe(true)
  await act(async () => {
    expect(await h.actions.submitText('retained intent', { fromQueue: true })).toBe(false)
  })
  await act(() => h.actions.cancelRun())
  expect(h.request.mock.calls.filter(call => call[0] === 'session.interrupt')).toHaveLength(2)
  h.event('session.info', { running: false })
  expect(h.state().busy).toBe(false)
  expect($sessionStates.get()[SID].messages).toHaveLength(1)
})
