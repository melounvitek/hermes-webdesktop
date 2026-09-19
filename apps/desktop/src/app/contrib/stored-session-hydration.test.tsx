// @vitest-environment jsdom
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { getLatestSessionMessages } from '@/hermes'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import { $todosBySession, clearSessionTodos, setSessionTodos } from '@/store/todos'

import { renderMessageStream } from '../session/hooks/use-message-stream/test-harness'

import { hydrateStoredSession } from './stored-session-hydration'

vi.mock('@/hermes', async actual => ({
  ...(await actual<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))
const SID = 'hydrate-runtime'
const STORED = 'hydrate-stored'
const user = (id: string): ChatMessage => ({ id, role: 'user', parts: [{ type: 'text', text: id }] })
const reply = (id: string): ChatMessage => ({ ...user(id), role: 'assistant' })

const rows = (texts: string[]) => ({
  session_id: STORED,
  messages: texts.map((text, i) => ({ id: i + 1, role: i % 2 ? 'assistant' : 'user', content: text }))
})

const todos = [{ id: 'live-task', content: 'new turn task', status: 'in_progress' as const }]

function mount() {
  const states = new Map([[SID, { ...createClientSessionState(STORED), busy: true, messages: [user('first')] }]])

  const update = (id: string, updater: (s: ClientSessionState) => ClientSessionState) => {
    const next = updater(states.get(id) ?? createClientSessionState(STORED))
    states.set(id, next)
    publishSessionState(id, next)

    return next
  }

  update(SID, s => s)

  const hydrate = () =>
    hydrateStoredSession({
      storedSessionId: STORED,
      runtimeSessionId: SID,
      profile: 'default',
      attempts: 1,
      updateSessionState: update
    })

  const invoked = vi.fn(hydrate)
  const stream = renderMessageStream(SID, { states, updateSessionState: update, hydrateFromStoredSession: invoked })

  const complete = () =>
    act(() =>
      stream.handleEvent({ type: 'message.complete', session_id: SID, payload: { text: 'first reply' } } as never)
    )

  return { stream, update, hydrate, invoked, complete, state: () => states.get(SID)! }
}

beforeEach(() => {
  clearAllSessionStates()
  clearSessionTodos(SID)
  vi.clearAllMocks()
  $activeSessionId.set(SID)
})
afterEach(() => {
  cleanup()
  clearAllSessionStates()
  clearSessionTodos(SID)
  $activeSessionId.set(null)
})

it('rejects an old real completion hydration after the subsequent turn finishes, including todos', async () => {
  let deliver!: (value: unknown) => void
  vi.mocked(getLatestSessionMessages).mockReturnValueOnce(
    new Promise(resolve => {
      deliver = resolve
    }) as never
  )
  const h = mount()
  h.complete()
  expect(h.invoked).toHaveBeenCalledTimes(1)
  expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
  h.update(SID, s => ({
    ...s,
    busy: true,
    awaitingResponse: true,
    interrupted: false,
    messages: [...s.messages, user('second')]
  }))
  h.update(SID, s => ({
    ...s,
    busy: false,
    awaitingResponse: false,
    sawAssistantPayload: true,
    messages: [...s.messages, reply('second reply')]
  }))
  setSessionTodos(SID, todos)
  const before = h.state().messages
  await act(async () => {
    deliver(rows(['first', 'first reply']))
    await h.invoked.mock.results[0].value
  })
  expect(h.state().messages).toBe(before)
  expect($todosBySession.get()[SID]).toEqual(todos)
  // Rejection is not a permanent latch: a later authoritative read converges.
  vi.mocked(getLatestSessionMessages).mockResolvedValueOnce(
    rows(['first', 'first reply', 'second', 'canonical reply']) as never
  )
  await h.hydrate()
  expect(h.state().messages.map(chatMessageText)).toEqual(['first', 'first reply', 'second', 'canonical reply'])
  expect($todosBySession.get()[SID]).toBeUndefined()
})

it.each(['busy', 'awaitingResponse', 'interrupted'] as const)(
  'does not start an old completion read against a newer %s owner',
  async flag => {
    const h = mount()
    h.update(SID, s => ({
      ...s,
      busy: flag !== 'awaitingResponse',
      awaitingResponse: flag === 'awaitingResponse',
      interrupted: flag === 'interrupted',
      messages: [user('second')]
    }))
    setSessionTodos(SID, todos)
    await h.hydrate()
    expect(getLatestSessionMessages).not.toHaveBeenCalled()
    expect($todosBySession.get()[SID]).toEqual(todos)
  }
)

it('rejects a held response after runtime rebinding or todo-only live progress', async () => {
  for (const change of ['owner', 'todo']) {
    let deliver!: (value: unknown) => void
    vi.mocked(getLatestSessionMessages).mockReturnValueOnce(
      new Promise(resolve => {
        deliver = resolve
      }) as never
    )
    const h = mount()
    h.complete()
    const before = h.state().messages

    if (change === 'owner') {
      h.update(SID, s => ({ ...s, storedSessionId: 'different-stored' }))
    } else {
      setSessionTodos(SID, todos)
    }

    const beforeTodos = $todosBySession.get()[SID]
    await act(async () => {
      deliver(rows(['stale', 'stale reply']))
      await h.invoked.mock.results[0].value
    })
    expect(h.state().messages).toBe(before)
    expect($todosBySession.get()[SID]).toBe(beforeTodos)
    cleanup()
  }
})

it('preserves older backfill, inline errors and the supplied exact owner on fresh hydration', async () => {
  const h = mount()
  const response = rows(['first', 'stored reply'])
  const converted = toChatMessages(response.messages as never)
  converted[1] = { ...converted[1], error: 'local failure' }
  h.update(SID, state => ({ ...state, busy: false, messages: [{ ...user('older page'), rowId: 0 }, ...converted] }))
  vi.mocked(getLatestSessionMessages).mockResolvedValueOnce(response as never)
  const owner = { connectionId: 'other-owner', profile: 'default' }
  await hydrateStoredSession({
    storedSessionId: STORED,
    runtimeSessionId: SID,
    profile: owner,
    attempts: 1,
    updateSessionState: h.update
  })
  expect(getLatestSessionMessages).toHaveBeenCalledWith(STORED, owner)
  expect(h.state().messages.map(chatMessageText)).toEqual(['older page', 'first', 'stored reply'])
  expect(h.state().messages.at(-1)?.error).toBe('local failure')
})

it('still hydrates a legitimate empty-stream completion and later authoritative empty transcript', async () => {
  const h = mount()
  vi.mocked(getLatestSessionMessages).mockResolvedValueOnce(rows(['first', 'persisted reply']) as never)
  h.complete()
  await act(async () => {
    await h.invoked.mock.results[0].value
  })
  expect(h.state().messages.map(chatMessageText)).toEqual(['first', 'persisted reply'])
  vi.mocked(getLatestSessionMessages).mockResolvedValueOnce(rows([]) as never)
  await h.hydrate()
  expect(h.state().messages).toEqual([])
})
