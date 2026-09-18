import { afterEach, describe, expect, it, vi } from 'vitest'

import { registerTerminalReader, setActiveTerminalId } from '@/app/right-sidebar/terminal/buffer'
import { $terminals, $activeTerminalId } from '@/app/right-sidebar/terminal/terminals'
import { $activeProfile } from '@/store/profile'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $toursEnabled } from '@/store/tours'

import { handleServerRequest } from './server-requests'
import type { ServerRequestContext } from './server-requests'

const deps = {
  activeSessionIdRef: { current: null },
  sessionInterrupted: () => false,
  updateSessionState: (_sessionId, update) => update(createClientSessionState('stored-session')),
  upsertToolCall: () => undefined
} as ServerRequestContext['deps']

function deliver(method: string, params: Record<string, unknown>, activeSessionId: null | string) {
  const respond = vi.fn()
  const fail = vi.fn()
  const handled = handleServerRequest({ fail, id: 'srq-1', method, params, profile: 'default', respond }, deps, activeSessionId)

  return { fail, handled, respond }
}

it('browser terminal reads fail closed across profiles and absent ownership, but allow the owner', () => {
  vi.stubEnv('VITE_BROWSER', '1')
  $activeProfile.set('beta')
  $terminals.set([{ id: 'b', profile: 'beta', title: 'B', auto: false, cwd: '', kind: 'user' }])
  $activeTerminalId.set('b')
  setActiveTerminalId('b')
  const read = vi.fn(() => ({ text: 'B secret' }) as never)
  const unregister = registerTerminalReader('b', read)
  const respond = vi.fn()
  const request = { fail: vi.fn(), id: 'read', method: 'terminal.read', params: {}, profile: 'alpha', respond }
  try {
    for (const profile of ['alpha', '']) {
      handleServerRequest({ ...request, profile }, deps, null)
      expect(respond).toHaveBeenLastCalledWith({ value: '' })
      expect(read).not.toHaveBeenCalled()
    }
    handleServerRequest({ ...request, profile: 'beta' }, deps, null)
    expect(read).toHaveBeenCalledOnce()
    expect(respond).toHaveBeenLastCalledWith({ value: JSON.stringify({ text: 'B secret' }) })
    read.mockClear()
    // Selection can advance before the workspace's reader subscription mounts.
    $terminals.set([{ ...$terminals.get()[0], id: 'new-tab' }])
    $activeTerminalId.set('new-tab')
    handleServerRequest({ ...request, profile: 'beta' }, deps, null)
    expect(respond).toHaveBeenLastCalledWith({ value: '' })
    expect(read).not.toHaveBeenCalled()
    $terminals.set([{ ...$terminals.get()[0], profile: undefined }])
    handleServerRequest({ ...request, profile: 'beta' }, deps, null)
    expect(respond).toHaveBeenLastCalledWith({ value: '' })
    expect(read).not.toHaveBeenCalled()
  } finally {
    unregister()
    setActiveTerminalId(null)
    $terminals.set([])
    $activeTerminalId.set(null)
    vi.unstubAllEnvs()
  }
})

describe('connection request routing', () => {
  it('does not route connection operations through the server-request rail', () => {
    const { handled, respond } = deliver(
      'connection',
      {
        deadline_at: 1_800_000_000,
        op_id: 'op-1',
        session_id: 'session-a',
        targets: [{ action: 'install', kind: 'mcp', name: 'linear' }],
        timeout_seconds: 60,
        tool_call_id: 'call-1'
      },
      'session-a'
    )

    expect(handled).toBe(false)
    expect(respond).not.toHaveBeenCalled()
  })
})

describe('preview action request routing', () => {
  it('leaves a scoped action request unanswered in a window showing another session', () => {
    const { handled, respond, fail } = deliver('preview.act', { action: 'elements', session_id: 'session-a' }, 'session-b')

    expect(handled).toBe(true)
    expect(respond).not.toHaveBeenCalled()
    expect(fail).not.toHaveBeenCalled()
  })

  it('fails fast for an unscoped request with no session in view', () => {
    const { respond } = deliver('preview.act', { action: 'elements' }, null)

    expect(respond).toHaveBeenCalledWith({
      value: JSON.stringify({
        error: 'The in-app browser only takes actions in the session the user is looking at.',
        success: false
      })
    })
  })
})

describe('tour request routing', () => {
  afterEach(() => {
    $toursEnabled.set(true)
  })

  it('leaves a scoped request unanswered in another session even when tours are disabled', () => {
    $toursEnabled.set(false)
    const { handled, respond } = deliver('tour', { action: 'discover', session_id: 'session-a' }, 'session-b')

    expect(handled).toBe(true)
    expect(respond).not.toHaveBeenCalled()
  })

  it('fails fast for an unscoped request with no session in view', () => {
    const { respond } = deliver('tour', { action: 'discover' }, null)

    expect(respond).toHaveBeenCalledWith({
      value: JSON.stringify({ error: 'Tours only run in the session the user is looking at.', success: false })
    })
  })
})
