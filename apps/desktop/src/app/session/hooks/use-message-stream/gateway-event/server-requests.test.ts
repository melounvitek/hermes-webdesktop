import { waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registerPreviewNav } from '@/app/chat/right-rail/preview-nav'
import { registerPreviewPageReader } from '@/app/chat/right-rail/preview-reader'
import { registerTerminalReader, setActiveTerminalId } from '@/app/right-sidebar/terminal/buffer'
import { $activeTerminalId, $terminals } from '@/app/right-sidebar/terminal/terminals'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $previewTabs, closeRightRail, openPreview } from '@/store/preview'
import { $activeProfile } from '@/store/profile'
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
  const handled = handleServerRequest(
    { fail, id: 'srq-1', method, params, profile: 'default', respond },
    deps,
    activeSessionId
  )

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
    const { handled, respond, fail } = deliver(
      'preview.act',
      { action: 'elements', session_id: 'session-a' },
      'session-b'
    )

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

describe('browser preview requests', () => {
  const disposers: (() => void)[] = []
  beforeEach(() => {
    vi.stubGlobal('hermesDesktop', { browser: { authRequired: false, signIn: vi.fn() } })
    closeRightRail()
  })
  afterEach(() => {
    disposers.splice(0).forEach(dispose => dispose())
    closeRightRail()
    vi.unstubAllGlobals()
  })

  it.each(['preview.read', 'preview.act'])('answers %s explicitly when a URL open was refused', async method => {
    openPreview(
      { kind: 'url', label: 'Page', source: 'https://example.invalid', url: 'https://example.invalid' },
      'tool-result'
    )
    const { respond } = deliver(method, { action: 'back', session_id: 'session-a' }, 'session-a')
    await waitFor(() => expect(respond).toHaveBeenCalledOnce())
    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({
      success: false,
      error: expect.stringMatching(/unavailable.*browser/i)
    })
    expect($previewTabs.get()).toHaveLength(0)
  })

  it.each(['preview.read', 'preview.act'])('does not let a non-owning session answer %s', async method => {
    const { respond, fail } = deliver(method, { action: 'back', session_id: 'session-a' }, 'session-b')
    await Promise.resolve()
    expect(respond).not.toHaveBeenCalled()
    expect(fail).not.toHaveBeenCalled()
  })

  it('does not expose a supported preview to an unscoped browser read', () => {
    openPreview({ kind: 'file', label: 'Owner file', source: '/work/file', url: 'file:///work/file' })
    const { respond } = deliver('preview.read', {}, 'session-a')
    expect(respond).toHaveBeenCalledOnce()
    expect(JSON.parse(respond.mock.calls[0][0].value)).toEqual({
      success: false,
      error: 'Preview reads require the session the user is looking at.'
    })
  })

  it.each(['preview.read', 'preview.act'])('never executes leftover native handles for browser %s', async method => {
    $previewTabs.set([
      { id: 'url:stale', target: { kind: 'url', label: 'Page', source: 'about:blank', url: 'about:blank' } }
    ])
    const back = vi.fn()
    const read = vi.fn(async () => ({ text: 'Stale page', title: 'Page', url: 'about:blank' }))
    disposers.push(registerPreviewNav('url:stale', { back, forward: vi.fn(), reload: vi.fn() }))
    disposers.push(registerPreviewPageReader('url:stale', read))
    const { respond } = deliver(method, { action: 'back', session_id: 'session-a' }, 'session-a')
    await waitFor(() => expect(respond).toHaveBeenCalledOnce())
    expect(JSON.parse(respond.mock.calls[0][0].value).success).toBe(false)
    expect(back).not.toHaveBeenCalled()
    expect(read).not.toHaveBeenCalled()
  })

  it.each(['file', 'artifact'] as const)('preserves owner %s identity reads without a webview', async kind => {
    openPreview({ kind, label: 'Supported preview', source: 'fixture', url: 'fixture', path: '/work/fixture' })
    const { respond } = deliver('preview.read', { session_id: 'session-a' }, 'session-a')
    await waitFor(() => expect(respond).toHaveBeenCalledOnce())
    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({
      kind,
      title: 'Supported preview',
      text: '',
      note: expect.any(String)
    })
  })

  it('preserves native page reads and action dispatch for Electron', async () => {
    vi.stubGlobal('hermesDesktop', {})
    openPreview({ kind: 'url', label: 'Page', source: 'https://example.invalid', url: 'https://example.invalid' })
    const tabId = $previewTabs.get()[0].id
    const back = vi.fn()
    disposers.push(registerPreviewNav(tabId, { back, forward: vi.fn(), reload: vi.fn() }))
    disposers.push(
      registerPreviewPageReader(tabId, async () => ({
        text: 'Native page',
        title: 'Page',
        url: 'https://example.invalid'
      }))
    )
    const read = deliver('preview.read', { session_id: 'session-a' }, 'session-a')
    const action = deliver('preview.act', { action: 'back', session_id: 'session-a' }, 'session-a')
    await waitFor(() => {
      expect(read.respond).toHaveBeenCalledOnce()
      expect(action.respond).toHaveBeenCalledOnce()
    })
    expect(JSON.parse(read.respond.mock.calls[0][0].value).text).toBe('Native page')
    expect(JSON.parse(action.respond.mock.calls[0][0].value).success).toBe(true)
    expect(back).toHaveBeenCalledOnce()
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
