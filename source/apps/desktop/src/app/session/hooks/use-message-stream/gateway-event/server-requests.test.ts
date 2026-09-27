import { waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registerPreviewNav } from '@/app/chat/right-rail/preview-nav'
import { registerPreviewPageReader } from '@/app/chat/right-rail/preview-reader'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $previewTabs, closeRightRail, openPreview } from '@/store/preview'
import { setActiveSessionId, setSessions } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { $toursEnabled } from '@/store/tours'
import type { SessionInfo } from '@/types/hermes'

import { handleServerRequest, previewSessionRoute } from './server-requests'
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

describe('approval request routing', () => {
  const notify = vi.fn().mockResolvedValue(true)
  const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

  beforeEach(() => {
    notify.mockClear()
    desktopWindow.hermesDesktop = { notify } as unknown as Window['hermesDesktop']
    setSessions([{ id: 'session-a', title: 'Fix the flaky test' } as SessionInfo])
    setActiveSessionId('session-b')
  })

  afterEach(() => {
    delete desktopWindow.hermesDesktop
    setSessions([])
    setActiveSessionId(null)
  })

  it('titles the parked approval toast with the session it belongs to', () => {
    deliver(
      'approval',
      { command: 'rm -rf /', description: 'dangerous', request_id: 'r1', session_id: 'session-a' },
      'session-b'
    )

    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'approval', title: expect.stringContaining('Fix the flaky test') })
    )
  })
})

describe('preview action request routing', () => {
  it('retries a replayed scoped request only while no session is bound yet', () => {
    expect(previewSessionRoute({ replayed: true, sessionId: 'session-a', activeSessionId: null })).toBe('retry')
    expect(previewSessionRoute({ replayed: true, sessionId: 'session-a', activeSessionId: 'session-a' })).toBe('run')
    expect(previewSessionRoute({ replayed: true, sessionId: 'session-a', activeSessionId: 'session-b' })).toBe('ignore')
    expect(previewSessionRoute({ replayed: true, sessionId: '', activeSessionId: null })).toBe('run')
  })

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

  it('leaves scoped pane reads unanswered in a window showing another session', async () => {
    const reads = ['preview.read', 'terminal.read', 'window.read'].map(method =>
      deliver(method, { session_id: 'session-a' }, 'session-b')
    )

    await Promise.resolve()

    for (const { handled, respond } of reads) {
      expect(handled).toBe(true)
      expect(respond).not.toHaveBeenCalled()
    }
  })

  it("answers pane reads for a session hosted in one of this window's tiles", async () => {
    // The tile session is not the active one, but this window hosts it: its
    // panes are here, so an 'ignore' would stall the tool until its deadline.
    $sessionTiles.set([{ runtimeId: 'session-a', storedSessionId: 'stored-a' } as never])

    try {
      const reads = ['preview.read', 'terminal.read', 'window.read'].map(method =>
        deliver(method, { session_id: 'session-a' }, 'session-b')
      )

      await new Promise(resolve => setTimeout(resolve, 0))

      for (const { handled, respond } of reads) {
        expect(handled).toBe(true)
        expect(respond).toHaveBeenCalledTimes(1)
      }
    } finally {
      $sessionTiles.set([])
    }
  })

  it('fails fast for an unscoped request with no session in view', () => {
    const { respond } = deliver('preview.act', { action: 'elements' }, null)

    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ success: false })
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
    $sessionTiles.set([])
    deps.activeSessionIdRef.current = null
    vi.useRealTimers()
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

  it.each([false, true])('keeps deferred native actions foreground-only (background tile: %s)', async background => {
    vi.stubGlobal('hermesDesktop', {})
    vi.useFakeTimers()
    openPreview({ kind: 'url', label: 'Page', source: 'https://example.invalid', url: 'https://example.invalid' })
    const back = vi.fn()
    disposers.push(registerPreviewNav($previewTabs.get()[0].id, { back, forward: vi.fn(), reload: vi.fn() }))
    const respond = vi.fn()
    const fail = vi.fn()

    expect(handleServerRequest({
      id: 'replayed-preview',
      method: 'preview.act',
      params: { action: 'back', session_id: 'session-a' },
      profile: 'default',
      replayed: true,
      respond,
      fail
    }, deps, null)).toBe(true)
    expect(respond).not.toHaveBeenCalled()

    if (background) {
      $sessionTiles.set([{ runtimeId: 'session-a', storedSessionId: 'stored-a' }])
    }

    deps.activeSessionIdRef.current = background ? 'session-b' : 'session-a'
    await vi.advanceTimersByTimeAsync(0)
    await vi.dynamicImportSettled()

    expect(respond).toHaveBeenCalledOnce()
    expect(JSON.parse(respond.mock.calls[0][0].value).success).toBe(!background)
    expect(back).toHaveBeenCalledTimes(background ? 0 : 1)
    expect(fail).not.toHaveBeenCalled()
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

    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ success: false })
  })
})
