import { act, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { stubThreadEnvironment } from '@/components/assistant-ui/test-utils'
import { type TranscriptWindowValue, useTranscriptWindow } from '@/components/assistant-ui/thread/transcript-window'
import type * as HermesApi from '@/hermes'
import type { ChatMessage } from '@/lib/chat-messages'
import type * as SessionStates from '@/store/session-states'
import { $transcriptTailBySessionId, recordTranscriptTail } from '@/store/transcript-tail'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from './session-view'
import { _resetTranscriptBackfillForTests } from './transcript-backfill'

import { ChatRuntimeBoundary } from '.'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getOlderSessionMessages: vi.fn()
}))
vi.mock('@/store/session-states', async importOriginal => ({
  ...(await importOriginal<typeof SessionStates>()),
  sessionTileDelegate: vi.fn()
}))
const { getOlderSessionMessages } = await import('@/hermes')
const { sessionTileDelegate } = await import('@/store/session-states')
stubThreadEnvironment()

const message = (id: string): ChatMessage => ({ id, role: 'user', parts: [{ type: 'text', text: id }] })

beforeEach(() => {
  $transcriptTailBySessionId.set({})
  _resetTranscriptBackfillForTests()
  vi.mocked(getOlderSessionMessages).mockReset()
})

// Mounts the live runtime over a tail whose older-page fetch stays pending
// until the test resolves it.
function renderPendingBackfill() {
  const $messages = atom([message('tail')])

  const view = {
    ...PRIMARY_SESSION_VIEW,
    $messages,
    $runtimeId: atom<string | null>('runtime'),
    $storedId: atom<string | null>('stored')
  }

  recordTranscriptTail('stored', {
    messages: Array.from({ length: 120 }, (_, index) => ({
      id: index + 120,
      role: 'user' as const,
      content: 'tail',
      timestamp: index + 2000
    })),
    pagination: { limit: 120, offset: 0, order: 'latest', returned: 120 }
  })
  let resolveFetch!: (value: unknown) => void
  vi.mocked(getOlderSessionMessages).mockReturnValue(
    new Promise(resolve => {
      resolveFetch = resolve
    }) as never
  )
  const events: string[] = []
  vi.mocked(sessionTileDelegate).mockReturnValue({
    updateSession: (_id: string, update: (state: { messages: ChatMessage[] }) => { messages: ChatMessage[] }) => {
      events.push('apply')
      $messages.set(update({ messages: $messages.get() }).messages)
    }
  } as never)
  let transcriptWindow!: TranscriptWindowValue

  function Observe() {
    transcriptWindow = useTranscriptWindow()

    return null
  }

  render(
    <SessionViewProvider value={view}>
      <ChatRuntimeBoundary
        busy={false}
        onCancel={() => {}}
        onEdit={async () => {}}
        onReload={async () => {}}
        onThreadMessagesChange={() => {}}
        suppressMessages={false}
      >
        <Observe />
      </ChatRuntimeBoundary>
    </SessionViewProvider>
  )

  const resolvePage = (rows: unknown[]) =>
    resolveFetch({
      session_id: 'stored',
      messages: rows,
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 }
    })

  return { $messages, events, resolvePage, window: () => transcriptWindow }
}

const olderRow = { id: 1, role: 'user', content: 'older', timestamp: 1000 }

describe('runtime older-page expansion', () => {
  it.each([true, false])(
    'captures only a page that adds history, after its network wait (adds rows: %s)',
    async addsRows => {
      const { $messages, events, resolvePage, window } = renderPendingBackfill()
      let pending: unknown
      act(() => {
        pending = window().expandWindow(() => events.push('anchor'))
      })
      expect(pending).toBeInstanceOf(Promise)
      expect(events).toEqual([])
      await act(async () => {
        resolvePage(addsRows ? [olderRow] : [])
        expect(await pending).toBe(addsRows)
      })
      expect(events).toEqual(addsRows ? ['anchor', 'apply'] : [])
      expect($messages.get()).toHaveLength(addsRows ? 2 : 1)
    }
  )

  it('discards a page that lands after the window collapsed', async () => {
    const { $messages, events, resolvePage, window } = renderPendingBackfill()
    let pending: unknown
    act(() => {
      pending = window().expandWindow(() => events.push('anchor'))
    })
    act(() => window().collapseWindow?.())
    await act(async () => {
      resolvePage([olderRow])
      expect(await pending).toBe(false)
    })
    expect(events).toEqual([])
    expect($messages.get()).toHaveLength(1)
  })
})
