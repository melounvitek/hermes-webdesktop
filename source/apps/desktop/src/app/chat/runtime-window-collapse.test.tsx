import { act, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { describe, expect, it } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize } from '@/components/assistant-ui/test-utils'
import { Thread } from '@/components/assistant-ui/thread'
import { useTranscriptWindow } from '@/components/assistant-ui/thread/transcript-window'
import type { ChatMessage } from '@/lib/chat-messages'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from './session-view'

import { ChatRuntimeBoundary } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const CLIENT_H = 600
const GROUP_H = 400
const groupsIn = (root: ParentNode) => root.querySelectorAll('[data-slot="aui_message-group"]').length
const scrollHeightNow = () => CLIENT_H + groupsIn(window.document) * GROUP_H

Object.defineProperty(HTMLElement.prototype, 'scrollHeight', { configurable: true, get: scrollHeightNow })
Object.defineProperty(HTMLElement.prototype, 'clientHeight', { configurable: true, get: () => CLIENT_H })

async function settle(ticks = 20) {
  await act(async () => {
    for (let tick = 0; tick < ticks; tick += 1) {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    }
  })
}

// The first page backfills in animation-frame steps; wait until it stops.
async function settledGroupCount(root: ParentNode): Promise<number> {
  let count = groupsIn(root)

  for (let round = 0; round < 20; round += 1) {
    await settle(10)
    const next = groupsIn(root)

    if (next === count) {
      return count
    }

    count = next
  }

  return count
}

const heavyText = 'x'.repeat(5000)

const transcript = (turns: number): ChatMessage[] =>
  Array.from({ length: turns }, (_, index): ChatMessage[] => [
    { id: `u-${index}`, role: 'user', parts: [{ type: 'text', text: heavyText }] },
    { id: `a-${index}`, role: 'assistant', parts: [{ type: 'text', text: heavyText }] }
  ]).flat()

describe('live transcript window', () => {
  it('releases the paged-in store window once the reader returns to the bottom', async () => {
    const view = {
      ...PRIMARY_SESSION_VIEW,
      $messages: atom(transcript(200)),
      $runtimeId: atom<string | null>('runtime'),
      $storedId: atom<string | null>(null)
    }

    let windowMessages = 0

    function Observe() {
      windowMessages = useTranscriptWindow().currentMessages.length

      return null
    }

    const { container } = render(
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
          <Thread sessionKey="long" />
        </ChatRuntimeBoundary>
      </SessionViewProvider>
    )

    const viewport = container.querySelector('[data-slot="aui_thread-viewport"]') as HTMLElement
    const initialGroups = await settledGroupCount(container)
    const initialWindow = windowMessages

    expect(initialWindow).toBeLessThan(400)

    viewport.scrollTop = scrollHeightNow() - CLIENT_H
    act(() => viewport.dispatchEvent(new Event('scroll')))

    for (let page = 0; page < 10 && windowMessages === initialWindow; page += 1) {
      viewport.scrollTop = 0
      act(() => viewport.dispatchEvent(new Event('scroll')))
      await settle(5)
      act(() => viewport.dispatchEvent(new WheelEvent('wheel', { deltaY: -120 })))
      await settle()
    }

    expect(windowMessages).toBeGreaterThan(initialWindow)

    viewport.scrollTop = scrollHeightNow() - CLIENT_H
    act(() => viewport.dispatchEvent(new Event('scroll')))
    await settle()

    expect(windowMessages).toBe(initialWindow)
    expect(groupsIn(container)).toBe(initialGroups)
  })
})
