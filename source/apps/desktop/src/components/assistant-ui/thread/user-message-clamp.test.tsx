import { AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

// Deliver resizes by hand; the thread's default stub never calls back.
const observers: Array<{ callback: ResizeObserverCallback; targets: Set<Element> }> = []

vi.stubGlobal(
  'ResizeObserver',
  class {
    readonly targets = new Set<Element>()

    constructor(readonly callback: ResizeObserverCallback) {
      observers.push(this)
    }

    observe(target: Element) {
      this.targets.add(target)
    }

    unobserve(target: Element) {
      this.targets.delete(target)
    }

    disconnect() {
      this.targets.clear()
    }
  }
)

afterEach(cleanup)

function Harness() {
  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: ExportedMessageRepository.fromArray([userMessage(), assistantMessage()]),
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onCancel: async () => {},
    onReload: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread cwd={null} gateway={null} sessionId="session-1" />
    </AssistantRuntimeProvider>
  )
}

it('judges the two-line clamp by the current line height after the UI scale changes', async () => {
  render(<Harness />)
  const body = (await screen.findByText('edit me please')).closest<HTMLElement>('.min-h-\\[1\\.25rem\\]')!
  const bubble = body.parentElement!

  const resize = (lineHeight: string, height: number) => {
    body.style.lineHeight = lineHeight
    const entry = { borderBoxSize: [{ blockSize: height }], target: body } as unknown as ResizeObserverEntry

    act(() => {
      for (const observer of observers.filter(({ targets }) => targets.has(body))) {
        observer.callback([entry], {} as ResizeObserver)
      }
    })
  }

  // Two lines at 100%, then the same two lines after a live change to 150%.
  resize('21.45px', 38)
  expect(bubble.dataset.clamped).toBeUndefined()
  resize('32.18px', 56)
  expect(bubble.dataset.clamped).toBeUndefined()
  resize('32.18px', 100)
  expect(bubble.dataset.clamped).toBe('true')
})
