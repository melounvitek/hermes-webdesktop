import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'

import { RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarState } from './types'

import { ChatBar } from './index'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness() {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: false,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <p>Last reply</p>
          <input aria-label="Elsewhere" />
          <ChatBar
            busy={false}
            disabled={false}
            gateway={null}
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
})

function renderComposer() {
  const view = render(<Harness />)
  const editor = view.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!

  return { ...view, editor }
}

function caretInside(editor: HTMLElement) {
  const selection = window.getSelection()!

  return selection.isCollapsed && editor.contains(selection.anchorNode)
}

describe('tapping the composer on a phone', () => {
  it('focuses the one-line editor from anywhere else on the bar', () => {
    const { container, editor, getByLabelText } = renderComposer()

    getByLabelText('Elsewhere').focus()
    fireEvent.click(container.querySelector('[data-slot="composer-fade"]')!)

    expect(document.activeElement).toBe(editor)
    expect(caretInside(editor)).toBe(true)
  })

  it('leaves taps on the bar buttons to the buttons', () => {
    const { editor, getByLabelText, getByRole } = renderComposer()

    getByLabelText('Elsewhere').focus()
    fireEvent.click(getByRole('button', { name: 'Open model picker' }))

    expect(document.activeElement).not.toBe(editor)
  })

  it('brings the caret back when a triple tap selected the reply above', () => {
    const { editor, getByText } = renderComposer()

    editor.focus()
    window.getSelection()!.selectAllChildren(getByText('Last reply'))
    fireEvent.click(editor, { detail: 3 })

    expect(document.activeElement).toBe(editor)
    expect(caretInside(editor)).toBe(true)
  })
})
