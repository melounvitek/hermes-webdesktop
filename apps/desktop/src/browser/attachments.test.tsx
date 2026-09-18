import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { ChatBar } from '@/app/chat/composer'
import { RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import { useComposerActions } from '@/app/chat/hooks/use-composer-actions'
import { I18nProvider } from '@/i18n'
import { mainComposerScope } from '@/store/composer'

function Composer() {
  const actions = useComposerActions({ activeSessionId: null, currentCwd: '', requestGateway: vi.fn() })

  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: false,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatBar
        busy={false}
        disabled={false}
        gateway={null}
        onAttachImageBlob={actions.attachImageBlob}
        onCancel={vi.fn()}
        onPasteClipboardImage={actions.pasteClipboardImage}
        onSubmit={vi.fn(async () => true)}
        state={{
          model: { canSwitch: false, model: '', provider: '' },
          tools: { enabled: false, label: '' },
          voice: { enabled: false, active: false }
        }}
      />
    </AssistantRuntimeProvider>
  )
}

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.unstubAllEnvs()
})

it('pastes event image bytes into the browser composer without native clipboard access', async () => {
  vi.stubEnv('VITE_BROWSER', '1')
  const saveClipboardImage = vi.fn()
  const saveImageBuffer = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveClipboardImage, saveImageBuffer } })

  const { container } = render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <Composer />
      </I18nProvider>
    </MemoryRouter>
  )

  const editor = container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
  const file = new File(['clipboard image'], 'pasted.png', { type: 'image/png' })
  const event = new Event('paste', { bubbles: true, cancelable: true })
  Object.defineProperty(event, 'clipboardData', {
    value: {
      getData: () => '',
      files: { length: 1, item: () => file },
      items: [{ kind: 'file', type: file.type, getAsFile: () => file }]
    }
  })
  await act(async () => {
    fireEvent(editor, event)
  })
  await waitFor(() => expect(mainComposerScope.$attachments.get()[0]?.thumbnailUrl).toBeTruthy())
  expect(mainComposerScope.$attachments.get()).toHaveLength(1)
  expect(mainComposerScope.$attachments.get()[0]?.blob).toBe(file)
  expect(mainComposerScope.$attachments.get()[0]?.path).toBeUndefined()
  expect(saveClipboardImage).not.toHaveBeenCalled()
  expect(saveImageBuffer).not.toHaveBeenCalled()
  expect(event.defaultPrevented).toBe(true)
})
