import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { LocalFilePreview } from './preview-file'

vi.mock('@/components/chat/code-editor', () => ({ CodeEditor: () => <textarea aria-label="File editor" /> }))
vi.mock('@/components/chat/shiki-highlighter', () => ({ LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre> }))
vi.mock('@/lib/desktop-fs', async importOriginal => ({
  ...await importOriginal<typeof import('@/lib/desktop-fs')>(),
  desktopGitRoot: async () => null,
  readDesktopFileText: async () => ({ text: 'fixture text', byteSize: 12 })
}))

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

it.each([true, false])('keeps browser=%s previews readable and only Electron editable', async browser => {
  vi.stubGlobal('hermesDesktop', { browser })
  const { container } = render(<LocalFilePreview reloadKey={0} target={{ kind: 'file', url: 'file:///tmp/read-only.txt', source: '/tmp/read-only.txt', previewKind: 'text' }} />)
  await screen.findByText('fixture text')
  if (browser) {
    expect(screen.queryByRole('button', { name: /^Edit/ })).toBeNull()
    fireEvent.mouseEnter(container.querySelector('[tabindex]') ?? container.firstChild!)
    fireEvent.keyDown(window, { key: 'e' })
    expect(screen.queryByRole('textbox', { name: 'File editor' })).toBeNull()
  } else {
    fireEvent.click(screen.getByRole('button', { name: /^Edit/ }))
    expect(screen.getByRole('textbox', { name: 'File editor' })).toBeTruthy()
  }
})
