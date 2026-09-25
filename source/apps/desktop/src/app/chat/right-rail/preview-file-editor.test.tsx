import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type * as DesktopFs from '@/lib/desktop-fs'
import { writeDesktopFileText } from '@/lib/desktop-fs'

import { LocalFilePreview } from './preview-file'

vi.mock('@/components/chat/code-editor', () => ({
  CodeEditor: ({
    initialValue,
    onChange,
    onSave
  }: {
    initialValue: string
    onChange: (value: string) => void
    onSave: () => void
  }) => (
    <textarea
      aria-label="File editor"
      defaultValue={initialValue}
      onChange={event => onChange(event.target.value)}
      onKeyDown={event => {
        if (event.ctrlKey && event.key === 's') onSave()
      }}
    />
  )
}))
vi.mock('@/components/chat/shiki-highlighter', () => ({ LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre> }))
vi.mock('@/lib/desktop-fs', async importOriginal => ({
  ...(await importOriginal<typeof DesktopFs>()),
  desktopGitRoot: async () => null,
  readDesktopFileText: vi.fn(async () => ({ text: 'fixture text', byteSize: 12 })),
  writeDesktopFileText: vi.fn(async (path: string) => ({ path }))
}))

const target = {
  kind: 'file' as const,
  label: 'notes.txt',
  url: 'file:///tmp/notes.txt',
  path: '/tmp/notes.txt',
  source: '/tmp/notes.txt',
  previewKind: 'text' as const
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

it.each([true, false])('offers explicit editing and saves (browser=%s)', async browser => {
  vi.stubGlobal('hermesDesktop', { browser })
  render(<LocalFilePreview reloadKey={0} target={target} />)
  await screen.findByText('fixture text')
  fireEvent.click(screen.getByRole('button', { name: /^Edit/ }))
  const editor = screen.getByRole('textbox', { name: 'File editor' })
  if (browser) {
    expect(screen.getByText(/may reset permissions/)).toBeTruthy()
    expect(screen.getByText(/No autosave/)).toBeTruthy()
  }
  fireEvent.change(editor, { target: { value: 'changed' } })
  expect(writeDesktopFileText).not.toHaveBeenCalled()
  if (browser) {
    fireEvent.click(screen.getByRole('button', { name: 'Save to server' }))
  } else {
    fireEvent.keyDown(editor, { key: 's', ctrlKey: true })
  }
  await waitFor(() => expect(writeDesktopFileText).toHaveBeenCalledWith('/tmp/notes.txt', 'changed'))
})
