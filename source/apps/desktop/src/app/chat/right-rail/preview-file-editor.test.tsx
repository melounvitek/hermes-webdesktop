import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type * as DesktopFs from '@/lib/desktop-fs'
import { desktopFsCacheKey, readDesktopFileDataUrl, writeDesktopFileText } from '@/lib/desktop-fs'
import { clearBrowserFileDraft } from '@/store/browser-file-drafts'
import { $confirmRequest, settleConfirm } from '@/store/confirm'
import { $connection } from '@/store/session'

import { LocalFilePreview } from './preview-file'

vi.mock('@/components/chat/code-editor', () => ({
  CodeEditor: ({
    initialValue,
    onChange,
    onSave,
    onCancel,
    disabled
  }: {
    initialValue: string
    onChange: (value: string) => void
    onSave: () => void
    onCancel: () => void
    disabled: boolean
  }) => (
    <textarea
      aria-label="File editor"
      defaultValue={initialValue}
      onChange={event => onChange(event.target.value)}
      onKeyDown={event => {
        if (event.ctrlKey && event.key === 's') {
          onSave()
        }

        if (event.key === 'Escape') {
          onCancel()
        }
      }}
      readOnly={disabled}
    />
  )
}))
vi.mock('@/components/chat/shiki-highlighter', () => ({ LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre> }))
vi.mock('@/lib/desktop-fs', async importOriginal => ({
  ...(await importOriginal<typeof DesktopFs>()),
  desktopGitRoot: async () => null,
  readDesktopFileText: vi.fn(async () => ({ text: 'fixture text', byteSize: 12 })),
  readDesktopFileDataUrl: vi.fn(async () => `data:text/plain;base64,${btoa('fixture text')}`),
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
  clearBrowserFileDraft(JSON.stringify([desktopFsCacheKey(null), target.path]))
  clearBrowserFileDraft(JSON.stringify([desktopFsCacheKey(), target.path]))
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  settleConfirm(false)
  $connection.set(null)
})

it.each([true, false])('offers explicit editing and saves (browser=%s)', async browser => {
  vi.stubGlobal('hermesDesktop', { browser })
  render(<LocalFilePreview reloadKey={0} target={target} />)
  await screen.findByText('fixture text')
  fireEvent.click(screen.getByRole('button', { name: /^Edit/ }))
  const editor = await screen.findByRole('textbox', { name: 'File editor' })

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

async function editBrowserFile() {
  vi.stubGlobal('hermesDesktop', { browser: true })
  const view = render(<LocalFilePreview reloadKey={0} target={target} />)
  await screen.findByText('fixture text')
  fireEvent.click(screen.getByRole('button', { name: /^Edit/ }))
  const editor = await screen.findByRole('textbox', { name: 'File editor' })
  fireEvent.change(editor, { target: { value: 'my draft' } })

  return { ...view, editor }
}

it('keeps the draft on failed checks, conflicts and failed writes; retries explicitly', async () => {
  const { editor } = await editBrowserFile()
  const save = () => fireEvent.click(screen.getByRole('button', { name: 'Save to server' }))
  vi.mocked(readDesktopFileDataUrl).mockRejectedValueOnce(new Error('Cannot read file'))
  save()
  await screen.findByText(/Cannot read file/)
  expect(writeDesktopFileText).not.toHaveBeenCalled()
  expect((editor as HTMLTextAreaElement).value).toBe('my draft')

  vi.mocked(readDesktopFileDataUrl).mockResolvedValueOnce(`data:text/plain;base64,${btoa('agent edit')}`)
  save()
  await screen.findByText('File changed on disk')
  expect(screen.queryByRole('button', { name: 'Overwrite' })).toBeNull()
  expect(writeDesktopFileText).not.toHaveBeenCalled()

  vi.mocked(writeDesktopFileText).mockRejectedValueOnce(new Error('Disk full'))
  save()
  await screen.findByText(/Disk full/)
  expect((editor as HTMLTextAreaElement).value).toBe('my draft')
  save()
  await screen.findByText('Saved to server')
  expect(screen.getByRole('textbox', { name: 'File editor' })).toBe(editor)
})

it('refuses invalid UTF-8 before opening the editor', async () => {
  vi.stubGlobal('hermesDesktop', { browser: true })
  vi.mocked(readDesktopFileDataUrl).mockResolvedValueOnce('data:text/plain;base64,/w==')
  render(<LocalFilePreview reloadKey={0} target={target} />)
  await screen.findByText('fixture text')
  fireEvent.click(screen.getByRole('button', { name: /^Edit/ }))
  await screen.findByText(/complete UTF-8 text file/)
  expect(screen.queryByRole('textbox', { name: 'File editor' })).toBeNull()
  expect(writeDesktopFileText).not.toHaveBeenCalled()
})

it('keeps drafts through refreshes and remounts, and warns before discard or browser close', async () => {
  const view = await editBrowserFile()
  view.rerender(<LocalFilePreview reloadKey={1} target={target} />)
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('my draft')
  view.unmount()
  const unload = new Event('beforeunload', { cancelable: true })
  window.dispatchEvent(unload)
  expect(unload.defaultPrevented).toBe(true)
  render(<LocalFilePreview reloadKey={0} target={target} />)
  const editor = await screen.findByRole('textbox', { name: 'File editor' })
  expect((editor as HTMLTextAreaElement).value).toBe('my draft')
  fireEvent.keyDown(editor, { key: 'Escape' })
  await waitFor(() => expect($confirmRequest.get()?.title).toBe('Discard unsaved edits?'))
  await act(async () => settleConfirm(false))
  expect((editor as HTMLTextAreaElement).value).toBe('my draft')
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await act(async () => settleConfirm(true))
  await waitFor(() => expect(screen.queryByRole('textbox')).toBeNull())
  const cleanUnload = new Event('beforeunload', { cancelable: true })
  window.dispatchEvent(cleanUnload)
  expect(cleanUnload.defaultPrevented).toBe(false)
})

it('reconciles a save that completes after the editor is reopened', async () => {
  const view = await editBrowserFile()
  let resolveWrite!: (value: { path: string }) => void
  vi.mocked(writeDesktopFileText).mockReturnValueOnce(
    new Promise(resolve => {
      resolveWrite = resolve
    })
  )
  fireEvent.click(screen.getByRole('button', { name: 'Save to server' }))
  await waitFor(() => expect(writeDesktopFileText).toHaveBeenCalledOnce())
  view.unmount()
  render(<LocalFilePreview reloadKey={0} target={target} />)
  const editor = await screen.findByRole('textbox')
  expect((editor as HTMLTextAreaElement).readOnly).toBe(true)
  await act(async () => {
    resolveWrite({ path: target.path })
  })
  await screen.findByText('Saved to server')
  expect((editor as HTMLTextAreaElement).readOnly).toBe(false)
  // Reverting to the pre-save text is now a NEW unsaved edit, not a clean draft.
  fireEvent.change(editor, { target: { value: 'fixture text' } })
  expect((screen.getByRole('button', { name: 'Save to server' }) as HTMLButtonElement).disabled).toBe(false)
  const unload = new Event('beforeunload', { cancelable: true })
  window.dispatchEvent(unload)
  expect(unload.defaultPrevented).toBe(true)
  vi.mocked(readDesktopFileDataUrl).mockResolvedValueOnce(`data:text/plain;base64,${btoa('my draft')}`)
  fireEvent.click(screen.getByRole('button', { name: 'Save to server' }))
  await screen.findByText('Saved to server')
  expect(writeDesktopFileText).toHaveBeenLastCalledWith(target.path, 'fixture text')
})

it('never sends a pending save to a newly selected profile', async () => {
  await editBrowserFile()
  let resolveRead!: (value: string) => void
  vi.mocked(readDesktopFileDataUrl).mockReturnValueOnce(
    new Promise(resolve => {
      resolveRead = resolve
    })
  )
  fireEvent.click(screen.getByRole('button', { name: 'Save to server' }))
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).readOnly).toBe(true)
  await act(async () => {
    $connection.set({ mode: 'remote', profile: 'other' } as never)
    resolveRead(`data:text/plain;base64,${btoa('fixture text')}`)
  })
  expect(writeDesktopFileText).not.toHaveBeenCalled()
  await act(async () => {
    $connection.set(null)
  })
  expect(((await screen.findByRole('textbox')) as HTMLTextAreaElement).value).toBe('my draft')
})
