import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { MarkdownImage } from '@/components/assistant-ui/markdown-text'
import { $connection } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { PreviewAttachment } from './preview-attachment'

const originalDesktop = window.hermesDesktop
const saveGatewayFile = vi.fn(async () => ({ saved: false, canceled: true }))

const view = {
  ...PRIMARY_SESSION_VIEW,
  kind: 'tile' as const,
  $storedId: atom<string | null>('session-A'),
  $runtimeId: atom<string | null>('runtime-A'),
  $cwd: atom('/workspace/A')
}

const ownerRoute = { connectionId: 'host-A', profile: 'route-A', targetProfile: 'profileA' }

beforeEach(() => {
  saveGatewayFile.mockClear()
  $connection.set({ mode: 'remote', connectionId: 'host-B', profile: 'profileB' } as never)
  $sessionTiles.set([{ storedSessionId: 'session-A', runtimeId: 'runtime-A', ownerRoute }])
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      saveGatewayFile,
      api: vi.fn(async () => {
        throw new Error('Image preview unavailable')
      })
    }
  })
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  $sessionTiles.set([])
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: originalDesktop })
})

it('downloads a relative file from its owning tile while another profile is foreground', async () => {
  render(
    <SessionViewProvider value={view}>
      <PreviewAttachment target="./report.txt" />
    </SessionViewProvider>
  )

  fireEvent.click(screen.getByRole('button', { name: 'Download' }))

  await waitFor(() =>
    expect(saveGatewayFile).toHaveBeenCalledWith({
      path: './report.txt',
      suggestedName: 'report.txt',
      sessionId: 'session-A',
      profile: 'profileA',
      connectionId: 'host-A'
    })
  )
})

it('keeps the markdown image download fallback on its owning session', async () => {
  render(
    <SessionViewProvider value={view}>
      <MarkdownImage alt="Report" src="/workspace/A/report.png" />
    </SessionViewProvider>
  )

  const openImage = await screen.findByRole('button', { name: 'Open image' })

  // A later foreground switch to a local connection must not bypass the download bridge either.
  for (const mode of ['remote', 'local'] as const) {
    $connection.set({ mode, connectionId: 'host-B', profile: 'profileB' } as never)
    saveGatewayFile.mockClear()
    fireEvent.click(openImage)

    await waitFor(() =>
      expect(saveGatewayFile).toHaveBeenCalledWith({
        path: '/workspace/A/report.png',
        suggestedName: 'report.png',
        sessionId: 'session-A',
        profile: 'profileA',
        connectionId: 'host-A'
      })
    )
  }
})
