import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $browserUpdate, closeBrowserUpdates, openBrowserUpdates } from '@/browser/updates'
import type { HermesApiRequest } from '@/global'
import { I18nProvider } from '@/i18n/context'

import { BrowserUpdatesDialog } from './browser-updates-dialog'

const loadedBuild = vi.hoisted(() => ({ revision: null as string | null, dirty: false }))
vi.mock('@/browser/build', () => ({ BROWSER_BUILD: loadedBuild }))

// The launcher's side of the stock file API: what the page asked for and the
// status the launcher answers that request with.
let requests: { id: string }[]
let directoryExists: boolean
let status: Record<string, unknown> | null

const api = vi.fn(async ({ path, body }: HermesApiRequest) => {
  if (path === '/api/fs/write-text') {
    if (!directoryExists) {
      throw new Error('HTTP 400: {"detail":"Parent directory does not exist"}')
    }

    requests.push(JSON.parse((body as { content: string }).content))

    return { ok: true }
  }

  if (!status) {
    throw new Error('HTTP 404: {"detail":"File not found"}')
  }

  return { path, text: JSON.stringify({ id: requests.at(-1)?.id, ...status }) }
})

function dialog() {
  return screen.getByRole('dialog', { name: 'Browser updates' })
}

function button(name: string) {
  return within(dialog()).getByRole<HTMLButtonElement>('button', { name })
}

async function click(element: HTMLElement) {
  await act(async () => {
    fireEvent.click(element)
  })
}

async function launcherReports(next: Record<string, unknown>, wait = 1000) {
  status = next
  await act(async () => {
    await vi.advanceTimersByTimeAsync(wait)
  })
}

async function mount() {
  await act(async () => {
    render(
      <I18nProvider
        configClient={{
          getConfig: async () => ({ display: { language: 'en' } }),
          saveConfig: async () => ({ ok: true })
        }}
        initialLocale="en"
      >
        <BrowserUpdatesDialog />
      </I18nProvider>
    )
  })
  expect(screen.queryByRole('dialog')).toBeNull()
  await act(async () => {
    openBrowserUpdates()
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  requests = []
  directoryExists = true
  status = null
  loadedBuild.revision = 'a'.repeat(40)
  vi.stubGlobal('hermesDesktop', { api, browser: { authRequired: false, signIn: vi.fn() } })
  $browserUpdate.set({ state: 'idle' })
})

afterEach(() => {
  cleanup()
  closeBrowserUpdates()
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

it('does nothing in desktop mode, even when the browser update store is opened', async () => {
  delete window.hermesDesktop.browser
  await mount()
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(api).not.toHaveBeenCalled()
})

it('opens idle with the loaded build and asks for nothing until Update is pressed', async () => {
  await mount()
  expect(within(dialog()).getByText('Browser build loaded in this tab').nextElementSibling?.textContent).toBe(
    'a'.repeat(40)
  )
  expect(dialog().textContent).toMatch(/checks for a newer release and installs it while Hermes keeps running/i)
  expect(button('Update').disabled).toBe(false)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(60_000)
  })
  expect(api).not.toHaveBeenCalled()
})

it('requests one update, shows progress, and offers a cancellable reload once the release is installed', async () => {
  await mount()
  await click(button('Update'))
  expect(requests).toHaveLength(1)
  expect(dialog().textContent).toMatch(/checking for a newer release and installing it/i)
  expect(button('Update').disabled).toBe(true)
  await click(button('Update'))

  await launcherReports({ state: 'running' }, 20_000)
  expect(button('Update').disabled).toBe(true)
  expect(requests).toHaveLength(1)

  // Progress and its result belong to the page, not to the open dialog.
  await click(button('Close'))
  expect(screen.queryByRole('dialog')).toBeNull()
  await launcherReports({ state: 'updated', release: 'browser-2' })
  await act(async () => {
    openBrowserUpdates()
  })
  expect(dialog().textContent).toContain('Updated to browser-2')
  expect(dialog().textContent).toMatch(/reload this tab and any other open tabs/i)
  expect(within(dialog()).queryByRole('button', { name: 'Update' })).toBeNull()

  const url = window.location.href
  await click(button('Reload this tab'))
  const confirmation = screen.getByRole('dialog', { name: 'Reload this tab?' })
  expect(confirmation.textContent).toMatch(/drafts and unsaved file changes/i)
  await click(within(confirmation).getByRole('button', { name: 'Cancel' }))
  expect(screen.queryByRole('dialog', { name: 'Reload this tab?' })).toBeNull()
  expect(dialog().textContent).toContain('Updated to browser-2')
  expect(window.location.href).toBe(url)
  expect(requests).toHaveLength(1)
})

it('says the installed release is already the newest and allows another check', async () => {
  await mount()
  await click(button('Update'))
  await launcherReports({ state: 'current', release: 'browser-1' })
  expect(dialog().textContent).toContain('browser-1 is already the newest release')
  expect(within(dialog()).queryByRole('button', { name: 'Reload this tab' })).toBeNull()

  await click(button('Update'))
  expect(requests).toHaveLength(2)
  expect(requests[1].id).not.toBe(requests[0].id)
  expect(dialog().textContent).not.toContain('browser-1')
})

it('shows the launcher failure as plain text and allows a retry', async () => {
  await mount()
  await click(button('Update'))
  await launcherReports({ state: 'failed', error: '<img src=x onerror=alert(1)> Download failed' })
  expect(within(dialog()).getByText('<img src=x onerror=alert(1)> Download failed')).toBeTruthy()
  expect(dialog().querySelector('img')).toBeNull()
  expect(within(dialog()).getByRole('heading', { name: 'Error' })).toBeTruthy()

  await click(button('Update'))
  expect(requests).toHaveLength(2)
})

it('fails when a running update reports no result within 10 minutes', async () => {
  await mount()
  await click(button('Update'))
  await launcherReports({ state: 'running' }, 599_000)
  expect(button('Update').disabled).toBe(true)
  await launcherReports({ state: 'running' })
  expect(dialog().textContent).toMatch(/did not finish within 10 minutes/i)
  expect(button('Update').disabled).toBe(false)
})

it.each([
  ['the launcher has no updates directory', false, 0],
  ['the launcher never answers', true, 15_000]
])('points to the terminal when %s', async (_label, exists, wait) => {
  directoryExists = exists
  await mount()
  await click(button('Update'))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(wait)
  })
  expect(dialog().textContent).toMatch(/updating from the browser is not available for this installation/i)
  expect(dialog().querySelector('pre code')?.textContent).toBe(
    'hermes-browser stop\nhermes-browser update\nhermes-browser start'
  )
  expect(button('Update').disabled).toBe(false)
})
