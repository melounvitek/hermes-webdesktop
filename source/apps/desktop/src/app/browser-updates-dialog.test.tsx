/** @vitest-environment jsdom
 * @vitest-environment-options {"url":"https://browser.example/"}
 */
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as BrowserUpdateClient from '@/browser/update-client'
import {
  applyBrowserUpdate,
  BrowserUpdateError,
  type BrowserUpdateOffer,
  type BrowserUpdatePhase,
  type BrowserUpdateSnapshot,
  checkBrowserUpdateSession,
  prepareBrowserUpdate,
  readBrowserUpdateStatus
} from '@/browser/update-client'
import {
  $browserUpdateView,
  closeBrowserUpdates,
  forgetBrowserUpdateTracking,
  openBrowserUpdates
} from '@/browser/updates'
import { I18nProvider } from '@/i18n/context'
import { reconnectGateway } from '@/store/gateway-reconnect'

import { BrowserUpdatesDialog } from './browser-updates-dialog'

vi.mock('@/browser/update-client', async importOriginal => ({
  ...(await importOriginal<typeof BrowserUpdateClient>()),
  readBrowserUpdateStatus: vi.fn(),
  prepareBrowserUpdate: vi.fn(),
  applyBrowserUpdate: vi.fn(),
  checkBrowserUpdateSession: vi.fn()
}))
vi.mock('@/store/gateway-reconnect', () => ({ reconnectGateway: vi.fn() }))
const loadedBuild = vi.hoisted(() => ({ revision: null as string | null, dirty: false }))
vi.mock('@/browser/build', () => ({ BROWSER_BUILD: loadedBuild }))

const status = vi.mocked(readBrowserUpdateStatus)
const prepare = vi.mocked(prepareBrowserUpdate)
const apply = vi.mocked(applyBrowserUpdate)
const checkSession = vi.mocked(checkBrowserUpdateSession)
const reconnect = vi.mocked(reconnectGateway)
const idle: BrowserUpdateSnapshot = { capabilities: ['install'], phase: 'idle' }
let offer: BrowserUpdateOffer
let originalUrl: string

function offered(): BrowserUpdateSnapshot {
  return { capabilities: ['install'], phase: 'offered', offer }
}

function job(phase: BrowserUpdatePhase): BrowserUpdateSnapshot {
  return { capabilities: ['install'], phase, offer, job: { id: offer.id, phase } }
}

function dialog() {
  return screen.getByRole('dialog', { name: 'Browser updates' })
}

async function click(element: HTMLElement) {
  await act(async () => {
    fireEvent.click(element)
  })
}

async function open() {
  await act(async () => {
    openBrowserUpdates()
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
  expect(status).not.toHaveBeenCalled()
  expect(prepare).not.toHaveBeenCalled()
  expect(apply).not.toHaveBeenCalled()
}

async function requestConfirmation() {
  await click(within(dialog()).getByRole('button', { name: 'Check and download' }))
  await click(within(dialog()).getByRole('button', { name: 'Update and restart' }))

  return screen.getByRole('dialog', { name: 'Update and restart web desktop?' })
}

async function cancelReload() {
  await click(within(dialog()).getByRole('button', { name: 'Reload this tab' }))
  const confirmation = screen.getByRole('dialog', { name: 'Reload this tab?' })
  expect(confirmation.textContent).toMatch(/drafts and unsaved file changes/i)
  expect(confirmation.textContent).toMatch(/only this tab.*server will not restart/i)
  await click(within(confirmation).getByRole('button', { name: 'Cancel' }))
  expect(screen.queryByRole('dialog', { name: 'Reload this tab?' })).toBeNull()
  expect(dialog()).toBeTruthy()
  expect(window.location.href).toBe(originalUrl)
}

async function cannotUpdate() {
  const button = within(dialog()).queryByRole<HTMLButtonElement>('button', { name: 'Update and restart' })

  if (button) {
    expect(button.disabled).toBe(true)
    await click(button)
  }

  expect(screen.queryByRole('dialog', { name: 'Update and restart web desktop?' })).toBeNull()
  expect(apply).not.toHaveBeenCalled()
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.resetAllMocks()
  loadedBuild.revision = null
  closeBrowserUpdates()
  forgetBrowserUpdateTracking()
  $browserUpdateView.set({ snapshot: null, error: null, busy: null, pendingId: null, unknownOutcome: false })
  vi.stubGlobal('hermesDesktop', { browser: { authRequired: true, signIn: vi.fn() } })
  offer = {
    id: 'a'.repeat(32),
    current_release: '1'.repeat(40),
    target_release: '2'.repeat(40),
    expires_at: Date.now() / 1000 + 600,
    warning: 'Restart disconnects browser users. Rollback may be required.',
    tested_backend: '3'.repeat(40),
    compatibility: 'not-exercised'
  }
  status.mockResolvedValue(idle)
  prepare.mockImplementation(async () => offered())
  apply.mockImplementation(async () => job('stopping'))
  checkSession.mockResolvedValue(undefined)
  reconnect.mockResolvedValue(undefined)
  originalUrl = window.location.href
})

afterEach(() => {
  cleanup()
  closeBrowserUpdates()
  forgetBrowserUpdateTracking()
  expect(window.location.href).toBe(originalUrl)
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it('does nothing in desktop mode, even when the browser update store is opened', async () => {
  delete window.hermesDesktop.browser
  await mount()
  await open()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(10_000)
  })
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(status).not.toHaveBeenCalled()
  expect(prepare).not.toHaveBeenCalled()
  expect(apply).not.toHaveBeenCalled()
  expect(checkSession).not.toHaveBeenCalled()
  expect(reconnect).not.toHaveBeenCalled()
})

it('reads status on open, downloads only on request, and requires a separate cancellable restart confirmation', async () => {
  await mount()
  await open()
  expect(status).toHaveBeenCalledTimes(1)
  expect(prepare).not.toHaveBeenCalled()
  expect(apply).not.toHaveBeenCalled()
  expect(within(dialog()).getByText('Browser build loaded in this tab')).toBeTruthy()
  expect(within(dialog()).getByText('Unknown build')).toBeTruthy()

  await click(within(dialog()).getByRole('button', { name: 'Check and download' }))
  expect(prepare).toHaveBeenCalledTimes(1)
  expect(within(dialog()).getByText('Release before this update')).toBeTruthy()
  expect(within(dialog()).getByText(offer.current_release)).toBeTruthy()
  expect(within(dialog()).getByText('Release offered for this update')).toBeTruthy()
  expect(within(dialog()).getByText(offer.target_release)).toBeTruthy()
  expect(dialog().textContent).toContain(offer.tested_backend)
  expect(dialog().textContent).toMatch(/does not certify compatibility with your current backend/i)

  await click(within(dialog()).getByRole('button', { name: 'Update and restart' }))
  const confirmation = screen.getByRole('dialog', { name: 'Update and restart web desktop?' })
  expect(screen.getAllByRole('dialog', { hidden: true })).toHaveLength(2)
  expect(confirmation.textContent).toContain(`Update from ${offer.current_release} to ${offer.target_release}?`)
  expect(confirmation.textContent).toMatch(/all browser users will be disconnected/i)
  expect(confirmation.textContent).toMatch(/chats may be interrupted and terminals may close/i)
  expect(confirmation.textContent).toMatch(/previous release will be restored and restarted/i)
  expect(confirmation.textContent).toMatch(/Hermes itself will not be upgraded/i)
  expect(confirmation.textContent).toMatch(/may need to sign in again/i)
  expect(apply).not.toHaveBeenCalled()

  const cancel = within(confirmation).getByRole('button', { name: 'Cancel' })
  act(() => {
    cancel.focus()
    fireEvent.keyDown(cancel, { key: 'Enter' })
  })
  expect(apply).not.toHaveBeenCalled()
  // jsdom does not synthesize the native button click from Enter.
  await click(cancel)
  expect(screen.queryByRole('dialog', { name: 'Update and restart web desktop?' })).toBeNull()
  expect(dialog()).toBeTruthy()
  expect(apply).not.toHaveBeenCalled()
  expect(reconnect).not.toHaveBeenCalled()
})

it('applies the pinned offer once, stops observing when closed, and offers a cancellable reload after success', async () => {
  let accept!: (snapshot: BrowserUpdateSnapshot) => void
  apply.mockReturnValueOnce(
    new Promise(resolve => {
      accept = resolve
    })
  )
  await mount()
  await open()
  const confirmation = await requestConfirmation()
  const confirm = within(confirmation).getByRole('button', { name: 'Update and restart' })
  act(() => {
    fireEvent.click(confirm)
    fireEvent.click(confirm)
  })
  expect(apply).toHaveBeenCalledExactlyOnceWith(offer.id)
  await click(confirm)
  expect(apply).toHaveBeenCalledTimes(1)
  await act(async () => {
    accept(job('stopping'))
  })
  expect(screen.queryByRole('dialog', { name: 'Update and restart web desktop?' })).toBeNull()
  expect(dialog().textContent).toMatch(/stopping web desktop/i)
  expect(dialog().textContent).toMatch(/closing this dialog stops status checks, not an update already accepted/i)

  status.mockResolvedValue(job('stopping'))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000)
  })
  expect(status).toHaveBeenCalledTimes(2)
  expect(reconnect).not.toHaveBeenCalled()

  await click(within(dialog()).getByRole('button', { name: 'Close' }))
  const readsAtClose = status.mock.calls.length
  await act(async () => {
    await vi.advanceTimersByTimeAsync(10_000)
  })
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(status).toHaveBeenCalledTimes(readsAtClose)
  expect(apply).toHaveBeenCalledTimes(1)

  status.mockResolvedValue(job('succeeded'))
  await open()
  expect(dialog().textContent).toMatch(/last update completed successfully/i)
  expect(within(dialog()).getByText('Unknown build')).toBeTruthy()
  expect(window.location.href).toBe(originalUrl)
  await cancelReload()
  expect(apply).toHaveBeenCalledTimes(1)
  expect(reconnect).not.toHaveBeenCalled()
})

it('distinguishes restored activation from failure and operator recovery, never applying their stale offers', async () => {
  await mount()

  for (const [phase, description] of [
    ['rolled_back', /last update did not activate.*previous release was restored and restarted/i],
    ['failed', /update failed.*check the server status/i],
    ['recovery_required', /server operator.*update journal.*do not retry.*automatically/i]
  ] as const) {
    status.mockResolvedValue(job(phase))
    await open()
    expect(dialog().textContent).toMatch(description)
    await cannotUpdate()

    if (phase === 'rolled_back') {
      await cancelReload()
    } else {
      expect(within(dialog()).queryByText(/previous release was restored and restarted/i)).toBeNull()
      expect(within(dialog()).queryByRole('button', { name: 'Reload this tab' })).toBeNull()
    }

    await click(within(dialog()).getByRole('button', { name: 'Close' }))
  }

  expect(prepare).not.toHaveBeenCalled()
  expect(apply).not.toHaveBeenCalled()
})

it('keeps release fields historical across every phase carrying an offer', async () => {
  await mount()

  for (const phase of [
    'offered', 'current', 'stopping', 'switching', 'starting',
    'rollback_stopping', 'rollback_switching', 'rollback_starting',
    'succeeded', 'rolled_back', 'failed', 'recovery_required'
  ] as const) {
    status.mockResolvedValue(job(phase))
    await open()
    expect(within(dialog()).getByText('Release before this update').nextElementSibling?.textContent)
      .toBe(offer.current_release)
    expect(within(dialog()).getByText('Release offered for this update').nextElementSibling?.textContent)
      .toBe(offer.target_release)
    expect(within(dialog()).queryByText('Installed release')).toBeNull()
    await click(within(dialog()).getByRole('button', { name: 'Close' }))
  }
})

it.each([
  ['current', /published release matched the installation at the last check/i],
  ['succeeded', /last update completed successfully/i],
  ['rolled_back', /last update did not activate.*previous release was restored and restarted/i]
] as const)('shows retained %s history independently of the build loaded after a reload or operator rollback', async (phase, message) => {
  const history = job(phase)
  status.mockResolvedValue(history)

  // The retained job cannot tell us whether an operator later changed the server.
  for (const revision of [offer.target_release, offer.current_release]) {
    loadedBuild.revision = revision
    status.mockClear()
    await mount()
    await open()
    expect(within(dialog()).getByText('Browser build loaded in this tab').nextElementSibling?.textContent).toBe(revision)
    expect(dialog().textContent).toMatch(message)
    await click(within(dialog()).getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    await open()
    expect(dialog().textContent).toMatch(message)
    expect(dialog().textContent).not.toMatch(/release is running|tab still has|installed release/i)
    await cannotUpdate()
    if (phase !== 'current') {
      expect(dialog().textContent).toMatch(/compare.*build loaded in this tab/i)
      expect(dialog().textContent).toMatch(/reload only after saving/i)
      await cancelReload()
    }
    cleanup()
    closeBrowserUpdates()
  }
})

it('disables an expired offer without sending another request when Update is clicked', async () => {
  offer.expires_at = Date.now() / 1000 - 1
  status.mockResolvedValue(offered())
  await mount()
  await open()
  expect(dialog().textContent).toMatch(/offer has expired.*check and download again/i)
  const update = within(dialog()).getByRole<HTMLButtonElement>('button', { name: 'Update and restart' })
  expect(update.disabled).toBe(true)
  await click(update)
  expect(status).toHaveBeenCalledTimes(1)
  expect(prepare).not.toHaveBeenCalled()
  expect(apply).not.toHaveBeenCalled()
  expect(screen.queryByRole('dialog', { name: 'Update and restart web desktop?' })).toBeNull()
})

it('keeps authorization recovery local, opening a fixed login tab and reconnecting only after an explicit successful session check', async () => {
  status.mockRejectedValue(new BrowserUpdateError('forbidden'))
  const openTab = vi.spyOn(window, 'open').mockReturnValue(null)
  await mount()
  await open()
  expect(dialog().textContent).toMatch(/session may have expired or lack permission/i)
  expect(dialog().textContent).toMatch(/authorized administrator/i)
  expect(window.hermesDesktop.browser!.signIn).not.toHaveBeenCalled()
  expect(checkSession).not.toHaveBeenCalled()
  expect(reconnect).not.toHaveBeenCalled()
  await click(within(dialog()).getByRole('button', { name: 'Sign in in another tab' }))
  expect(openTab).toHaveBeenCalledExactlyOnceWith('/login?next=/', '_blank', 'noopener,noreferrer')
  expect(checkSession).not.toHaveBeenCalled()
  expect(status).toHaveBeenCalledTimes(1)

  checkSession.mockRejectedValueOnce(new BrowserUpdateError('forbidden'))
  await click(within(dialog()).getByRole('button', { name: 'Check session and retry' }))
  expect(checkSession).toHaveBeenCalledTimes(1)
  expect(status).toHaveBeenCalledTimes(1)
  expect(reconnect).not.toHaveBeenCalled()

  status.mockResolvedValue(idle)
  await click(within(dialog()).getByRole('button', { name: 'Check session and retry' }))
  expect(checkSession).toHaveBeenCalledTimes(2)
  expect(status).toHaveBeenCalledTimes(2)
  expect(reconnect).toHaveBeenCalledTimes(1)
  expect(within(dialog()).getByRole('button', { name: 'Check and download' })).toBeTruthy()
  expect(prepare).not.toHaveBeenCalled()
  expect(apply).not.toHaveBeenCalled()
  expect(window.hermesDesktop.browser!.signIn).not.toHaveBeenCalled()
})

it('reports an absent helper, malformed responses, and network failures without reusing a stale install offer', async () => {
  await mount()

  for (const [kind, description] of [
    ['unsupported', /unavailable.*helper is not configured/i],
    ['invalid', /invalid response.*no new update will be started/i],
    ['network', /status is unavailable.*accepted update may still be running/i]
  ] as const) {
    status.mockResolvedValue(offered())
    await open()
    expect(within(dialog()).getByRole<HTMLButtonElement>('button', { name: 'Update and restart' }).disabled).toBe(false)
    await click(within(dialog()).getByRole('button', { name: 'Close' }))
    status.mockRejectedValue(new BrowserUpdateError(kind))
    await open()
    expect(dialog().textContent).toMatch(description)
    await cannotUpdate()
    await click(within(dialog()).getByRole('button', { name: 'Close' }))
  }

  expect(prepare).not.toHaveBeenCalled()
  expect(checkSession).not.toHaveBeenCalled()
  expect(reconnect).not.toHaveBeenCalled()
})

it('dismisses confirmation after an unknown apply error and exposes the unconfirmed outcome without resubmitting', async () => {
  apply.mockRejectedValueOnce(new Error('Connection lost after sending request'))
  await mount()
  await open()
  const confirmation = await requestConfirmation()
  await click(within(confirmation).getByRole('button', { name: 'Update and restart' }))
  expect(apply).toHaveBeenCalledExactlyOnceWith(offer.id)
  expect(screen.queryByRole('dialog', { name: 'Update and restart web desktop?' })).toBeNull()
  expect(dialog().textContent).toMatch(
    /not known whether the server accepted.*do not automatically send the request again/i
  )
  expect($browserUpdateView.get()).toMatchObject({ error: 'network', pendingId: offer.id, unknownOutcome: true })
  await click(within(dialog()).getByRole('button', { name: 'Close' }))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(10_000)
  })
  expect(status).toHaveBeenCalledTimes(1)
  expect(apply).toHaveBeenCalledTimes(1)
  expect(reconnect).not.toHaveBeenCalled()
})
