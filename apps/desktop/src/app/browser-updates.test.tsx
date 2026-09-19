import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DesktopUpdateStatus } from '@/global'
import { I18nProvider } from '@/i18n/context'
import { $connection } from '@/store/session'
import {
  $backendUpdateStatus,
  $updateOverlayOpen,
  $updateOverlayTarget,
  $updateStatus,
  checkBackendUpdates,
  resetUpdateApplyState
} from '@/store/updates'
import type { StatusResponse } from '@/types/hermes'

import { CommandCenterView } from './command-center'
import { AboutSettings } from './settings/about-settings'
import { useStatusbarItems } from './shell/hooks/use-statusbar-items'
import { UpdatesOverlay } from './updates-overlay'

vi.mock('./command-center/maintenance', () => ({ MaintenancePanel: () => null }))

const check = vi.fn()
const apply = vi.fn()
const api = vi.fn()
const offered: DesktopUpdateStatus = { supported: true, behind: 2, updateAvailable: true, fetchedAt: 1 }

beforeEach(() => {
  check.mockReset().mockResolvedValue(offered)
  apply.mockReset().mockResolvedValue({ ok: false })
  api.mockReset().mockImplementation(async ({ path }: { path: string }) => {
    if (path === '/api/hermes/update') {
      throw new Error('Fixture stops before any real updater')
    }

    if (path.startsWith('/api/logs')) {
      return { lines: [] }
    }

    return {}
  })
  vi.stubGlobal('hermesDesktop', { api, updates: { check, apply } })
  $connection.set({
    baseUrl: 'http://fixture.invalid',
    token: '',
    wsUrl: '',
    mode: 'remote',
    logs: [],
    isFullscreen: false,
    nativeOverlayWidth: 0,
    windowButtonPosition: null
  })
  $updateStatus.set(offered)
  $backendUpdateStatus.set(null)
  $updateOverlayOpen.set(false)
  resetUpdateApplyState()
})

afterEach(() => {
  cleanup()
  $updateOverlayOpen.set(false)
  $updateOverlayTarget.set('client')
  $backendUpdateStatus.set(null)
  $updateStatus.set(null)
  $connection.set(null)
  vi.unstubAllGlobals()
})

async function mount(ui: React.ReactNode) {
  await act(async () => {
    render(
      <MemoryRouter>
        <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
          {ui}
        </I18nProvider>
      </MemoryRouter>
    )
  })
}

it.each([true, false])('gates About and Command Center update gestures for browser=%s', async browser => {
  if (browser) {
    window.hermesDesktop.browser = { authRequired: true, signIn: vi.fn() }
  }

  await mount(
    <>
      <AboutSettings />
      <CommandCenterView
        initialSection="system"
        onClose={() => {}}
        onDeleteSession={async () => {}}
        onOpenSession={() => {}}
      />
    </>
  )

  expect(screen.getByRole('link', { name: 'Release notes' })).toBeTruthy()

  if (browser) {
    expect(screen.queryByRole('button', { name: 'Check now' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Update now' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Update Hermes' })).toBeNull()
    expect(screen.queryByText(/Hermes checks for updates automatically/)).toBeNull()
    expect(api.mock.calls.some(([request]) => request.path === '/api/hermes/update')).toBe(false)
    expect(check).not.toHaveBeenCalled()
  } else {
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Check now' })))
    expect(check).toHaveBeenCalledWith({ force: true })
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Update Hermes' })))
    expect(api.mock.calls.some(([request]) => request.path === '/api/hermes/update' && request.method === 'POST')).toBe(
      true
    )
  }
})

it.each([true, false])('gates statusbar update gestures despite cached offers for browser=%s', async browser => {
  if (browser) {
    window.hermesDesktop.browser = { authRequired: true, signIn: vi.fn() }
  }

  $backendUpdateStatus.set(offered)
  let items: ReturnType<typeof useStatusbarItems>['statusbarItems'] = []

  const options = {
    agentsOpen: false,
    chatOpen: false,
    commandCenterOpen: false,
    extraLeftItems: [],
    extraRightItems: [],
    freshDraftReady: false,
    gatewayState: 'connected',
    inferenceStatus: null,
    openAgents: vi.fn(),
    openCommandCenterSection: vi.fn(),
    requestGateway: vi.fn(),
    statusSnapshot: { version: 'fixture' } as StatusResponse,
    toggleCommandCenter: vi.fn()
  }

  function StatusbarProbe() {
    items = useStatusbarItems(options).statusbarItems

    return null
  }

  await mount(<StatusbarProbe />)
  const backend = items.find(item => item.id === 'version-backend')!
  expect(backend.variant).toBe(browser ? 'text' : 'action')

  if (browser) {
    expect(backend.onSelect).toBeUndefined()
    expect(backend.title).not.toMatch(/update|latest|compatible/i)
    expect(items.find(item => item.id === 'version-client')?.hidden).toBe(true)
    expect(api).not.toHaveBeenCalled()
  } else {
    await act(async () => backend.onSelect?.({ shiftKey: false }))
    expect($updateOverlayOpen.get()).toBe(true)
    expect(api.mock.calls.some(([request]) => request.path.startsWith('/api/hermes/update/check'))).toBe(true)
  }
})

it('does not render a forced browser updater overlay or check on its behalf', async () => {
  window.hermesDesktop.browser = { authRequired: true, signIn: vi.fn() }
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)
  await mount(<UpdatesOverlay />)
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(api).not.toHaveBeenCalled()
  expect(apply).not.toHaveBeenCalled()
})

it.each([
  { behind: null, update_available: false, failed: true },
  { behind: -1, update_available: false, failed: true },
  { behind: 0, update_available: false, failed: false },
  { behind: null, update_available: true, failed: false }
])('renders honest backend check state for %j', async result => {
  api.mockResolvedValue({
    ...result,
    can_apply: true,
    current_version: 'fixture',
    message: 'Update source unavailable'
  })
  await checkBackendUpdates()
  $updateOverlayTarget.set('backend')
  $updateOverlayOpen.set(true)
  await mount(<UpdatesOverlay />)

  expect(Boolean(screen.queryByText('The backend is running the latest version.'))).toBe(result.behind === 0)
  expect(Boolean(screen.queryByText('Update source unavailable'))).toBe(result.failed)
  expect(Boolean(screen.queryByRole('button', { name: 'Update now' }))).toBe(result.update_available)
  expect(api.mock.calls.every(([request]) => request.method !== 'POST')).toBe(true)
})
