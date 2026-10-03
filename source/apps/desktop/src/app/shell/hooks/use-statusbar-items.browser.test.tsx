import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type * as I18n from '@/i18n'

// Each case cold-imports the statusbar module graph, which is slow in a full run.
vi.setConfig({ testTimeout: 60_000 })

const { openBrowserUpdates, copy } = vi.hoisted(() => ({
  openBrowserUpdates: vi.fn(),
  copy: {
    label: (version: string) => `Browser fixture ${version}`,
    unknownBuild: 'Unknown fixture build',
    localChanges: 'Fixture local changes',
    title: 'Fixture browser updates'
  }
}))

vi.mock('@/browser/updates', () => ({ openBrowserUpdates }))
vi.mock('@/i18n', async importOriginal => {
  const actual = await importOriginal<typeof I18n>()

  return {
    ...actual,
    useI18n: () => {
      const value = actual.useI18n()

      return { ...value, t: { ...value.t, browserUpdates: copy } }
    }
  }
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

it.each([
  { browser: true, build: { revision: 'abcdef0123456789abcdef0123456789abcdef01', dirty: false } },
  { browser: true, build: { revision: 'abcdef0123456789abcdef0123456789abcdef01', dirty: true } },
  { browser: true, build: { revision: null, dirty: null } },
  { browser: true, build: undefined },
  { browser: false, build: { revision: 'abcdef0123456789abcdef0123456789abcdef01', dirty: false } }
])('renders loaded browser provenance without fetching (%j)', async ({ browser, build }) => {
  vi.resetModules()
  vi.stubGlobal('__HERMES_BROWSER_BUILD__', build)
  const fetch = vi.fn()
  vi.stubGlobal('fetch', fetch)
  vi.stubGlobal('hermesDesktop', browser ? { browser: {} } : {})

  const { MemoryRouter } = await import('react-router')
  const { useStatusbarItems } = await import('./use-statusbar-items')
  const { StatusbarControls } = await import('../statusbar-controls')
  const { $statusbarHiddenIds } = await import('@/store/statusbar-prefs')
  // Update visibility is not a preference, including a stale persisted hidden id.
  const hiddenIds = $statusbarHiddenIds.get()
  $statusbarHiddenIds.set([...hiddenIds, 'version-browser'])

  function Footer() {
    const { statusbarItems } = useStatusbarItems({
      agentsOpen: false,
      chatOpen: false,
      commandCenterOpen: false,
      extraLeftItems: [],
      extraRightItems: [],
      freshDraftReady: false,
      gatewayState: 'closed',
      inferenceStatus: null,
      openAgents: () => {},
      openCommandCenterSection: () => {},
      requestGateway: async () => undefined as never,
      statusSnapshot: null,
      toggleCommandCenter: () => {}
    })

    return <StatusbarControls items={statusbarItems} />
  }

  render(
    <MemoryRouter>
      <Footer />
    </MemoryRouter>
  )
  expect(fetch).not.toHaveBeenCalled()
  expect(openBrowserUpdates).not.toHaveBeenCalled()

  if (browser) {
    const version = build?.revision?.slice(0, 10) ?? copy.unknownBuild
    const label = `${copy.label(version)}${build?.dirty ? ` · ${copy.localChanges}` : ''}`
    fireEvent.click(screen.getByRole('button', { name: label }))
    expect(openBrowserUpdates).toHaveBeenCalledExactlyOnceWith()
    expect(fetch).not.toHaveBeenCalled()
  } else {
    expect(screen.queryByRole('button', { name: /Browser fixture/ })).toBeNull()
  }

  $statusbarHiddenIds.set(hiddenIds)
})
