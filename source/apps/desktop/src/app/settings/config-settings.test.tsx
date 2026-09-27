import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as ConfigApi from '@/api/config'
import { $settingsRequestProfile } from '@/store/settings-scope'

import type { ConfigSettings as ConfigSettingsType } from './config-settings'

// The vi.mock factory below replaces the computed (read-only) atom with a
// writable one; narrow the import back so tests can drive it.
const scopeProfileMock = $settingsRequestProfile as unknown as { set: (value: string) => void }

const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()
const saveHermesConfig = vi.fn()
const getElevenLabsVoices = vi.fn()

// Keep the read-origin helpers and concrete owner cache keys live.
vi.mock('@/hermes', async () => ({
  ...(await vi.importActual<typeof ConfigApi>('@/api/config')),
  profileScopeKey: (await import('@/api/client')).profileScopeKey,
  getHermesConfigRecord: (profile?: unknown) => getHermesConfigRecord(profile),
  getHermesConfigSchema: () => getHermesConfigSchema(),
  saveHermesConfigRecord: (config: unknown, profile?: unknown) => saveHermesConfig(config, profile),
  getElevenLabsVoices: () => getElevenLabsVoices(),
  setApiRequestProfile: () => {}
}))

vi.mock('../hooks/use-on-profile-switch', () => ({
  useOnProfileSwitch: () => {}
}))

// The real stores pull in the gateway/profile stack, which needs a live
// backend connection. This page only reads the "applies to" scope override
// and the repo-discovery signature, neither of which this test touches. The
// scope chip it renders also reads the selected profile and the loud-note
// selector, so those are stubbed to the single-profile default shape.
vi.mock('@/store/settings-scope', () => ({
  // The real store derives this from the displayed $settingsScopeProfile
  // (never undefined for a real profile — settings-scope.test.ts pins that);
  // here it is a plain atom so the page's threading of it can be driven.
  $settingsRequestProfile: atom<string | undefined>('default'),
  $settingsScopeEditsNonDefault: atom(false),
  $settingsScopeOverride: atom<null | string>(null),
  $settingsScopeProfile: atom<string>('default')
}))

vi.mock('@/store/projects', () => ({
  repoDiscoveryPolicyFromConfig: () => ({ enabled: true, roots: [], exclude_paths: [] }),
  repoDiscoveryPolicySignature: (policy: unknown) => JSON.stringify(policy),
  scanAndRecordRepos: vi.fn().mockResolvedValue(undefined)
}))

// The module graph behind ConfigSettings is large (1.5s cold here, >10s on a
// saturated CI runner); load it once under the hook timeout so the 15s test
// budget is spent on the autosave behaviour, not on transform + import.
let ConfigSettings: typeof ConfigSettingsType

beforeAll(async () => {
  ;({ ConfigSettings } = await import('./config-settings'))
}, 60_000)

beforeEach(() => {
  scopeProfileMock.set('default')
  getElevenLabsVoices.mockResolvedValue({ available: false })
  getHermesConfigSchema.mockResolvedValue({ fields: {} })
  saveHermesConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

function renderConfigSettings(activeSectionId = 'safety', subpage?: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const importInputRef = createRef<HTMLInputElement>()

  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ConfigSettings activeSectionId={activeSectionId} importInputRef={importInputRef} subpage={subpage} />
      </QueryClientProvider>
    </MemoryRouter>
  )

  return { importInputRef }
}

describe('ConfigSettings autosave', () => {
  it.each([
    [true, undefined],
    [false, undefined],
    [true, 'desktop'],
    [false, 'desktop']
  ] as const)('only offers native preferences outside browser=%s on page %s', async (browser, subpage) => {
    vi.stubGlobal('hermesDesktop', { browser })
    getHermesConfigRecord.mockResolvedValue({})
    const { importInputRef } = renderConfigSettings('advanced', subpage)
    await waitFor(() => expect(importInputRef.current).not.toBeNull())
    expect(Boolean(screen.queryByText('Keep computer awake'))).toBe(!browser)
    expect(Boolean(screen.queryByText('Disable F12 DevTools'))).toBe(!browser)
    expect(Boolean(screen.queryByText('Always open links in external browser'))).toBe(!browser)
  })

  it('renders and saves the Codex compression auto-raise setting', async () => {
    getHermesConfigRecord.mockResolvedValue({
      compression: { codex_gpt55_autoraise: true }
    })
    getHermesConfigSchema.mockResolvedValue({
      fields: {
        'compression.codex_gpt55_autoraise': { type: 'boolean' }
      }
    })

    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      renderConfigSettings('memory')

      expect(await screen.findByText('Codex Compression Auto-Raise')).toBeTruthy()
      expect(screen.getByText('Raise compression to 85% for supported ChatGPT Codex OAuth models.')).toBeTruthy()

      screen.getByRole('switch').click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() =>
        expect(saveHermesConfig).toHaveBeenCalledWith(
          { compression: { codex_gpt55_autoraise: false } },
          expect.objectContaining({ connectionId: null, profile: 'default' })
        )
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it('sends a later revert instead of diffing it away against the stale page-load baseline', async () => {
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false }, other: 'untouched' })

    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      renderConfigSettings()

      const toggle = await screen.findByRole('switch')

      // Edit: flip checkpoints.enabled on, let the debounced autosave fire.
      toggle.click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(1))
      expect(saveHermesConfig.mock.calls[0][0]).toEqual({ checkpoints: { enabled: true } })

      // Revert: flip it back to its original value and let autosave fire again.
      toggle.click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(2))
      // Must still explicitly send the reverted value — diffing against the
      // never-advanced page-load baseline would produce an empty patch here
      // (the field is back to its original value) and leave disk stuck at
      // `enabled: true` from the first save.
      expect(saveHermesConfig.mock.calls[1][0]).toEqual({ checkpoints: { enabled: false } })
    } finally {
      vi.useRealTimers()
    }
  })

  it('threads the "Applies to" request scope into both the config read and the autosave write', async () => {
    // #118432: the request scope is the concrete profile the page displays
    // (see settings-scope.test.ts). The page must carry it into the read AND
    // the write — a read scoped to B with a write that falls back to the
    // ambient (launch) profile is exactly the silent cross-profile write.
    scopeProfileMock.set('nash')
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false } })

    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      renderConfigSettings()

      await vi.waitFor(() =>
        expect(getHermesConfigRecord).toHaveBeenCalledWith(
          expect.objectContaining({ connectionId: null, profile: 'nash' })
        )
      )

      ;(await screen.findByRole('switch')).click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() =>
        expect(saveHermesConfig).toHaveBeenCalledWith(
          { checkpoints: { enabled: true } },
          getHermesConfigRecord.mock.calls[0][0]
        )
      )
    } finally {
      vi.useRealTimers()
    }
  })
})
