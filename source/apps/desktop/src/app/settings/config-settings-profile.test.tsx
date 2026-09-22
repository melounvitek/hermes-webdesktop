import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'
import type { queryClient as QueryClientInstance } from '@/lib/query-client'
import type * as Profile from '@/store/profile'
import type * as SettingsScope from '@/store/settings-scope'
import type { HermesConfigRecord } from '@/types/hermes'

import type * as ConfigRecord from '../hooks/use-config-record'

import type { ConfigSettings as ConfigSettingsType } from './config-settings'

vi.mock('./config-field', () => ({
  ConfigField: ({
    schemaKey,
    value,
    onChange
  }: {
    schemaKey: string
    value: number
    onChange: (v: number) => void
  }) => <input aria-label={schemaKey} onChange={event => onChange(Number(event.target.value))} value={value} />
}))
vi.mock('./model-settings', () => ({ ModelSettings: () => null, ModelSettingsSkeleton: () => null }))
vi.mock('./pool-limits-setting', () => ({ PoolLimitsSetting: () => null }))
vi.mock('./quick-entry-settings', () => ({ QuickEntrySettings: () => null }))
vi.mock('./profile-scope', () => ({ SettingsProfileScope: () => null }))
vi.mock('./memory/connect', () => ({ MemoryConnect: () => null }))
vi.mock('./memory/provider-config-panel', () => ({ ProviderConfigPanel: () => null }))
vi.mock('@/store/projects', () => ({
  repoDiscoveryPolicyFromConfig: () => ({}),
  repoDiscoveryPolicySignature: () => '',
  scanAndRecordRepos: vi.fn()
}))

let ConfigSettings: typeof ConfigSettingsType
let configCache: typeof ConfigRecord
let queryClient: typeof QueryClientInstance
let profile: typeof Profile
let settings: typeof SettingsScope
const api = vi.fn<(request: HermesApiRequest) => Promise<unknown>>()
const saved = vi.fn()
const a = { agent: { max_turns: 11 } }
const b = { agent: { max_turns: 22 } }

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

beforeAll(async () => {
  ;({ ConfigSettings } = await import('./config-settings'))
  configCache = await import('../hooks/use-config-record')
  ;({ queryClient } = await import('@/lib/query-client'))
  profile = await import('@/store/profile')
  settings = await import('@/store/settings-scope')
}, 60_000)

beforeEach(() => {
  vi.useFakeTimers()
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  profile.$activeGatewayProfile.set('A')
  settings.$settingsScopeOverride.set(null)
  api.mockImplementation(async request => {
    if (request.path === '/api/config/schema') {
      return { fields: { 'agent.max_turns': { type: 'integer' } } }
    }

    if (request.path === '/api/config' && request.method === 'PUT') {
      return { ok: true }
    }

    if (request.path === '/api/config') {
      return request.profile === 'B' ? b : a
    }

    return { available: false }
  })
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.useRealTimers()
  vi.clearAllMocks()
})

const flush = () =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(1)
  })

const debounce = () =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(600)
  })

const input = () => screen.getByRole('textbox', { name: 'agent.max_turns' }) as HTMLInputElement
const writes = () => api.mock.calls.map(([request]) => request).filter(request => request.method === 'PUT')

function mount() {
  return render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <ConfigSettings
          activeSectionId="advanced"
          importInputRef={createRef<HTMLInputElement>()}
          onConfigSaved={saved}
        />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

function switchProfile(name: string) {
  act(() => {
    profile.$activeGatewayProfile.set(name)
  })
}

describe('ConfigSettings edit ownership', () => {
  it.each(['mounted', 'reopened'] as const)('never seeds a sibling cache or saves on navigation (%s)', async mode => {
    let view = mount()
    await flush()
    expect(input().value).toBe('11')
    // Exercise the production cache writer and provider, not two query clients.
    act(() => configCache.hermesConfigCacheWriter()({ agent: { max_turns: 13 } }))
    await flush()
    expect(input().value).toBe('11') // background changes cannot replace this draft
    fireEvent.change(input(), { target: { value: '12' } })
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: configCache.hermesConfigKey('A') })
    })
    await flush()
    expect(input().value).toBe('12') // same-profile revalidation preserves unsaved edits

    const pendingB = deferred<HermesConfigRecord>()
    api.mockImplementation(async request => {
      if (request.path === '/api/config/schema') {
        return { fields: { 'agent.max_turns': { type: 'integer' } } }
      }

      if (request.path === '/api/config') {
        return request.profile === 'B' ? pendingB.promise : a
      }

      return { available: false }
    })

    if (mode === 'reopened') {
      view.unmount()
    }

    switchProfile('B')

    if (mode === 'reopened') {
      view = mount()
    }

    await flush()
    expect(screen.queryByRole('textbox', { name: 'agent.max_turns' })).toBeNull()

    switchProfile('A')
    await flush()
    expect(input().value).not.toBe('22')
    await act(async () => {
      pendingB.resolve(b)
    })
    await flush()
    expect(input().value).not.toBe('22')
    switchProfile('B')
    await flush()
    expect(input().value).toBe('22')

    // Explicit selector changes use the same concrete-target reset.
    act(() => settings.$settingsScopeOverride.set('A'))
    await flush()
    expect(input().value).toBe('11')
    await debounce()
    expect(writes()).toEqual([])
    expect(saved).not.toHaveBeenCalled()
  })

  it.each(['unmount', 'scope switch'])(
    'cancels pending saves on %s; an in-flight save only updates its own cache',
    async mode => {
      const flight = deferred<{ ok: boolean }>()
      api.mockImplementation(async request => {
        if (request.path === '/api/config/schema') {
          return { fields: { 'agent.max_turns': { type: 'integer' } } }
        }

        if (request.method === 'PUT') {
          return flight.promise
        }

        if (request.path === '/api/config') {
          return request.profile === 'B' ? b : a
        }

        return { available: false }
      })
      const view = mount()
      await flush()
      fireEvent.change(input(), { target: { value: '12' } })
      await debounce()
      expect(writes()).toHaveLength(1)
      fireEvent.change(input(), { target: { value: '13' } })
      await debounce() // queued behind the first request
      fireEvent.change(input(), { target: { value: '14' } }) // still debouncing

      if (mode === 'unmount') {
        view.unmount()
        switchProfile('B')
        mount()
      } else {
        act(() => settings.$settingsScopeOverride.set('B'))
      }

      await flush()
      await act(async () => {
        flight.resolve({ ok: true })
      })
      await debounce()
      expect(writes()).toHaveLength(1)
      expect(writes()[0].profile).toBe('A')
      expect(input().value).toBe('22')
      expect(queryClient.getQueryData(configCache.hermesConfigKey('A'))).toEqual({ agent: { max_turns: 12 } })
      expect(queryClient.getQueryData(configCache.hermesConfigKey('B'))).toEqual(b)
      expect(saved).not.toHaveBeenCalled()
    }
  )
})
