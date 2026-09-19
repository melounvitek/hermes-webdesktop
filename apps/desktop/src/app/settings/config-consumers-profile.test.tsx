import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { $terminalFontFamily } from '@/app/right-sidebar/terminal/terminal-font'
import type { HermesApiRequest } from '@/global'
import type { queryClient as QueryClientInstance } from '@/lib/query-client'
import type * as Profile from '@/store/profile'
import { $chatFontFamily } from '@/themes/chat-font'

import type * as ConfigRecord from '../hooks/use-config-record'

import type { ChatFontSetting as ChatFontSettingType } from './chat-font-setting'
import type { ConfigSettings as ConfigSettingsType } from './config-settings'
import type { TerminalFontSetting as TerminalFontSettingType } from './terminal-font-setting'

let ChatFontSetting: typeof ChatFontSettingType
let TerminalFontSetting: typeof TerminalFontSettingType
let ConfigSettings: typeof ConfigSettingsType
let configCache: typeof ConfigRecord
let queryClient: typeof QueryClientInstance
let profile: typeof Profile
const api = vi.fn<(request: HermesApiRequest) => Promise<unknown>>()

beforeAll(async () => {
  ;({ ChatFontSetting } = await import('./chat-font-setting'))
  ;({ TerminalFontSetting } = await import('./terminal-font-setting'))
  ;({ ConfigSettings } = await import('./config-settings'))
  configCache = await import('../hooks/use-config-record')
  ;({ queryClient } = await import('@/lib/query-client'))
  profile = await import('@/store/profile')
}, 60_000)

beforeEach(() => {
  vi.useFakeTimers()
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  profile.$activeGatewayProfile.set('A')
  profile.$profiles.set([])
  api.mockImplementation(async request => {
    if (request.path === '/api/profiles') {
      return { profiles: [] }
    }

    if (request.path === '/api/config') {
      return { desktop: { font_family: request.profile }, terminal: { font_family: request.profile } }
    }

    if (request.path === '/api/config/schema') {
      return { fields: {} }
    }

    if (request.path === '/api/model/info') {
      return { provider: 'test', model: 'model' }
    }

    if (request.path.startsWith('/api/model/options')) {
      return { providers: [{ name: 'Test', slug: 'test', models: ['model'], authenticated: true }] }
    }

    if (request.path === '/api/model/auxiliary') {
      return { main: { provider: 'test', model: 'model' }, tasks: [] }
    }

    if (request.path === '/api/model/moa') {
      return null
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
    await vi.advanceTimersByTimeAsync(10)
  })

it.each(['chat', 'terminal'])('uses the new owner’s cached %s font even when revalidation fails', async kind => {
  const FontSetting = kind === 'chat' ? ChatFontSetting : TerminalFontSetting
  render(
    <QueryClientProvider client={queryClient}>
      <FontSetting />
    </QueryClientProvider>
  )
  await flush()
  expect((screen.getByRole('combobox') as HTMLInputElement).value).toBe('A')
  act(() => configCache.hermesConfigCacheWriter('B')({ desktop: { font_family: 'B' }, terminal: { font_family: 'B' } }))
  api.mockRejectedValue(new Error('offline'))
  act(() => profile.$activeGatewayProfile.set('B'))
  await flush()
  const input = screen.getByRole('combobox') as HTMLInputElement
  expect(input.disabled).toBe(false)
  expect(input.value).toBe('B')
  expect(api.mock.calls.every(([request]) => request.method !== 'PUT')).toBe(true)
})

it.each(['chat', 'terminal'])('revalidates the %s font after its save finishes in another profile', async kind => {
  const FontSetting = kind === 'chat' ? ChatFontSetting : TerminalFontSetting
  const font = kind === 'chat' ? $chatFontFamily : $terminalFontFamily
  const section = kind === 'chat' ? 'desktop' : 'terminal'
  let finishSave!: (value: unknown) => void
  let finishRead!: (value: unknown) => void

  const pendingSave = new Promise(resolve => {
    finishSave = resolve
  })

  const pendingRead = new Promise(resolve => {
    finishRead = resolve
  })

  render(
    <QueryClientProvider client={queryClient}>
      <FontSetting />
    </QueryClientProvider>
  )
  await flush()
  const input = screen.getByRole('combobox') as HTMLInputElement
  expect(input.value).toBe('A')
  api.mockImplementationOnce(() => pendingSave)
  fireEvent.change(input, { target: { value: 'A saved' } })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(550)
  })
  expect(api.mock.calls.at(-1)?.[0]).toMatchObject({
    method: 'PUT',
    path: '/api/config',
    profile: 'A',
    body: { config: { [section]: { font_family: 'A saved' } } }
  })

  act(() => profile.$activeGatewayProfile.set('B'))
  await flush()
  expect(input.value).toBe('B')
  await act(async () => finishSave({ ok: true }))
  await flush()
  expect(input.value).toBe('B')
  expect(font.get()).toBe('B')

  const defaultApi = api.getMockImplementation()!
  api.mockImplementation(request =>
    request.path === '/api/config' && request.profile === 'A' ? pendingRead : defaultApi(request)
  )
  act(() => profile.$activeGatewayProfile.set('A'))
  await flush()
  expect(api.mock.calls.at(-1)?.[0]).toMatchObject({ path: '/api/config', profile: 'A' })
  expect(input.value).toBe('A')
  await act(async () => finishRead({ [section]: { font_family: 'A saved' } }))
  await flush()
  expect(queryClient.getQueryData(configCache.hermesConfigKey('A'))).toEqual({
    [section]: { font_family: 'A saved' }
  })
  expect(input.value).toBe('A saved')
  expect(font.get()).toBe('A saved')
  expect(api.mock.calls.filter(([request]) => request.method === 'PUT')).toHaveLength(1)
})

it.each(['chat', 'terminal'])('preserves an edited %s font when revalidation finishes', async kind => {
  const FontSetting = kind === 'chat' ? ChatFontSetting : TerminalFontSetting
  const font = kind === 'chat' ? $chatFontFamily : $terminalFontFamily
  const section = kind === 'chat' ? 'desktop' : 'terminal'
  let finishRead!: (value: unknown) => void

  const pendingRead = new Promise(resolve => {
    finishRead = resolve
  })

  render(
    <QueryClientProvider client={queryClient}>
      <FontSetting />
    </QueryClientProvider>
  )
  await flush()
  api.mockImplementationOnce(() => pendingRead)
  act(() => void configCache.invalidateHermesConfig('A'))
  await flush()
  const input = screen.getByRole('combobox') as HTMLInputElement
  fireEvent.change(input, { target: { value: 'My draft' } })
  await act(async () => finishRead({ [section]: { font_family: 'Server font' } }))
  await flush()
  expect(queryClient.getQueryData(configCache.hermesConfigKey('A'))).toEqual({
    [section]: { font_family: 'Server font' }
  })
  expect(input.value).toBe('My draft')
  expect(font.get()).toBe('My draft')
  api.mockResolvedValueOnce({ ok: true })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(550)
  })
  expect(api.mock.calls.filter(([request]) => request.method === 'PUT').map(([request]) => request)).toEqual([
    expect.objectContaining({ profile: 'A', body: { config: { [section]: { font_family: 'My draft' } } } })
  ])
})

it('does not publish an old model save after ConfigSettings remounts for another profile', async () => {
  let finish!: (value: unknown) => void

  const pending = new Promise(resolve => {
    finish = resolve
  })

  const changed = vi.fn()
  render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <ConfigSettings
          activeSectionId="model"
          importInputRef={createRef<HTMLInputElement>()}
          onMainModelChanged={changed}
        />
      </QueryClientProvider>
    </MemoryRouter>
  )
  await flush()
  api.mockImplementationOnce(() => pending)
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
  expect(api.mock.calls.at(-1)?.[0].path).toBe('/api/model/set')
  act(() => profile.$activeGatewayProfile.set('B'))
  await flush()
  await act(async () => {
    finish({ ok: true, provider: 'test', model: 'model' })
  })
  await flush()
  expect(changed).not.toHaveBeenCalled()
})
