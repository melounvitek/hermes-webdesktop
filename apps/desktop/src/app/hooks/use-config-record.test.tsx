import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import type { PropsWithChildren } from 'react'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import type * as ApiClient from '@/api/client'
import type { HermesApiRequest } from '@/global'
import type { ProfileScope } from '@/hermes'
import type { queryClient as QueryClientInstance } from '@/lib/query-client'
import type * as Profile from '@/store/profile'

import type * as ConfigRecord from './use-config-record'

let config: typeof ConfigRecord
let client: typeof ApiClient
let profile: typeof Profile
let queryClient: typeof QueryClientInstance
const api = vi.fn<(request: HermesApiRequest) => Promise<unknown>>()

beforeAll(async () => {
  config = await import('./use-config-record')
  client = await import('@/api/client')
  profile = await import('@/store/profile')
  ;({ queryClient } = await import('@/lib/query-client'))
}, 60_000)

beforeEach(() => {
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  profile.$activeGatewayProfile.set('A')
  client.setApiRequestProfile('A')
  client.setApiRequestConnection(null)
  api.mockImplementation(async request => ({ owner: request.profile, connection: request.connectionId }))
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.clearAllMocks()
})

function wrapper({ children }: PropsWithChildren) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}

it('isolates implicit and explicit owners; captured writers cannot roll back the newly active sibling', async () => {
  const { result } = renderHook(() => config.useHermesConfigRecord(), { wrapper })
  await waitFor(() => expect(result.current.data).toEqual({ owner: 'A' }))
  const keyA = config.hermesConfigKey()
  const writeA = config.hermesConfigCacheWriter(result.current.scope)
  const previousA = result.current.data
  act(() => writeA({ owner: 'optimistic A' }))

  act(() => profile.$activeGatewayProfile.set('B'))
  await waitFor(() => expect(result.current.data).toEqual({ owner: 'B' }))
  const keyB = config.hermesConfigKey()
  expect(config.hermesConfigKey(result.current.scope)).toEqual(keyB)
  expect(keyB).not.toEqual(keyA)
  expect(config.hermesConfigKey('A')).toEqual(keyA)
  expect(config.hermesConfigKey(null)).toEqual(keyB)
  act(() => writeA(previousA)) // delayed rollback belongs to A
  expect(queryClient.getQueryData(keyA)).toEqual({ owner: 'A' })
  expect(queryClient.getQueryData(keyB)).toEqual({ owner: 'B' })

  client.setApiRequestConnection('remote')
  expect(config.hermesConfigKey('B')).toEqual(config.hermesConfigKey({ connectionId: 'remote', profile: 'B' }))
  expect(config.hermesConfigKey('B')).not.toEqual(keyB)
  expect(config.hermesConfigKey({ connectionId: 'local', profile: 'B' })).not.toEqual(keyB)
  // No argument retains the broad invalidation contract for all scopes.
  await config.invalidateHermesConfig()
  expect(queryClient.getQueryState(keyA)?.isInvalidated).toBe(true)
})

it('keeps the captured request owner when an old query is refetched after a profile switch', async () => {
  const scopes: ProfileScope[] = [undefined, 'A', { connectionId: 'remote', profile: 'A' }]

  for (const scope of scopes) {
    const { result, unmount } = renderHook(() => config.useHermesConfigRecord(scope), { wrapper })
    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(api.mock.calls.at(-1)?.[0].priority).toBe(scope == null ? undefined : 'foreground')
    const key = config.hermesConfigKey(scope)
    unmount()
    client.setApiRequestProfile('B')
    client.setApiRequestConnection('sibling')
    api.mockClear()
    await queryClient.refetchQueries({ queryKey: key, exact: true, type: 'all' })
    expect(api.mock.calls[0][0].profile).toBe('A')
    expect(api.mock.calls[0][0].priority).toBe(scope == null ? undefined : 'foreground')
    expect(api.mock.calls[0][0].connectionId).toBe(typeof scope === 'object' ? 'remote' : undefined)
    client.setApiRequestProfile('A')
    client.setApiRequestConnection(null)
    queryClient.clear()
  }
})
