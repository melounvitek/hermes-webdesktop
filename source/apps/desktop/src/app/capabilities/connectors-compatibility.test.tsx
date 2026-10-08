import { JsonRpcGatewayError } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

const { rpc } = vi.hoisted(() => ({ rpc: vi.fn() }))
vi.mock('@/store/gateway', async () => {
  const { atom } = await import('nanostores')

  return { $gateway: atom<unknown>(null), requestGatewayForAgent: rpc }
})

import { $notifications, clearNotifications } from '@/store/notifications'

import { useHostedConnectors } from './connectors/data/queries'

afterEach(() => {
  cleanup()
  clearNotifications()
  rpc.mockReset()
})

it.each(['stock', 'transient'] as const)(
  'distinguishes an unsupported account API from a %s failure',
  async failure => {
    const methods: string[] = []
    rpc.mockImplementation(async (_connection: string, _profile: string, method: string, params: unknown) => {
      methods.push(method)

      if (failure === 'transient') {
        throw new Error('Connection timed out')
      }

      if (method === 'connectors.list') {
        expect(params).toEqual({ owner: { type: 'account' } })
        throw new JsonRpcGatewayError('session_id required', { code: 4000, data: { reason: 'INVALID_PARAMS' } })
      }

      throw new JsonRpcGatewayError('Method not found', { code: -32601 })
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    const { result } = renderHook(() => useHostedConnectors({ connectionId: 'local', profile: 'default' }), {
      wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
    })

    await waitFor(() => expect(result.current.phase).toBe(failure === 'stock' ? 'unsupported' : 'failed'))
    expect(result.current.rows).toEqual([])
    expect(methods.sort()).toEqual([
      'connectors.accounts',
      'connectors.catalog',
      'connectors.list',
      'connectors.policy.get'
    ])
    expect($notifications.get()).toEqual([])
    client.clear()
  }
)
