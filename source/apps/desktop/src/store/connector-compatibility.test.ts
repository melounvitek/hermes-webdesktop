import { JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { openConnectionDoneLink, reissueConnectionTarget } from '@/components/assistant-ui/connector-tool'
import { queryClient } from '@/lib/query-client'

import {
  $connectionRequests,
  normalizeConnectionRequest,
  respondToConnectionRequest,
  setConnectionRequest
} from './connection-request'
import { prefetchConnectorCatalog } from './connector-catalog'
import { setPrimaryGateway, setPrimaryGatewayConnectionId } from './gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from './session'

const owner = { connectionId: 'local', profile: 'default' }

const request = normalizeConnectionRequest(
  {
    deadline_at: 1800000000,
    timeout_seconds: 120,
    op_id: 'op',
    seq: 1,
    tool_call_id: 'call',
    targets: [{ name: 'gmail', kind: 'connector', action: 'connect', state: 'expired' }]
  },
  'runtime'
)!

afterEach(() => {
  queryClient.clear()
  setPrimaryGateway(null)
  $connectionRequests.set({})
  _resetSessionOwnerHintsForTests({ storage: true })
})

describe('session connector wire compatibility', () => {
  it.each(['stock', 'owner'] as const)('lists, retries, wakes and responds on the %s contract', async version => {
    const accepted: string[] = []

    const rpc = vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (version === 'stock') {
        if (method === 'connectors.operation.wake') {
          throw new JsonRpcGatewayError('Method not found', { code: -32601 })
        }

        // Stock validates session ownership before any operation, and list/connect reject extras.
        if (!params.session_id) {
          throw new JsonRpcGatewayError('session_id required', { code: 4000, data: { reason: 'INVALID_PARAMS' } })
        }

        expect(params.session_id).toBe('runtime')
        expect(params).not.toHaveProperty('owner')
      } else {
        expect(params.owner).toEqual({ type: 'session', session_id: 'runtime' })
      }

      accepted.push(method)

      return {
        available: true,
        connectors: [{ connector: 'gmail' }],
        targets: [{ name: 'gmail', connect_url: 'https://example.com/connect' }]
      }
    })

    setPrimaryGateway({ request: rpc } as never)
    setPrimaryGatewayConnectionId('local')
    setSessionOwnerHint('runtime', owner)
    setSessionOwnerHint('stored', owner)
    setConnectionRequest(request)

    prefetchConnectorCatalog('stored', 'runtime')
    await vi.waitFor(() =>
      expect(queryClient.getQueryData(['onboarding', 'connectors.list', 'stored', 'runtime'])).toMatchObject({
        status: 'ready'
      })
    )
    expect(await reissueConnectionTarget(owner, request, 'gmail')).toBe('https://example.com/connect')
    await openConnectionDoneLink('op', vi.fn(), () => 'stored')
    expect(await respondToConnectionRequest(request, { settled_by: 'continue' })).toBe(true)
    expect(accepted).toEqual(
      version === 'stock'
        ? ['connectors.list', 'connectors.connect', 'connection.respond']
        : ['connectors.list', 'connectors.connect', 'connectors.operation.wake', 'connection.respond']
    )
    expect(rpc.mock.calls.filter(([method]) => method === 'connectors.operation.wake')).toHaveLength(1)
  })
})
