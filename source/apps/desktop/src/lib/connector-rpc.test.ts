import { JsonRpcGatewayError } from '@hermes/shared'
import { expect, it, vi } from 'vitest'

import { requestSessionConnector } from './connector-rpc'

it('never retries uncertain mutations, ownership refusals, missing methods, or other invalid params', async () => {
  for (const error of [
    new Error('Connection timed out'),
    new JsonRpcGatewayError('session not found or not owned by this transport', { code: 4001 }),
    new JsonRpcGatewayError('Method not found', { code: -32601 }),
    new JsonRpcGatewayError('connectors must be nonempty slugs; reconnect must be boolean', { code: 4000 }),
    new JsonRpcGatewayError('session_id required', { code: 5034 })
  ]) {
    const request = vi.fn().mockRejectedValue(error)
    await expect(requestSessionConnector(request, 'runtime')).rejects.toBe(error)
    expect(request).toHaveBeenCalledTimes(1)
  }
})

it('falls back once and surfaces a stock refusal without another retry', async () => {
  const refusal = new JsonRpcGatewayError('session not found or not owned by this transport', { code: 4001 })

  const request = vi
    .fn()
    .mockRejectedValueOnce(new JsonRpcGatewayError('session_id required', { code: 4000 }))
    .mockRejectedValueOnce(refusal)

  await expect(requestSessionConnector(request, 'runtime', { op_id: 'op' })).rejects.toBe(refusal)
  expect(request).toHaveBeenCalledTimes(2)
  expect(request).toHaveBeenLastCalledWith({ op_id: 'op', session_id: 'runtime' })
})
