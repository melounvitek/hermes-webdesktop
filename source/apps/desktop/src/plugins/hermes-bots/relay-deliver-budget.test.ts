import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { startBotRelay, stopBotRelay } from './relay'
import type { ProfileRoute } from './types'

const { hostMock, clearBotAttentionMock, noteBotAttentionMock } = vi.hoisted(() => ({
  hostMock: { profileRoutes: vi.fn(), requestProfile: vi.fn() },
  clearBotAttentionMock: vi.fn(),
  noteBotAttentionMock: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async () => ({
  host: hostMock,
  LruCache: (await import('../../lib/lru-cache')).LruCache
}))

vi.mock('./data', () => ({
  botHandle: vi.fn(),
  clearBotAttention: clearBotAttentionMock,
  noteBotAttention: noteBotAttentionMock
}))

// Chosen default API contract: 120s lock wait + two 600s attempts, then
// settlement/transport headroom. This does not discover backend defaults or
// verify every stock version; compatibility belongs in external stock tests.
const DEFAULT_WORK_BOUND_MS = (120 + 2 * 600) * 1000
const SETTLEMENT_HEADROOM_MS = 180_000

const sender: ProfileRoute = {
  connectionId: 'a',
  mode: 'remote',
  profile: 'default',
  targetProfile: 'default'
}

const target: ProfileRoute = { ...sender, connectionId: 'b' }

const envelope = {
  id: 'env-1',
  message: 'Research this',
  from_profile: 'research',
  from_handle: 'researcher',
  target_connection: 'b',
  target_profile: 'ops'
}

const backendFailure = Object.assign(new Error('turn exhausted its attempts'), {
  data: { reason: 'turn_timeout' }
})

beforeEach(() => {
  vi.useFakeTimers()
  vi.resetAllMocks()
  hostMock.profileRoutes.mockResolvedValue([sender, target])
})

afterEach(() => {
  stopBotRelay()
  vi.clearAllTimers()
  vi.useRealTimers()
})

async function startDelivery(responseAfterMs?: number, failure?: Error) {
  let queued = true
  hostMock.requestProfile.mockImplementation(
    (route: ProfileRoute, method: string, _params: unknown, timeoutMs = 30_000) => {
      if (method === 'bot_relay.outbox.drain') {
        const envelopes = route.connectionId === sender.connectionId && queued ? [envelope] : []

        if (envelopes.length) {
          queued = false
        }

        return Promise.resolve({ envelopes })
      }

      if (method === 'bot_relay.deliver') {
        // Model the host's request deadline, including its generic fallback.
        // The real relay must supply its longer budget at this seam.
        return new Promise((resolve, reject) => {
          const deadline = setTimeout(() => reject(new Error('request timed out')), timeoutMs)

          if (responseAfterMs !== undefined) {
            setTimeout(() => {
              clearTimeout(deadline)

              if (failure) {
                reject(failure)
              } else {
                resolve({ reply: 'Research complete' })
              }
            }, responseAfterMs)
          }
        })
      }

      return Promise.resolve({})
    }
  )

  startBotRelay()
  await vi.advanceTimersByTimeAsync(30_000)

  const deliveries = hostMock.requestProfile.mock.calls.filter(([, method]) => method === 'bot_relay.deliver')
  expect(deliveries).toHaveLength(1)
  const [route, , params, timeoutMs] = deliveries[0]
  expect(route).toEqual(target)
  expect(params).toEqual({
    profile: 'ops',
    message: envelope.message,
    from_profile: envelope.from_profile,
    from_handle: envelope.from_handle,
    from_connection: 'a'
  })
  expect(Number.isFinite(timeoutMs)).toBe(true)
  expect(timeoutMs).toBeGreaterThanOrEqual(DEFAULT_WORK_BOUND_MS + SETTLEMENT_HEADROOM_MS)

  return timeoutMs as number
}

function replies() {
  return hostMock.requestProfile.mock.calls.filter(([, method]) => method === 'bot_relay.reply')
}

describe('bot_relay.deliver request budget', () => {
  it.each([
    {
      name: 'a long response before the default work bound',
      delay: DEFAULT_WORK_BOUND_MS - 1,
      failure: undefined
    },
    {
      name: 'a typed failure during settlement headroom',
      delay: DEFAULT_WORK_BOUND_MS + SETTLEMENT_HEADROOM_MS - 1,
      failure: backendFailure
    }
  ])('forwards $name instead of a generic client timeout', async ({ delay, failure }) => {
    const timeoutMs = await startDelivery(delay, failure)
    await vi.advanceTimersByTimeAsync(delay - 1)
    expect(replies()).toEqual([])
    expect(noteBotAttentionMock).not.toHaveBeenCalled()
    expect(clearBotAttentionMock).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(1)
    expect(replies()).toEqual([
      [
        sender,
        'bot_relay.reply',
        failure
          ? { id: envelope.id, error: failure.message, reason: 'turn_timeout' }
          : { id: envelope.id, reply: 'Research complete' }
      ]
    ])

    if (failure) {
      expect(noteBotAttentionMock).toHaveBeenCalledExactlyOnceWith('b::ops', 'turn_timeout')
      expect(clearBotAttentionMock).not.toHaveBeenCalled()
    } else {
      expect(clearBotAttentionMock).toHaveBeenCalledExactlyOnceWith('b::ops')
      expect(noteBotAttentionMock).not.toHaveBeenCalled()
    }

    // A settled request must not send a second reply when its deadline passes.
    await vi.advanceTimersByTimeAsync(timeoutMs - delay)
    expect(replies()).toHaveLength(1)
  })

  it('posts a timeout to the sender only when the supplied request budget expires', async () => {
    const timeoutMs = await startDelivery()
    await vi.advanceTimersByTimeAsync(timeoutMs - 1)
    expect(replies()).toEqual([])
    expect(noteBotAttentionMock).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(1)
    expect(replies()).toEqual([
      [sender, 'bot_relay.reply', { id: envelope.id, error: 'request timed out' }]
    ])
    expect(noteBotAttentionMock).toHaveBeenCalledExactlyOnceWith('b::ops', 'request timed out')
    expect(clearBotAttentionMock).not.toHaveBeenCalled()
  })
})
