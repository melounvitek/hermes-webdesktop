import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'

import { $browserUpdate, requestBrowserUpdate } from './updates'

const DIRECTORY = '~/.local/lib/hermes-browser/installation.updates'
const REQUEST = `${DIRECTORY}/request.json`
const STATUS = `${DIRECTORY}/status.json`

// The stock file API over the launcher's updates directory.
let files: Record<string, string>

const api = vi.fn(async ({ path, body }: HermesApiRequest) => {
  if (path === '/api/fs/write-text') {
    const { path: file, content } = body as { path: string; content: string }
    files[file] = content

    return { ok: true, path: file }
  }

  const file = decodeURIComponent(path.replace('/api/fs/read-text?path=', ''))

  if (!(file in files)) {
    throw new Error('HTTP 404: {"detail":"File not found"}')
  }

  return { path: file, text: files[file] }
})

const requestId = () => (JSON.parse(files[REQUEST]) as { id: string }).id
const writes = () => api.mock.calls.filter(([request]) => request.path === '/api/fs/write-text')

function launcherReports(status: Record<string, unknown>) {
  files[STATUS] = JSON.stringify({ id: requestId(), ...status })
}

beforeEach(() => {
  vi.useFakeTimers()
  files = {}
  vi.stubGlobal('hermesDesktop', { api })
  $browserUpdate.set({ state: 'idle' })
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

it('asks the launcher through the stock file API and follows the update to its release', async () => {
  const done = requestBrowserUpdate()
  expect($browserUpdate.get()).toEqual({ state: 'working' })
  await vi.advanceTimersByTimeAsync(0)

  expect(files[REQUEST]).toMatch(/^\{"id":"[0-9a-f]{32}"\}$/)
  // Unscoped: a profile would send the files to that profile's workspace.
  expect(writes()).toEqual([
    [{ path: '/api/fs/write-text', method: 'POST', body: { path: REQUEST, content: files[REQUEST] } }]
  ])

  launcherReports({ state: 'running' })
  await vi.advanceTimersByTimeAsync(20_000)
  expect(api).toHaveBeenLastCalledWith({ path: `/api/fs/read-text?path=${encodeURIComponent(STATUS)}` })
  expect($browserUpdate.get()).toEqual({ state: 'working' })

  launcherReports({ state: 'updated', release: 'browser-2.0_rc-1' })
  await vi.advanceTimersByTimeAsync(1000)
  await done
  expect($browserUpdate.get()).toEqual({ state: 'updated', release: 'browser-2.0_rc-1' })

  const requests = api.mock.calls.length
  await vi.advanceTimersByTimeAsync(60_000)
  expect(api).toHaveBeenCalledTimes(requests)
})

it('truncates a long failure message', async () => {
  const done = requestBrowserUpdate()
  await vi.advanceTimersByTimeAsync(0)
  launcherReports({ state: 'failed', error: 'x'.repeat(5000) })
  await vi.advanceTimersByTimeAsync(1000)
  await done
  expect($browserUpdate.get()).toEqual({ state: 'failed', error: 'x'.repeat(500) })
})

it.each([
  ['a status left by an earlier request', JSON.stringify({ id: 'a'.repeat(32), state: 'updated', release: 'old' })],
  ['an unparsable status', '{"id":'],
  ['a status that is not an object', 'null']
])('is unavailable when only %s appears within 15 seconds', async (_label, status) => {
  files[STATUS] = status
  const done = requestBrowserUpdate()
  await vi.advanceTimersByTimeAsync(14_000)
  expect($browserUpdate.get()).toEqual({ state: 'working' })
  await vi.advanceTimersByTimeAsync(1000)
  await done
  expect($browserUpdate.get()).toEqual({ state: 'unavailable' })
})

it.each([
  ['an unknown state', { state: 'done', release: 'browser-2' }],
  ['no release', { state: 'updated' }],
  ['a release that is a path', { state: 'updated', release: '../browser-2' }],
  ['a release with markup', { state: 'updated', release: '<b>browser-2</b>' }],
  ['a release longer than 80 characters', { state: 'current', release: 'b'.repeat(81) }],
  ['an error that is not text', { state: 'failed', error: { message: 'Download failed' } }]
])('does not trust a status with %s', async (_label, status) => {
  const done = requestBrowserUpdate()
  await vi.advanceTimersByTimeAsync(0)
  launcherReports(status)
  await vi.advanceTimersByTimeAsync(15_000)
  await done
  expect($browserUpdate.get()).toEqual({ state: 'unavailable' })
})

it('waits while the launcher finishes an update requested from another tab', async () => {
  files[STATUS] = JSON.stringify({ id: 'a'.repeat(32), state: 'running' })
  const done = requestBrowserUpdate()
  await vi.advanceTimersByTimeAsync(20_000)
  expect($browserUpdate.get()).toEqual({ state: 'working' })

  files[STATUS] = JSON.stringify({ id: 'a'.repeat(32), state: 'updated', release: 'browser-2' })
  await vi.advanceTimersByTimeAsync(5000)
  expect($browserUpdate.get()).toEqual({ state: 'working' })

  launcherReports({ state: 'current', release: 'browser-2' })
  await vi.advanceTimersByTimeAsync(1000)
  await done
  expect($browserUpdate.get()).toEqual({ state: 'current', release: 'browser-2' })
})

it('sends no second request while one is in progress, then a fresh id that ignores the previous status', async () => {
  const first = requestBrowserUpdate()
  await requestBrowserUpdate()
  await vi.advanceTimersByTimeAsync(0)
  expect(writes()).toHaveLength(1)

  const firstId = requestId()
  launcherReports({ state: 'current', release: 'browser-1' })
  await vi.advanceTimersByTimeAsync(1000)
  await first
  expect($browserUpdate.get()).toEqual({ state: 'current', release: 'browser-1' })

  const second = requestBrowserUpdate()
  await vi.advanceTimersByTimeAsync(5000)
  expect(writes()).toHaveLength(2)
  expect(requestId()).not.toBe(firstId)
  expect($browserUpdate.get()).toEqual({ state: 'working' })

  launcherReports({ state: 'updated', release: 'browser-2' })
  await vi.advanceTimersByTimeAsync(1000)
  await second
  expect($browserUpdate.get()).toEqual({ state: 'updated', release: 'browser-2' })
})
