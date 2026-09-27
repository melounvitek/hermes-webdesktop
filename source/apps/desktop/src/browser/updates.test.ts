/** @vitest-environment jsdom
 * @vitest-environment-options {"url":"https://browser.example/"}
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  applyBrowserUpdate,
  BrowserUpdateError,
  type BrowserUpdateOffer,
  type BrowserUpdatePhase,
  type BrowserUpdateSnapshot,
  checkBrowserUpdateSession,
  prepareBrowserUpdate,
  readBrowserUpdateStatus
} from './update-client'

vi.mock(import('./update-client'), async importOriginal => ({
  ...(await importOriginal()),
  readBrowserUpdateStatus: vi.fn(),
  prepareBrowserUpdate: vi.fn(),
  applyBrowserUpdate: vi.fn(),
  checkBrowserUpdateSession: vi.fn()
}))

const KEY = 'hermes:browser-update-request'
const NOW = new Date('2026-09-01T12:00:00Z')

const offer: BrowserUpdateOffer = {
  id: 'a'.repeat(32),
  current_release: 'browser-1',
  target_release: 'browser-2',
  expires_at: NOW.getTime() / 1000 + 600,
  warning: 'Restart disconnects browser users. Rollback may be required.',
  tested_backend: 'b'.repeat(40),
  compatibility: 'not-exercised'
}

const idle: BrowserUpdateSnapshot = { capabilities: ['install'], phase: 'idle' }
const offered: BrowserUpdateSnapshot = { capabilities: ['install'], phase: 'offered', offer }
const status = vi.mocked(readBrowserUpdateStatus)
const prepare = vi.mocked(prepareBrowserUpdate)
const apply = vi.mocked(applyBrowserUpdate)
const session = vi.mocked(checkBrowserUpdateSession)
const cleanups: (() => void)[] = []
const hiddenDescriptor = Object.getOwnPropertyDescriptor(document, 'hidden')
const visibilityDescriptor = Object.getOwnPropertyDescriptor(document, 'visibilityState')
let originalUrl: string

function job(phase: BrowserUpdatePhase, updateOffer = offer): BrowserUpdateSnapshot {
  return { capabilities: ['install'], phase, offer: updateOffer, job: { id: updateOffer.id, phase } }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

function visibility(hidden: boolean) {
  Object.defineProperty(document, 'hidden', { configurable: true, value: hidden })
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    value: hidden ? 'hidden' : 'visible'
  })
  document.dispatchEvent(new Event('visibilitychange'))
}

async function controller(marker: string | null = null) {
  vi.resetModules()
  sessionStorage.clear()

  if (marker !== null) {
    sessionStorage.setItem(KEY, marker)
  }

  const store = await import('./updates')
  cleanups.push(() => store.closeBrowserUpdates())

  return store
}

function noMutations() {
  expect(prepare).not.toHaveBeenCalled()
  expect(apply).not.toHaveBeenCalled()
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(NOW)
  vi.resetAllMocks()
  status.mockResolvedValue(idle)
  prepare.mockResolvedValue(offered)
  apply.mockResolvedValue(job('stopping'))
  session.mockResolvedValue(undefined)
  visibility(false)
  originalUrl = window.location.href
  vi.spyOn(window.history, 'pushState')
  vi.spyOn(window.history, 'replaceState')
})

afterEach(() => {
  for (const cleanup of cleanups.splice(0).reverse()) {
    cleanup()
  }

  expect(window.location.href).toBe(originalUrl)
  expect(window.history.pushState).not.toHaveBeenCalled()
  expect(window.history.replaceState).not.toHaveBeenCalled()
  sessionStorage.clear()

  for (const [key, descriptor] of [
    ['hidden', hiddenDescriptor],
    ['visibilityState', visibilityDescriptor]
  ] as const) {
    if (descriptor) {
      Object.defineProperty(document, key, descriptor)
    } else {
      Reflect.deleteProperty(document, key)
    }
  }

  vi.clearAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('browser update controller', () => {
  it('loads only a request marker, never cached authority, and opening alone never starts work', async () => {
    for (const marker of [
      null,
      '',
      'a'.repeat(31),
      'a'.repeat(33),
      'z'.repeat(32),
      `${offer.id}\n`,
      JSON.stringify(offered),
      offer.id
    ]) {
      const store = await controller(marker)
      expect(store.$browserUpdatesOpen.get()).toBe(false)
      expect(store.$browserUpdateView.get()).toMatchObject({
        snapshot: null,
        error: null,
        busy: null,
        pendingId: marker === offer.id ? offer.id : null
      })

      if (marker !== offer.id) {
        expect(store.$browserUpdateView.get().unknownOutcome).toBe(false)
      }

      store.openBrowserUpdates()
      expect(store.$browserUpdatesOpen.get()).toBe(true)
      await vi.advanceTimersByTimeAsync(10_000)
      expect(status).not.toHaveBeenCalled()
      expect(session).not.toHaveBeenCalled()
      noMutations()
      store.forgetBrowserUpdateTracking()
      expect(store.$browserUpdateView.get()).toMatchObject({ pendingId: null, unknownOutcome: false })
      expect(sessionStorage.getItem(KEY)).toBeNull()
      store.closeBrowserUpdates()
      expect(store.$browserUpdatesOpen.get()).toBe(false)
    }
  })

  it('prepares only explicitly and confirms only the current, unexpired offered release pair', async () => {
    const store = await controller()
    store.openBrowserUpdates()
    await store.prepareBrowserUpdateOffer()
    expect(prepare).toHaveBeenCalledTimes(1)
    expect(store.$browserUpdateView.get()).toMatchObject({ snapshot: offered, error: null, busy: null })
    expect(apply).not.toHaveBeenCalled()

    const invalidOffers = [
      { ...offer, id: 'c'.repeat(32) },
      { ...offer, current_release: 'browser-other' },
      { ...offer, target_release: 'browser-other' }
    ]

    for (const candidate of invalidOffers) {
      status.mockResolvedValue(offered)
      await store.refreshBrowserUpdates()
      await expect(store.confirmBrowserUpdateOffer(candidate)).rejects.toBeInstanceOf(BrowserUpdateError)
      expect(store.$browserUpdateView.get().error).not.toBeNull()
      expect(sessionStorage.getItem(KEY)).toBeNull()
    }

    for (const snapshot of [idle, { ...offered, phase: 'current' } satisfies BrowserUpdateSnapshot, job('starting')]) {
      status.mockResolvedValue(snapshot)
      await store.refreshBrowserUpdates()
      await expect(store.confirmBrowserUpdateOffer(offer)).rejects.toBeInstanceOf(BrowserUpdateError)
    }

    status.mockResolvedValue(offered)
    await store.refreshBrowserUpdates()
    vi.setSystemTime(offer.expires_at * 1000)
    await expect(store.confirmBrowserUpdateOffer(offer)).rejects.toBeInstanceOf(BrowserUpdateError)
    expect(apply).not.toHaveBeenCalled()
    expect(sessionStorage.getItem(KEY)).toBeNull()

    vi.setSystemTime(NOW)
    await store.refreshBrowserUpdates()
    apply.mockImplementation(async id => {
      expect(id).toBe(offer.id)
      expect(sessionStorage.getItem(KEY)).toBe(id)
      expect(store.$browserUpdateView.get()).toMatchObject({ pendingId: id, busy: 'apply' })

      return job('stopping')
    })
    await store.confirmBrowserUpdateOffer({ ...offer })
    expect(apply).toHaveBeenCalledTimes(1)
    expect(store.$browserUpdateView.get()).toMatchObject({ snapshot: job('stopping'), busy: null })
  })

  it('rejects conflicting actions throughout status, preparation, apply, and session requests', async () => {
    for (const busy of ['status', 'offer', 'apply', 'session'] as const) {
      vi.clearAllMocks()
      const store = await controller()
      status.mockResolvedValue(offered)
      await store.refreshBrowserUpdates()
      const waiting = deferred<BrowserUpdateSnapshot>()
      const checking = deferred<void>()

      if (busy === 'status') {
        status.mockReturnValueOnce(waiting.promise)
      }

      if (busy === 'offer') {
        prepare.mockReturnValueOnce(waiting.promise)
      }

      if (busy === 'apply') {
        apply.mockReturnValueOnce(waiting.promise)
      }

      if (busy === 'session') {
        session.mockReturnValueOnce(checking.promise)
      }

      const actions = {
        status: () => store.refreshBrowserUpdates(),
        offer: () => store.prepareBrowserUpdateOffer(),
        apply: () => store.confirmBrowserUpdateOffer(offer),
        session: () => store.checkBrowserSessionAndRetry()
      }

      const pending = actions[busy]()
      await vi.advanceTimersByTimeAsync(0)
      expect(store.$browserUpdateView.get().busy).toBe(busy)
      const counts = [status, prepare, apply, session].map(mock => mock.mock.calls.length)

      for (const [name, action] of Object.entries(actions)) {
        if (name === busy) {
          continue
        }

        await expect(action()).rejects.toMatchObject({ kind: 'conflict' })
        expect(store.$browserUpdateView.get()).toMatchObject({ busy, error: 'conflict' })
      }

      expect([status, prepare, apply, session].map(mock => mock.mock.calls.length)).toEqual(counts)
      waiting.resolve(busy === 'apply' ? job('stopping') : offered)
      checking.resolve(undefined)
      await pending
      expect(store.$browserUpdateView.get().busy).toBeNull()
    }
  })

  it('keeps a closed-dialog apply alive, retains unknown outcomes, and clears only matching terminal tracking', async () => {
    const store = await controller()
    status.mockResolvedValue(offered)
    await store.refreshBrowserUpdates()
    store.openBrowserUpdates()
    const stop = store.observeBrowserUpdates()
    cleanups.push(stop)
    await vi.advanceTimersByTimeAsync(0)
    const waiting = deferred<BrowserUpdateSnapshot>()
    apply.mockReturnValueOnce(waiting.promise)
    const result = expect(store.confirmBrowserUpdateOffer(offer)).rejects.toMatchObject({ kind: 'network' })
    expect(sessionStorage.getItem(KEY)).toBe(offer.id)
    store.closeBrowserUpdates()
    stop()
    expect(store.$browserUpdateView.get().busy).toBe('apply')
    // The real apply API has no cancellation parameter; closing must not add one.
    expect(apply).toHaveBeenCalledWith(offer.id)
    waiting.reject(new BrowserUpdateError('network'))
    await result
    expect(store.$browserUpdateView.get()).toMatchObject({
      pendingId: offer.id,
      unknownOutcome: true,
      error: 'network',
      busy: null
    })
    await vi.advanceTimersByTimeAsync(180_000)
    expect(apply).toHaveBeenCalledTimes(1)
    expect(prepare).not.toHaveBeenCalled()
    expect(sessionStorage.getItem(KEY)).toBe(offer.id)
    status.mockResolvedValue(job('starting'))
    await store.refreshBrowserUpdates()
    expect(store.$browserUpdateView.get().pendingId).toBe(offer.id)
    expect(sessionStorage.getItem(KEY)).toBe(offer.id)

    for (const phase of ['succeeded', 'rolled_back', 'failed', 'recovery_required'] as const) {
      const restored = await controller(offer.id)
      status.mockResolvedValue(job(phase))
      await restored.refreshBrowserUpdates()
      expect(restored.$browserUpdateView.get()).toMatchObject({
        snapshot: job(phase),
        pendingId: null,
        unknownOutcome: false,
        busy: null
      })
      expect(sessionStorage.getItem(KEY)).toBeNull()
    }

    expect(apply).toHaveBeenCalledTimes(1)
  })

  it('never treats a different offer or job as the tracked request or permits mutations until tracking is forgotten', async () => {
    const other = { ...offer, id: 'c'.repeat(32) }

    for (const snapshot of [{ ...offered, offer: other }, job('starting', other), job('succeeded', other)]) {
      const store = await controller(offer.id)
      status.mockResolvedValue(snapshot)
      await store.refreshBrowserUpdates()
      expect(store.$browserUpdateView.get()).toMatchObject({ error: 'differentJob', pendingId: offer.id })
      expect(sessionStorage.getItem(KEY)).toBe(offer.id)
      await expect(store.prepareBrowserUpdateOffer()).rejects.toBeInstanceOf(BrowserUpdateError)
      await expect(store.confirmBrowserUpdateOffer(other)).rejects.toBeInstanceOf(BrowserUpdateError)
      noMutations()
      store.forgetBrowserUpdateTracking()
      expect(sessionStorage.getItem(KEY)).toBeNull()
      expect(store.$browserUpdateView.get()).toMatchObject({ pendingId: null, unknownOutcome: false })
    }

    noMutations()
  })

  it('invalidates stale actionability on errors and checks the stock session only on an explicit retry', async () => {
    const store = await controller()

    for (const kind of ['unsupported', 'forbidden', 'network', 'invalid', 'conflict', 'insecure'] as const) {
      status.mockResolvedValueOnce(offered)
      await store.refreshBrowserUpdates()
      const error = new BrowserUpdateError(kind)
      status.mockRejectedValueOnce(error)
      await expect(store.refreshBrowserUpdates()).rejects.toBe(error)
      expect(store.$browserUpdateView.get()).toMatchObject({ snapshot: null, error: kind, busy: null })
      await expect(store.confirmBrowserUpdateOffer(offer)).rejects.toBeInstanceOf(BrowserUpdateError)
      noMutations()
    }

    expect(session).not.toHaveBeenCalled()
    const denied = new BrowserUpdateError('forbidden')
    prepare.mockRejectedValueOnce(denied)
    await expect(store.prepareBrowserUpdateOffer()).rejects.toBe(denied)
    expect(store.$browserUpdateView.get()).toMatchObject({ error: 'forbidden', busy: null })
    await vi.advanceTimersByTimeAsync(30_000)
    expect(prepare).toHaveBeenCalledTimes(1)
    expect(session).not.toHaveBeenCalled()

    const checking = deferred<void>()
    session.mockReturnValueOnce(checking.promise)
    const reads = status.mock.calls.length
    const retry = store.checkBrowserSessionAndRetry()
    expect(store.$browserUpdateView.get().busy).toBe('session')
    expect(session).toHaveBeenCalledTimes(1)
    expect(status).toHaveBeenCalledTimes(reads)
    status.mockResolvedValue(offered)
    checking.resolve(undefined)
    await retry
    expect(status).toHaveBeenCalledTimes(reads + 1)
    expect(store.$browserUpdateView.get()).toMatchObject({ snapshot: offered, error: null, busy: null })
    session.mockRejectedValueOnce(denied)
    await expect(store.checkBrowserSessionAndRetry()).rejects.toBe(denied)
    expect(store.$browserUpdateView.get()).toMatchObject({ error: 'forbidden', busy: null })
    expect(status).toHaveBeenCalledTimes(reads + 1)
    expect(prepare).toHaveBeenCalledTimes(1)
    expect(apply).not.toHaveBeenCalled()
  })

  it('aborts observation reads on cleanup and ignores late responses after a fresh dialog read', async () => {
    const store = await controller()
    const stale = deferred<BrowserUpdateSnapshot>()
    // Deliberately finish after abort to exercise the controller's generation guard.
    status.mockReturnValueOnce(stale.promise)
    store.openBrowserUpdates()
    const stop = store.observeBrowserUpdates()
    cleanups.push(stop)
    await vi.advanceTimersByTimeAsync(0)
    expect(status).toHaveBeenCalledTimes(1)
    const signal = status.mock.calls[0][0]
    expect(signal).toBeInstanceOf(AbortSignal)
    expect(signal?.aborted).toBe(false)
    store.closeBrowserUpdates()
    stop()
    expect(signal?.aborted).toBe(true)
    const calls = status.mock.calls.length
    await vi.advanceTimersByTimeAsync(10_000)
    expect(status).toHaveBeenCalledTimes(calls)
    store.openBrowserUpdates()
    status.mockResolvedValue(job('succeeded'))
    await store.refreshBrowserUpdates()
    expect(store.$browserUpdateView.get().snapshot).toEqual(job('succeeded'))
    stale.resolve(offered)
    await vi.advanceTimersByTimeAsync(0)
    expect(store.$browserUpdateView.get()).toMatchObject({ snapshot: job('succeeded'), error: null, busy: null })
    noMutations()
  })

  it('releases tracking after definitive apply rejection but preserves uncertain responses', async () => {
    for (const kind of ['conflict', 'forbidden', 'network', 'invalid'] as const) {
      const store = await controller()
      status.mockResolvedValue(offered)
      await store.refreshBrowserUpdates()
      apply.mockRejectedValueOnce(new BrowserUpdateError(kind))
      await expect(store.confirmBrowserUpdateOffer(offer)).rejects.toMatchObject({ kind })
      const uncertain = kind === 'network' || kind === 'invalid'
      expect(store.$browserUpdateView.get()).toMatchObject({
        pendingId: uncertain ? offer.id : null,
        unknownOutcome: uncertain
      })
      expect(sessionStorage.getItem(KEY)).toBe(uncertain ? offer.id : null)

      if (!uncertain) {
        await store.refreshBrowserUpdates()
        await store.prepareBrowserUpdateOffer()
        expect(store.$browserUpdateView.get().snapshot).toEqual(offered)
      }
    }
  })

  it('refreshes idle, offered and completed dialogs on visibility restoration without exhausting an inactive poll', async () => {
    for (const snapshot of [idle, offered, job('succeeded')]) {
      const store = await controller()
      status.mockResolvedValue(snapshot)
      const stop = store.observeBrowserUpdates()
      cleanups.push(stop)
      await vi.advanceTimersByTimeAsync(0)
      visibility(true)
      document.dispatchEvent(new Event('visibilitychange'))
      await vi.advanceTimersByTimeAsync(121_000)
      visibility(false)
      document.dispatchEvent(new Event('visibilitychange'))
      await vi.advanceTimersByTimeAsync(0)
      expect(store.$browserUpdateView.get()).toMatchObject({ snapshot, error: null, busy: null })
      stop()
    }
  })

  it('keeps the last observed phase visible during a refresh but disables mutations until it completes', async () => {
    const store = await controller()
    status.mockResolvedValue(job('rollback_starting'))
    await store.refreshBrowserUpdates()
    const waiting = deferred<BrowserUpdateSnapshot>()
    status.mockReturnValueOnce(waiting.promise)
    const refresh = store.refreshBrowserUpdates()
    expect(store.$browserUpdateView.get()).toMatchObject({ snapshot: job('rollback_starting'), busy: 'status' })
    await expect(store.confirmBrowserUpdateOffer(offer)).rejects.toMatchObject({ kind: 'conflict' })
    waiting.resolve(job('rolled_back'))
    await refresh
    expect(store.$browserUpdateView.get()).toMatchObject({ snapshot: job('rolled_back'), error: null, busy: null })
  })

  it('polls only while observed and visible, bounds active/network recovery, and pauses without retrying mutations', async () => {
    for (const outcome of ['active', 'network'] as const) {
      vi.clearAllMocks()
      vi.setSystemTime(NOW)
      const store = await controller(offer.id)

      if (outcome === 'active') {
        status.mockResolvedValue(job('starting'))
      } else {
        status.mockRejectedValue(new BrowserUpdateError('network'))
      }

      visibility(true)
      store.openBrowserUpdates()
      const stop = store.observeBrowserUpdates()
      cleanups.push(stop)
      await vi.advanceTimersByTimeAsync(10_000)
      expect(status).not.toHaveBeenCalled()
      visibility(false)
      await vi.advanceTimersByTimeAsync(10_000)
      expect(status.mock.calls.length).toBeGreaterThan(1)
      expect(status.mock.calls.length).toBeLessThan(12)
      visibility(true)
      const hiddenCalls = status.mock.calls.length
      await vi.advanceTimersByTimeAsync(10_000)
      expect(status).toHaveBeenCalledTimes(hiddenCalls)
      visibility(false)
      await vi.advanceTimersByTimeAsync(125_000)
      expect(store.$browserUpdateView.get()).toMatchObject({ error: 'paused', busy: null, pendingId: offer.id })
      const boundedCalls = status.mock.calls.length
      expect(boundedCalls).toBeGreaterThan(hiddenCalls)
      expect(boundedCalls).toBeLessThan(80)
      await vi.advanceTimersByTimeAsync(300_000)
      expect(status).toHaveBeenCalledTimes(boundedCalls)
      expect(session).not.toHaveBeenCalled()
      noMutations()
      stop()
      visibility(true)
      visibility(false)
      await vi.advanceTimersByTimeAsync(10_000)
      expect(status).toHaveBeenCalledTimes(boundedCalls)
      store.closeBrowserUpdates()
    }
  })
})
