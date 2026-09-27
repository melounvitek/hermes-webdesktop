import { atom } from 'nanostores'

import {
  applyBrowserUpdate,
  BrowserUpdateError,
  type BrowserUpdateOffer,
  type BrowserUpdateSnapshot,
  checkBrowserUpdateSession,
  prepareBrowserUpdate,
  readBrowserUpdateStatus
} from './update-client'

interface BrowserUpdateView {
  snapshot: BrowserUpdateSnapshot | null
  error: BrowserUpdateError['kind'] | 'paused' | 'differentJob' | null
  busy: 'status' | 'offer' | 'apply' | 'session' | null
  pendingId: string | null
  unknownOutcome: boolean
}

const TRACKING_KEY = 'hermes:browser-update-request'

const ACTIVE = new Set([
  'preparing',
  'stopping',
  'switching',
  'starting',
  'rollback_stopping',
  'rollback_switching',
  'rollback_starting'
])

const TERMINAL = new Set(['succeeded', 'rolled_back', 'failed', 'recovery_required'])
const OBSERVATION_MS = 120_000

function readTracking(): string | null {
  try {
    const value = sessionStorage.getItem(TRACKING_KEY)

    return value && /^[a-f0-9]{32}$/.test(value) ? value : null
  } catch {
    return null // Private browsing may disable storage; observation still works in this tab.
  }
}

const pendingId = readTracking()
export const $browserUpdatesOpen = atom(false)
export const $browserUpdateView = atom<BrowserUpdateView>({
  snapshot: null,
  error: null,
  busy: null,
  pendingId,
  unknownOutcome: pendingId !== null
})

let requestId = 0
let statusController: AbortController | null = null
let observing = false
let timer: ReturnType<typeof setTimeout> | undefined
let deadline = 0

function patch(value: Partial<BrowserUpdateView>) {
  $browserUpdateView.set({ ...$browserUpdateView.get(), ...value })
}

function track(id: string | null) {
  patch({ pendingId: id })

  try {
    if (id) {
      sessionStorage.setItem(TRACKING_KEY, id)
    } else {
      sessionStorage.removeItem(TRACKING_KEY)
    }
  } catch {
    // Do not lose the in-memory correlation if browser storage is unavailable.
  }
}

function publish(snapshot: BrowserUpdateSnapshot) {
  const id = $browserUpdateView.get().pendingId
  const matchingJob = id !== null && snapshot.job?.id === id
  const differentJob = id !== null && !matchingJob && snapshot.offer?.id !== id

  if (matchingJob && TERMINAL.has(snapshot.phase ?? '')) {
    track(null)
  }

  patch({ snapshot, error: differentJob ? 'differentJob' : null, unknownOutcome: id !== null && !matchingJob })
}

function needsObservation() {
  const view = $browserUpdateView.get()

  return view.error === 'network' || (!view.error && (ACTIVE.has(view.snapshot?.phase ?? '') || view.unknownOutcome))
}

function schedule(delay: number) {
  clearTimeout(timer)

  if (observing && !document.hidden && needsObservation()) {
    timer = setTimeout(() => void tick(), delay)
  }
}

function conflict() {
  patch({ error: 'conflict' })

  return new BrowserUpdateError('conflict')
}

async function run(
  busy: NonNullable<BrowserUpdateView['busy']>,
  action: (signal?: AbortSignal) => Promise<BrowserUpdateSnapshot>
) {
  if ($browserUpdateView.get().busy) {
    throw conflict()
  }

  const id = ++requestId
  const controller = busy === 'status' ? new AbortController() : null
  statusController = controller
  patch({ busy, error: null, ...(busy === 'status' ? {} : { snapshot: null }) })

  try {
    const snapshot = await action(controller?.signal)

    if (id === requestId) {
      publish(snapshot)
    }
  } catch (error) {
    if (id === requestId && !controller?.signal.aborted) {
      patch({ snapshot: null, error: error instanceof BrowserUpdateError ? error.kind : 'network' })
    }

    throw error
  } finally {
    if (id === requestId) {
      statusController = null
      patch({ busy: null })
      schedule($browserUpdateView.get().error ? 5000 : 2000)
    }
  }
}

async function tick() {
  if (!observing || document.hidden) {
    return
  }

  if (Date.now() >= deadline) {
    patch({ error: 'paused' })

    return
  }

  if ($browserUpdateView.get().busy) {
    timer = setTimeout(() => void tick(), 2000)

    return
  }

  try {
    await run('status', readBrowserUpdateStatus)
  } catch {
    // The view owns the error. Only status reads are retried by observation.
  }
}

export function openBrowserUpdates() {
  $browserUpdatesOpen.set(true)
}

export function closeBrowserUpdates() {
  $browserUpdatesOpen.set(false)
}

export function observeBrowserUpdates(): () => void {
  observing = true
  deadline = Date.now() + OBSERVATION_MS

  const visibility = () => {
    clearTimeout(timer)

    if (!document.hidden && $browserUpdateView.get().error !== 'paused') {
      // An idle/terminal dialog was not polling; returning to it is a fresh
      // status read, not an exhausted maintenance observation.
      if (!needsObservation()) {
        deadline = Date.now() + OBSERVATION_MS
      }

      void tick()
    }
  }

  document.addEventListener('visibilitychange', visibility)
  void tick()

  return () => {
    observing = false
    clearTimeout(timer)
    document.removeEventListener('visibilitychange', visibility)

    if (statusController) {
      ++requestId
      statusController.abort()
      statusController = null
      patch({ busy: null })
    }
  }
}

export function refreshBrowserUpdates() {
  deadline = Date.now() + OBSERVATION_MS

  return run('status', readBrowserUpdateStatus)
}

export function prepareBrowserUpdateOffer() {
  const view = $browserUpdateView.get()

  if (
    view.pendingId ||
    view.busy ||
    view.snapshot?.capabilities.length === 0 ||
    ACTIVE.has(view.snapshot?.phase ?? '') ||
    view.snapshot?.phase === 'recovery_required'
  ) {
    return Promise.reject(conflict())
  }

  deadline = Date.now() + OBSERVATION_MS

  return run('offer', prepareBrowserUpdate)
}

export function confirmBrowserUpdateOffer(offer: BrowserUpdateOffer) {
  const view = $browserUpdateView.get()
  const current = view.snapshot?.offer

  if (
    view.busy ||
    view.error ||
    view.snapshot?.phase !== 'offered' ||
    !view.snapshot.capabilities.includes('install') ||
    current?.id !== offer.id ||
    current.current_release !== offer.current_release ||
    current.target_release !== offer.target_release ||
    current.expires_at <= Date.now() / 1000 ||
    offer.expires_at <= Date.now() / 1000
  ) {
    return Promise.reject(conflict())
  }

  track(offer.id)
  patch({ unknownOutcome: true })
  deadline = Date.now() + OBSERVATION_MS

  return run('apply', () => applyBrowserUpdate(offer.id)).catch(error => {
    // Authorization/offer refusals occur before acceptance. A lost or malformed
    // response, however, cannot prove that the job did not start.
    if (
      error instanceof BrowserUpdateError &&
      ['conflict', 'forbidden', 'unsupported', 'insecure'].includes(error.kind)
    ) {
      track(null)
      patch({ unknownOutcome: false })
    }

    throw error
  })
}

export function checkBrowserSessionAndRetry() {
  deadline = Date.now() + OBSERVATION_MS

  return run('session', async () => {
    await checkBrowserUpdateSession()

    return readBrowserUpdateStatus()
  })
}

export function forgetBrowserUpdateTracking() {
  track(null)
  patch({
    unknownOutcome: false,
    error: $browserUpdateView.get().error === 'differentJob' ? null : $browserUpdateView.get().error
  })
}
