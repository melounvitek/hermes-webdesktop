import { atom } from 'nanostores'

import { translateNow } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { isBrowserClient } from '@/lib/platform'
import { persistBoolean, storedBoolean } from '@/lib/storage'

export type BrowserAttentionStatus = keyof Translations['settings']['notifications']['browser']['status']

// One opt-in for every tab on this origin. Only Disable clears it.
const STORAGE_KEY = 'hermes:browser:global:attentionNotifications'

export const $browserAttentionStatus = atom<BrowserAttentionStatus>('off')
export const $browserAttentionWanted = atom(false)
let connected = false
let generation = 0
let listening = false
let pending = false
let releaseLock: (() => void) | undefined
const notifications = new Set<Notification>()
// Document lifetime, including disabled periods: re-enabling must not announce
// a previously observed request. On exhaustion require reload, not silent eviction.
const seen = new Set<string>()
const MAX_SEEN = 1024

/** Release this tab's alerts without changing the stored opt-in. */
export function stopBrowserAttention(status: BrowserAttentionStatus = 'off'): void {
  generation++
  pending = false
  releaseLock?.()
  releaseLock = undefined

  for (const notification of notifications) {
    notification.close()
  }

  notifications.clear()
  $browserAttentionStatus.set(seen.size >= MAX_SEEN ? 'limit' : status)
}

export function disableBrowserAttention(): void {
  persistBoolean(STORAGE_KEY, false)
  $browserAttentionWanted.set(false)
  stopBrowserAttention()
}

export function setBrowserAttentionConnected(value: boolean): void {
  connected = value

  if (!value) {
    stopBrowserAttention('disconnected')
  }
}

function unavailable(): BrowserAttentionStatus | undefined {
  if (!isBrowserClient()) {
    return 'unsupported'
  }

  if (!window.isSecureContext) {
    return 'insecure'
  }

  if (
    typeof Notification !== 'function' ||
    typeof Notification.requestPermission !== 'function' ||
    !navigator.locks?.request
  ) {
    return 'unsupported'
  }

  if (!connected) {
    return 'disconnected'
  }

  if (seen.size >= MAX_SEEN) {
    return 'limit'
  }

  if (Notification.permission === 'denied') {
    return 'denied'
  }
}

/** Start or stop this tab to match the stored opt-in. Never prompts. */
function sync(): void {
  const wanted = storedBoolean(STORAGE_KEY, false)
  $browserAttentionWanted.set(wanted)
  const reason = wanted ? (unavailable() ?? (Notification.permission === 'granted' ? undefined : 'default')) : 'off'

  if (reason) {
    // Without an opt-in, keep the message from a declined Enable.
    if (wanted || releaseLock || pending) {
      stopBrowserAttention(reason)
    }

    return
  }

  if (!releaseLock && !pending) {
    acquire()
  }
}

/** Recheck the opt-in and permission before acting as the alerting tab. */
function running(): boolean {
  sync()

  return Boolean(releaseLock)
}

function acquire(): void {
  const epoch = ++generation
  pending = true

  // Do not await the lock's lifetime. ifAvailable refuses peers, never queues
  // automatic takeover; generation also cancels a late granted callback.
  void navigator.locks
    .request('hermes:browser-attention', { mode: 'exclusive', ifAvailable: true }, async lock => {
      if (epoch !== generation) {
        return
      }

      pending = false

      if (!lock) {
        stopBrowserAttention('otherTab')

        return
      }

      const hold = new Promise<void>(resolve => {
        releaseLock = resolve
      })

      if (running()) {
        $browserAttentionStatus.set('enabled')
      }

      await hold
    })
    .catch(() => {
      if (epoch === generation) {
        stopBrowserAttention('failed')
      }
    })
}

function listen(): void {
  if (listening) {
    return
  }

  listening = true
  const reset = () => stopBrowserAttention('reset')
  window.addEventListener('pagehide', reset)
  document.addEventListener('freeze', reset)
  window.addEventListener('focus', sync)
}

/** Browser boot: resume a stored opt-in once the bridge is installed. */
export function startBrowserAttention(): void {
  if (isBrowserClient()) {
    listen()
    sync()
  }
}

/** Call directly from the Enable click: no await before requestPermission. */
export async function enableBrowserAttention(): Promise<void> {
  if (releaseLock || pending || $browserAttentionStatus.get() === 'enabling') {
    return
  }

  const reason = unavailable()

  if (reason) {
    stopBrowserAttention(reason)

    return
  }

  listen()
  const epoch = ++generation
  $browserAttentionStatus.set('enabling')

  try {
    const permission = Notification.permission === 'granted' ? 'granted' : await Notification.requestPermission()

    if (epoch !== generation) {
      return
    }

    if (permission !== 'granted') {
      stopBrowserAttention(permission)

      return
    }

    persistBoolean(STORAGE_KEY, true)

    if (!storedBoolean(STORAGE_KEY, false)) {
      // Storage is blocked, so the opt-in could not be kept.
      stopBrowserAttention('failed')

      return
    }

    sync()
  } catch {
    if (epoch === generation) {
      stopBrowserAttention('failed')
    }
  }
}

function show(): boolean {
  if (!releaseLock || !running()) {
    return false
  }

  const epoch = generation

  try {
    const notification = new Notification(translateNow('settings.notifications.browser.title'), {
      body: translateNow('settings.notifications.browser.body')
    })

    notifications.add(notification)

    notification.onclick = event => {
      event.preventDefault()
      notification.close()
      notifications.delete(notification)

      if (epoch === generation && running()) {
        window.focus()
      }
    }

    notification.onclose = () => notifications.delete(notification)

    notification.onerror = () => {
      if (epoch === generation) {
        stopBrowserAttention('failed')
      }
    }

    return true
  } catch {
    stopBrowserAttention('failed')

    return false
  }
}

export function testBrowserAttention(): void {
  if (show()) {
    $browserAttentionStatus.set('requested')
  }
}

/** Only the fresh blocking-request handlers call this; no native payload crosses here. */
export function notifyBrowserAttention(request: {
  id: string
  method: string
  sessionId: string
  owned: boolean
  replayed?: boolean
  active: boolean
}): void {
  if (!isBrowserClient() || !request.id || !request.sessionId) {
    return
  }

  // One same-origin backend per document. Remember even unresolved ownership:
  // a later binding (or reordered profile rows) cannot make this request new.
  const key = JSON.stringify([request.sessionId, request.method, request.id])

  if (seen.has(key)) {
    return
  }

  if (seen.size < MAX_SEEN) {
    seen.add(key)
  }

  if (seen.size >= MAX_SEEN) {
    stopBrowserAttention('limit')

    return
  }

  if (!request.owned || request.replayed || (request.active && !document.hidden && document.hasFocus())) {
    return
  }

  show()
}
