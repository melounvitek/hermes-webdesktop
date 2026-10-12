import { atom } from 'nanostores'

import { translateNow } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { isBrowserClient } from '@/lib/platform'
import { readKey, writeKey } from '@/lib/storage'

export type BrowserAttentionStatus = keyof Translations['settings']['notifications']['browser']['status']

// One opt-in for every tab on this origin. Only Disable clears it.
const STORAGE_KEY = 'hermes:browser:global:attentionNotifications'
// Held by the one tab that alerts; the other tabs queue behind it.
const LOCK_NAME = 'hermes:browser-attention'

export const $browserAttentionStatus = atom<BrowserAttentionStatus>('off')
export const $browserAttentionWanted = atom(false)
let connected = false
// The Enable click waiting for permission. Only Disable cancels it; pauses do not.
let enabling: object | undefined
let generation = 0
let listening = false
let pending: AbortController | undefined
let signedOut = false
let releaseLock: (() => void) | undefined
const notifications = new Set<Notification>()
// Document lifetime, including disabled periods: re-enabling must not announce
// a previously observed request. On exhaustion require reload, not silent eviction.
const seen = new Set<string>()
const MAX_SEEN = 1024

/** Release this tab's alerts without changing the stored opt-in. */
function stop(status: BrowserAttentionStatus = 'off'): void {
  generation++
  pending?.abort()
  pending = undefined
  releaseLock?.()
  releaseLock = undefined

  for (const notification of notifications) {
    notification.close()
  }

  notifications.clear()
  $browserAttentionStatus.set(seen.size >= MAX_SEEN ? 'limit' : status)
}

function turnOff(): void {
  enabling = undefined
  $browserAttentionWanted.set(false)
  stop()
}

export function disableBrowserAttention(): void {
  writeKey(STORAGE_KEY, null)
  turnOff()
}

/** Pause while the active socket is unavailable; resume when it opens again. */
export function setBrowserAttentionConnected(value: boolean): void {
  connected = value
  sync()
}

/** A failed sign-in pauses this tab until it reloads; the opt-in stays. */
export function pauseBrowserAttentionForSignIn(): void {
  signedOut = true
  sync()
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

  if (signedOut) {
    return 'signedOut'
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
  if (readKey(STORAGE_KEY) !== 'true') {
    // Disabled in another tab. Without an opt-in, keep a declined Enable's message.
    if ($browserAttentionWanted.get()) {
      turnOff()
    }

    return
  }

  $browserAttentionWanted.set(true)
  const reason = unavailable() ?? (Notification.permission === 'granted' ? undefined : 'default')

  if (reason) {
    stop(reason)

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
  const controller = new AbortController()
  pending = controller

  // Do not await the lock's lifetime. A late grant after a stop finds a newer
  // generation and returns, which releases the lock again.
  const run = async (lock: Lock | null): Promise<void> => {
    if (epoch !== generation) {
      return
    }

    if (!lock) {
      $browserAttentionStatus.set('otherTab')

      return navigator.locks.request(LOCK_NAME, { signal: controller.signal }, run)
    }

    pending = undefined

    const hold = new Promise<void>(resolve => {
      releaseLock = resolve
    })

    if (running()) {
      $browserAttentionStatus.set('enabled')
    }

    await hold
  }

  // Try first, so a free lock never shows the waiting status.
  void navigator.locks.request(LOCK_NAME, { ifAvailable: true }, run).catch(() => {
    if (epoch === generation) {
      stop('failed')
    }
  })
}

function listen(): void {
  if (listening) {
    return
  }

  listening = true
  // A hidden or frozen page cannot alert; let another tab take over meanwhile.
  const leave = () => stop()
  window.addEventListener('pagehide', leave)
  document.addEventListener('freeze', leave)
  window.addEventListener('pageshow', sync)
  document.addEventListener('resume', sync)
  window.addEventListener('focus', sync)
  // Another tab turned alerts on or off; a null key means storage was cleared.
  window.addEventListener('storage', event => {
    if (event.key === STORAGE_KEY || event.key === null) {
      sync()
    }
  })
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
  if (releaseLock || pending || enabling) {
    return
  }

  const reason = unavailable()

  if (reason) {
    stop(reason)

    return
  }

  listen()
  const attempt = (enabling = {})
  $browserAttentionStatus.set('enabling')

  try {
    const permission = Notification.permission === 'granted' ? 'granted' : await Notification.requestPermission()
    // Disable in another tab may land before its storage event does.
    sync()

    if (attempt !== enabling) {
      return
    }

    enabling = undefined

    if (permission !== 'granted') {
      stop(permission)

      return
    }

    writeKey(STORAGE_KEY, 'true')

    if (readKey(STORAGE_KEY) !== 'true') {
      // Storage is blocked, so the opt-in could not be kept.
      stop('failed')

      return
    }

    sync()
  } catch {
    if (attempt === enabling) {
      enabling = undefined
      stop('failed')
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
        stop('failed')
      }
    }

    return true
  } catch {
    stop('failed')

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
    stop('limit')

    return
  }

  if (!request.owned || request.replayed || (request.active && !document.hidden && document.hasFocus())) {
    return
  }

  show()
}
