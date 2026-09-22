import { atom } from 'nanostores'

import { translateNow } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { isBrowserClient } from '@/lib/platform'

export type BrowserAttentionStatus = keyof Translations['settings']['notifications']['browser']['status']

export const $browserAttentionStatus = atom<BrowserAttentionStatus>('off')
let connected = false
let generation = 0
let releaseLock: (() => void) | undefined
let removeListeners: (() => void) | undefined
const notifications = new Set<Notification>()
// Document lifetime, including disabled periods: re-enabling must not announce
// a previously observed request. On exhaustion require reload, not silent eviction.
const seen = new Set<string>()
const MAX_SEEN = 1024

export function disableBrowserAttention(status: BrowserAttentionStatus = 'off'): void {
  generation++
  releaseLock?.()
  releaseLock = undefined
  removeListeners?.()
  removeListeners = undefined

  for (const notification of notifications) {
    notification.close()
  }

  notifications.clear()
  $browserAttentionStatus.set(seen.size >= MAX_SEEN ? 'limit' : status)
}

export function setBrowserAttentionConnected(value: boolean): void {
  connected = value

  if (!value) {
    disableBrowserAttention('disconnected')
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

function allowed(): boolean {
  const reason = unavailable() ?? (Notification.permission !== 'granted' ? 'default' : undefined)

  if (reason) {
    disableBrowserAttention(reason)
  }

  return !reason && Boolean(releaseLock)
}

/** Call directly from the Enable click: no await before requestPermission. */
export async function enableBrowserAttention(): Promise<void> {
  if (releaseLock || $browserAttentionStatus.get() === 'enabling') {
    return
  }

  const reason = unavailable()

  if (reason) {
    disableBrowserAttention(reason)

    return
  }

  const epoch = ++generation
  $browserAttentionStatus.set('enabling')
  const reset = () => disableBrowserAttention('reset')

  const recheck = () => {
    if (releaseLock) {
      allowed()
    }
  }

  window.addEventListener('pagehide', reset)
  document.addEventListener('freeze', reset)
  window.addEventListener('focus', recheck)

  removeListeners = () => {
    window.removeEventListener('pagehide', reset)
    document.removeEventListener('freeze', reset)
    window.removeEventListener('focus', recheck)
  }

  try {
    const permission = Notification.permission === 'granted' ? 'granted' : await Notification.requestPermission()

    if (epoch !== generation) {
      return
    }

    if (permission !== 'granted') {
      disableBrowserAttention(permission)

      return
    }

    // Do not await the lock's lifetime. ifAvailable refuses peers, never queues
    // automatic takeover; generation also cancels a late granted callback.
    void navigator.locks
      .request('hermes:browser-attention', { mode: 'exclusive', ifAvailable: true }, async lock => {
        if (epoch !== generation) {
          return
        }

        if (!lock) {
          disableBrowserAttention('otherTab')

          return
        }

        const hold = new Promise<void>(resolve => {
          releaseLock = resolve
        })

        if (allowed()) {
          $browserAttentionStatus.set('enabled')
        }

        await hold
      })
      .catch(() => {
        if (epoch === generation) {
          disableBrowserAttention('failed')
        }
      })
  } catch {
    if (epoch === generation) {
      disableBrowserAttention('failed')
    }
  }
}

function show(): boolean {
  if (!releaseLock || !allowed()) {
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

      if (epoch === generation && allowed()) {
        window.focus()
      }
    }

    notification.onclose = () => notifications.delete(notification)

    notification.onerror = () => {
      if (epoch === generation) {
        disableBrowserAttention('failed')
      }
    }

    return true
  } catch {
    disableBrowserAttention('failed')

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
    disableBrowserAttention('limit')

    return
  }

  if (!request.owned || request.replayed || (request.active && !document.hidden && document.hasFocus())) {
    return
  }

  show()
}
