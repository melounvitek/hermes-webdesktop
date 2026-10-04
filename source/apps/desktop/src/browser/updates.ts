import { atom } from 'nanostores'

import { hermesApi } from '@/api/client'

export type BrowserUpdate =
  | { state: 'idle' | 'working' | 'unavailable' }
  | { state: 'updated' | 'current'; release: string }
  // No error: the launcher never reported how the update ended.
  | { state: 'failed'; error?: string }

// The launcher serving this page watches this directory. Stock Hermes neither
// knows about it nor restarts: the page only reads and writes files there.
// The requests carry no profile: the active profile's workspace may be on
// another machine, while these files belong to the launcher's own home.
const DIRECTORY = '~/.local/lib/hermes-browser/installation.updates'
const RELEASE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/
const ACCEPT_MS = 15_000
const FINISH_MS = 600_000

export const $browserUpdatesOpen = atom(false)
export const $browserUpdate = atom<BrowserUpdate>({ state: 'idle' })

export function openBrowserUpdates() {
  $browserUpdatesOpen.set(true)
}

export function closeBrowserUpdates() {
  $browserUpdatesOpen.set(false)
}

async function readStatus(id: string): Promise<BrowserUpdate | null> {
  try {
    const { text } = await hermesApi<{ text: string }>({
      path: `/api/fs/read-text?path=${encodeURIComponent(`${DIRECTORY}/status.json`)}`
    })

    const status = JSON.parse(text) as Record<string, unknown>

    // A status with another id was left by an earlier request.
    if (status.id !== id) {
      return null
    }

    if (status.state === 'running') {
      return { state: 'working' }
    }

    if (
      (status.state === 'updated' || status.state === 'current') &&
      typeof status.release === 'string' &&
      RELEASE.test(status.release)
    ) {
      return { state: status.state, release: status.release }
    }

    if (status.state === 'failed' && typeof status.error === 'string') {
      return { state: 'failed', error: status.error.slice(0, 500) }
    }
  } catch {
    // Not written yet, or caught while the launcher was replacing it.
  }

  return null
}

export async function requestBrowserUpdate() {
  if ($browserUpdate.get().state === 'working') {
    return
  }

  $browserUpdate.set({ state: 'working' })

  const id = Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('')

  try {
    await hermesApi({
      path: '/api/fs/write-text',
      method: 'POST',
      body: { path: `${DIRECTORY}/request.json`, content: JSON.stringify({ id }) }
    })
  } catch {
    // A launcher without live updates has no such directory, so stock refuses the write.
    $browserUpdate.set({ state: 'unavailable' })

    return
  }

  const started = Date.now()
  let running = false

  for (;;) {
    await new Promise(resolve => setTimeout(resolve, 1000))
    const status = await readStatus(id)

    if (status?.state === 'working') {
      running = true
    } else if (status) {
      $browserUpdate.set(status)

      return
    }

    if (Date.now() - started >= (running ? FINISH_MS : ACCEPT_MS)) {
      $browserUpdate.set({ state: running ? 'failed' : 'unavailable' })

      return
    }
  }
}
