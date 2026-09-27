const PHASES = [
  'idle',
  'preparing',
  'offered',
  'current',
  'failed',
  'stopping',
  'switching',
  'starting',
  'rollback_stopping',
  'rollback_switching',
  'rollback_starting',
  'succeeded',
  'rolled_back',
  'recovery_required'
] as const

export type BrowserUpdatePhase = (typeof PHASES)[number]

export interface BrowserUpdateOffer {
  id: string
  current_release: string
  target_release: string
  expires_at: number
  warning: string
  tested_backend: string
  compatibility: 'not-exercised'
}

export interface BrowserUpdateSnapshot {
  capabilities: string[]
  phase?: BrowserUpdatePhase
  offer?: BrowserUpdateOffer
  job?: { id: string; phase: BrowserUpdatePhase }
  error?: string
}

type ErrorKind = 'unsupported' | 'forbidden' | 'network' | 'invalid' | 'conflict' | 'insecure'

const ERROR_MESSAGES: Record<ErrorKind, string> = {
  unsupported: 'Browser updater is unavailable for this deployment.',
  forbidden: 'Request authorization was rejected.',
  network: 'Browser updater request did not complete.',
  invalid: 'Browser updater data is invalid.',
  conflict: 'Browser updater state conflicts with this request.',
  insecure: 'Browser updater requires HTTPS.'
}

export class BrowserUpdateError extends Error {
  constructor(public readonly kind: ErrorKind) {
    super(ERROR_MESSAGES[kind])
    this.name = 'BrowserUpdateError'
  }
}

const ID = /^[a-f0-9]{32}$/
const REVISION = /^[a-f0-9]{40}$/
const RELEASE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/

const JOB_PHASES = new Set<BrowserUpdatePhase>([
  'stopping',
  'switching',
  'starting',
  'rollback_stopping',
  'rollback_switching',
  'rollback_starting',
  'succeeded',
  'rolled_back',
  'failed',
  'recovery_required'
])

const MAX_BODY_BYTES = 32 * 1024
const TIMEOUT_MS = 15_000

function valid(condition: unknown): asserts condition {
  if (!condition) {
    throw new BrowserUpdateError('invalid')
  }
}

function object(value: unknown): asserts value is Record<string, unknown> {
  valid(value !== null && typeof value === 'object' && !Array.isArray(value))
}

function text(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value.length > 0 &&
    value.length <= 4096 &&
    // Reject markup and non-printing controls; ordinary whitespace is allowed.
    // eslint-disable-next-line no-control-regex
    !/[<>\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value)
  )
}

function matches(value: unknown, pattern: RegExp): value is string {
  return typeof value === 'string' && pattern.test(value)
}

function parseOffer(value: unknown): BrowserUpdateOffer {
  object(value)
  valid(matches(value.id, ID))
  valid(matches(value.current_release, RELEASE) && matches(value.target_release, RELEASE))
  valid(typeof value.expires_at === 'number' && Number.isFinite(value.expires_at))
  valid(text(value.warning) && matches(value.tested_backend, REVISION))
  valid(value.compatibility === 'not-exercised')

  return {
    id: value.id,
    current_release: value.current_release,
    target_release: value.target_release,
    expires_at: value.expires_at,
    warning: value.warning,
    tested_backend: value.tested_backend,
    compatibility: value.compatibility
  }
}

function parseSnapshot(value: unknown): BrowserUpdateSnapshot {
  object(value)
  valid(Array.isArray(value.capabilities))
  valid(value.capabilities.length === 0 || (value.capabilities.length === 1 && value.capabilities[0] === 'install'))
  const snapshot: BrowserUpdateSnapshot = { capabilities: [...value.capabilities] }

  if (value.capabilities.length === 0) {
    valid(
      value.phase === undefined && value.offer === undefined && value.job === undefined && value.error === undefined
    )

    return snapshot
  }

  valid(typeof value.phase === 'string' && PHASES.includes(value.phase as BrowserUpdatePhase))
  const phase = value.phase as BrowserUpdatePhase
  snapshot.phase = phase

  if (value.offer !== undefined) {
    snapshot.offer = parseOffer(value.offer)
  }

  if (value.error !== undefined) {
    valid(text(value.error))
    snapshot.error = value.error
  }

  if (phase === 'idle' || phase === 'preparing') {
    valid(!snapshot.offer && value.job === undefined)
  }

  if (phase === 'offered' || phase === 'current') {
    valid(snapshot.offer && value.job === undefined)
  }

  if (value.job !== undefined) {
    object(value.job)
    valid(JOB_PHASES.has(phase) && snapshot.offer && value.job.id === snapshot.offer.id && value.job.phase === phase)
    snapshot.job = { id: snapshot.offer.id, phase }
  }

  // A failed preflight may already be confirmed. Interrupted preparation may
  // instead report failed/recovery_required without any offer or job.
  if (JOB_PHASES.has(phase) && phase !== 'failed' && phase !== 'recovery_required') {
    valid(snapshot.job)
  }

  return snapshot
}

async function request(path: string, body?: object, signal?: AbortSignal): Promise<unknown> {
  if (window.location.protocol !== 'https:') {
    throw new BrowserUpdateError('insecure')
  }

  // The browser uses hash routing. A pathname prefix is not a supported mount.
  if (window.location.pathname !== '/') {
    throw new BrowserUpdateError('unsupported')
  }

  if (signal?.aborted) {
    throw new DOMException('Request cancelled.', 'AbortError')
  }

  const controller = new AbortController()
  const cancel = () => controller.abort(new DOMException('Request cancelled.', 'AbortError'))
  signal?.addEventListener('abort', cancel, { once: true })
  const timer = setTimeout(() => controller.abort(new BrowserUpdateError('network')), TIMEOUT_MS)
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
  let onAbort!: () => void

  const aborted = new Promise<never>((_, reject) => {
    onAbort = () => reject(controller.signal.reason)
    controller.signal.addEventListener('abort', onAbort, { once: true })
  })

  try {
    return await Promise.race([
      aborted,
      (async () => {
        const response = await fetch(path, {
          method: body === undefined ? 'GET' : 'POST',
          headers:
            body === undefined
              ? { Accept: 'application/json' }
              : { Accept: 'application/json', 'Content-Type': 'application/json' },
          body: body === undefined ? undefined : JSON.stringify(body),
          credentials: 'same-origin',
          mode: 'same-origin',
          redirect: 'error',
          cache: 'no-store',
          signal: controller.signal
        })

        controller.signal.throwIfAborted()
        reader = response.body?.getReader()

        if (!response.ok) {
          const kinds: Record<number, ErrorKind> = {
            404: 'unsupported',
            401: 'forbidden',
            403: 'forbidden',
            409: 'conflict'
          }

          throw new BrowserUpdateError(kinds[response.status] ?? 'network')
        }

        valid(!response.redirected)
        valid(response.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() === 'application/json')
        valid(reader)
        const decoder = new TextDecoder()
        let bytes = 0
        let raw = ''

        while (true) {
          const chunk = await reader.read()
          controller.signal.throwIfAborted()

          if (chunk.done) {
            break
          }

          bytes += chunk.value.byteLength
          valid(bytes <= MAX_BODY_BYTES)
          raw += decoder.decode(chunk.value, { stream: true })
        }

        raw += decoder.decode()

        try {
          return JSON.parse(raw) as unknown
        } catch {
          throw new BrowserUpdateError('invalid')
        }
      })()
    ])
  } catch (error) {
    if (signal?.aborted) {
      throw new DOMException('Request cancelled.', 'AbortError')
    }

    if (error instanceof BrowserUpdateError) {
      throw error
    }

    throw new BrowserUpdateError('network')
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', cancel)
    controller.signal.removeEventListener('abort', onAbort)
    // Cancellation must not await a stalled stream's cleanup. The request's
    // error is already authoritative; cleanup errors cannot replace it.
    void reader?.cancel().catch(() => {})
  }
}

export async function readBrowserUpdateStatus(signal?: AbortSignal): Promise<BrowserUpdateSnapshot> {
  return parseSnapshot(await request('/browser-updater/api/status', {}, signal))
}

export async function prepareBrowserUpdate(signal?: AbortSignal): Promise<BrowserUpdateSnapshot> {
  return parseSnapshot(await request('/browser-updater/api/offer', {}, signal))
}

export async function applyBrowserUpdate(offerId: string): Promise<BrowserUpdateSnapshot> {
  valid(matches(offerId, ID))

  const snapshot = parseSnapshot(
    await request('/browser-updater/api/apply', {
      offer_id: offerId,
      confirm_restart_and_rollback: true
    })
  )

  valid(!snapshot.offer || snapshot.offer.id === offerId)

  return snapshot
}

// Explicit UI action only: stock cookie authentication may refresh the session,
// but proves identity, not membership of the helper's administrator allowlist.
export async function checkBrowserUpdateSession(signal?: AbortSignal): Promise<void> {
  const value = await request('/api/auth/me', undefined, signal)
  object(value)
  valid(text(value.provider) && text(value.user_id))
  valid(value.org_id === null || (typeof value.org_id === 'string' && (value.org_id === '' || text(value.org_id))))
}
