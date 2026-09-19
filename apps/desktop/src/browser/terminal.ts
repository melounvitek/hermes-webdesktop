import { isGatewayReauthRequired } from '@hermes/shared'

import type { HermesApiRequest, HermesTerminalExit, HermesTerminalSession, HermesTerminalStatus } from '../global'

const BASE = '/api/plugins/browser-terminal'
const TIMEOUT_MS = 10_000
const MAX_INPUT_BYTES = 65_536
const RETRY_DELAYS = [500, 1500, 3000]

type TerminalApi = Window['hermesDesktop']['terminal']
type Reason = NonNullable<HermesTerminalStatus['reason']>

interface Session {
  profile: string
  status: HermesTerminalStatus
  data: Set<(data: string) => void>
  exits: Set<(exit: HermesTerminalExit) => void>
  statuses: Set<(status: HermesTerminalStatus) => void>
  socket?: WebSocket
  cancelDial?: () => void
  retryTimer?: ReturnType<typeof setTimeout>
  heartbeatTimer?: ReturnType<typeof setInterval>
  stableTimer?: ReturnType<typeof setTimeout>
  retries: number
  generation: number
  size: { cols: number; rows: number }
}

function failure(error: unknown, missing: Reason = 'missing-session'): Error & { reason: Reason } {
  const message = error instanceof Error ? error.message : String(error)

  // Without the plugin, the SPA fallback rejects POST /sessions with 405.
  const missingStatus = missing === 'missing-plugin' ? /^HTTP 40[45]\b/ : /^HTTP 404\b/
  const auth = isGatewayReauthRequired(error) || /^HTTP 40[13]\b/.test(message)
  const reason: Reason = auth ? 'auth' : missingStatus.test(message) ? missing : 'connection'

  return Object.assign(error instanceof Error ? error : new Error(message), { reason })
}

/** Runtime-only ownership. A reconnect can attach only to an id created here,
 * with the profile captured at creation, never to the currently selected profile. */
export function createBrowserTerminal(api: <T>(request: HermesApiRequest) => Promise<T>): TerminalApi {
  const sessions = new Map<string, Session>()

  const publish = (session: Session, status: HermesTerminalStatus) => {
    session.status = status
    session.statuses.forEach(callback => callback(status))
  }

  const stop = (session: Session) => {
    session.generation++
    clearTimeout(session.retryTimer)
    clearInterval(session.heartbeatTimer)
    clearTimeout(session.stableTimer)
    session.cancelDial?.()
    session.cancelDial = undefined
    const socket = session.socket
    session.socket = undefined

    if (socket) {
      socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null
      socket.close()
    }
  }

  const request = <T>(id: string, session: Session, method: 'POST' | 'DELETE', suffix = '') =>
    api<T>({
      path: `${BASE}/sessions/${encodeURIComponent(id)}${suffix}`,
      profile: session.profile,
      method,
      timeoutMs: TIMEOUT_MS
    })

  const reconnect = (id: string, session: Session) => {
    stop(session)

    if (session.retries >= RETRY_DELAYS.length) {
      publish(session, { state: 'disconnected', reason: 'connection' })

      return
    }

    publish(session, { state: 'reconnecting' })
    session.retryTimer = setTimeout(() => void dial(id, session, true), RETRY_DELAYS[session.retries++])
  }

  const dial = async (id: string, session: Session, automatic: boolean): Promise<boolean> => {
    const generation = session.generation
    const current = () => sessions.get(id) === session && session.generation === generation

    try {
      const { ticket } = await request<{ ticket: string }>(id, session, 'POST', '/ticket')

      if (!current()) {
        return false
      }

      if (!ticket) {
        throw new Error('No terminal WebSocket ticket returned')
      }

      const url = new URL(`${BASE}/ws`, window.location.origin)
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
      url.searchParams.set('ticket', ticket)
      const socket = new WebSocket(url)
      session.socket = socket
      socket.binaryType = 'arraybuffer'
      const decoder = new TextDecoder()

      return await new Promise<boolean>(resolve => {
        const timer = setTimeout(() => lost(), TIMEOUT_MS)

        const settled = (result: boolean) => {
          clearTimeout(timer)
          session.cancelDial = undefined
          resolve(result)
        }

        session.cancelDial = () => settled(false)

        const lost = () => {
          if (!current()) {
            return
          }

          settled(false)
          reconnect(id, session)
        }

        socket.onopen = () => {
          if (!current()) {
            return
          }

          // The server replays its bounded buffer on every attach. Reset first,
          // including scrollback, rather than appending duplicate output.
          session.data.forEach(callback => callback('\x1bc'))
          publish(session, { state: 'open' })

          try {
            socket.send(JSON.stringify({ type: 'resize', ...session.size }))
            settled(true)
            session.stableTimer = setTimeout(() => {
              session.retries = 0
            }, 10_000)
            session.heartbeatTimer = setInterval(() => void heartbeat(), 20_000)
          } catch {
            lost()
          }
        }

        const heartbeat = async () => {
          if (!current() || socket.readyState !== 1) {
            return
          }

          try {
            await request(id, session, 'POST', '/heartbeat')
          } catch (error) {
            if (!current()) {
              return
            }

            const { reason } = failure(error)

            if (reason === 'connection') {
              reconnect(id, session)
            } else {
              stop(session)
              publish(session, { state: 'disconnected', reason })
            }
          }
        }

        socket.onmessage = event => {
          if (!current() || typeof event.data === 'string') {
            return
          }

          const text = decoder.decode(event.data, { stream: true })
          session.data.forEach(callback => callback(text))
        }

        socket.onerror = lost

        socket.onclose = event => {
          if (!current()) {
            return
          }

          settled(false)

          const reason = ({ 4401: 'auth', 4409: 'superseded', 4410: 'exited' } as const)[
            event.code as 4401 | 4409 | 4410
          ]

          if (!reason) {
            reconnect(id, session)

            return
          }

          stop(session)

          if (reason === 'exited') {
            // Exit closes the tab through onExit; it is not a recoverable error.
            session.status = { state: 'disconnected' }
            session.exits.forEach(callback => callback({ code: null, signal: null }))
          } else {
            publish(session, { state: 'disconnected', reason })
          }
        }
      })
    } catch (error) {
      if (!current()) {
        return false
      }

      const { reason } = failure(error)

      if (automatic && reason === 'connection') {
        reconnect(id, session)
      } else {
        publish(session, { state: 'disconnected', reason })
      }

      return false
    }
  }

  const send = (id: string, value: string | Uint8Array<ArrayBuffer>) => {
    const session = sessions.get(id)

    if (!session || session.socket?.readyState !== 1) {
      return false
    }

    try {
      session.socket.send(value)

      return true
    } catch {
      reconnect(id, session)

      return false
    }
  }

  return {
    async start(options = {}) {
      const { profile = 'default', cwd, cols = 80, rows = 24 } = options
      let result: HermesTerminalSession

      try {
        result = await api<HermesTerminalSession>({
          path: `${BASE}/sessions`,
          profile,
          method: 'POST',
          body: { cwd: cwd || undefined, cols, rows },
          timeoutMs: TIMEOUT_MS
        })

        if (!result?.id) {
          throw new Error('No terminal session returned')
        }
      } catch (error) {
        throw failure(error, 'missing-plugin')
      }

      sessions.set(result.id, {
        profile,
        status: { state: 'connecting' },
        data: new Set(),
        exits: new Set(),
        statuses: new Set(),
        retries: 0,
        generation: 0,
        size: { cols, rows }
      })

      return result
    },
    async attach(id) {
      const session = sessions.get(id)

      if (!session) {
        return false
      }

      stop(session)
      session.retries = 0
      publish(session, { state: 'connecting' })

      return dial(id, session, false)
    },
    async dispose(id) {
      const session = sessions.get(id)

      if (!session) {
        return false
      }

      sessions.delete(id)
      stop(session)
      session.data.clear()
      session.exits.clear()
      session.statuses.clear()

      // Closing is synchronous locally; an unreachable server must not strand
      // retries or produce an unhandled rejection during React cleanup.
      try {
        await request(id, session, 'DELETE')

        return true
      } catch {
        return false
      }
    },
    async cwd() {
      return null
    },
    onData(id, callback) {
      const callbacks = sessions.get(id)?.data
      callbacks?.add(callback)

      return () => {
        callbacks?.delete(callback)
      }
    },
    onExit(id, callback) {
      const callbacks = sessions.get(id)?.exits
      callbacks?.add(callback)

      return () => {
        callbacks?.delete(callback)
      }
    },
    onStatus(id, callback) {
      const session = sessions.get(id)
      session?.statuses.add(callback)

      if (session) {
        callback(session.status)
      }

      return () => {
        session?.statuses.delete(callback)
      }
    },
    async resize(id, size) {
      const session = sessions.get(id)

      if (session) {
        session.size = size
      }

      return send(id, JSON.stringify({ type: 'resize', ...size }))
    },
    async write(id, data) {
      const bytes = new TextEncoder().encode(data)
      let offset = 0

      do {
        if (!send(id, bytes.subarray(offset, offset + MAX_INPUT_BYTES))) {
          return false
        }

        offset += MAX_INPUT_BYTES
      } while (offset < bytes.byteLength)

      return true
    }
  }
}
