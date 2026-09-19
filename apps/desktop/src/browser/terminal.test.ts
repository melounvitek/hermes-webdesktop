import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createBrowserBridge } from './bridge'

class Socket {
  static instances: Socket[] = []
  readyState = 0
  binaryType = ''
  sent: unknown[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: ArrayBuffer }) => void) | null = null
  onclose: ((event: { code: number }) => void) | null = null
  onerror: (() => void) | null = null
  constructor(readonly url: string) {
    Socket.instances.push(this)
  }
  open() {
    this.readyState = 1
    this.onopen?.()
  }
  output(text: string) {
    this.onmessage?.({ data: new TextEncoder().encode(text).buffer })
  }
  close(code = 1006) {
    this.readyState = 3
    this.onclose?.({ code })
  }
  send(data: unknown) {
    this.sent.push(data)
  }
}

let requests: { url: URL; init: RequestInit }[]
let ticketStatus: number
let heartbeatStatus: number
beforeEach(() => {
  vi.useFakeTimers()
  Socket.instances = []
  requests = []
  ticketStatus = 200
  heartbeatStatus = 200
  vi.stubGlobal('WebSocket', Socket)
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit) => {
      requests.push({ url: new URL(url), init })

      if (url.includes('/heartbeat')) {
        if (heartbeatStatus === 0) {
          throw new TypeError('Network failed')
        }

        return new Response(JSON.stringify({ ok: true }), { status: heartbeatStatus })
      }

      if (url.includes('/ticket')) {
        return new Response(JSON.stringify({ ticket: `ticket-${requests.length}` }), { status: ticketStatus })
      }

      return new Response(
        JSON.stringify(init.method === 'DELETE' ? { ok: true } : { id: 'owned', shell: 'bash', cwd: '/repo' })
      )
    })
  )
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})
const flush = () => vi.advanceTimersByTimeAsync(0)
const bridge = () => createBrowserBridge({ token: 'secret', authRequired: true }).terminal

it.each([401, 403, 503, 0])('renews only connected leases under their owner and handles heartbeat failure %s', async httpStatus => {
  const terminal = bridge()
  const options = { profile: 'alpha' }
  const { id } = await terminal.start(options)
  options.profile = 'beta'
  const status = vi.fn()
  terminal.onStatus!(id, status)
  await vi.advanceTimersByTimeAsync(20_000)
  expect(requests).toHaveLength(1)
  const attached = terminal.attach(id)
  await flush()
  Socket.instances[0].open()
  await attached
  await vi.advanceTimersByTimeAsync(60_000)
  const beats = requests.filter(({ url }) => url.pathname.endsWith('/heartbeat'))
  expect(beats).toHaveLength(3)
  expect(beats.every(({ url, init }) => url.searchParams.get('profile') === 'alpha' && init.method === 'POST' && new Headers(init.headers).get('X-Hermes-Session-Token') === 'secret')).toBe(true)
  heartbeatStatus = httpStatus
  await vi.advanceTimersByTimeAsync(20_000)
  expect(Socket.instances[0].readyState).toBe(3)
  const reconnects = httpStatus === 503 || httpStatus === 0
  expect(status).toHaveBeenLastCalledWith(reconnects ? { state: 'reconnecting' } : { state: 'disconnected', reason: 'auth' })
  await vi.advanceTimersByTimeAsync(500)
  expect(Socket.instances).toHaveLength(reconnects ? 2 : 1)
  await terminal.dispose(id)
  const count = requests.length
  await vi.advanceTimersByTimeAsync(120_000)
  expect(requests).toHaveLength(count)
  expect(requests.filter(({ url }) => url.pathname.endsWith('/sessions'))).toHaveLength(1)
})

it('ignores a stale heartbeat failure after explicit reattach and stops heartbeats on disposal', async () => {
  const terminal = bridge()
  const { id } = await terminal.start({ profile: 'alpha' })
  const status = vi.fn()
  terminal.onStatus!(id, status)
  const attached = terminal.attach(id)
  await flush()
  Socket.instances[0].open()
  await attached
  let finishHeartbeat!: (value: Response) => void
  vi.mocked(fetch).mockImplementationOnce(() => new Promise(resolve => { finishHeartbeat = resolve }))
  await vi.advanceTimersByTimeAsync(20_000)
  const reattached = terminal.attach(id)
  await flush()
  Socket.instances[1].open()
  await reattached
  finishHeartbeat(new Response('expired', { status: 401 }))
  await flush()
  expect(status).toHaveBeenLastCalledWith({ state: 'open' })
  expect(Socket.instances[1].readyState).toBe(1)
  await terminal.dispose(id)
  const count = requests.length
  await vi.advanceTimersByTimeAsync(120_000)
  expect(requests).toHaveLength(count)
})

it('refunds retries only after a stable connection, while rapid open-close stays bounded', async () => {
  const terminal = bridge()
  const { id } = await terminal.start({ profile: 'alpha' })
  const status = vi.fn()
  terminal.onStatus!(id, status)
  const attached = terminal.attach(id)
  await flush()
  Socket.instances[0].open()
  await attached
  for (const delay of [500, 1500, 3000]) {
    Socket.instances.at(-1)!.close()
    await vi.advanceTimersByTimeAsync(delay)
    Socket.instances.at(-1)!.open()
  }
  await vi.advanceTimersByTimeAsync(10_000)
  Socket.instances.at(-1)!.close()
  expect(status).toHaveBeenLastCalledWith({ state: 'reconnecting' })
  for (const delay of [500, 1500, 3000]) {
    await vi.advanceTimersByTimeAsync(delay)
    Socket.instances.at(-1)!.open()
    Socket.instances.at(-1)!.close()
  }
  expect(status).toHaveBeenLastCalledWith({ state: 'disconnected', reason: 'connection' })
  const count = requests.length
  await vi.advanceTimersByTimeAsync(120_000)
  expect(requests).toHaveLength(count)
  await terminal.dispose(id)
})

it('attaches after listeners, preserves the first prompt, replays with reset, and never queues input across reconnects', async () => {
  const terminal = bridge()
  const options = { profile: 'alpha', cwd: '/repo', cols: 90, rows: 30 }
  const session = await terminal.start(options)
  options.profile = 'beta'
  expect(Socket.instances).toHaveLength(0)
  const output: string[] = []
  terminal.onData(session.id, text => output.push(text))
  const attached = terminal.attach(session.id)
  await flush()
  const first = Socket.instances[0]
  first.open()
  first.output('first prompt$ ')
  expect(await attached).toBe(true)
  expect(output.join('')).toBe('\x1bcfirst prompt$ ')
  await terminal.write(session.id, 'pwd\r')
  expect(new TextDecoder().decode(first.sent.find(value => typeof value !== 'string') as Uint8Array)).toBe('pwd\r')
  await terminal.resize(session.id, { cols: 100, rows: 40 })
  expect(JSON.parse(first.sent.at(-1) as string)).toEqual({ type: 'resize', cols: 100, rows: 40 })
  first.close()
  expect(await terminal.write(session.id, 'do-not-replay')).toBe(false)
  await vi.advanceTimersByTimeAsync(5000)
  const second = Socket.instances[1]
  expect(new URL(second.url).searchParams.get('ticket')).not.toBe(new URL(first.url).searchParams.get('ticket'))
  expect(new URL(second.url).searchParams.has('profile')).toBe(false)
  second.open()
  second.output('first prompt$ retained output')
  expect(output.join('')).toBe('\x1bcfirst prompt$ \x1bcfirst prompt$ retained output')
  expect(second.sent.every(value => typeof value === 'string')).toBe(true)
  expect(requests.every(({ url }) => url.searchParams.get('profile') === 'alpha')).toBe(true)
  expect(requests.every(({ init }) => new Headers(init.headers).get('X-Hermes-Session-Token') === 'secret')).toBe(true)
  expect(requests.filter(({ url }) => url.pathname.endsWith('/sessions'))).toHaveLength(1)
  await terminal.dispose(session.id)
  expect(requests.at(-1)?.init.method).toBe('DELETE')
  expect(await terminal.dispose('not-owned')).toBe(false)
})

it.each([
  ['exact limit', 'x'.repeat(65_536)],
  ['over limit', 'x'.repeat(65_537)],
  ['multibyte boundary', 'x'.repeat(65_535) + '🙂漢é'.repeat(20_000)]
])('sends %s input in bounded binary frames without changing UTF-8 bytes', async (_label, input) => {
  const terminal = bridge()
  const { id } = await terminal.start({ profile: 'alpha' })
  const attached = terminal.attach(id)
  await flush()
  const socket = Socket.instances[0]
  socket.open()
  await attached

  expect(await terminal.write(id, input)).toBe(true)
  const frames = socket.sent.filter((value): value is Uint8Array => ArrayBuffer.isView(value))
  const encoded = new TextEncoder().encode(input)
  expect(frames).toHaveLength(Math.ceil(encoded.byteLength / 65_536))
  const received = new Uint8Array(encoded.byteLength)
  let offset = 0

  for (const frame of frames) {
    expect(frame.byteLength).toBeGreaterThan(0)
    expect(frame.byteLength).toBeLessThanOrEqual(65_536)
    received.set(frame, offset)
    offset += frame.byteLength
  }

  expect(offset).toBe(encoded.byteLength)
  expect(received.every((byte, index) => byte === encoded[index])).toBe(true)
  expect(new TextDecoder().decode(received)).toBe(input)
  await terminal.dispose(id)
})

it.each(['send failure', 'disconnect', 'dispose'] as const)(
  'drops the remainder of a paste on %s and never replays it',
  async interruption => {
    const terminal = bridge()
    const { id } = await terminal.start({ profile: 'alpha' })
    const attached = terminal.attach(id)
    await flush()
    const socket = Socket.instances[0]
    socket.open()
    await attached
    const send = vi.spyOn(socket, 'send')
    send.mockImplementationOnce(data => {
      socket.sent.push(data)

      if (interruption === 'disconnect') {
        socket.close()
      } else if (interruption === 'dispose') {
        void terminal.dispose(id)
      }
    })

    if (interruption === 'send failure') {
      send.mockImplementationOnce(() => {
        throw new Error('Connection lost')
      })
    }

    const input = '🙂'.repeat(50_000)
    expect(await terminal.write(id, input)).toBe(false)
    expect(send).toHaveBeenCalledTimes(interruption === 'send failure' ? 2 : 1)
    expect(socket.sent.filter(value => ArrayBuffer.isView(value))).toEqual([
      new TextEncoder().encode(input).subarray(0, 65_536)
    ])
    expect(await terminal.write(id, 'do-not-replay')).toBe(false)
    await vi.advanceTimersByTimeAsync(500)

    if (interruption === 'dispose') {
      expect(Socket.instances).toHaveLength(1)
    } else {
      const reconnected = Socket.instances[1]
      reconnected.open()
      expect(reconnected.sent.every(value => typeof value === 'string')).toBe(true)
    }

    await terminal.dispose(id)
  }
)

it('bounds retry, cancels pending dials on close, and never creates a replacement shell for a lost server session', async () => {
  const terminal = bridge()
  const { id } = await terminal.start({ profile: 'alpha' })
  const status = vi.fn()
  terminal.onStatus!(id, status)
  const attached = terminal.attach(id)
  await flush()
  Socket.instances[0].open()
  await attached
  ticketStatus = 503
  Socket.instances[0].close()
  await vi.advanceTimersByTimeAsync(120_000)
  const count = requests.length
  expect(status).toHaveBeenLastCalledWith(expect.objectContaining({ state: 'disconnected' }))
  expect(count).toBeLessThan(8)
  await vi.advanceTimersByTimeAsync(120_000)
  expect(requests).toHaveLength(count)
  ticketStatus = 404
  expect(await terminal.attach(id)).toBe(false)
  expect(status).toHaveBeenLastCalledWith(expect.objectContaining({ reason: 'missing-session' }))
  expect(requests.filter(({ url }) => url.pathname.endsWith('/sessions'))).toHaveLength(1)
  ticketStatus = 200
  const pending = terminal.attach(id)
  await flush()
  const socket = Socket.instances.at(-1)!
  await terminal.dispose(id)
  expect(await pending).toBe(false)
  socket.open()
  await vi.advanceTimersByTimeAsync(120_000)
  expect(requests.at(-1)?.init.method).toBe('DELETE')
  expect(requests.filter(({ init }) => init.method === 'DELETE')).toHaveLength(1)
})

it.each([4401, 4409, 4410])('does not auto-reconnect terminal close code %s', async code => {
  const terminal = bridge()
  const { id } = await terminal.start({ profile: 'alpha' })
  const exit = vi.fn()
  terminal.onExit(id, exit)
  const attached = terminal.attach(id)
  await flush()
  Socket.instances[0].open()
  await attached
  Socket.instances[0].close(code)
  await vi.advanceTimersByTimeAsync(120_000)
  expect(Socket.instances).toHaveLength(1)
  expect(exit).toHaveBeenCalledTimes(code === 4410 ? 1 : 0)
})

it('times out a silent dial and exposes missing-plugin separately from authentication errors', async () => {
  const terminal = bridge()
  const { id } = await terminal.start()
  const attached = terminal.attach(id)
  await vi.advanceTimersByTimeAsync(30_000)
  expect(await attached).toBe(false)
  vi.mocked(fetch).mockResolvedValueOnce(new Response('not found', { status: 404 }))
  await expect(terminal.start()).rejects.toMatchObject({ reason: 'missing-plugin' })
  vi.mocked(fetch).mockResolvedValueOnce(new Response('expired', { status: 401 }))
  await expect(terminal.start()).rejects.toMatchObject({ reason: 'auth', needsOauthLogin: true })
})

it.each([
  [404, 'missing-plugin', 'missing-session'],
  [405, 'missing-plugin', 'connection'],
  [401, 'auth', 'auth'],
  [403, 'auth', 'auth'],
  [503, 'connection', 'connection']
] as const)('classifies HTTP %s separately for start and attach', async (httpStatus, startReason, attachReason) => {
  const terminal = bridge()
  vi.mocked(fetch).mockResolvedValueOnce(new Response('request failed', { status: httpStatus }))
  await expect(terminal.start()).rejects.toMatchObject({ reason: startReason })

  const { id } = await terminal.start({ profile: 'alpha' })
  const status = vi.fn()
  terminal.onStatus!(id, status)
  ticketStatus = httpStatus
  expect(await terminal.attach(id)).toBe(false)
  expect(status).toHaveBeenLastCalledWith({ state: 'disconnected', reason: attachReason })
  await terminal.dispose(id)
})

it.each([undefined, '', '/repo'])('only sends an explicit nonempty cwd (%s)', async cwd => {
  const terminal = bridge()
  const { id } = await terminal.start({ profile: 'alpha', cwd, cols: 90, rows: 30 })
  const { url, init } = requests[0]
  const body = JSON.parse(init.body as string)
  expect(url.searchParams.get('profile')).toBe('alpha')
  expect(body).toMatchObject({ cols: 90, rows: 30 })

  if (cwd) {
    expect(body.cwd).toBe(cwd)
  } else {
    expect(body).not.toHaveProperty('cwd')
  }

  await terminal.dispose(id)
})

it('aborts a stalled start through the stock authenticated HTTP path', async () => {
  vi.spyOn(AbortSignal, 'timeout').mockImplementation(ms => {
    const controller = new AbortController()
    setTimeout(() => controller.abort(new Error('Timed out')), ms)

    return controller.signal
  })
  vi.mocked(fetch).mockImplementationOnce(
    (_url, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(init.signal?.reason))
      })
  )
  const rejected = expect(bridge().start()).rejects.toMatchObject({ reason: 'connection' })
  await vi.advanceTimersByTimeAsync(30_000)
  await rejected
  expect(Socket.instances).toHaveLength(0)
})

it('closing during retry delay or ticket fetch prevents any later socket or shell creation', async () => {
  const terminal = bridge()
  const { id } = await terminal.start({ profile: 'alpha' })
  const attached = terminal.attach(id)
  await flush()
  Socket.instances[0].open()
  await attached
  Socket.instances[0].close()
  await terminal.dispose(id)
  const count = requests.length
  await vi.advanceTimersByTimeAsync(120_000)
  expect(requests).toHaveLength(count)

  const next = await terminal.start({ profile: 'beta' })
  let resolveTicket!: (response: Response) => void
  vi.mocked(fetch).mockImplementationOnce(
    () =>
      new Promise(resolve => {
        resolveTicket = resolve
      })
  )
  const pending = terminal.attach(next.id)
  await terminal.dispose(next.id)
  resolveTicket(new Response('{"ticket":"late"}'))
  expect(await pending).toBe(false)
  expect(Socket.instances).toHaveLength(1)
  expect(requests.at(-1)?.url.searchParams.get('profile')).toBe('beta')
})
