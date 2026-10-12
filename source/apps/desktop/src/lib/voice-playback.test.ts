// Teardown contract for voice-playback (#91991): a barge-in (stopVoicePlayback)
// must drain EVERY live playback — streaming sessions (WebSocket + AudioContext)
// and data-URL audio elements — and a settled session must never resume or
// write stale state into a newer turn.

import type * as SharedTypes from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { speakText } from '@/hermes'
import { $voicePlayback } from '@/store/voice-playback'

import {
  playSpeechText,
  SpeechAudioBlockedError,
  startSpeechStream,
  stopVoicePlayback,
  unlockSpeechAudio
} from './voice-playback'

const gateway = vi.hoisted(() => {
  let nextUrl: string | null = 'ws://gateway.test/api/ws'

  return {
    currentUrl: () => nextUrl,
    setNextUrl: (url: string | null) => {
      nextUrl = url
    }
  }
})

vi.mock('@hermes/shared', async importOriginal => {
  const actual = await importOriginal<typeof SharedTypes>()

  return {
    ...actual,
    resolveGatewayWsUrl: async () => gateway.currentUrl()
  }
})

vi.mock('@/hermes', () => ({
  getApiRequestProfile: () => null,
  getApiRequestConnection: () => null,
  speakText: vi.fn(async () => ({
    data_url: 'data:audio/mpeg;base64,AAAA',
    mime_type: 'audio/mpeg',
    ok: true
  }))
}))

class FakeSocket {
  static readonly OPEN = 1
  static readonly CONNECTING = 0
  static instances: FakeSocket[] = []

  binaryType = 'blob'
  readyState = FakeSocket.CONNECTING
  sent: string[] = []
  url: string

  close = vi.fn(() => {
    this.readyState = 3
  })
  send = vi.fn((data: string) => {
    this.sent.push(data)
  })

  onopen: (() => void) | null = null
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null

  constructor(url: string) {
    this.url = url
    FakeSocket.instances.push(this)
  }

  serverOpen() {
    this.readyState = FakeSocket.OPEN
    this.onopen?.()
  }

  serverFrame(frame: object) {
    this.onmessage?.({ data: JSON.stringify(frame) })
  }

  serverPcm(bytes: number) {
    this.onmessage?.({ data: new ArrayBuffer(bytes) })
  }

  serverClose() {
    this.onclose?.()
  }

  serverError() {
    this.onerror?.()
  }
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []
  // iOS WebKit: a context starts only during a user gesture, and resume()
  // outside one never settles. Stricter than WebKit, which also allows the
  // first seconds after a tap.
  static requireGesture = false
  static inGesture = false

  currentTime = 0
  destination = {}
  gains: { connect: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }[] = []
  state = FakeAudioContext.requireGesture ? 'suspended' : 'running'
  private listeners = new Set<() => void>()

  addEventListener = (_type: string, listener: () => void) => this.listeners.add(listener)
  removeEventListener = (_type: string, listener: () => void) => this.listeners.delete(listener)
  close = vi.fn(() => {
    this.setState('closed')

    return Promise.resolve()
  })
  resume = vi.fn(() => {
    if (FakeAudioContext.requireGesture && !FakeAudioContext.inGesture) {
      return new Promise<void>(() => undefined)
    }

    this.setState('running')

    return Promise.resolve()
  })
  createBuffer = vi.fn((_channels: number, length: number, rate: number) => ({
    duration: length / rate,
    getChannelData: () => new Float32Array(length)
  }))
  createBufferSource = vi.fn(() => ({
    buffer: null as unknown,
    connect: vi.fn(),
    start: vi.fn()
  }))
  createGain = vi.fn(() => {
    const gain = { connect: vi.fn(), disconnect: vi.fn() }
    this.gains.push(gain)

    return gain
  })

  constructor() {
    FakeAudioContext.instances.push(this)
  }

  setState(state: string) {
    this.state = state
    this.listeners.forEach(listener => listener())
  }
}

function tap(action: () => void) {
  FakeAudioContext.inGesture = true

  try {
    action()
  } finally {
    FakeAudioContext.inGesture = false
  }
}

class FakeAudio {
  static instances: FakeAudio[] = []

  paused = false
  src: string

  addEventListener = vi.fn()
  load = vi.fn()
  pause = vi.fn(() => {
    this.paused = true
  })
  play = vi.fn(() => {
    this.paused = false

    return Promise.resolve()
  })
  removeEventListener = vi.fn()

  constructor(src: string) {
    this.src = src
    FakeAudio.instances.push(this)
  }
}

async function liveSession() {
  const session = await startSpeechStream({ source: 'voice-conversation' })

  if (!session) {
    throw new Error('expected a live speech stream')
  }

  return session
}

describe('voice-playback teardown (#91991)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    FakeSocket.instances = []
    FakeAudioContext.instances = []
    FakeAudio.instances = []
    gateway.setNextUrl('ws://gateway.test/api/ws')
    vi.stubGlobal('WebSocket', FakeSocket)
    vi.stubGlobal('AudioContext', FakeAudioContext)
    vi.stubGlobal('Audio', FakeAudio)
    vi.stubGlobal('hermesDesktop', { getConnection: vi.fn(async () => ({ profile: 'test' })) })
    $voicePlayback.set({ audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' })
  })

  afterEach(() => {
    stopVoicePlayback()
    // The app keeps one context for the window; start each test with a new one.
    FakeAudioContext.instances.forEach(context => void context.close())
    FakeAudioContext.requireGesture = false
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('abandons a stream start that crosses a stop instead of resuming playback', async () => {
    // Stream discovery (gateway connection) is in flight when the user barges.
    const pending = startSpeechStream({ source: 'voice-conversation' })

    stopVoicePlayback()

    const session = await pending

    expect(session).toBeNull()
    expect(FakeSocket.instances).toHaveLength(0)
    expect($voicePlayback.get().status).toBe('idle')
  })

  it('does not let a settled session reset the playback state of a newer turn', async () => {
    const first = await liveSession()
    const second = await liveSession()

    // Starting the second session settles the first (its socket closes), but
    // the settled session's completion callback must not stomp the state the
    // newer session just wrote.
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalled()
    expect($voicePlayback.get().status).toBe('preparing')

    stopVoicePlayback()
  })

  it('a barge-in closes the socket and cuts the audio of a live streaming session and settles it', async () => {
    const session = await liveSession()
    const ws = FakeSocket.instances[0]

    ws.serverOpen()
    ws.serverFrame({ type: 'start', sample_rate: 24_000 })
    ws.serverPcm(64)

    expect($voicePlayback.get().status).toBe('speaking')

    stopVoicePlayback()

    expect(ws.close).toHaveBeenCalled()
    expect(FakeAudioContext.instances[0]?.gains[0]?.disconnect).toHaveBeenCalled()
    expect(FakeAudioContext.instances[0]?.state).toBe('running')
    expect($voicePlayback.get().status).toBe('idle')
    await expect(session.done).resolves.toBe('done')

    // A settled session can no longer feed text — nothing reaches the socket.
    const sentBefore = ws.sent.length

    session.append('stale text')
    session.finish()

    expect(ws.sent.length).toBe(sentBefore)
  })

  it('a barge-in during the drain window settles the session and the late drain is a no-op', async () => {
    const session = await liveSession()
    const ws = FakeSocket.instances[0]

    ws.serverOpen()
    ws.serverFrame({ type: 'start', sample_rate: 24_000 })
    ws.serverPcm(64)
    ws.serverFrame({ type: 'end' }) // drains, then settles — timer pending

    stopVoicePlayback()

    await expect(session.done).resolves.toBe('done')
    expect(ws.close).toHaveBeenCalled()

    // The pending drain timer fires later — it must not double-settle or
    // disturb anything that started since.
    await vi.runOnlyPendingTimersAsync()

    expect($voicePlayback.get().status).toBe('idle')
  })

  it('a barge-in cuts a data-url playback and clears the audio element', async () => {
    gateway.setNextUrl(null) // no streaming endpoint → data-url fallback

    const playback = playSpeechText('hello world', { messageId: 'm1', source: 'read-aloud' })

    await vi.advanceTimersByTimeAsync(0)

    expect($voicePlayback.get().status).toBe('speaking')
    expect(FakeAudio.instances).toHaveLength(1)

    stopVoicePlayback()

    expect(FakeAudio.instances[0]?.pause).toHaveBeenCalled()
    expect(FakeAudio.instances[0]?.src).toBe('')
    expect(FakeAudio.instances[0]?.load).toHaveBeenCalled()
    expect($voicePlayback.get().status).toBe('idle')
    await expect(playback).resolves.toBe(false)
  })

  it('one stop drains BOTH of two overlapping streaming sessions (#91991)', async () => {
    // The reported overlap: session A is live and playing;
    // session B starts (its stopVoicePlayback drains A under the Set
    // registry — the old single currentStop slot left A's socket open and
    // its done unsettled). A final explicit stop must still tear down
    // everything that is live.
    const first = await liveSession()

    const firstWs = FakeSocket.instances[0]

    firstWs.serverOpen()
    firstWs.serverFrame({ type: 'start', sample_rate: 24_000 })
    firstWs.serverPcm(64)
    first.append('turn one')

    const second = await liveSession()

    const secondWs = FakeSocket.instances[1]

    secondWs.serverOpen()
    secondWs.serverFrame({ type: 'start', sample_rate: 24_000 })
    secondWs.serverPcm(64)
    second.append('turn two')

    // Starting B drained A: its socket is closed and its audio is cut.
    await expect(first.done).resolves.toBe('done')
    expect(firstWs.close).toHaveBeenCalled()
    expect(FakeAudioContext.instances[0]?.gains[0]?.disconnect).toHaveBeenCalled()

    stopVoicePlayback()

    await expect(second.done).resolves.toBe('done')
    expect(secondWs.close).toHaveBeenCalled()
    expect(FakeAudioContext.instances[0]?.gains[1]?.disconnect).toHaveBeenCalled()
    expect($voicePlayback.get().status).toBe('idle')
  })
})

describe('voice-playback on iOS WebKit, which plays audio only after a tap', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    FakeSocket.instances = []
    FakeAudioContext.instances = []
    FakeAudioContext.requireGesture = true
    FakeAudio.instances = []
    vi.mocked(speakText).mockClear()
    gateway.setNextUrl('ws://gateway.test/api/ws')
    vi.stubGlobal('WebSocket', FakeSocket)
    vi.stubGlobal('AudioContext', FakeAudioContext)
    vi.stubGlobal('Audio', FakeAudio)
    vi.stubGlobal('hermesDesktop', { getConnection: vi.fn(async () => ({ profile: 'test' })) })
    $voicePlayback.set({ audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' })
  })

  afterEach(() => {
    stopVoicePlayback()
    FakeAudioContext.instances.forEach(context => void context.close())
    FakeAudioContext.requireGesture = false
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  async function streamReply() {
    await vi.advanceTimersByTimeAsync(0)
    const ws = FakeSocket.instances.at(-1)!

    ws.serverOpen()
    ws.serverFrame({ type: 'start', sample_rate: 24_000 })
    ws.serverPcm(4_800)

    return ws
  }

  it('plays a reply read aloud when its audio arrives after the tap', async () => {
    let playback!: Promise<boolean>

    tap(() => {
      playback = playSpeechText('Hello there.', { messageId: 'm1', source: 'read-aloud' })
    })

    const ws = await streamReply()

    expect(FakeAudioContext.instances).toHaveLength(1)
    expect(FakeAudioContext.instances[0]?.state).toBe('running')
    expect($voicePlayback.get().status).toBe('speaking')

    ws.serverFrame({ type: 'end' })
    await vi.runAllTimersAsync()

    await expect(playback).resolves.toBe(true)
    expect($voicePlayback.get().status).toBe('idle')
  })

  it('plays every later reply through the context the first tap unlocked', async () => {
    tap(() => void playSpeechText('First.', { messageId: 'm1', source: 'read-aloud' }))
    ;(await streamReply()).serverFrame({ type: 'end' })
    await vi.runAllTimersAsync()

    // An automatic read: no tap at all.
    void playSpeechText('Second.', { messageId: 'm2', source: 'read-aloud' })
    await streamReply()

    expect(FakeAudioContext.instances).toHaveLength(1)
    expect(FakeAudioContext.instances[0]?.state).toBe('running')
    expect($voicePlayback.get()).toMatchObject({ messageId: 'm2', status: 'speaking' })
  })

  it('plays an automatic read after an earlier unrelated tap unlocked audio', async () => {
    tap(unlockSpeechAudio)

    void playSpeechText('A new reply.', { messageId: 'm1', source: 'read-aloud' })
    await streamReply()

    expect(FakeAudioContext.instances[0]?.state).toBe('running')
    expect($voicePlayback.get().status).toBe('speaking')
  })

  it('never shows speaking while the browser blocks audio, then reports it blocked', async () => {
    // An automatic read, so it carries a turn key.
    const outcome = playSpeechText('Nobody tapped.', { messageId: 'm1', source: 'read-aloud', turnKey: 't1' }).catch(
      error => error
    )
    const ws = await streamReply()

    expect($voicePlayback.get().status).toBe('preparing')

    ws.serverFrame({ type: 'end' })
    ws.serverClose()
    await vi.advanceTimersByTimeAsync(3_000)

    expect(await outcome).toBeInstanceOf(SpeechAudioBlockedError)
    expect($voicePlayback.get().status).toBe('idle')
    // Not retried over POST: that audio element would be blocked as well.
    expect(speakText).not.toHaveBeenCalled()
  })

  it('starts speaking once a tap unlocks audio that was waiting', async () => {
    const playback = playSpeechText('Tap to hear.', { messageId: 'm1', source: 'read-aloud' })
    const ws = await streamReply()

    ws.serverFrame({ type: 'end' })
    tap(unlockSpeechAudio)

    expect($voicePlayback.get().status).toBe('speaking')

    await vi.runAllTimersAsync()

    await expect(playback).resolves.toBe(true)
  })

  it('resumes the unlocked context after iOS interrupted it', async () => {
    tap(unlockSpeechAudio)
    FakeAudioContext.requireGesture = false // once unlocked, WebKit resumes it without a tap
    FakeAudioContext.instances[0]?.setState('interrupted')

    void playSpeechText('After a phone call.', { messageId: 'm1', source: 'read-aloud' })
    await streamReply()

    expect(FakeAudioContext.instances[0]?.state).toBe('running')
    expect($voicePlayback.get().status).toBe('speaking')
  })
})
