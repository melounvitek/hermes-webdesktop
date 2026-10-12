import { act, cleanup, fireEvent, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { clearSpokenRepliesForTests, markAssistantIdSpoken, resolveSpokenReply } from '@/lib/spoken-reply'
import { playSpeechText, SpeechAudioBlockedError, unlockSpeechAudio } from '@/lib/voice-playback'
import type * as Notifications from '@/store/notifications'
import { notify, notifyError } from '@/store/notifications'
import { $voicePlayback, setVoicePlaybackState } from '@/store/voice-playback'
import { $autoSpeakReplies } from '@/store/voice-prefs'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useAutoSpeakReplies } from './use-auto-speak-replies'

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: vi.fn(),
  SpeechAudioBlockedError: class SpeechAudioBlockedError extends Error {},
  unlockSpeechAudio: vi.fn()
}))
vi.mock('@/store/notifications', async importOriginal => ({
  ...(await importOriginal<typeof Notifications>()),
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const SESSION_ID = 'session-under-test'
const IDLE_STATE = { audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' as const }

function assistantMessage(id: string, text: string): ChatMessage {
  return { id, parts: [assistantTextPart(text)], role: 'assistant' }
}

// #93515 — Edge TTS has no chunked-PCM API, so the WS attempt in
// playSpeechText's fallback ladder settles 'fallback' before any audio plays
// and the client retries over the POST endpoint. While that POST round-trip
// is in flight, the backend can rewrite the just-completed reply's renderer
// id (`assistant-stream-*`) to its durable id. The issue claims
// `resolveSpokenReply()` fails to follow that rewrite and the reply gets
// spoken a second time once `$voicePlayback` goes idle.
describe('useAutoSpeakReplies — Edge TTS fallback chain (#93515)', () => {
  afterEach(() => {
    cleanup()
    clearSpokenRepliesForTests()
    $autoSpeakReplies.set(false)
    setVoicePlaybackState({ ...IDLE_STATE })
    vi.clearAllMocks()
  })

  it('does not re-speak the reply once playback goes idle after an id rewrite mid-fallback', async () => {
    $autoSpeakReplies.set(true)

    const $messages = atom<ChatMessage[]>([])

    // The exact pendingReply/markSpoken contract use-composer-voice.ts wires
    // up for this hook, backed by the real ordinal-anchored dedupe.
    const pendingReply = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)
      const spoken = resolveSpokenReply(SESSION_ID, messages)

      if (!last || last.id === spoken?.id) {
        return null
      }

      return { id: last.id, pending: Boolean(last.pending), text: chatMessageText(last) }
    }

    const markSpoken = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

      if (last) {
        markAssistantIdSpoken(SESSION_ID, messages, last.id)
      }
    }

    let settleFallback: (() => void) | null = null

    vi.mocked(playSpeechText).mockImplementation(async () => {
      setVoicePlaybackState({
        audioElement: null,
        messageId: 'assistant-stream-1',
        sequence: 0,
        source: 'read-aloud',
        status: 'preparing'
      })

      // Holds mid-ladder — the WS-fallback-then-POST round trip the issue
      // describes — until the test rewrites the message id underneath it.
      await new Promise<void>(resolve => {
        settleFallback = resolve
      })

      $messages.set([assistantMessage('durable-42', 'hello there')])

      setVoicePlaybackState({
        audioElement: null,
        messageId: 'durable-42',
        sequence: 0,
        source: 'read-aloud',
        status: 'idle'
      })

      return true
    })

    renderHook(
      () =>
        useAutoSpeakReplies({
          blockedLabel: 'press read aloud',
          conversationActive: false,
          failureLabel: 'read-aloud failed',
          markSpoken,
          pendingReply,
          sessionId: SESSION_ID
        }),
      {
        wrapper: ({ children }) => (
          <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages }}>{children}</ComposerScopeProvider>
        )
      }
    )

    act(() => {
      $messages.set([assistantMessage('assistant-stream-1', 'hello there')])
    })

    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))

    await act(async () => {
      settleFallback?.()
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect($voicePlayback.get().status).toBe('idle')
    expect(playSpeechText).toHaveBeenCalledTimes(1)
  })

  it('speaks a tool-folded rewrite once when the assistant ordinal drifts', async () => {
    $autoSpeakReplies.set(true)

    const $messages = atom<ChatMessage[]>([])

    const pendingReply = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)
      const spoken = resolveSpokenReply(SESSION_ID, messages)

      if (!last || last.id === spoken?.id) {
        return null
      }

      return { id: last.id, pending: Boolean(last.pending), text: chatMessageText(last), turnKey: `${SESSION_ID}:0` }
    }

    const markSpoken = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

      if (last) {
        markAssistantIdSpoken(SESSION_ID, messages, last.id)
      }
    }

    let settlePlayback: (() => void) | null = null

    vi.mocked(playSpeechText).mockImplementation(async () => {
      setVoicePlaybackState({
        audioElement: null,
        messageId: 'assistant-stream-1',
        sequence: 1,
        source: 'read-aloud',
        status: 'preparing'
      })

      await new Promise<void>(resolve => {
        settlePlayback = resolve
      })

      // Tool rows fold into one durable bubble. The assistant ordinal moves.
      $messages.set([
        { id: 'u1', parts: [assistantTextPart('do the thing')], role: 'user' },
        assistantMessage('durable-42', 'the folded answer')
      ])
      setVoicePlaybackState({ ...IDLE_STATE })

      return true
    })

    renderHook(
      () =>
        useAutoSpeakReplies({
          blockedLabel: 'press read aloud',
          conversationActive: false,
          failureLabel: 'read-aloud failed',
          markSpoken,
          pendingReply,
          sessionId: SESSION_ID
        }),
      {
        wrapper: ({ children }) => (
          <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages }}>{children}</ComposerScopeProvider>
        )
      }
    )

    act(() => {
      $messages.set([
        { id: 'u1', parts: [assistantTextPart('do the thing')], role: 'user' },
        assistantMessage('narration', 'checking'),
        assistantMessage('tool-segment', 'ran the tool'),
        assistantMessage('assistant-stream-1', 'the folded answer')
      ])
    })

    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))

    await act(async () => {
      settlePlayback?.()
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect($voicePlayback.get().status).toBe('idle')
    expect(playSpeechText).toHaveBeenCalledTimes(1)
  })

  it('does not consume a reply when markSpoken runs before a play that never starts', async () => {
    $autoSpeakReplies.set(true)

    const $messages = atom<ChatMessage[]>([])

    const pendingReply = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)
      const spoken = resolveSpokenReply(SESSION_ID, messages)

      if (!last || last.id === spoken?.id) {
        return null
      }

      return { id: last.id, pending: false, text: chatMessageText(last) }
    }

    const markSpoken = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

      if (last) {
        markAssistantIdSpoken(SESSION_ID, messages, last.id)
      }
    }

    vi.mocked(playSpeechText).mockResolvedValue(false)

    renderHook(
      () =>
        useAutoSpeakReplies({
          blockedLabel: 'press read aloud',
          conversationActive: false,
          failureLabel: 'read-aloud failed',
          markSpoken,
          pendingReply,
          sessionId: SESSION_ID
        }),
      {
        wrapper: ({ children }) => (
          <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages }}>{children}</ComposerScopeProvider>
        )
      }
    )

    act(() => {
      $messages.set([assistantMessage('assistant-stream-1', 'hello there')])
    })

    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(resolveSpokenReply(SESSION_ID, $messages.get())).toBeNull())
  })

  it('speaks a later turn that says the same thing', async () => {
    $autoSpeakReplies.set(true)

    const $messages = atom<ChatMessage[]>([])

    const pendingReply = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)
      const spoken = resolveSpokenReply(SESSION_ID, messages)

      if (!last || last.id === spoken?.id) {
        return null
      }

      return { id: last.id, pending: false, text: chatMessageText(last) }
    }

    const markSpoken = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

      if (last) {
        markAssistantIdSpoken(SESSION_ID, messages, last.id)
      }
    }

    vi.mocked(playSpeechText).mockResolvedValue(true)

    renderHook(
      () =>
        useAutoSpeakReplies({
          blockedLabel: 'press read aloud',
          conversationActive: false,
          failureLabel: 'read-aloud failed',
          markSpoken,
          pendingReply,
          sessionId: SESSION_ID
        }),
      {
        wrapper: ({ children }) => (
          <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages }}>{children}</ComposerScopeProvider>
        )
      }
    )

    act(() => {
      $messages.set([
        { id: 'u1', parts: [assistantTextPart('again')], role: 'user' },
        assistantMessage('assistant-stream-1', 'Done.')
      ])
    })
    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))

    act(() => {
      $messages.set([
        { id: 'u1', parts: [assistantTextPart('again')], role: 'user' },
        assistantMessage('durable-1', 'Done.'),
        { id: 'u2', parts: [assistantTextPart('again')], role: 'user' },
        assistantMessage('assistant-stream-2', 'Done.')
      ])
    })
    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(2))
  })
})

// iOS WebKit plays audio only after a tap, and a reply lands long after the
// tap that asked for it.
describe('useAutoSpeakReplies — unlocking audio for later replies', () => {
  afterEach(() => {
    cleanup()
    $autoSpeakReplies.set(false)
    vi.clearAllMocks()
  })

  const renderAutoSpeak = () =>
    renderHook(
      () =>
        useAutoSpeakReplies({
          blockedLabel: 'press read aloud',
          conversationActive: false,
          failureLabel: 'read-aloud failed',
          markSpoken: () => undefined,
          pendingReply: () => null,
          sessionId: SESSION_ID
        }),
      {
        wrapper: ({ children }) => (
          <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages: atom<ChatMessage[]>([]) }}>
            {children}
          </ComposerScopeProvider>
        )
      }
    )

  it('unlocks speech audio on any tap or key press while it is on', () => {
    $autoSpeakReplies.set(true)
    renderAutoSpeak()

    fireEvent.touchEnd(document.body)
    fireEvent.click(document.body)
    fireEvent.keyDown(document.body, { key: 'Enter' })

    expect(unlockSpeechAudio).toHaveBeenCalledTimes(3)
  })

  it('keeps unlocking audio for the composers that stay open', () => {
    $autoSpeakReplies.set(true)
    const first = renderAutoSpeak()
    renderAutoSpeak()

    first.unmount()
    fireEvent.click(document.body)

    expect(unlockSpeechAudio).toHaveBeenCalled()
  })

  it('leaves audio alone while it is off', () => {
    renderAutoSpeak()

    fireEvent.click(document.body)

    expect(unlockSpeechAudio).not.toHaveBeenCalled()
  })

  it('keeps a reply the browser would not play marked spoken and points at Read aloud', async () => {
    $autoSpeakReplies.set(true)

    const $messages = atom<ChatMessage[]>([])

    const pendingReply = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

      if (!last || last.id === resolveSpokenReply(SESSION_ID, messages)?.id) {
        return null
      }

      return { id: last.id, pending: false, text: chatMessageText(last) }
    }

    const markSpoken = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

      if (last) {
        markAssistantIdSpoken(SESSION_ID, messages, last.id)
      }
    }

    vi.mocked(playSpeechText).mockRejectedValue(new SpeechAudioBlockedError())

    renderHook(
      () =>
        useAutoSpeakReplies({
          blockedLabel: 'press read aloud',
          conversationActive: false,
          failureLabel: 'read-aloud failed',
          markSpoken,
          pendingReply,
          sessionId: SESSION_ID
        }),
      {
        wrapper: ({ children }) => (
          <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages }}>{children}</ComposerScopeProvider>
        )
      }
    )

    act(() => {
      $messages.set([assistantMessage('assistant-stream-1', 'hello there')])
    })

    await waitFor(() => expect(notify).toHaveBeenCalledWith(expect.objectContaining({ message: 'press read aloud' })))
    expect(notifyError).not.toHaveBeenCalled()
    // Released, it would be read out late, when the next message is sent.
    expect(resolveSpokenReply(SESSION_ID, $messages.get())?.id).toBe('assistant-stream-1')
    clearSpokenRepliesForTests()
  })
})
