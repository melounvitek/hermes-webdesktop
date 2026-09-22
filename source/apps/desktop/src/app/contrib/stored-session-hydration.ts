import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import type { ClientSessionState } from '@/app/types'
import { getLatestSessionMessages, type ProfileScope } from '@/hermes'
import { preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { latestSessionTodos } from '@/lib/todos'
import { $sessionStates } from '@/store/session-states'
import { $todosBySession, clearSessionTodos, setSessionTodos, todosForHydration } from '@/store/todos'

interface HydrationOptions {
  storedSessionId: string
  runtimeSessionId: string
  profile: ProfileScope
  attempts: number
  updateSessionState: (
    runtimeId: string,
    update: (state: ClientSessionState) => ClientSessionState,
    storedId?: string
  ) => ClientSessionState
}

export async function hydrateStoredSession({
  storedSessionId,
  runtimeSessionId,
  profile,
  attempts,
  updateSessionState
}: HydrationOptions) {
  const snapshot = $sessionStates.get()[runtimeSessionId]
  const todosAtRequest = $todosBySession.get()[runtimeSessionId]

  const ownsSnapshot = (state: ClientSessionState | undefined) =>
    Boolean(
      snapshot &&
      state &&
      !state.busy &&
      !state.awaitingResponse &&
      state.storedSessionId === storedSessionId &&
      state.messages === snapshot.messages &&
      $todosBySession.get()[runtimeSessionId] === todosAtRequest
    )

  // A late old completion may start hydration AFTER optimistic submission.
  // Capturing that new message array alone would incorrectly bless this read.
  for (let index = 0; index < Math.max(1, attempts); index += 1) {
    if (!ownsSnapshot($sessionStates.get()[runtimeSessionId])) {return}

    try {
      const latest = await getLatestSessionMessages(storedSessionId, profile)
      const messages = toChatMessages(latest.messages)

      if (!ownsSnapshot($sessionStates.get()[runtimeSessionId])) {return}
      let applied = false
      updateSessionState(
        runtimeSessionId,
        state => {
          if (!ownsSnapshot(state)) {return state}
          applied = true

          return {
            ...state,
            // Keep only legitimate older backfill and local errors, not a union
            // of arbitrary live messages with an obsolete server transcript.
            messages: preserveLocalAssistantErrors(
              graftRefreshedTailOntoBackfill(messages, state.messages),
              state.messages
            )
          }
        },
        storedSessionId
      )

      if (!applied) {return}
      const restored = todosForHydration(latestSessionTodos(messages))

      if (restored) {setSessionTodos(runtimeSessionId, restored)}
      else {clearSessionTodos(runtimeSessionId)}

      return
    } catch {
      // Best-effort fallback when live stream payloads are empty.
    }

    if (index < attempts - 1) {await new Promise(resolve => window.setTimeout(resolve, 250))}
  }
}
