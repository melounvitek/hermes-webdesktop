import {
  extendRefreshPageToOverlap,
  graftRefreshedTailOntoBackfill,
  olderPageReader
} from '@/app/chat/transcript-backfill'
import { preserveLocalPendingTurnMessages } from '@/app/session/hooks/use-session-actions/utils'
import type { ClientSessionState } from '@/app/types'
import { getLatestSessionMessages, type ProfileScope } from '@/hermes'
import {
  type ChatMessage,
  preserveLocalAssistantErrors,
  preserveLocalSystemNotices,
  toChatMessages
} from '@/lib/chat-messages'
import { latestSessionTodos, latestSessionTodoSnapshot } from '@/lib/todos'
import { pendingSessionReplay } from '@/store/gateway'
import { $sessionStates } from '@/store/session-states'
import {
  $todosBySession,
  clearActiveSessionTodos,
  clearSessionTodos,
  restoreSessionTodosFromSnapshot,
  setSessionTodos,
  todosForHydration
} from '@/store/todos'

/** Backfill/retention may prepend or release a prefix without changing the tail. */
export function transcriptChangedDuringRead(
  before: ChatMessage[] | undefined,
  after: ChatMessage[] | undefined
): boolean {
  if (before === after) {
    return false
  }

  if (!before?.length || !after?.length) {
    return true
  }

  // Require the entire shorter transcript to be an unchanged suffix. Matching
  // only the last row would miss edits/tool updates earlier in the current turn.
  const overlap = Math.min(before.length, after.length)

  for (let offset = 1; offset <= overlap; offset += 1) {
    if (before[before.length - offset] !== after[after.length - offset]) {
      return true
    }
  }

  return false
}

interface HydrationOptions {
  storedSessionId: string
  runtimeSessionId: string
  storedProfile: ProfileScope
  attempts: number
  updateSessionState: (
    runtimeId: string,
    update: (state: ClientSessionState) => ClientSessionState,
    storedId?: string
  ) => ClientSessionState
}

export async function hydrateStoredSessionTranscript({
  storedSessionId,
  runtimeSessionId,
  storedProfile,
  attempts,
  updateSessionState
}: HydrationOptions) {
  const snapshot = $sessionStates.get()[runtimeSessionId]
  const todosAtRequest = $todosBySession.get()[runtimeSessionId]
  const replay = pendingSessionReplay(runtimeSessionId)

  if (replay && !(await replay)) {
    return
  }

  const ownsSnapshot = (state: ClientSessionState | undefined) =>
    Boolean(
      snapshot &&
      state &&
      !state.busy &&
      !state.awaitingResponse &&
      !state.needsInput &&
      !state.turnLive &&
      state.storedSessionId === storedSessionId &&
      !transcriptChangedDuringRead(snapshot.messages, state.messages) &&
      $todosBySession.get()[runtimeSessionId] === todosAtRequest
    )

  // A late old completion may start hydration AFTER optimistic submission.
  // Capturing that new message array alone would incorrectly bless this read.
  for (let index = 0; index < Math.max(1, attempts); index += 1) {
    if (index > 0) {
      await new Promise(resolve => window.setTimeout(resolve, 250))
    }

    if (!ownsSnapshot($sessionStates.get()[runtimeSessionId])) {
      return
    }

    try {
      const latest = await getLatestSessionMessages(storedSessionId, storedProfile)
      const replayAtReturn = pendingSessionReplay(runtimeSessionId)

      if (replayAtReturn && !(await replayAtReturn)) {
        return
      }

      const current = $sessionStates.get()[runtimeSessionId]

      if (!ownsSnapshot(current)) {
        return
      }

      // A respawning backend can return a transient empty page. Stop is the
      // exception: ownsSnapshot already requires its backend-confirmed idle.
      if (latest.messages.length === 0 && current?.messages.length && !current.interrupted) {
        continue
      }

      const messages = await extendRefreshPageToOverlap(
        toChatMessages(latest.messages),
        current?.messages ?? [],
        olderPageReader(storedSessionId, storedProfile, latest)
      )

      if (!ownsSnapshot($sessionStates.get()[runtimeSessionId])) {
        return
      }

      let applied = false
      updateSessionState(
        runtimeSessionId,
        state => {
          if (!ownsSnapshot(state)) {
            return state
          }

          applied = true

          return {
            ...state,
            // Keep backfill, un-acked optimistic input, local errors and trailing
            // client-local system notices, not arbitrary live messages from an
            // obsolete server transcript.
            messages: preserveLocalSystemNotices(
              preserveLocalAssistantErrors(
                preserveLocalPendingTurnMessages(
                  graftRefreshedTailOntoBackfill(messages, state.messages),
                  state.messages
                ),
                state.messages
              ),
              state.messages
            )
          }
        },
        storedSessionId
      )

      if (!applied) {
        return
      }

      const todoSnapshot = latestSessionTodoSnapshot(messages)

      if (todoSnapshot) {
        // Deferred Desktop resume sends no todo_state on its initial ACK.
        // The persisted tool result is the first authoritative snapshot.
        restoreSessionTodosFromSnapshot(runtimeSessionId, todoSnapshot, false)
      }

      const latestTodos = latestSessionTodos(messages)
      const restored = todosForHydration(latestTodos)

      if (latestTodos?.length === 0) {
        // An explicit empty result retires the list; missing paged history does not.
        // A valid older snapshot must not mask a newer legacy clear without a revision.
        if (!todoSnapshot || todoSnapshot.todos.length > 0) {
          clearSessionTodos(runtimeSessionId)
        }
      } else if (restored) {
        setSessionTodos(runtimeSessionId, restored)
      } else {
        clearActiveSessionTodos(runtimeSessionId)
      }

      return
    } catch {
      // Best-effort fallback when live stream payloads are empty.
    }
  }
}
