import type { ClientSessionState } from '@/app/types'
import { translateNow } from '@/i18n'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { requestForSessionProfile } from '@/store/session-request-router'
import { $sessionStates, knownOwnerForSession } from '@/store/session-states'

import { resolveSessionOwner } from '../use-session-actions/utils'

import { finalizeInterruptedMessages } from './rewind'
import {
  type GatewayRequest,
  isSessionNotFoundError,
  markSessionRecentlyInterrupted,
  withSessionNotFoundResume
} from './utils'

interface StopRecoveryOptions {
  sessionId: string
  storedSessionId: string | null
  requestGateway: GatewayRequest
  updateSessionState: (
    id: string,
    change: (state: ClientSessionState) => ClientSessionState,
    storedId?: string | null,
    rebindFrom?: string
  ) => unknown
  onRecovered: (id: string, previousId: string) => void
}

// Stop's identity transition is shared by the main composer and tiles. An ACK
// is not idle: only a missing-runtime error retires an obsolete claim here.
export async function interruptStoppedSession({
  sessionId,
  storedSessionId,
  requestGateway,
  updateSessionState,
  onRecovered
}: StopRecoveryOptions): Promise<void> {
  const stopped = $sessionStates.get()[sessionId]

  const owner =
    knownOwnerForSession(sessionId) ??
    knownOwnerForSession(storedSessionId) ??
    (storedSessionId ? await resolveSessionOwner(storedSessionId) : undefined)

  assertSessionOwnerResolved(owner, { method: 'session.interrupt', sessionId })

  const request: GatewayRequest = (method, params, timeout) =>
    requestForSessionProfile(owner, requestGateway, method, params, timeout)

  const missing = new Set<string>()
  let bindingFrom = sessionId

  const retireMissing = () => {
    for (const id of missing) {
      updateSessionState(id, state =>
        state.busy && state.interrupted && !state.awaitingResponse && state.storedSessionId === storedSessionId
          ? { ...state, busy: false, turnLive: false, turnStartedAt: null }
          : state
      )
    }
  }

  try {
    const result = await withSessionNotFoundResume(
      sessionId,
      storedSessionId,
      async liveId => {
        try {
          return await request('session.interrupt', { session_id: liveId })
        } catch (error) {
          if (isSessionNotFoundError(error)) {
            missing.add(liveId)
          }

          throw error
        }
      },
      {
        requestGateway: request,
        resolveProfile: async () => (typeof owner === 'string' ? owner : owner?.profile),
        onRecovered: recoveredId => {
          const existing = $sessionStates.get()[recoveredId]

          if (existing?.storedSessionId && existing.storedSessionId !== storedSessionId) {
            throw new Error(translateNow('desktop.stopUnconfirmed'))
          }

          // Seed the blocked replacement before publishing its binding or retiring
          // A, so neither the queue nor an old Stop waiter can expose an idle gap.
          markSessionRecentlyInterrupted(recoveredId)
          updateSessionState(
            recoveredId,
            state => ({
              ...state,
              storedSessionId,
              messages: finalizeInterruptedMessages(
                state.messages.length ? state.messages : (stopped?.messages ?? []),
                state.streamId
              ),
              busy: true,
              interrupted: true,
              awaitingResponse: false,
              streamId: null,
              pendingBranchGroup: null,
              needsInput: false,
              turnLive: false,
              turnStartedAt: null
            }),
            storedSessionId,
            bindingFrom
          )
          missing.delete(recoveredId)
          onRecovered(recoveredId, bindingFrom)
          bindingFrom = recoveredId
          retireMissing()
        }
      }
    )

    if (!result.recovered) {
      return
    }

    // Resuming an already idle session need not emit another idle transition.
    // This fresh read is pinned to the same owner as both interrupt attempts.
    const snapshot = $sessionStates.get()[result.sessionId]

    if (!snapshot?.busy || !snapshot.interrupted) {
      return
    }

    const status = await request<{ sessions?: Array<{ id: string; status?: string }> }>('session.active_list', {})

    if (!Array.isArray(status.sessions)) {
      throw new Error(translateNow('desktop.stopUnconfirmed'))
    }

    const live = status.sessions.find(session => session.id === result.sessionId)

    if (!live || live.status === 'idle') {
      updateSessionState(result.sessionId, state =>
        state === snapshot
          ? { ...state, busy: false, awaitingResponse: false, turnLive: false, turnStartedAt: null }
          : state
      )
    }
  } finally {
    // Resume/retry failure does not undo the owning backend's proof that A is
    // absent. A failed retry on a live B, however, leaves B blocked and retryable.
    retireMissing()
  }
}
