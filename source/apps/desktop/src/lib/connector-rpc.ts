/** Stock session-only connector RPCs reject the owner envelope before doing any work. */
export function isSessionOnlyConnectorBackend(error: unknown): boolean {
  const code = error && typeof error === 'object' ? (error as { code?: unknown }).code : undefined
  const message = error instanceof Error ? error.message : String(error)

  return (code === undefined || code === 4000) && message === 'session_id required'
}

/** Retry only the stock pre-operation validation failure, never a failed or timed-out mutation. */
export async function requestSessionConnector<T>(
  request: (params: Record<string, unknown>) => Promise<T>,
  sessionId: string,
  params: Record<string, unknown> = {}
): Promise<T> {
  try {
    return await request({ ...params, owner: { session_id: sessionId, type: 'session' } })
  } catch (error) {
    if (!isSessionOnlyConnectorBackend(error)) {
      throw error
    }

    // Stock list/connect also reject unknown fields, so the two wire shapes cannot be combined.
    return request({ ...params, session_id: sessionId })
  }
}
