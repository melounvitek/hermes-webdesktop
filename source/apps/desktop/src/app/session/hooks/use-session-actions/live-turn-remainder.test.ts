import { expect, it } from 'vitest'

import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage, SessionResumeResult } from '@/types/hermes'

import { mergeLiveAssistantRun } from './live-turn-remainder'
import { reconcilePersistedLiveTurn } from './persisted-live-turn'

const assistant = (id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role: 'assistant',
  parts: [{ type: 'text', text }],
  ...extra
})

it('extends one response across arbitrary chunk cuts without changing its tool or text occurrences', () => {
  const tool = {
    type: 'tool-call' as const,
    toolCallId: 'local-tool',
    toolName: 'read_file',
    args: {},
    argsText: '{}',
    result: 'fixture'
  }

  const remote = [assistant('snapshot', 'Checking.\n\nThe result.', { pending: true })]

  for (let cut = 1; cut < 'The result.'.length; cut++) {
    let local = [
      assistant('live', '', {
        parts: [
          { type: 'text', text: 'Checking.' },
          tool,
          { type: 'text', text: `\n\n${'The result.'.slice(0, cut)}` }
        ],
        pending: true
      })
    ]

    for (let resume = 0; resume < 3; resume++) {
      local = mergeLiveAssistantRun(remote, local)
      expect(local.map(chatMessageText)).toEqual(['Checking.\n\nThe result.'])
      expect(local.flatMap(row => row.parts.filter(part => part.type === 'tool-call'))).toEqual([tool])
      expect(local.map(row => row.id)).toEqual(['live'])
    }
  }

  const sealed = assistant('sealed', '', {
    parts: [{ type: 'text', text: 'The res', completedAt: 10 }]
  })

  const terminal = [assistant('snapshot', 'The result.', { error: 'Connection reset' })]
  const settled = mergeLiveAssistantRun(terminal, [sealed])
  expect(settled.map(chatMessageText)).toEqual(['The result.'])
  expect(mergeLiveAssistantRun(terminal, settled)).toEqual(settled)
  expect(settled[0].parts[0].completedAt).toBe(10)
})

it('settles a matching partial error without discarding richer local parts or distinct failures', () => {
  const tool = { type: 'tool-call' as const, toolCallId: 'local-tool', toolName: 'read_file', args: {}, argsText: '{}' }
  const surface = { layer: 'streaming' as const, code: 'stream_drop', retryable: true }

  const failed = assistant('snapshot', 'The result.', {
    error: 'Connection reset',
    errorSurface: surface,
    pending: false
  })

  const local = assistant('live', '', {
    parts: [tool, { type: 'text', text: 'The result. More local detail.' }],
    pending: true
  })

  let rows = mergeLiveAssistantRun([failed], [local])

  for (let resume = 0; resume < 3; resume++) {
    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({
      id: local.id,
      parts: local.parts,
      error: failed.error,
      errorSurface: surface,
      pending: false
    })
    rows = mergeLiveAssistantRun([failed], rows)
  }

  const earlier = assistant('earlier-failure', 'A different partial reply', { error: 'Read failed', pending: false })
  rows = mergeLiveAssistantRun([failed], [earlier])
  expect(rows).toEqual([earlier, failed])
  expect(mergeLiveAssistantRun([failed], rows)).toEqual(rows)

  const equalTextDifferentFailure = assistant('another-failure', chatMessageText(failed), {
    error: 'Permission denied'
  })

  expect(mergeLiveAssistantRun([failed], [equalTextDifferentFailure])).toEqual([equalTextDifferentFailure, failed])
  expect(mergeLiveAssistantRun([assistant('distinct', 'Unrelated reply')], [local])).toEqual([
    local,
    assistant('distinct', 'Unrelated reply')
  ])
})

it('pairs only the queue projection, preserving equal corrections and different local queued occurrences', () => {
  const prompt = 'Inspect this file'
  const correction = 'Also inspect its tests'
  const rows: SessionMessage[] = [{ id: 1, role: 'user', content: prompt }]

  const projection: Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id'> = {
    session_id: 'runtime',
    inflight: { user: prompt, assistant: '', corrections: [correction], correction_offsets: [0], streaming: true },
    queued: { user: correction }
  }

  const reconcile = (previous: ChatMessage[]) =>
    reconcilePersistedLiveTurn(toChatMessages(rows), previous, rows, projection)!

  let current = reconcile([])

  const unrelatedUser: ChatMessage = { id: 'local-user', role: 'user', parts: [{ type: 'text', text: correction }] }

  const oldQueue: ChatMessage = {
    id: 'user-queued-older-runtime',
    role: 'user',
    parts: [{ type: 'text', text: 'A different queued request' }]
  }

  const response = assistant('local-response', 'An unrepresented later reply')
  current.push(unrelatedUser, oldQueue, response)

  for (let resume = 0; resume < 3; resume++) {
    current = reconcile(current)
    expect(current.filter(row => row.role === 'user').map(chatMessageText)).toEqual([
      prompt,
      correction,
      correction,
      correction,
      chatMessageText(oldQueue)
    ])
    expect(current.filter(row => row.id === 'user-queued-runtime')).toHaveLength(1)
    expect(current).toContainEqual(unrelatedUser)
    expect(current).toContainEqual(oldQueue)
    expect(current).toContainEqual(response)
    expect(new Set(current.map(row => row.id)).size).toBe(current.length)
  }
})

it('keeps the live view of a running turn that already shows every stored tool call', () => {
  const prompt = 'Run the checks'
  const call = (id: string) => ({ id, type: 'function', function: { name: 'terminal', arguments: '{}' } })
  const tool = (toolCallId: string, done: boolean) => ({
    type: 'tool-call' as const,
    toolCallId,
    toolName: 'terminal',
    args: {},
    argsText: '{}',
    ...(done ? { result: 'done', completedAt: 3 } : {})
  })

  // Stored before each tool runs: the running tool has no result row yet.
  const rows: SessionMessage[] = [
    { id: 11, role: 'user', content: prompt },
    { id: 12, role: 'assistant', content: 'Step 1.', tool_calls: [call('call-1')] },
    { id: 13, role: 'tool', content: 'done', tool_call_id: 'call-1' },
    { id: 14, role: 'assistant', content: 'Step 2.', tool_calls: [call('call-2')] }
  ]

  const live: ChatMessage[] = [
    { id: 'local-user', role: 'user', rowId: 11, parts: [{ type: 'text', text: prompt }] },
    assistant('live-1', '', {
      parts: [
        { type: 'reasoning', text: 'First, look.' },
        { type: 'text', text: 'Step 1. ' }
      ]
    }),
    assistant('live-2', '', {
      parts: [tool('call-1', true), { type: 'reasoning', text: 'Then run.' }, { type: 'text', text: '\n\nStep 2.' }]
    }),
    assistant('live-3', '', { parts: [tool('call-2', false)], pending: true })
  ]

  const projection: Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id'> = {
    session_id: 'runtime',
    inflight: { user: prompt, assistant: 'Step 1. \n\nStep 2.', streaming: true }
  }

  const current = reconcilePersistedLiveTurn(toChatMessages(rows), live, rows, projection)!
  expect(current.slice(1)).toEqual(live.slice(1))

  // A live view that missed events loses to the stored rows: here the running
  // tool, then the first tool's completion.
  const behind = reconcilePersistedLiveTurn(toChatMessages(rows), live.slice(0, 3), rows, projection)!
  expect(behind.flatMap(row => row.parts).some(part => part.type === 'tool-call' && part.toolCallId === 'call-2')).toBe(
    true
  )

  const missedCompletion = live.map(row =>
    row.id === 'live-2' ? { ...row, parts: [tool('call-1', false), ...row.parts.slice(1)] } : row
  )

  const repaired = reconcilePersistedLiveTurn(toChatMessages(rows), missedCompletion, rows, projection)!
  expect(
    repaired
      .flatMap(row => row.parts)
      .some(part => part.type === 'tool-call' && part.toolCallId === 'call-1' && part.result !== undefined)
  ).toBe(true)
})
