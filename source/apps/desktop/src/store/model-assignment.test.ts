import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestProfile } from '@/api/client'
import type * as HermesApi from '@/hermes'
import { $cronReviewRequest } from '@/store/cron'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { deferred } from '@/test/deferred'

const { assign } = vi.hoisted(() => ({ assign: vi.fn() }))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  setModelAssignment: assign
}))

import { setMainModelAssignment } from './model-assignment'

const impact = {
  available: true,
  affected_count: 1,
  truncated: false,
  jobs: [{ id: 'job', name: 'Morning summary', drifted_axes: ['model'] }]
}

const response = { ok: true, cron_model_impact: impact }
const request = { provider: 'nous', model: 'new-model' }

function profile(name: string) {
  setApiRequestProfile(name)
  $activeGatewayProfile.set(name)
}

beforeEach(() => {
  assign.mockReset()
  profile('default')
  clearNotifications()
})
afterEach(() => {
  profile('default')
  $connection.set(null)
  clearNotifications()
})

it('warns only for reported cron impact, preserves confirmation, and clears explicit zero impact', async () => {
  assign
    .mockResolvedValueOnce({ ok: false, confirm_required: true, confirm_message: 'Please confirm' })
    .mockResolvedValueOnce(response)
  const pending = setMainModelAssignment(request)
  await vi.waitFor(() => expect($notifications.get().some(n => n.id.startsWith('model-warning-confirm-'))).toBe(true))
  $notifications
    .get()
    .find(n => n.id.startsWith('model-warning-confirm-'))!
    .action!.onClick()
  await pending
  expect(assign).toHaveBeenLastCalledWith({ ...request, scope: 'main', confirm_expensive_model: true })
  const warning = $notifications.get().find(n => n.id === 'cron-model-impact')
  expect(warning?.message).toContain('1 unpinned scheduled job')
  expect(warning?.detail).toContain(impact.jobs[0].name)
  const reviews = $cronReviewRequest.get()
  warning!.action!.onClick()
  expect($cronReviewRequest.get()).toBe(reviews + 1)

  for (const result of [{ ok: true }, { ok: true, cron_model_impact: { ...impact, affected_count: 2 } }]) {
    assign.mockResolvedValueOnce(result)
    await setMainModelAssignment(request)
    expect($notifications.get().find(n => n.id === warning!.id)).toBe(warning)
  }

  assign.mockResolvedValueOnce({ ok: true, cron_model_impact: { ...impact, affected_count: 0, jobs: [] } })
  await setMainModelAssignment(request)
  expect($notifications.get()).toEqual([])
})

it('keeps warnings and review actions bound to the assigning profile and connection, including reversed replies', async () => {
  assign.mockResolvedValue(response)
  await setMainModelAssignment(request, { connectionId: 'other', profile: 'other' })
  expect($notifications.get()).toEqual([])

  const first = deferred<typeof response>()
  assign.mockReturnValueOnce(first.promise)
  const stale = setMainModelAssignment(request)
  await setMainModelAssignment(request)
  const warning = $notifications.get().find(n => n.id === 'cron-model-impact')!
  expect(warning).toBeDefined()
  first.resolve({ ...response, cron_model_impact: { ...impact, affected_count: 0, jobs: [] } })
  await stale
  expect($notifications.get().find(n => n.id === warning.id)).toBe(warning)

  const next = deferred<typeof response>()
  assign.mockReturnValueOnce(next.promise)
  const pending = setMainModelAssignment(request)
  profile('other')
  profile('default')
  next.resolve(response)
  await pending
  expect($notifications.get()).toEqual([])
  const reviews = $cronReviewRequest.get()
  warning.action!.onClick()
  expect($cronReviewRequest.get()).toBe(reviews)

  await setMainModelAssignment(request)
  const connectionWarning = $notifications.get().find(n => n.id === 'cron-model-impact')!
  $connection.set({ mode: 'remote', baseUrl: 'https://other.example', connectionId: 'other' } as never)
  expect($notifications.get()).toEqual([])
  connectionWarning.action!.onClick()
  expect($cronReviewRequest.get()).toBe(reviews)
})
