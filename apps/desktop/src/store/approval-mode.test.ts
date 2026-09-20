import { beforeEach, describe, expect, it, vi } from 'vitest'

import { deferred } from '../test/deferred'

import {
  $approvalModes,
  approvalModeForProfile,
  reconcileApprovalModeForProfile,
  setApprovalModeForProfile,
  syncApprovalModeForProfile
} from './approval-mode'

describe('profile-scoped approval mode cache', () => {
  beforeEach(() => $approvalModes.set({}))

  it('keeps unread policy unknown and reads the explicit owner', async () => {
    expect(approvalModeForProfile('default')).toBeUndefined()

    const request = vi.fn(async () => ({ value: 'manual' }))
    await syncApprovalModeForProfile(request, 'default')

    expect(request).toHaveBeenCalledWith('config.get', { key: 'approvals.mode', profile: 'default' })
    expect(approvalModeForProfile('default')).toBe('manual')
  })

  it('keeps profile values isolated', async () => {
    await syncApprovalModeForProfile(
      vi.fn(async () => ({ value: 'manual' })),
      'work'
    )
    await syncApprovalModeForProfile(
      vi.fn(async () => ({ value: 'off' })),
      'personal'
    )

    expect(approvalModeForProfile('work')).toBe('manual')
    expect(approvalModeForProfile('personal')).toBe('off')
    expect(approvalModeForProfile('default')).toBeUndefined()
  })

  it('does not present stale or malformed reads as authoritative policy', async () => {
    reconcileApprovalModeForProfile('read-failure', 'off')
    await expect(
      syncApprovalModeForProfile(vi.fn().mockRejectedValue(new Error('unavailable')), 'read-failure')
    ).rejects.toThrow('unavailable')
    expect(approvalModeForProfile('read-failure')).toBeUndefined()
    await expect(syncApprovalModeForProfile(vi.fn().mockResolvedValue({}), 'read-failure')).rejects.toThrow()
    expect(approvalModeForProfile('read-failure')).toBeUndefined()
  })

  it('rolls consecutive failed writes back to the last authoritative value', async () => {
    await syncApprovalModeForProfile(
      vi.fn(async () => ({ value: 'smart' })),
      'default'
    )
    const first = deferred<{ value: string }>()
    const second = deferred<{ value: string }>()

    const request = vi
      .fn()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise)

    const staleWrite = setApprovalModeForProfile(request, 'default', 'manual')
    const currentWrite = setApprovalModeForProfile(request, 'default', 'off')
    expect(approvalModeForProfile('default')).toBe('off')

    first.reject(new Error('old failure'))
    await expect(staleWrite).rejects.toThrow('old failure')
    expect(approvalModeForProfile('default')).toBe('off')

    second.reject(new Error('current failure'))
    await expect(currentWrite).rejects.toThrow('current failure')
    expect(approvalModeForProfile('default')).toBe('smart')
  })

  it('lets a backend event supersede an optimistic write and its later failure', async () => {
    const write = deferred<{ value: string }>()

    const pending = setApprovalModeForProfile(
      vi.fn(() => write.promise),
      'work',
      'off'
    )

    reconcileApprovalModeForProfile('work', 'smart')
    expect(approvalModeForProfile('work')).toBe('smart')

    write.reject(new Error('late failure'))
    await expect(pending).rejects.toThrow('late failure')
    expect(approvalModeForProfile('work')).toBe('smart')
  })

  it('ignores a stale initial read after a newer write succeeds', async () => {
    const read = deferred<{ value: string }>()

    const request = vi
      .fn()
      .mockImplementationOnce(() => read.promise)
      .mockResolvedValueOnce({ value: 'off' })

    const staleRead = syncApprovalModeForProfile(request, 'default')
    await setApprovalModeForProfile(request, 'default', 'off')
    read.resolve({ value: 'manual' })
    await staleRead

    expect(approvalModeForProfile('default')).toBe('off')
  })
})
