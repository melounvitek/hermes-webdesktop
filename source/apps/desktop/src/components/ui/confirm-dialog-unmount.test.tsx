import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { ConfirmDialog } from '@/components/ui/confirm-dialog'

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel', confirm: 'Confirm', delete: 'Delete', done: 'Done', loading: 'Working' },
      errors: { genericFailure: 'Something failed' }
    }
  })
}))

// ConfirmDialog schedules window.setTimeout(onClose, 600) after a successful
// confirm. The timer had no cleanup, so an unmount inside that window left it
// pending. In CI it came due after the environment was gone. The setState
// path of React then touched `window`:
//
//   ReferenceError: window is not defined
//    at resolveUpdatePriority (react-dom-client.development.js:1308)
//    at dispatchSetState
//    at Timeout.t4 [as _onTimeout] session-actions-menu.tsx:574
//
// The frame at session-actions-menu.tsx:574 is the `onClose` prop of
// DeleteSessionDialog. The owner of the timer is this component.
//
// This test confirms, unmounts inside the 600ms window, and then lets the
// timer come due on the dead tree.
test('the close timer does not fire after unmount', async () => {
  vi.useFakeTimers()
  const onClose = vi.fn()
  const onConfirm = vi.fn()

  render(<ConfirmDialog confirmLabel="Delete" onClose={onClose} onConfirm={onConfirm} open title="Delete session" />)

  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Delete' })))
  expect(onConfirm).toHaveBeenCalled()

  // Unmount while the close timer is still pending.
  cleanup()

  // Let the timer come due on the unmounted tree.
  vi.advanceTimersByTime(1000)

  expect(onClose).not.toHaveBeenCalled()
})

test.each([false, true])('ignores async completion after unmount (dismissOnConfirm=%s)', async dismissOnConfirm => {
  vi.useFakeTimers()
  let resolve!: () => void

  const pending = new Promise<void>(done => {
    resolve = done
  })

  const onConfirm = vi.fn(() => pending)
  const onClose = vi.fn()

  const { unmount } = render(
    <ConfirmDialog dismissOnConfirm={dismissOnConfirm} onClose={onClose} onConfirm={onConfirm} open title="Update?" />
  )

  fireEvent.click(screen.getByRole('button', { name: 'Confirm' }))
  expect(onConfirm).toHaveBeenCalledTimes(1)
  unmount()
  vi.runAllTimers()
  expect(vi.getTimerCount()).toBe(0)

  await act(async () => resolve())
  expect(vi.getTimerCount()).toBe(0)
  expect(onClose).not.toHaveBeenCalled()
})
