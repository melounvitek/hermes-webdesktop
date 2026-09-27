import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ConfirmDialog } from './confirm-dialog'

afterEach(cleanup)

describe('ConfirmDialog', () => {
  function renderWithSecondary() {
    const onConfirm = vi.fn()
    const onClose = vi.fn()
    const onSecondary = vi.fn()

    render(
      <ConfirmDialog
        onClose={onClose}
        onConfirm={onConfirm}
        open
        secondaryAction={{ label: 'Remove from sidebar', onClick: onSecondary }}
        title="Remove worktree?"
      />
    )

    return { onClose, onConfirm, onSecondary }
  }

  it('runs the secondary action and closes without confirming', async () => {
    const { onClose, onConfirm, onSecondary } = renderWithSecondary()

    fireEvent.click(await screen.findByRole('button', { name: 'Remove from sidebar' }))

    expect(onSecondary).toHaveBeenCalledTimes(1)
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it.each(['Enter', ' '])('leaves %j to native activation on the focused Confirm button', async key => {
    const { onConfirm, onSecondary } = renderWithSecondary()

    const confirm = await screen.findByRole('button', { name: /confirm/i })

    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(document.activeElement).toBe(confirm))
    expect(fireEvent.keyDown(confirm, { key })).toBe(true)
    expect(onConfirm).not.toHaveBeenCalled()
    // jsdom does not synthesize a native button click from keyDown.
    fireEvent.click(confirm)

    await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(1))
    expect(onSecondary).not.toHaveBeenCalled()
  })

  it.each([
    ['Cancel', 'Enter'],
    ['Cancel', ' '],
    ['Remove from sidebar', 'Enter'],
    ['Remove from sidebar', ' ']
  ])('leaves %j keyboard activation with %j to the focused button', async (name, key) => {
    const { onClose, onConfirm, onSecondary } = renderWithSecondary()
    const button = await screen.findByRole('button', { name })
    button.focus()

    const unhandled = fireEvent.keyDown(button, { key })
    expect(onConfirm).not.toHaveBeenCalled()
    expect(unhandled).toBe(true)
    // jsdom does not synthesize a native button click from keyDown.
    expect(onClose).not.toHaveBeenCalled()
    expect(onSecondary).not.toHaveBeenCalled()

    fireEvent.click(button)
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onSecondary).toHaveBeenCalledTimes(name === 'Cancel' ? 0 : 1)
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('blocks duplicate dismiss-on-confirm submissions and allows retry after failure', async () => {
    let reject!: (error: Error) => void
    let resolve!: () => void

    const failedAttempt = new Promise<void>((_, rejectPromise) => {
      reject = rejectPromise
    })

    const retry = new Promise<void>(resolvePromise => {
      resolve = resolvePromise
    })

    const onConfirm = vi.fn().mockReturnValueOnce(failedAttempt).mockReturnValueOnce(retry)
    const onClose = vi.fn()

    render(
      <ConfirmDialog
        busyLabel="Updating"
        dismissOnConfirm
        onClose={onClose}
        onConfirm={onConfirm}
        open
        title="Update?"
      />
    )
    const confirm = await screen.findByRole('button', { name: /confirm/i })
    expect((confirm as HTMLButtonElement).disabled).toBe(false)

    act(() => {
      fireEvent.click(confirm)
      fireEvent.click(confirm)
    })
    expect(onConfirm).toHaveBeenCalledTimes(1)
    expect((screen.getByRole('button', { name: 'Updating' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(confirm)
    expect(onConfirm).toHaveBeenCalledTimes(1)
    expect(onClose).not.toHaveBeenCalled()

    await act(async () => reject(new Error('Update failed')))
    expect(screen.getByText('Update failed')).toBeTruthy()
    expect((screen.getByRole('button', { name: /confirm/i }) as HTMLButtonElement).disabled).toBe(false)
    expect(onClose).not.toHaveBeenCalled()

    fireEvent.click(confirm)
    expect(onConfirm).toHaveBeenCalledTimes(2)
    expect(screen.queryByText('Update failed')).toBeNull()
    expect((confirm as HTMLButtonElement).disabled).toBe(true)

    await act(async () => resolve())
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
