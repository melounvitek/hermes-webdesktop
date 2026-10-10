import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { startDragSession } from './drag-session'

afterEach(cleanup)

const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve))

function renderHandle() {
  const spec = { onCommit: vi.fn(), onEngage: vi.fn(), onTap: vi.fn(), resolveMove: vi.fn(() => null) }
  const otherAction = vi.fn()

  const { getByText } = render(
    <>
      <button onPointerDown={event => startDragSession(event, spec)}>Handle</button>
      <button onClick={otherAction}>Other</button>
    </>
  )

  return { handle: getByText('Handle'), other: getByText('Other'), otherAction, spec }
}

describe('startDragSession on touch', () => {
  it.each([
    ['mouse', true],
    ['touch', false]
  ])('a %s press that wobbles 8px engages a drag: %s', (pointerType, engages) => {
    const { handle, spec } = renderHandle()

    fireEvent.pointerDown(handle, { button: 0, clientX: 10, clientY: 10, pointerType })
    fireEvent.pointerMove(handle, { clientX: 18, clientY: 10, pointerType })
    fireEvent.pointerUp(handle, { clientX: 18, clientY: 10, pointerType })

    expect(spec.onEngage).toHaveBeenCalledTimes(engages ? 1 : 0)
    expect(spec.onTap).toHaveBeenCalledTimes(engages ? 0 : 1)
  })

  it('does not swallow the next tap after the browser takes a drag over as a scroll', async () => {
    const { handle, other, otherAction, spec } = renderHandle()

    fireEvent.pointerDown(handle, { button: 0, clientX: 10, clientY: 10, pointerType: 'touch' })
    fireEvent.pointerMove(handle, { clientX: 10, clientY: 60, pointerType: 'touch' })
    await nextFrame()
    expect(spec.onEngage).toHaveBeenCalledOnce()
    fireEvent.pointerCancel(handle, { pointerType: 'touch' })

    fireEvent.pointerDown(other, { button: 0, pointerType: 'touch' })
    fireEvent.pointerUp(other, { button: 0, pointerType: 'touch' })
    fireEvent.click(other, { button: 0 })

    expect(otherAction).toHaveBeenCalledOnce()
    expect(spec.onCommit).not.toHaveBeenCalled()
  })
})
