import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ESCAPE_PRIORITY, pushEscapeLayer } from '@/lib/escape-layers'

import { OverlayView } from './overlay-view'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('overlay interaction ownership', () => {
  it('keeps content clicks and higher layers separate from dismissal', () => {
    const onClose = vi.fn()

    const { getByRole } = render(
      <OverlayView closeLabel="Close settings" onClose={onClose}>
        <input aria-label="Draft" defaultValue="Keep me" />
      </OverlayView>
    )

    fireEvent.click(getByRole('textbox'))
    expect(onClose).not.toHaveBeenCalled()
    const release = pushEscapeLayer(ESCAPE_PRIORITY.drag)
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).not.toHaveBeenCalled()
    release()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledOnce()
    onClose.mockClear()
    fireEvent.click(getByRole('button', { name: 'Close settings' }))
    expect(onClose).toHaveBeenCalledOnce()
  })
})
