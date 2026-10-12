import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { FileDiffPanel } from './diff-lines'

afterEach(() => {
  cleanup()
  globalThis.document.documentElement.style.removeProperty('font-size')
  vi.unstubAllGlobals()
})

it('renders the scrolled-to rows when the UI scale grows the root font size', () => {
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    callback(0)

    return 1
  })
  // UI scale 125%: rows are 1.25rem = 25px instead of 20px.
  globalThis.document.documentElement.style.fontSize = '20px'
  const lines = Array.from({ length: 10_000 }, (_, index) => ` row ${index}`)
  const diff = ['--- a/file', '+++ b/file', '@@ -1,10000 +1,10000 @@', ...lines].join('\n')
  const { container } = render(<FileDiffPanel diff={diff} virtualized />)
  const scroller = container.querySelector<HTMLElement>('[data-slot="file-diff-panel"] > div')!
  Object.defineProperty(scroller, 'clientHeight', { configurable: true, value: 500 })
  Object.defineProperty(scroller, 'scrollTop', { configurable: true, value: 5000 * 25 })

  act(() => {
    fireEvent.scroll(scroller)
  })

  expect(container.textContent?.includes('row 5000')).toBe(true)
})
