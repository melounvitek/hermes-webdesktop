import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

vi.mock('@streamdown/code', () => {
  throw new TypeError('Failed to fetch dynamically imported module')
})

afterEach(cleanup)

it('keeps complete rich markdown readable after an optional code plugin failure, including on remount', async () => {
  const text = '# Still readable\n\nThe entire first paragraph.\n\nThe final paragraph.'

  for (let mount = 0; mount < 2; mount++) {
    const view = render(<MarkdownTextContent isRunning={false} text={text} />)
    await vi.dynamicImportSettled()

    expect(screen.getByRole('heading', { name: 'Still readable' })).toBeTruthy()
    expect(view.container.textContent).toBe('Still readableThe entire first paragraph.The final paragraph.')
    view.unmount()
  }
})
