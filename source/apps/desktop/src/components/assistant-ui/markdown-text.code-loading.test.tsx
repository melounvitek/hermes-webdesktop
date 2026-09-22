import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

const loads = vi.hoisted(() => ({ plugin: vi.fn(), highlighter: vi.fn() }))

vi.mock('@streamdown/code', () => {
  loads.plugin()
  throw new TypeError('Optional code plugin must not be needed to render markdown')
})

vi.mock('@/components/chat/shiki-block', () => {
  loads.highlighter()

  return { default: ({ code }: { code: string }) => <code data-testid="highlighted-code">{code}</code> }
})

beforeEach(() => vi.clearAllMocks())
afterEach(cleanup)

it('renders complete prose on mount and remount without loading syntax highlighting', async () => {
  const text = '# Still readable\n\nThe entire first paragraph.\n\nThe final paragraph.'

  for (let mount = 0; mount < 2; mount++) {
    const view = render(<MarkdownTextContent isRunning={false} text={text} />)
    await vi.dynamicImportSettled()

    expect(screen.getByRole('heading', { name: 'Still readable' })).toBeTruthy()
    expect(view.container.textContent).toBe('Still readableThe entire first paragraph.The final paragraph.')
    expect(loads.plugin).not.toHaveBeenCalled()
    expect(loads.highlighter).not.toHaveBeenCalled()
    view.unmount()
  }
})

it('keeps incomplete streaming code readable and loads highlighting only when it settles', async () => {
  const code = 'const answer = 42'
  const text = `\`\`\`js\n${code}`
  const view = render(<MarkdownTextContent isRunning text={text} />)
  await vi.dynamicImportSettled()

  expect(view.container.querySelector('[data-slot="code-card"]')?.textContent).toContain(code)
  expect(screen.getByRole('button', { name: 'Copy code' })).toBeTruthy()
  expect(loads.plugin).not.toHaveBeenCalled()
  expect(loads.highlighter).not.toHaveBeenCalled()

  view.rerender(<MarkdownTextContent isRunning={false} text={`${text}\n\`\`\``} />)
  await waitFor(() => expect(screen.getByTestId('highlighted-code').textContent).toContain(code))
  expect(loads.highlighter).toHaveBeenCalledTimes(1)
  expect(loads.plugin).not.toHaveBeenCalled()
})
