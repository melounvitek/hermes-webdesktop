import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { quietPasswordFields } from './password-fields'

const nativeType = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'type')!

afterEach(() => {
  cleanup()
  Object.defineProperty(HTMLInputElement.prototype, 'type', nativeType)
  vi.unstubAllGlobals()
})

function LoginPrompt({ reveal = false }: { reveal?: boolean }) {
  const [password, setPassword] = useState('')

  return (
    <form>
      <input aria-label="Username" autoComplete="username" />
      <input
        aria-label="Password"
        autoComplete="current-password"
        onChange={event => setPassword(event.target.value)}
        type={reveal ? 'text' : 'password'}
        value={password}
      />
    </form>
  )
}

// jsdom drops the property from the parsed rule, so check it in the source text.
function isMasked(input: HTMLElement) {
  const style = input.ownerDocument.querySelector('style')!

  expect(style.textContent).toContain('-webkit-text-security: disc')

  return input.matches((style.sheet!.cssRules[0] as CSSStyleRule).selectorText)
}

function expectOptedOut(input: HTMLElement) {
  for (const name of ['data-1p-ignore', 'data-bwignore', 'data-lpignore']) {
    expect(input.getAttribute(name)).toBe('true')
  }

  expect(input.getAttribute('data-form-type')).toBe('other')
  expect(input.getAttribute('autocomplete')).toBe('off')
}

it('renders secret fields as masked text fields that password managers ignore', async () => {
  vi.stubGlobal('CSS', { supports: (property: string, value: string) => `${property}:${value}` === '-webkit-text-security:disc' })
  quietPasswordFields()
  const { rerender } = render(<LoginPrompt />)
  const password = screen.getByLabelText('Password')

  fireEvent.change(password, { target: { value: 'hunter2' } })

  expect(password.getAttribute('type')).toBe('text')
  expect(isMasked(password)).toBe(true)
  expect(password.getAttribute('spellcheck')).toBe('false')
  expectOptedOut(password)
  await waitFor(() => expectOptedOut(screen.getByLabelText('Username')))

  rerender(<LoginPrompt reveal />)

  expect(password.getAttribute('type')).toBe('text')
  expect(isMasked(password)).toBe(false)
  expect(password).toHaveProperty('value', 'hunter2')

  rerender(<LoginPrompt />)

  expect(password.getAttribute('type')).toBe('text')
  expect(isMasked(password)).toBe(true)
})

it('keeps password fields where text masking is unavailable', () => {
  vi.stubGlobal('CSS', { supports: () => false })
  quietPasswordFields()
  render(<LoginPrompt />)
  const password = screen.getByLabelText('Password')

  expect(password.getAttribute('type')).toBe('password')
  expectOptedOut(password)
})
