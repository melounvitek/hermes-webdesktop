import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { HostedNotice } from './hosted-notice'

afterEach(cleanup)

it('explains a backend without account-wide connector management', () => {
  render(
    <I18nProvider>
      <HostedNotice hasGuest phase="unsupported" />
    </I18nProvider>
  )

  expect(
    screen.getByText(
      'This backend does not support account-wide connector management. Manage connections in a chat session instead.'
    )
  ).toBeTruthy()
})
