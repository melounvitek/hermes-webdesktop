import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { I18nProvider } from '@/i18n'
import { $approvalModes } from '@/store/approval-mode'
import { $notifications } from '@/store/notifications'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { useApprovalModeStatusbarItem } from './approval-mode-menu'

const profileRequest = vi.hoisted(() => vi.fn())
vi.mock('@/store/gateway', async original => ({
  ...await original<Record<string, unknown>>(),
  requestGatewayForProfile: profileRequest
}))

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
  $approvalModes.set({})
  $notifications.set([])
  vi.unstubAllGlobals()
  profileRequest.mockReset()
})

function Harness({
  profile = 'default',
  requestGateway
}: {
  profile?: string
  requestGateway: (method: string, params?: Record<string, unknown>) => Promise<unknown>
}) {
  const item = useApprovalModeStatusbarItem(profile, requestGateway)

  return (
    <MemoryRouter>
      <StatusbarControls items={[item]} />
    </MemoryRouter>
  )
}

describe('approval mode statusbar item', () => {
  it('routes browser A/B policy reads and writes by displayed profile, not the session dispatcher', async () => {
    vi.stubGlobal('hermesDesktop', { browser: { authRequired: false, signIn: vi.fn() } })
    const policies: Record<string, string> = { a: 'manual', b: 'off' }
    profileRequest.mockImplementation(async (profile, method, params) => {
      if (method === 'config.set') policies[profile] = params.value
      return { value: policies[profile] }
    })
    const sessionRequest = vi.fn().mockRejectedValue(new Error('Wrong session route'))
    const view = render(<Harness profile="a" requestGateway={sessionRequest} />)
    await screen.findByRole('button', { name: 'Manual', exact: true })
    view.rerender(<Harness profile="b" requestGateway={sessionRequest} />)
    fireEvent.pointerDown(await screen.findByRole('button', { name: 'Off', exact: true }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /smart/i }))
    await waitFor(() => expect(policies).toEqual({ a: 'manual', b: 'smart' }))
    expect(sessionRequest).not.toHaveBeenCalled()
  })

  it('uses the shared statusbar menu trigger without a nested bespoke button', async () => {
    const response = new Promise<never>(() => undefined)
    render(<Harness requestGateway={vi.fn(() => response)} />)

    const statusbar = screen.getByRole('contentinfo')
    const trigger = within(statusbar).getByRole('button', { name: /unknown/i })
    expect(within(statusbar).getAllByRole('button')).toHaveLength(1)

    fireEvent.pointerDown(trigger, { button: 0 })

    expect(await screen.findByRole('menuitemradio', { name: /manual/i })).toBeTruthy()
    expect(trigger.getAttribute('aria-haspopup')).toBe('menu')
    expect(screen.getByRole('menuitemradio', { name: /smart/i })).toBeTruthy()
    expect(screen.getByRole('menuitemradio', { name: /off/i })).toBeTruthy()
  })

  it('writes the selected mode through the gateway and updates its shared trigger label', async () => {
    const requestGateway = vi.fn(async (_method, params) => ({ value: params?.value ?? 'smart' }))
    render(<Harness profile="work" requestGateway={requestGateway} />)

    fireEvent.pointerDown(await screen.findByRole('button', { name: /smart/i }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /manual/i }))

    await waitFor(() => {
      expect(requestGateway).toHaveBeenCalledWith('config.set', {
        key: 'approvals.mode',
        value: 'manual'
      })
      expect(screen.getByRole('button', { name: /manual/i })).toBeTruthy()
    })
  })

  it('shows failed loads as unknown, allows retry and reports failed writes with rollback', async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new Error('read denied'))
      .mockResolvedValueOnce({ value: 'off' })
      .mockRejectedValueOnce(new Error('write denied'))

    render(<Harness profile="failure" requestGateway={request} />)
    await waitFor(() => expect($notifications.get().at(-1)?.message).toBe('read denied'))
    fireEvent.pointerDown(screen.getByRole('button', { name: /unknown/i }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitem', { name: /retry/i }))
    fireEvent.pointerDown(await screen.findByRole('button', { name: /^off$/i }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /manual/i }))
    await waitFor(() => expect($notifications.get()[0]?.message).toBe('write denied'))
    expect(screen.getByRole('button', { name: /^off$/i })).toBeTruthy()
  })

  it('renders the shared trigger and menu in the active locale', async () => {
    const response = Promise.resolve({ value: 'smart' })
    render(
      <I18nProvider configClient={null} initialLocale="ja">
        <Harness requestGateway={vi.fn(() => response)} />
      </I18nProvider>
    )

    fireEvent.pointerDown(await screen.findByRole('button', { name: 'スマート' }), { button: 0 })

    expect(await screen.findByText('必要な場合にのみ確認します')).toBeTruthy()
    expect(screen.getByText('承認プロンプトなしで実行します')).toBeTruthy()
  })
})
