import { afterEach, expect, it, vi } from 'vitest'

import type { TerminalReadResult } from '@/app/right-sidebar/terminal/buffer'

import type { ServerRequestContext } from './server-requests'

afterEach(() => vi.unstubAllEnvs())

it('reads only the selected agent mirror owned by the requesting profile through A → B → A', async () => {
  vi.stubEnv('VITE_BROWSER', '1')
  const { $activeGatewayProfile } = await import('@/store/profile')
  const store = await import('@/app/right-sidebar/terminal/terminals')
  const { registerTerminalReader, setActiveTerminalId } = await import('@/app/right-sidebar/terminal/buffer')
  const { handleServerRequest } = await import('./server-requests')
  const deps: ServerRequestContext['deps'] = {
    activeSessionIdRef: { current: null },
    sessionInterrupted: () => false,
    updateSessionState: vi.fn(),
    upsertToolCall: vi.fn()
  }
  const originalProfile = $activeGatewayProfile.get()
  const a = store.ensureAgentTerminal('proc', 'A job', 'alpha')!
  const b = store.ensureAgentTerminal('proc', 'B job', 'beta')!
  const output: TerminalReadResult = {
    total_lines: 1,
    start: 0,
    end: 1,
    viewport_rows: 24,
    cursor_row: 0,
    text: 'A secret'
  }
  const outputB = { ...output, text: 'B secret' }
  const readA = vi.fn(() => output)
  const readB = vi.fn(() => outputB)
  const unregisterA = registerTerminalReader(a, readA)
  const unregisterB = registerTerminalReader(b, readB)

  function requestRead(profile: string, value: string) {
    const respond = vi.fn()
    const fail = vi.fn()
    expect(
      handleServerRequest({ fail, id: 'read', method: 'terminal.read', params: {}, profile, respond }, deps, null)
    ).toBe(true)
    expect(respond).toHaveBeenCalledExactlyOnceWith({ value })
    expect(fail).not.toHaveBeenCalled()
  }

  try {
    $activeGatewayProfile.set('alpha')
    store.selectTerminal(a)
    for (const profile of ['alpha', 'beta', 'alpha']) {
      $activeGatewayProfile.set(profile)
      const isAlpha = profile === 'alpha'
      setActiveTerminalId(isAlpha ? a : b)
      readA.mockClear()
      readB.mockClear()

      requestRead(isAlpha ? 'beta' : 'alpha', '')
      requestRead('', '')
      expect(readA).not.toHaveBeenCalled()
      expect(readB).not.toHaveBeenCalled()

      requestRead(profile, JSON.stringify(isAlpha ? output : outputB))
      expect(isAlpha ? readA : readB).toHaveBeenCalledOnce()
      expect(isAlpha ? readB : readA).not.toHaveBeenCalled()
    }

    readA.mockClear()
    readB.mockClear()
    // Selection can advance before the workspace's reader subscription mounts.
    const next = store.ensureAgentTerminal('next-proc', 'Next job', 'alpha')!
    store.selectTerminal(next)
    requestRead('alpha', '')
    expect(readA).not.toHaveBeenCalled()
    expect(readB).not.toHaveBeenCalled()

    // Align selection and buffer again so missing ownership alone must block the read.
    store.selectTerminal(a)
    store.$terminals.set([{ ...store.$terminals.get().find(term => term.id === a)!, profile: undefined }])
    requestRead('alpha', '')
    expect(readA).not.toHaveBeenCalled()
    expect(readB).not.toHaveBeenCalled()
  } finally {
    unregisterA()
    unregisterB()
    setActiveTerminalId(null)
    store.$terminals.set([])
    store.$activeTerminalId.set(null)
    $activeGatewayProfile.set(originalProfile)
  }
})
