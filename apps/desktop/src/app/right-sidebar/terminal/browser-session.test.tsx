import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { HermesTerminalStatus } from '@/global'

import { TerminalInstance } from './instance'
import { closeTerminal } from './terminals'

const mocks = vi.hoisted(() => ({ write: vi.fn(), dispose: vi.fn(), focus: vi.fn(), fit: vi.fn() }))
vi.mock('@xterm/xterm', () => ({
  Terminal: class {
    cols = 80
    rows = 24
    options = {}
    unicode = {}
    parser = { registerOscHandler: () => ({ dispose() {} }) }
    element?: HTMLElement
    loadAddon(addon: { activate?: (terminal: unknown) => void }) {
      addon.activate?.(this)
    }
    open(host: HTMLElement) {
      this.element = host
    }
    focus = mocks.focus
    refresh() {}
    clearSelection() {}
    attachCustomKeyEventHandler() {}
    getSelection() {
      return ''
    }
    hasSelection() {
      return false
    }
    onData() {
      return { dispose() {} }
    }
    onSelectionChange() {
      return { dispose() {} }
    }
    write = mocks.write
    dispose = mocks.dispose
  }
}))
vi.mock('@xterm/addon-fit', () => ({
  FitAddon: class {
    terminal?: { element?: HTMLElement; cols: number; rows: number }
    activate(terminal: { element?: HTMLElement; cols: number; rows: number }) {
      this.terminal = terminal
    }
    fit() {
      const terminal = this.terminal!
      terminal.cols = Math.floor(terminal.element!.clientWidth / 10)
      terminal.rows = Math.floor(terminal.element!.clientHeight / 10)
      mocks.fit()
    }
  }
}))
vi.mock('@xterm/addon-serialize', () => ({
  SerializeAddon: class {
    serialize() {
      return 'secret'
    }
  }
}))
vi.mock('@xterm/addon-unicode11', () => ({ Unicode11Addon: class {} }))
vi.mock('@xterm/addon-web-links', () => ({ WebLinksAddon: class {} }))
vi.mock('@xterm/addon-webgl', () => ({
  WebglAddon: class {
    onContextLoss() {}
    clearTextureAtlas() {}
  }
}))
vi.mock('@/themes/context', () => ({ useTheme: () => ({ renderedMode: 'dark', theme: {}, themeName: 'test' }) }))
vi.mock('./buffer', () => ({ registerTerminalReader: () => () => {}, makeTerminalReader: vi.fn() }))
vi.mock('./terminals', () => ({
  reportTerminalShell: vi.fn(),
  closeTerminal: vi.fn(),
  updateTerminalRestoreCwd: vi.fn(),
  updateTerminalReviveBuffer: vi.fn()
}))
vi.mock('./terminal-font', async importOriginal => ({
  ...(await importOriginal<object>()),
  prepareTerminalFontFamily: async () => 'monospace'
}))

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

it('shows actionable start errors and reconnect state, retries the owned session, and keeps it alive while hidden', async () => {
  vi.stubEnv('VITE_BROWSER', '1')
  let status!: (value: HermesTerminalStatus) => void

  const start = vi
    .fn()
    .mockRejectedValueOnce(Object.assign(new Error('HTTP 404'), { reason: 'missing-plugin' }))
    .mockResolvedValue({ id: 'owned', cwd: '/repo', shell: 'bash' })

  const attach = vi.fn(async () => {
    status({ state: 'open' })

    return true
  })

  const dispose = vi.fn(async () => true)
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      disconnect() {}
    }
  )
  window.hermesDesktop = {
    ...window.hermesDesktop,
    terminal: {
      start,
      attach,
      dispose,
      cwd: async () => null,
      write: async () => true,
      resize: async () => true,
      onData: () => () => {},
      onExit: () => () => {},
      onStatus: (_id, callback) => {
        status = callback

        return () => {}
      }
    }
  }
  const props = { id: 'tab', cwd: '/repo', profile: 'alpha', active: true, onAddSelectionToChat: vi.fn() }
  const view = render(<TerminalInstance {...props} />)
  expect(await screen.findByText(/Install and enable browser-terminal/)).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(attach).toHaveBeenCalledTimes(1))
  expect(start).toHaveBeenLastCalledWith(expect.objectContaining({ profile: 'alpha' }))
  const composer = document.createElement('textarea')
  document.body.append(composer)
  composer.focus()
  mocks.focus.mockClear()
  act(() => status({ state: 'reconnecting' }))
  expect(screen.getByText(/Reconnecting to terminal/)).toBeTruthy()
  act(() => status({ state: 'open' }))
  await act(async () => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
  expect(mocks.focus).not.toHaveBeenCalled()
  expect(document.activeElement).toBe(composer)
  view.rerender(<TerminalInstance {...props} active={false} />)
  view.rerender(<TerminalInstance {...props} active />)
  await waitFor(() => expect(mocks.focus).toHaveBeenCalled())
  composer.remove()
  act(() => status({ state: 'disconnected', reason: 'missing-session' }))
  expect(screen.getByText(/server no longer has this terminal/)).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  expect(closeTerminal).toHaveBeenCalledWith('tab')
  expect(attach).toHaveBeenCalledTimes(1)
  expect(start).toHaveBeenCalledTimes(2) // no invisible replacement shell
  view.rerender(<TerminalInstance {...props} active={false} />)
  expect(dispose).not.toHaveBeenCalled()
  view.unmount()
  expect(dispose).toHaveBeenCalledExactlyOnceWith('owned')
})

it('deletes a shell whose start response arrives after the tab was closed, without attaching', async () => {
  vi.stubEnv('VITE_BROWSER', '1')
  let resolveStart!: (session: { id: string; shell: string; cwd: string }) => void
  const start = vi.fn(
    () =>
      new Promise<{ id: string; shell: string; cwd: string }>(resolve => {
        resolveStart = resolve
      })
  )
  const attach = vi.fn()
  const dispose = vi.fn(async () => true)
  window.hermesDesktop = {
    ...window.hermesDesktop,
    terminal: { ...window.hermesDesktop.terminal, start, attach, dispose }
  }
  const view = render(
    <TerminalInstance active={false} cwd="/repo" id="closed-tab" onAddSelectionToChat={vi.fn()} profile="alpha" />
  )
  await waitFor(() => expect(start).toHaveBeenCalledOnce())
  view.unmount()
  await act(async () => resolveStart({ id: 'late-owned', shell: 'bash', cwd: '/repo' }))
  expect(attach).not.toHaveBeenCalled()
  expect(dispose).toHaveBeenCalledExactlyOnceWith('late-owned')
})

it.each(['0', '1'])('reconciles a host that settles before the shell opens without session or focus churn (browser=%s)', async browser => {
  vi.stubEnv('VITE_BROWSER', browser)
  let width = 300
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(() => width)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(100)
  const frames = new Map<number, FrameRequestCallback>()
  let nextFrame = 0
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    frames.set(++nextFrame, callback)

    return nextFrame
  })
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))

  const flush = () => act(() => {
    const pending = [...frames.values()]
    frames.clear()
    pending.forEach(callback => callback(0))
  })

  let deliverResize!: () => void
  const disconnect = vi.fn()
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback: ResizeObserverCallback) {
      deliverResize = () => callback([], this as unknown as ResizeObserver)
    }
    observe() {}
    disconnect = disconnect
  })
  let opened!: (attached: boolean) => void
  const start = vi.fn(async () => ({ id: 'owned-fit', cwd: '/repo', shell: 'bash' }))
  const attach = vi.fn(() => new Promise<boolean>(resolve => { opened = resolve }))
  const resize = vi.fn(async () => true)
  const dispose = vi.fn(async () => true)
  window.hermesDesktop = {
    ...window.hermesDesktop,
    terminal: {
      start, attach, resize, dispose,
      cwd: async () => null, write: async () => true,
      onData: () => () => {}, onExit: () => () => {}
    }
  }
  const props = { id: 'fit-tab', cwd: '/repo', active: true, onAddSelectionToChat: vi.fn() }
  const view = render(<TerminalInstance {...props} />)
  await waitFor(() => expect(attach).toHaveBeenCalledOnce())
  expect(start).toHaveBeenCalledWith(expect.objectContaining({ cols: 30, rows: 10 }))
  expect(mocks.fit).toHaveBeenCalledOnce()
  const focusCount = mocks.focus.mock.calls.length

  width = 200 // Overlay settled while start/attach was pending, before observation.
  await act(async () => opened(true))
  act(() => deliverResize()) // Initial delivery before the activation frame.
  flush()
  expect(resize).toHaveBeenLastCalledWith('owned-fit', { cols: 20, rows: 10 })
  expect(start).toHaveBeenCalledOnce()
  expect(attach).toHaveBeenCalledOnce()
  expect(mocks.focus).toHaveBeenCalledTimes(focusCount)

  act(() => deliverResize())
  flush()
  expect(resize).toHaveBeenCalledOnce() // Same size never sends another PTY resize.
  view.rerender(<TerminalInstance {...props} active={false} />)
  expect(disconnect).toHaveBeenCalledOnce()
  const fitCount = mocks.fit.mock.calls.length
  width = 400
  act(() => deliverResize())
  flush()
  expect(mocks.fit).toHaveBeenCalledTimes(fitCount)
  expect(dispose).not.toHaveBeenCalled()
  view.rerender(<TerminalInstance {...props} active />)
  flush()
  expect(resize).toHaveBeenLastCalledWith('owned-fit', { cols: 40, rows: 10 })
  expect(start).toHaveBeenCalledOnce()
  view.unmount()
  expect(dispose).toHaveBeenCalledExactlyOnceWith('owned-fit')
})
