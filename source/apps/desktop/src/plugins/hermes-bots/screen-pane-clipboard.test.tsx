/**
 * Explicit-paste clipboard bridge (#123089): a native `paste` gesture over the
 * live screen must forward the client's text clipboard to noVNC's
 * `clipboardPasteFrom`, but only while this viewer holds the lease — never on
 * a watch-only stream, and never via polling. The platform paste chord must
 * reach the browser as a native paste (noVNC would otherwise swallow it) and
 * then paste that text into the focused remote app.
 */

import { fireEvent, render, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DisplayStatus } from './screen-connection'
import type * as ScreenConnection from './screen-connection'
import type { RosterRow } from './types'

type FakeRfb = {
  target: HTMLElement
  canvas: HTMLCanvasElement
  viewOnly: boolean
  clipboardPasteFrom: (text: string) => void
  // What the remote X server saw: cut texts and non-modifier key presses with the modifiers
  // held at that moment, plus the keys still down (by code, as Xvnc tracks QEMU key events).
  remote: { seen: string[]; down: Map<string, number> }
}

const rfbs = vi.hoisted(() => [] as FakeRfb[])

vi.mock('@hermes/plugin-sdk', async () => {
  const { useStore } = await import('@nanostores/react')
  const { onGatewayEvent } = await import('../../contrib/events')

  return {
    Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
      <button {...props}>{children}</button>
    ),
    Codicon: () => null,
    GlyphSpinner: () => null,
    Tip: ({ children }: { children: ReactNode }) => <>{children}</>,
    EmptyState: () => null,
    useValue: useStore,
    host: { onEvent: onGatewayEvent, retainProfile: async () => () => {} }
  }
})
vi.mock('./routing', () => {
  const route = { connectionId: 'host-a', mode: 'remote', profile: 'default', targetProfile: 'default' }

  return { botConnectionRoute: () => route, resolveBotConnectionRoute: () => ({ status: 'resolved', route }) }
})
vi.mock('./data', () => ({ botSelectionKey: (bot: RosterRow) => bot.name }))
vi.mock('./i18n', () => ({
  useBots: () => ({
    screen: {
      title: 'Screen',
      youControl: 'You control',
      handBack: 'Hand back',
      takeOver: 'Take over',
      reconnect: 'Reconnect',
      streamLost: 'Stream lost'
    }
  })
}))
vi.mock('./screen-connection', async importActual => ({
  ...(await importActual<typeof ScreenConnection>()),
  displayRequest: vi.fn(),
  resolveScreenWsUrl: vi.fn(async () => 'ws://localhost/api/display/ws'),
  isEventForBotScreen: () => false
}))
vi.mock('@novnc/novnc', async () => {
  const { createRequire } = await import('node:module')
  const { dirname, join } = await import('node:path')
  // noVNC's real keyboard handler (the package exports only rfb.js), so the chord is tested
  // against what noVNC itself sends for the surrounding modifier presses and releases.
  const rfbPath = createRequire(import.meta.url).resolve('@novnc/novnc')
  const { default: Keyboard } = await import(/* @vite-ignore */ join(dirname(rfbPath), 'input/keyboard.js'))

  const modifiers: Record<number, string> = {
    0xffe1: 'Shift_L',
    0xffe2: 'Shift_R',
    0xffe3: 'Control_L',
    0xffe4: 'Control_R',
    0xffe9: 'Alt_L',
    0xffea: 'Alt_R',
    0xffeb: 'Super_L',
    0xffec: 'Super_R',
    0xff7e: 'Mode_switch',
    0xfe03: 'ISO_Level3_Shift'
  }

  return {
    default: class {
      canvas = globalThis.document.createElement('canvas')
      keyboard = new Keyboard(this.canvas)
      remote = { seen: [] as string[], down: new Map<string, number>() }
      clipboardPasteFrom = vi.fn((text: string) => this.remote.seen.push(`cut:${text}`))
      #viewOnly = false

      constructor(public target: HTMLElement) {
        target.append(this.canvas)
        this.keyboard.onkeyevent = (keysym: number, code: string, down: boolean) => this.sendKey(keysym, code, down)
        this.keyboard.grab()
        rfbs.push(this as never)
      }

      get viewOnly() {
        return this.#viewOnly
      }

      set viewOnly(viewOnly: boolean) {
        this.#viewOnly = viewOnly

        if (viewOnly) {
          this.keyboard.ungrab()
        } else {
          this.keyboard.grab()
        }
      }

      sendKey(keysym: number, code: string, down?: boolean) {
        if (down === undefined) {
          this.sendKey(keysym, code, true)
          this.sendKey(keysym, code, false)

          return
        }

        if (!down) {
          // A release of a key that is not down is ignored, as the X server does.
          this.remote.down.delete(code)

          return
        }

        if (!(keysym in modifiers)) {
          const held = [...this.remote.down.values()].map(held => modifiers[held] ?? String.fromCharCode(held))
          this.remote.seen.push(`key:${[...held, String.fromCharCode(keysym)].join('+')}`)
        }

        this.remote.down.set(code, keysym)
      }

      addEventListener(type: string, callback: (event: { detail?: unknown }) => void) {
        if (type === 'connect') {
          queueMicrotask(() => callback({}))
        }
      }
      disconnect() {}
      focus() {}
    }
  }
})

import { displayRequest } from './screen-connection'
import { BotScreenPane } from './screen-pane'
import { $screenState } from './screen-state'

const bot: RosterRow = { name: 'default' }

// `viewer_hash` is the first 12 hex of sha256("this-viewer"); leaseHeldBy() matches on it,
// which is what makes this viewer the lease holder (`iHold`) and puts `client.viewOnly = false`.
const status: DisplayStatus = {
  profile: 'default',
  profile_key: '/home/hermes/.hermes',
  supported: true,
  installed: true,
  missing: [],
  running: true,
  pid: 42,
  display: ':20',
  socket: '/tmp/rfb.sock',
  geometry: '1440x900',
  install_command: null,
  lease: { holder: 'human', viewer_id: null, viewer_hash: 'e0f9a555d558', since: 1, reason: '', epoch: 1 }
}

const clipboard = (text: string) => ({ clipboardData: { getData: () => text } })

async function renderInControl() {
  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs).toHaveLength(1))
  await waitFor(() => expect(rfbs[0].viewOnly).toBe(false))

  return { view, rfb: rfbs[0] }
}

/** Ctrl(+Shift)+V the way a browser delivers it to the focused remote canvas: keydown, then
 *  (only if nothing cancelled the keydown) a native `paste` carrying the local clipboard. */
function pressCtrlV(canvas: HTMLElement, text: string, { shift = false } = {}) {
  fireEvent.keyDown(canvas, { key: 'Control', code: 'ControlLeft', ctrlKey: true })

  if (shift) {
    fireEvent.keyDown(canvas, { key: 'Shift', code: 'ShiftLeft', ctrlKey: true, shiftKey: true })
  }

  const key = shift ? 'V' : 'v'
  const browserPastes = fireEvent.keyDown(canvas, { key, code: 'KeyV', ctrlKey: true, shiftKey: shift })

  if (browserPastes) {
    fireEvent.paste(canvas, clipboard(text))
  }

  fireEvent.keyUp(canvas, { key, code: 'KeyV', ctrlKey: true, shiftKey: shift })

  if (shift) {
    fireEvent.keyUp(canvas, { key: 'Shift', code: 'ShiftLeft', ctrlKey: true })
  }

  fireEvent.keyUp(canvas, { key: 'Control', code: 'ControlLeft' })
}

beforeEach(() => {
  $screenState.set({})
  rfbs.length = 0
  vi.mocked(displayRequest)
    .mockReset()
    .mockResolvedValue({ ...status, ticket: 'test-ticket', viewer_id: 'this-viewer' })
  vi.stubGlobal(
    'WebSocket',
    class {
      binaryType = ''
      addEventListener() {}
      close() {}
    }
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it('forwards an explicit paste over the canvas to noVNC while holding the lease', async () => {
  const { view, rfb } = await renderInControl()

  // E.g. Edit > Paste: the text is offered to the remote clipboard, nothing is typed.
  fireEvent.paste(rfb.target, clipboard('clipboard-test-123'))

  expect(rfb.clipboardPasteFrom).toHaveBeenCalledWith('clipboard-test-123')
  expect(rfb.remote.seen).toEqual(['cut:clipboard-test-123'])
  view.unmount()
})

it('pastes the local clipboard into the focused remote app on Ctrl+V', async () => {
  const { view, rfb } = await renderInControl()

  pressCtrlV(rfb.canvas, 'https://example.test/secret')

  expect(rfb.remote.seen).toEqual(['cut:https://example.test/secret', 'key:Control_L+v'])
  expect(rfb.remote.down.size).toBe(0)
  view.unmount()
})

it('keeps Shift for Ctrl+Shift+V so remote terminals paste too', async () => {
  const { view, rfb } = await renderInControl()

  pressCtrlV(rfb.canvas, 'echo hi', { shift: true })

  expect(rfb.remote.seen).toEqual(['cut:echo hi', 'key:Control_L+Shift_L+V'])
  expect(rfb.remote.down.size).toBe(0)
  view.unmount()
})

it('turns ⌘V on macOS into Ctrl+V on the remote without leaving ⌘ held there', async () => {
  vi.spyOn(navigator, 'platform', 'get').mockReturnValue('MacIntel')
  const { view, rfb } = await renderInControl()

  fireEvent.keyDown(rfb.canvas, { key: 'Meta', code: 'MetaLeft', metaKey: true })
  // Not cancelled, so the Edit menu's Paste (Electron's role, the browser's own) can fire `paste`.
  expect(fireEvent.keyDown(rfb.canvas, { key: 'v', code: 'KeyV', metaKey: true })).toBe(true)
  fireEvent.paste(rfb.canvas, clipboard('secret'))
  // macOS sends no keyup for V while ⌘ is held; ⌘'s own release follows.
  fireEvent.keyUp(rfb.canvas, { key: 'Meta', code: 'MetaLeft' })

  expect(rfb.remote.seen).toEqual(['cut:secret', 'key:Control_L+v'])
  expect(rfb.remote.down.size).toBe(0)
  view.unmount()
})

it('drops a paste while this viewer only watches (no lease)', async () => {
  vi.mocked(displayRequest).mockResolvedValue({
    ...status,
    lease: { ...status.lease, viewer_hash: 'someone-else' },
    ticket: 'test-ticket',
    viewer_id: 'this-viewer'
  })

  const view = render(<BotScreenPane bot={bot} />)
  await waitFor(() => expect(rfbs).toHaveLength(1))
  await waitFor(() => expect(rfbs[0].viewOnly).toBe(true))

  pressCtrlV(rfbs[0].canvas, 'clipboard-test-123')
  fireEvent.paste(rfbs[0].target, clipboard('clipboard-test-123'))

  expect(rfbs[0].clipboardPasteFrom).not.toHaveBeenCalled()
  expect(rfbs[0].remote.seen).toEqual([])
  view.unmount()
})

it("drops a paste over the bridge's 256 KiB cut-text cap instead of forwarding it", async () => {
  const { view, rfb } = await renderInControl()

  pressCtrlV(rfb.canvas, 'a'.repeat(256 * 1024 + 1))

  expect(rfb.clipboardPasteFrom).not.toHaveBeenCalled()
  expect(rfb.remote.seen).toEqual([])
  expect(rfb.remote.down.size).toBe(0)
  view.unmount()
})
