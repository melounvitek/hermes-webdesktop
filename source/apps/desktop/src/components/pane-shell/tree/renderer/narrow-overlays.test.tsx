import { act, cleanup, fireEvent, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { registry } from '@/contrib/registry'
import { ESCAPE_PRIORITY, pushEscapeLayer } from '@/lib/escape-layers'
import * as platform from '@/lib/platform'
import { stubResizeObserver } from '@/test/jsdom'

import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree, $narrowViewport, declareDefaultTree } from '../store'

import { NarrowOverlays } from './narrow-overlays'

// Ground truth for "the Bots tab is still visible when the sessions sidebar
// collapses on a narrow window". A collapsible pane DOCKED into the sessions
// zone (SESSIONS | BOTS) must leave the grid with the zone, and the narrow
// edge overlay must mirror the zone's tab strip so the docked pane stays
// reachable — not just the zone's first pane.

beforeAll(() => {
  stubResizeObserver()
})

const disposers: (() => void)[] = []

const registerPane = (id: string, title: string, data: Record<string, unknown>, body: ReactNode) => {
  disposers.push(
    registry.register({
      area: 'panes',
      data,
      id,
      render: () => <div data-testid={`${id}-body`}>{body}</div>,
      title
    })
  )
}

beforeEach(() => {
  window.localStorage.clear()
  vi.spyOn(platform, 'isBrowserClient').mockReturnValue(true)
  $hiddenTreePanes.set(new Set())

  registerPane('sessions', 'sessions', { collapsible: true, placement: 'left', width: '237px' }, 'session rows')
  registerPane('bots', 'Bots', { collapsible: true, placement: 'left', width: '260px' }, 'bot roster')
  registerPane('workspace', 'workspace', { placement: 'main', uncloseable: true }, 'chat')

  declareDefaultTree(split('row', [group(['sessions', 'bots']), group(['workspace'])]))
  $narrowViewport.set(true)
})

afterEach(() => {
  cleanup()
  $narrowViewport.set(false)
  $layoutTree.set(null)
  disposers.splice(0).forEach(dispose => dispose())
  vi.restoreAllMocks()
})

const revealPane = (id: string) => {
  act(() => {
    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id, mode: 'open' } }))
  })
}

const overlayTab = (paneId: string) => document.querySelector<HTMLElement>(`[data-narrow-overlay-tab="${paneId}"]`)

const click = (target: HTMLElement) => {
  fireEvent.pointerDown(target, { button: 0 })
  fireEvent.pointerUp(target, { button: 0 })
  fireEvent.click(target, { button: 0 })
}

describe('browser narrow sidebar dismissal', () => {
  it.each([true, false])(
    'dismisses only the browser reveal without eating the outside action (browser=%s)',
    browser => {
      vi.mocked(platform.isBrowserClient).mockReturnValue(browser)
      const outsideAction = vi.fn()

      const { getByText, getByTestId, queryByTestId } = render(
        <>
          <NarrowOverlays />
          <button onClick={outsideAction}>Outside action</button>
        </>
      )

      revealPane('sessions')
      const pane = getByTestId('sessions-body').closest('[data-glass-opaque]') as HTMLElement
      expect(pane.style.top).toBe(browser ? `${TITLEBAR_HEIGHT}px` : '')
      click(getByText('Outside action'))
      expect(outsideAction).toHaveBeenCalledOnce()
      expect(Boolean(queryByTestId('sessions-body'))).toBe(!browser)
    }
  )

  it('preserves inside/portalled actions, gestures, toggles and higher layer ownership', () => {
    registerPane(
      'files',
      'Files',
      { collapsible: true, placement: 'right' },
      <>
        <input aria-label="Sidebar draft" defaultValue="Keep me" />
        {createPortal(
          <>
            <button>Portalled action</button>
            <button onClick={e => e.stopPropagation()}>Stopped action</button>
          </>,
          document.body
        )}
      </>
    )
    $layoutTree.set(split('row', [group(['sessions', 'bots']), group(['workspace']), group(['files'])]))

    const { getByText, getByRole, getByTestId, queryByTestId } = render(
      <>
        <NarrowOverlays />
        <button
          onClick={() => {
            window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: 'files' } }))
          }}
        >
          Toggle files
        </button>
        <button>Outside action</button>
      </>
    )

    revealPane('sessions')
    // Pane reveals also come from ordinary uncovered controls, not just titlebar toggles.
    click(getByText('Toggle files'))
    const draft = getByRole('textbox') as HTMLInputElement
    expect((draft.closest('[data-glass-opaque]') as HTMLElement).style.top).toBe(`${TITLEBAR_HEIGHT}px`)
    click(draft)
    click(getByText('Portalled action'))
    click(getByText('Stopped action'))
    fireEvent.wheel(getByText('Outside action'))
    fireEvent.pointerDown(getByText('Outside action'), { button: 0 })
    fireEvent.pointerCancel(getByText('Outside action'))
    expect(getByRole('textbox')).toBe(draft)
    expect(draft.value).toBe('Keep me')
    const selection = window.getSelection()!
    const range = document.createRange()
    range.selectNodeContents(getByText('Outside action'))
    selection.addRange(range)
    click(getByText('Outside action'))
    expect(getByRole('textbox')).toBe(draft)
    selection.removeAllRanges()

    // A menu may unmount on pointerdown, before its eventual outside click.
    const menu = document.createElement('div')
    menu.setAttribute('role', 'menu')
    document.body.append(menu)
    fireEvent.pointerDown(getByText('Outside action'), { button: 0 })
    menu.remove()
    fireEvent.click(getByText('Outside action'), { button: 0 })
    expect(getByTestId('files-body')).toBeTruthy()

    const release = pushEscapeLayer(ESCAPE_PRIORITY.overlay)
    click(getByText('Outside action'))
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(getByTestId('files-body')).toBeTruthy()
    release()
    click(getByText('Toggle files'))
    expect(queryByTestId('files-body')).toBeNull()
    click(getByText('Toggle files'))
    expect(getByTestId('files-body')).toBeTruthy()
    click(getByText('Outside action'))
    expect(queryByTestId('files-body')).toBeNull()

    revealPane('files')
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(queryByTestId('files-body')).toBeNull()
    const dockedTree = $layoutTree.get()
    act(() => $narrowViewport.set(false))
    revealPane('sessions')
    click(getByText('Outside action'))
    expect(queryByTestId('sessions-body')).toBeNull()
    expect($layoutTree.get()).toBe(dockedTree)
  })
})

describe('narrow overlay of a stacked zone', () => {
  it('mirrors the zone tab strip so every stacked collapsible stays reachable', () => {
    const { getByTestId, queryByTestId } = render(<NarrowOverlays />)

    revealPane('sessions')

    // Both zone-mates surface as tabs; the revealed pane's body is on screen.
    expect(overlayTab('sessions')).toBeTruthy()
    expect(overlayTab('bots')).toBeTruthy()
    expect(getByTestId('sessions-body')).toBeTruthy()
    expect(queryByTestId('bots-body')).toBeNull()

    // Clicking the BOTS tab swaps the overlay to the docked pane.
    fireEvent.pointerDown(overlayTab('bots')!, { button: 0 })
    expect(getByTestId('bots-body')).toBeTruthy()
    expect(queryByTestId('sessions-body')).toBeNull()
  })

  it('keeps the stripless form for a zone with a single collapsible', () => {
    // Direct set: declareDefaultTree only ADOPTS into an existing tree — it
    // would keep the beforeEach zone (with bots) instead of replacing it.
    $layoutTree.set(split('row', [group(['sessions']), group(['workspace'])]))

    const { getByTestId } = render(<NarrowOverlays />)

    revealPane('sessions')

    expect(getByTestId('sessions-body')).toBeTruthy()
    expect(overlayTab('sessions')).toBeNull()
  })
})
