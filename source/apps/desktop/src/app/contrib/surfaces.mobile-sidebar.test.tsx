import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { group, split } from '@/components/pane-shell/tree/model'
import { NarrowOverlays } from '@/components/pane-shell/tree/renderer/narrow-overlays'
import { $hiddenTreePanes, $layoutTree, $narrowViewport } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import * as platform from '@/lib/platform'
import { makeSessionInfo } from '@/test/session-info'

import { SidebarSurface } from './surfaces'
import type { SidebarActions } from './types'

const otherSession = makeSessionInfo({ id: 'other', title: 'Other session' })

vi.mock('../chat/sidebar', () => ({
  ChatSidebar: ({ onResumeSession }: SidebarActions) => (
    <nav aria-label="Sessions">
      <button aria-current="page" onClick={() => onResumeSession('selected')}>
        Selected session
      </button>
      <button onClick={() => onResumeSession('other', otherSession)}>Other session</button>
    </nav>
  )
}))
vi.mock('../chat', () => ({ ChatView: () => null }))
vi.mock('../right-sidebar/terminal/chrome', () => ({ TerminalPaneChrome: () => null }))
vi.mock('../shell/hooks/use-status-snapshot', () => ({ useStatusSnapshot: vi.fn() }))
vi.mock('../shell/hooks/use-statusbar-items', () => ({ useStatusbarItems: vi.fn() }))
vi.mock('../shell/statusbar-controls', () => ({ StatusbarControls: () => null }))
vi.mock('../shell/model-menu-panel', () => ({ ModelMenuPanel: () => null }))
vi.mock('../shell/reasoning-menu-panel', () => ({ ReasoningMenuPanel: () => null }))
vi.mock('./panes', () => ({ setStatusbarItemGroup: vi.fn(), useStatusbarContributions: () => [] }))

let unregister: () => void

beforeEach(() => {
  vi.spyOn(platform, 'isBrowserClient').mockReturnValue(true)
  $hiddenTreePanes.set(new Set())
  $layoutTree.set(split('row', [group(['sessions']), group(['workspace'])]))
  $narrowViewport.set(true)
})

afterEach(() => {
  cleanup()
  unregister?.()
  $narrowViewport.set(false)
  $layoutTree.set(null)
  vi.restoreAllMocks()
})

function mountSidebar(actions: SidebarActions, narrow = true) {
  unregister = registry.register({
    area: 'panes',
    data: { collapsible: true, placement: 'left' },
    id: 'sessions',
    render: () => <SidebarSurface actions={actions} currentView="chat" />
  })
  $narrowViewport.set(narrow)

  return render(
    <>
      <NarrowOverlays />
      {!narrow && <SidebarSurface actions={actions} currentView="chat" />}
    </>
  )
}

function openSidebar() {
  act(() => {
    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: 'sessions', mode: 'open' } }))
  })
}

describe('SidebarSurface session selection', () => {
  it.each([true, false])('closes the narrow reveal only in browser clients (browser=%s)', browser => {
    vi.mocked(platform.isBrowserClient).mockReturnValue(browser)
    const originalResume = vi.fn()
    const actions = { onResumeSession: originalResume } as unknown as SidebarActions
    mountSidebar(actions)
    openSidebar()

    // The controller replaces fields without rerendering the memoized surface.
    const resume = vi.fn(() => new Promise<void>(() => {}))
    actions.onResumeSession = resume
    const dockedTree = $layoutTree.get()
    const hiddenPanes = $hiddenTreePanes.get()
    const persist = vi.spyOn(Storage.prototype, 'setItem')

    fireEvent.click(screen.getByRole('button', { name: 'Other session' }))
    expect(resume).toHaveBeenLastCalledWith('other', otherSession)
    expect(Boolean(screen.queryByRole('navigation', { name: 'Sessions' }))).toBe(!browser)

    openSidebar()
    fireEvent.click(screen.getByRole('button', { name: 'Selected session' }))
    expect(resume).toHaveBeenLastCalledWith('selected')
    expect(resume).toHaveBeenCalledTimes(2)
    expect(originalResume).not.toHaveBeenCalled()
    expect(Boolean(screen.queryByRole('navigation', { name: 'Sessions' }))).toBe(!browser)
    expect($layoutTree.get()).toBe(dockedTree)
    expect($hiddenTreePanes.get()).toBe(hiddenPanes)
    expect(persist).not.toHaveBeenCalled()
  })

  it('keeps the wide browser sidebar docked when selecting a session', () => {
    const resume = vi.fn()
    mountSidebar({ onResumeSession: resume } as unknown as SidebarActions, false)
    const dockedTree = $layoutTree.get()
    const hiddenPanes = $hiddenTreePanes.get()
    const persist = vi.spyOn(Storage.prototype, 'setItem')

    fireEvent.click(screen.getByRole('button', { name: 'Other session' }))

    expect(resume).toHaveBeenCalledWith('other', otherSession)
    expect(screen.getByRole('navigation', { name: 'Sessions' })).toBeTruthy()
    expect($layoutTree.get()).toBe(dockedTree)
    expect($hiddenTreePanes.get()).toBe(hiddenPanes)
    expect(persist).not.toHaveBeenCalled()
  })
})
