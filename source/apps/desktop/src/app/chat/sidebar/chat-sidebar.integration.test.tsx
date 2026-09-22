// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { group, split } from '@/components/pane-shell/tree/model'
import { NarrowOverlays } from '@/components/pane-shell/tree/renderer/narrow-overlays'
import { $layoutTree, $narrowViewport, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { SidebarProvider } from '@/components/ui/sidebar'
import { registry } from '@/contrib/registry'
import * as platform from '@/lib/platform'
import { $selectedStoredSessionId, $sessions } from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'
import { makeSessionInfo } from '@/test/session-info'

import { SidebarSurface } from '../../contrib/surfaces'
import type { SidebarActions } from '../../contrib/types'
import { type AppView, NEW_CHAT_ROUTE, ROUTES_AREA, SIDEBAR_NAV_AREA } from '../../routes'

import { ChatSidebar } from './index'

const noop = () => {}

const noopAsync = async () => {}

const sessionRows = [
  makeSessionInfo({ id: 'tile-one', last_active: 2, profile: 'default', started_at: 1, title: 'Tile one' }),
  makeSessionInfo({ id: 'tile-two', last_active: 2, profile: 'default', started_at: 1, title: 'Tile two' })
]

const renderSidebar = (pathname: string, currentView: AppView) =>
  render(
    <MemoryRouter initialEntries={[pathname]}>
      <SidebarProvider>
        <ChatSidebar
          currentView={currentView}
          onArchiveSession={noop}
          onBranchSession={noop}
          onDeleteSession={noop}
          onLoadMoreSessions={noop}
          onManageCronJob={noop}
          onNavigate={noop}
          onNewSessionInWorkspace={noop}
          onNewSessionSplit={noop}
          onResumeSession={noop}
          onTriggerCronJob={noopAsync}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

const currentButtons = () =>
  screen.queryAllByRole('button').filter(button => button.classList.contains('bg-(--ui-control-active-background)'))

const expectOnlyCurrent = (label: string | null) => {
  const button = label ? screen.getByRole('button', { name: label }) : null

  expect(currentButtons()).toEqual(button ? [button] : [])
}

const expectOnlySelectedSession = (title: string | null) => {
  const rows = ['Tile one', 'Tile two']
    .map(label => screen.queryByText(label)?.closest('.group.row-hover'))
    .filter(row => row !== undefined)

  const selectedRows = rows.filter(row => row?.className.includes('bg-(--ui-row-active-background)'))
  const expected = title ? [screen.getByText(title).closest('.group.row-hover')] : []

  expect(selectedRows).toEqual(expected)
}

const focus = (groupId: null | string) => act(() => noteActiveTreeGroup(groupId))

describe('ChatSidebar navigation activity', () => {
  let disposeContributions: () => void

  beforeEach(() => {
    disposeContributions = registry.registerMany([
      { area: ROUTES_AREA, id: 'kanban-page', data: { path: '/kanban' }, render: () => null },
      { area: ROUTES_AREA, id: 'reports-page', data: { path: '/reports' }, render: () => null },
      { area: SIDEBAR_NAV_AREA, id: 'kanban-nav', data: { codicon: 'project', label: 'Kanban', path: '/kanban' } },
      { area: SIDEBAR_NAV_AREA, id: 'reports-nav', data: { codicon: 'graph', label: 'Reports', path: '/reports' } }
    ])
    $selectedStoredSessionId.set('tile-one')
    $sessions.set(sessionRows)
    $removedSessionIds.set(new Set())
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'workspace-group' }),
        group(['session-tile:tile-one'], { active: 'session-tile:tile-one', id: 'tile-one-group' }),
        group(['session-tile:tile-two'], { active: 'session-tile:tile-two', id: 'tile-two-group' })
      ])
    )
    noteActiveTreeGroup('workspace-group')
  })

  afterEach(() => {
    cleanup()
    disposeContributions()
    $selectedStoredSessionId.set(null)
    $sessions.set([])
    $removedSessionIds.set(new Set())
    $layoutTree.set(null)
    noteActiveTreeGroup(null)
  })

  describe('mobile browser navigation', () => {
    let disposePane: () => void

    beforeEach(() => {
      vi.spyOn(platform, 'isBrowserClient').mockReturnValue(true)
      $narrowViewport.set(true)
      $layoutTree.set(split('row', [group(['sessions']), group(['workspace'])]))
      $sessions.set([
        ...sessionRows,
        makeSessionInfo({ id: 'recent', last_active: Date.now() / 1000, profile: 'default', title: 'Recent session' })
      ])
    })

    afterEach(() => {
      cleanup()
      disposePane?.()
      $narrowViewport.set(false)
      vi.restoreAllMocks()
    })

    const openSidebar = () => {
      act(() => {
        window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: 'sessions', mode: 'open' } }))
      })
    }

    const mountMobileSidebar = (pathname = NEW_CHAT_ROUTE, currentView: AppView = 'chat') => {
      const actions = {
        onArchiveSession: noop,
        onBranchSession: noop,
        onDeleteSession: noop,
        onLoadMoreSessions: noop,
        onManageCronJob: noop,
        // Navigation can remain pending; dismissing the overlay must not wait for it.
        onNavigate: vi.fn(() => new Promise<void>(() => {})),
        onNewSessionInWorkspace: vi.fn(() => new Promise<void>(() => {})),
        onNewSessionSplit: noop,
        onResumeSession: vi.fn(),
        onTriggerCronJob: noopAsync
      } satisfies SidebarActions

      disposePane = registry.register({
        area: 'panes',
        data: { collapsible: true, placement: 'left' },
        id: 'sessions',
        render: () => <SidebarSurface actions={actions} currentView={currentView} />
      })
      render(
        <MemoryRouter initialEntries={[pathname]}>
          <SidebarProvider>
            <NarrowOverlays />
          </SidebarProvider>
        </MemoryRouter>
      )
      openSidebar()

      return actions
    }

    for (const alreadyActive of [false, true]) {
      it.each([
        ['New session', 'new-session', NEW_CHAT_ROUTE, 'chat'],
        ['Capabilities', 'skills', '/skills', 'skills'],
        ['Messaging', 'messaging', '/messaging', 'messaging'],
        ['Artifacts', 'artifacts', '/artifacts', 'artifacts'],
        ['Scheduled jobs', 'cron', '/cron', 'cron'],
        ['Kanban', 'kanban-nav', '/kanban', 'extension']
      ] as const)(`closes immediately for %s (already active: ${alreadyActive})`, (label, id, pathname, view) => {
        const actions = mountMobileSidebar(alreadyActive ? pathname : '/reports', alreadyActive ? view : 'extension')

        fireEvent.click(screen.getByRole('button', { name: label === 'New session' ? /^New session.+/ : label }))

        expect(actions.onNavigate).toHaveBeenCalledWith(expect.objectContaining({ id }))
        expect(screen.queryByRole('button', { name: 'Capabilities' })).toBeNull()
      })
    }

    it.each([
      ['header', 'Sessions'],
      ['date divider', /^Hide .* sessions$/]
    ] as const)('closes immediately for the %s New session button', (_label, sectionName) => {
      const actions = mountMobileSidebar()
      const sectionHeader = screen.getByRole('button', { name: sectionName }).parentElement!

      fireEvent.click(within(sectionHeader).getByRole('button', { name: 'New session' }))

      expect(actions.onNewSessionInWorkspace).toHaveBeenCalledWith(null)
      expect(screen.queryByRole('button', { name: 'Capabilities' })).toBeNull()
    })

    it('keeps search, filters, session menus and rename interactions open', async () => {
      const actions = mountMobileSidebar()
      const expectSidebarOpen = () => expect(screen.getByText('Capabilities')).toBeTruthy()
      const search = screen.getByRole('textbox', { name: /search/i })
      fireEvent.click(search)
      fireEvent.change(search, { target: { value: 'Tile one' } })
      expectSidebarOpen()
      fireEvent.change(search, { target: { value: '' } })

      const filters = screen.getByRole('button', { name: 'Filters' })
      fireEvent.pointerDown(filters, { button: 0, pointerType: 'mouse' })
      fireEvent.pointerUp(filters, { button: 0, pointerType: 'mouse' })
      fireEvent.click(filters)
      expect(await screen.findByRole('menu')).toBeTruthy()
      expectSidebarOpen()
      fireEvent.keyDown(screen.getByRole('menu'), { key: 'Escape' })
      expectSidebarOpen()

      fireEvent.contextMenu(screen.getByText('Tile one'))
      const rename = await screen.findByRole('menuitem', { name: /rename/i })
      expectSidebarOpen()
      fireEvent.click(rename)
      const dialog = await screen.findByRole('dialog', { name: 'Rename session' })
      fireEvent.change(within(dialog).getByRole('textbox'), { target: { value: 'Renamed tile' } })
      expectSidebarOpen()
      fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }))
      expect(screen.queryByRole('dialog')).toBeNull()
      expectSidebarOpen()
      expect(actions.onNavigate).not.toHaveBeenCalled()
      expect(actions.onResumeSession).not.toHaveBeenCalled()
    })
  })

  it('keeps navigation and session activity coherent with the focused pane', () => {
    renderSidebar('/kanban', 'extension')
    expectOnlyCurrent('Kanban')
    expectOnlySelectedSession(null)

    focus('tile-one-group')
    expectOnlyCurrent(null)
    expectOnlySelectedSession('Tile one')

    focus('tile-two-group')
    expectOnlyCurrent(null)
    expectOnlySelectedSession('Tile two')

    focus(null)
    expectOnlyCurrent('Kanban')
    expectOnlySelectedSession(null)

    focus('tile-two-group')
    act(() => {
      $removedSessionIds.set(new Set(['tile-two']))
      $sessions.set([sessionRows[0]])
    })
    expectOnlyCurrent(null)
    expectOnlySelectedSession(null)

    act(() => {
      $removedSessionIds.set(new Set())
      $sessions.set(sessionRows)
    })

    for (const [pathname, currentView, label] of [
      ['/skills', 'skills', 'Capabilities'],
      ['/messaging', 'messaging', 'Messaging'],
      ['/artifacts', 'artifacts', 'Artifacts'],
      ['/cron', 'cron', 'Scheduled jobs']
    ] as const) {
      cleanup()
      focus('workspace-group')
      renderSidebar(pathname, currentView)
      expectOnlyCurrent(label)
      expectOnlySelectedSession(null)

      focus('tile-one-group')
      expectOnlyCurrent(null)
      expectOnlySelectedSession('Tile one')
    }

    cleanup()
    focus('workspace-group')
    renderSidebar('/reports', 'extension')
    expectOnlyCurrent('Reports')

    cleanup()
    disposeContributions()
    disposeContributions = noop
    focus('workspace-group')
    renderSidebar('/kanban', 'extension')
    expect(screen.queryByRole('button', { name: 'Kanban' })).toBeNull()
    expectOnlyCurrent(null)
    expectOnlySelectedSession(null)
  })
})
