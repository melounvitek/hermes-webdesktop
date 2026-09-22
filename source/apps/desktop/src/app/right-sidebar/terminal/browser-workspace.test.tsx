import { act, cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

const bootProfile = atom('default')
const profile = atom('alpha')
const background = atom<Record<string, { id: string; title: string; output?: string }[]>>({})
vi.mock('@/store/session-states', () => ({
  knownOwnerForSession: (id: string) => ({ 'session-a': 'alpha', 'session-b': { profile: 'beta' } })[id]
}))
const unmount = vi.fn()
const startShell = vi.fn()
const takeover = atom(true)
vi.mock('@/store/profile', () => ({
  $activeProfile: bootProfile,
  $activeGatewayProfile: profile
}))
vi.mock('@/store/session', () => ({ $currentCwd: atom('/repo') }))
vi.mock('@/store/composer-status', () => ({ $backgroundStatusBySession: background }))
vi.mock('../store', () => ({ $terminalTakeover: takeover, setTerminalTakeover: vi.fn() }))
vi.mock('./buffer', () => ({ setActiveTerminalId: vi.fn() }))
vi.mock('./instance', async () => {
  const { useEffect } = await import('react')

  return {
    TerminalInstance: () => {
      startShell()

      return null
    },
    AgentTerminalInstance: ({ id, active, profile }: { id: string; active: boolean; profile: string }) => {
      useEffect(() => () => unmount(id), [id])

      return (
        <div data-testid={id} hidden={!active}>
          {profile} mirror
        </div>
      )
    }
  }
})
afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
})

it('restores an open browser pane without shells or creation controls and keeps agent mirrors profile-isolated', async () => {
  vi.stubEnv('VITE_BROWSER', '1')
  vi.spyOn(document, 'hasFocus').mockReturnValue(true)
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    top: 0,
    left: 0,
    right: 600,
    bottom: 400,
    width: 600,
    height: 400
  } as DOMRect)
  const store = await import('./terminals')
  const { TerminalPaneChrome } = await import('./chrome')
  const { PersistentTerminal } = await import('./persistent')
  render(
    <>
      <TerminalPaneChrome />
      <PersistentTerminal onAddSelectionToChat={vi.fn()} />
    </>
  )
  expect(await screen.findByText('No terminal in this profile')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'New terminal' })).toBeNull()
  expect(startShell).not.toHaveBeenCalled()

  // A stale runtime tab must not mount an interactive instance either.
  act(() =>
    store.$terminals.set([{ id: 'stale', kind: 'user', title: 'old', auto: true, cwd: '/repo', profile: 'alpha' }])
  )
  expect(startShell).not.toHaveBeenCalled()
  expect(screen.queryByRole('tab', { name: /old/ })).toBeNull()
  expect(screen.queryByRole('button', { name: 'New terminal' })).toBeNull()
  // A background B event arrives while A is foreground; unowned events stay hidden.
  act(() =>
    background.set({
      'session-b': [{ id: 'proc', title: 'B job', output: 'B secret' }],
      unknown: [{ id: 'unowned', title: 'unknown job' }]
    })
  )
  const mirrorB = store.$terminals.get().find(term => term.procId === 'proc')!
  expect(mirrorB.profile).toBe('beta')
  expect(store.$terminals.get().some(term => term.procId === 'unowned')).toBe(false)
  expect(store.$visibleTerminals.get()).not.toContain(mirrorB)
  expect(screen.getByTestId(mirrorB.id).hidden).toBe(true)
  act(() => profile.set('beta'))
  expect(store.$visibleTerminals.get()).toContain(mirrorB)
  act(() => store.selectTerminal(mirrorB.id))
  expect(screen.getByTestId(mirrorB.id).hidden).toBe(false)
  act(() =>
    background.set({
      ...background.get(),
      'session-a': [{ id: 'proc', title: 'A job', output: 'A secret' }]
    })
  )
  const mirrorA = store.$terminals.get().find(term => term.procId === 'proc' && term.profile === 'alpha')!
  expect(mirrorA.id).not.toBe(mirrorB.id)
  expect(store.$visibleTerminals.get()).not.toContain(mirrorA)
  act(() => store.closeAgentTerminalByProc('proc', 'alpha'))
  expect(store.$terminals.get()).toContain(mirrorB)
  act(() => profile.set('alpha'))
  expect(store.$visibleTerminals.get()).not.toContain(mirrorB)
  expect(bootProfile.get()).toBe('default')
  expect(unmount).toHaveBeenCalledWith(mirrorA.id)
  expect(unmount).not.toHaveBeenCalledWith(mirrorB.id)
  expect(startShell).not.toHaveBeenCalled()
})
