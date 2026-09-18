import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

const bootProfile = atom('default')
const profile = atom('alpha')
const background = atom<Record<string, { id: string; title: string; output?: string }[]>>({})
vi.mock('@/store/session-states', () => ({
  knownOwnerForSession: (id: string) => ({ 'session-a': 'alpha', 'session-b': { profile: 'beta' } })[id]
}))
const unmount = vi.fn()
vi.mock('@/store/profile', () => ({
  $activeProfile: bootProfile,
  $activeGatewayProfile: profile
}))
vi.mock('@/store/session', () => ({ $currentCwd: atom('/repo') }))
vi.mock('@/store/composer-status', () => ({ $backgroundStatusBySession: background }))
vi.mock('../store', () => ({ setTerminalTakeover: vi.fn() }))
vi.mock('./buffer', () => ({ setActiveTerminalId: vi.fn() }))
vi.mock('./instance', async () => {
  const { useEffect } = await import('react')

  return {
    TerminalInstance: ({ id, active, profile }: { id: string; active: boolean; profile: string }) => {
      useEffect(() => () => unmount(id), [id])

      return (
        <div data-testid={id} hidden={!active}>
          {profile} output
        </div>
      )
    },
    AgentTerminalInstance: ({ id, active, profile }: { id: string; active: boolean; profile: string }) => (
      <div data-testid={id} hidden={!active}>{profile} mirror</div>
    )
  }
})
afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
})

it('hides other profiles without unmounting their shells and offers a new terminal instead of a blank pane', async () => {
  vi.stubEnv('VITE_BROWSER', '1')
  const store = await import('./terminals')
  const { TerminalWorkspace } = await import('./workspace')
  const a = store.createTerminal()
  render(<TerminalWorkspace onAddSelectionToChat={vi.fn()} />)
  const aNode = screen.getByTestId(a)
  expect(aNode.hidden).toBe(false)
  act(() => profile.set('beta'))
  expect(aNode.hidden).toBe(true)
  expect(screen.getByText('No terminal in this profile')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'New terminal' }))
  const b = store.$visibleActiveTerminalId.get()!
  expect(screen.getByTestId(b).hidden).toBe(false)
  expect(unmount).not.toHaveBeenCalled()
  act(() => profile.set('alpha'))
  expect(screen.getByTestId(a)).toBe(aNode)
  expect(aNode.hidden).toBe(false)
  expect(screen.getByTestId(b).hidden).toBe(true)
  expect(unmount).not.toHaveBeenCalled()
  // A background B event arrives while A is foreground; unowned events stay hidden.
  act(() => background.set({
    'session-b': [{ id: 'proc', title: 'B job', output: 'B secret' }],
    unknown: [{ id: 'unowned', title: 'unknown job' }]
  }))
  const mirrorB = store.$terminals.get().find(term => term.procId === 'proc')!
  expect(mirrorB.profile).toBe('beta')
  expect(store.$terminals.get().some(term => term.procId === 'unowned')).toBe(false)
  expect(store.$visibleTerminals.get()).not.toContain(mirrorB)
  expect(screen.getByTestId(mirrorB.id).hidden).toBe(true)
  act(() => profile.set('beta'))
  expect(store.$visibleTerminals.get()).toContain(mirrorB)
  act(() => store.selectTerminal(mirrorB.id))
  expect(screen.getByTestId(mirrorB.id).hidden).toBe(false)
  act(() => background.set({
    ...background.get(),
    'session-a': [{ id: 'proc', title: 'A job', output: 'A secret' }]
  }))
  const mirrorA = store.$terminals.get().find(term => term.procId === 'proc' && term.profile === 'alpha')!
  expect(mirrorA.id).not.toBe(mirrorB.id)
  expect(store.$visibleTerminals.get()).not.toContain(mirrorA)
  act(() => store.closeAgentTerminalByProc('proc', 'alpha'))
  expect(store.$terminals.get()).toContain(mirrorB)
  act(() => profile.set('alpha'))
  expect(store.$visibleTerminals.get()).not.toContain(mirrorB)
  expect(bootProfile.get()).toBe('default')
})
