import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

const bootProfile = atom('default')
const gatewayProfile = atom('alpha')
vi.mock('@/store/profile', () => ({
  $activeProfile: bootProfile,
  $activeGatewayProfile: gatewayProfile
}))
vi.mock('@/store/session', () => ({ $currentCwd: atom('/repo') }))
vi.mock('../store', () => ({ setTerminalTakeover: vi.fn() }))

afterEach(() => vi.unstubAllEnvs())

it('keeps browser terminals runtime-only and isolates selection and close operations through A → B → A', async () => {
  vi.stubEnv('VITE_BROWSER', '1')
  localStorage.setItem(
    'hermes.desktop.terminals.v1',
    JSON.stringify({
      activeTerminalId: 'unowned',
      terminals: [{ id: 'unowned', cwd: '/repo', title: 'old', reviveBuffer: 'secret' }]
    })
  )
  const store = await import('./terminals')
  expect(store.$terminals.get()).toEqual([])
  const a = store.createTerminal()
  store.updateTerminalReviveBuffer(a, 'password output')
  expect(store.$terminals.get().find(term => term.id === a)?.profile).toBe('alpha')
  gatewayProfile.set('beta')
  expect(store.$visibleTerminals.get()).toEqual([])
  expect(store.$visibleActiveTerminalId.get()).toBeNull()
  const b = store.createTerminal()
  expect(store.$visibleTerminals.get().map(term => term.id)).toEqual([b])
  expect(store.$terminals.get().find(term => term.id === b)?.profile).toBe('beta')
  store.selectTerminal(a)
  expect(store.$visibleActiveTerminalId.get()).toBe(b)
  gatewayProfile.set('alpha')
  expect(store.$visibleTerminals.get().map(term => term.id)).toEqual([a])
  expect(store.$visibleActiveTerminalId.get()).toBe(a)
  expect(store.$terminals.get().find(term => term.id === a)?.profile).toBe('alpha')
  store.closeAllTerminals()
  expect(store.$terminals.get().map(term => term.id)).toEqual([b])
  gatewayProfile.set('beta')
  expect(store.$visibleActiveTerminalId.get()).toBe(b)
  expect(bootProfile.get()).toBe('default')
  expect(localStorage.getItem('hermes.desktop.terminals.v1')).toBeNull()
})

it('remembers only live shells, never their output, in this browser tab across a reload', async () => {
  vi.stubEnv('VITE_BROWSER', '1')
  vi.resetModules()
  gatewayProfile.set('alpha')
  const key = 'hermes.desktop.terminals.window.v1'
  const first = await import('./terminals')
  const live = first.createTerminal()
  first.createTerminal() // Its start failed: no shell to come back to.
  first.rememberTerminalSession(live, { id: 'sid-live', shell: 'zsh' })
  first.renameTerminal(live, 'build')
  first.updateTerminalReviveBuffer(live, 'password output')
  first.selectTerminal(live)
  expect(sessionStorage.getItem(key)).not.toContain('password output')
  expect(localStorage.getItem('hermes.desktop.terminals.v1')).toBeNull()
  const stored = JSON.parse(sessionStorage.getItem(key)!)
  stored.terminals.push({ id: 'no-shell', title: 'x', auto: true, cwd: '/repo', profile: 'alpha' })
  sessionStorage.setItem(key, JSON.stringify(stored))

  vi.resetModules() // Page reload.
  const reloaded = await import('./terminals')
  expect(reloaded.$terminals.get()).toEqual([
    {
      id: live,
      title: 'build',
      auto: false,
      cwd: '/repo',
      kind: 'user',
      profile: 'alpha',
      session: { id: 'sid-live', shell: 'zsh' }
    }
  ])
  expect(reloaded.$activeTerminalId.get()).toBe(live)
  reloaded.closeTerminal(live)
  expect(sessionStorage.getItem(key)).toBeNull()
})
