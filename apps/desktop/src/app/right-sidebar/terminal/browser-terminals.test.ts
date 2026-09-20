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
  // Restoring an open pane and invoking a stale creation action cannot spawn a shell.
  store.ensureTerminal()
  expect(store.$terminals.get()).toEqual([])
  expect(store.createTerminal()).toBeNull()
  const { defaultBindings } = await import('@/lib/keybinds/actions')
  expect(defaultBindings()).not.toHaveProperty('view.newTerminal')
  const { createBrowserBridge } = await import('@/browser/bridge')
  expect(createBrowserBridge({ token: '', authRequired: false })).not.toHaveProperty('terminal')

  const a = store.ensureAgentTerminal('proc', 'A job', 'alpha')!
  store.updateTerminalReviveBuffer(a, 'password output')
  expect(store.$terminals.get().find(term => term.id === a)?.profile).toBe('alpha')
  gatewayProfile.set('beta')
  expect(store.$visibleTerminals.get()).toEqual([])
  expect(store.$visibleActiveTerminalId.get()).toBeNull()
  const b = store.ensureAgentTerminal('proc', 'B job', 'beta')!
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
