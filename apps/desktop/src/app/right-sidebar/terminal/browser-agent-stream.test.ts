import { afterEach, expect, it, vi } from 'vitest'

import { registerAgentTerminalWriter, seedAgentTerminalCommand, syncAgentTerminalSnapshot, writeAgentTerminalChunk } from './agent-terminal-stream'

afterEach(() => vi.unstubAllEnvs())

it('isolates mirror backlog and live output by verified profile, even with identical process ids', () => {
  vi.stubEnv('VITE_BROWSER', '1')
  const a = vi.fn()
  const b = vi.fn()
  seedAgentTerminalCommand('same', 'A command', 'alpha')
  syncAgentTerminalSnapshot('same', 'A secret', 'alpha')
  writeAgentTerminalChunk('same', 'unowned secret')
  const offB = registerAgentTerminalWriter('same', b, 'beta')
  const offA = registerAgentTerminalWriter('same', a, 'alpha')
  expect(a).toHaveBeenCalledWith('$ A command\r\nA secret')
  expect(b).not.toHaveBeenCalled()
  writeAgentTerminalChunk('same', 'B secret', 'beta')
  expect(b).toHaveBeenCalledExactlyOnceWith('B secret')
  expect(a).toHaveBeenCalledOnce()
  offA()
  offB()
})
