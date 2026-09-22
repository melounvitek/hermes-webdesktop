import assert from 'node:assert/strict'
import { execFileSync, spawnSync } from 'node:child_process'
import { mkdtemp, mkdir, writeFile, rm, symlink } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

const launcher = fileURLToPath(new URL('./backend.py', import.meta.url))
// Stdlib-only preflight: never import an agent or start a server/model/browser.
const python = '/usr/bin/python3'

test('stock preflight requires explicit source/interpreter and rejects modified code or dotenv', async () => {
  const temp = await mkdtemp(path.join(os.tmpdir(), 'compat-source-test-'))
  const stock = path.join(temp, 'stock')
  const dist = path.join(temp, 'dist')
  const git = (...args) => execFileSync('git', ['-C', stock, ...args], { encoding: 'utf8' }).trim()
  const run = (...extra) => spawnSync(python, [launcher, '--backend-root', stock, '--python', python,
    '--web-dist', dist, '--check-source', ...extra], { encoding: 'utf8', env: { PATH: '/usr/bin:/bin' } })
  try {
    await mkdir(stock)
    await mkdir(dist)
    await writeFile(path.join(dist, 'index.html'), '<html></html>')
    for (const name of ['hermes_cli/main.py', 'tui_gateway/server.py', 'run_agent.py']) {
      await mkdir(path.dirname(path.join(stock, name)), { recursive: true })
      await writeFile(path.join(stock, name), '# preflight path only; never imported\n')
    }
    git('init', '-q')
    git('add', '.')
    git('-c', 'user.name=Harness Test', '-c', 'user.email=harness@example.invalid', 'commit', '-qm', 'Fixture paths')
    const clean = run()
    assert.equal(clean.status, 0, clean.stderr)
    assert.equal(JSON.parse(clean.stdout).revision, git('rev-parse', 'HEAD'))
    assert.equal(JSON.parse(clean.stdout).root, stock)
    await writeFile(path.join(stock, 'run_agent.py'), '# modified\n')
    assert.match(run().stderr, /clean stock checkout/)
    git('add', 'run_agent.py')
    assert.match(run().stderr, /clean stock checkout/)
    git('reset', '--hard', '-q', 'HEAD')
    await writeFile(path.join(stock, '.env'), 'NOT_A_REAL_SECRET=test\n')
    assert.match(run().stderr, /\.env/)
    await rm(path.join(stock, '.env'))
    await symlink(path.join(temp, 'absent'), path.join(stock, '.env'))
    assert.match(run().stderr, /\.env/)
    const missing = spawnSync(python, [launcher, '--web-dist', dist], { encoding: 'utf8' })
    assert.notEqual(missing.status, 0)
    assert.match(missing.stderr, /--python/)
    assert.match(missing.stderr, /--backend-root/)
  } finally {
    await rm(temp, { recursive: true, force: true })
  }
})
