import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { childEnvironment, options, run } from './compat.mjs'
import { startProcess, stopProcess, waitFor } from './processes.mjs'

test('explicit preflight fails privately, reports skipped groups, and never inherits credentials', async () => {
  assert.throws(() => options([]), /--backend-root/)
  assert.throws(() => options(['--public-url', 'https://example.invalid']), /Unknown option/)
  const temp = await mkdtemp(path.join(os.tmpdir(), 'compat-preflight-test-'))
  try {
    const inputs = options(['--backend-root', temp, '--python', path.join(temp, 'missing-python'),
      '--chrome', process.execPath, '--web-dist', temp, '--evidence', path.join(temp, 'evidence')])
    const oldSecret = process.env.OPENAI_API_KEY
    process.env.OPENAI_API_KEY = 'harness-sentinel-not-a-real-key'
    try {
      const env = childEnvironment(temp, inputs)
      assert.equal(env.OPENAI_API_KEY, undefined)
      assert.equal(env.NODE_OPTIONS, undefined)
      assert.equal(env.PYTHONPATH, undefined)
      assert.equal(env.HOME, temp)
      assert.equal(env.SPIKE_BACKEND_ROOT, temp)
    } finally {
      if (oldSecret === undefined) delete process.env.OPENAI_API_KEY
      else process.env.OPENAI_API_KEY = oldSecret
    }
    const receipt = await run(inputs)
    assert.equal(receipt.status, 'FAIL')
    assert.ok(receipt.groups.every(group => group.status === 'SKIP'))
    assert.match(receipt.error, /missing-python/)
    assert.equal((await stat(inputs.evidence)).mode & 0o777, 0o700)
    const summary = path.join(inputs.evidence, 'summary.json')
    assert.equal((await stat(summary)).mode & 0o777, 0o600)
    assert.deepEqual(JSON.parse(await readFile(summary, 'utf8')), JSON.parse(JSON.stringify(receipt)))
    await assert.rejects(run(inputs), { code: 'EEXIST' })
  } finally { await rm(temp, { recursive: true, force: true }) }
})

test('owned teardown catches detached descendants after launcher loss, escalates, and leaves unrelated children alone', async () => {
  const temp = await mkdtemp(path.join(os.tmpdir(), 'compat-lifecycle-test-'))
  const owned = []
  const launch = (code, name) => {
    const child = startProcess(process.execPath, ['-e', code], {
      cwd: temp, env: { PATH: '/usr/bin:/bin', HOME: temp }, logPath: path.join(temp, name)
    })
    owned.push(child)
    return child
  }
  try {
    const unrelated = launch("console.log('READY'); setInterval(() => {}, 1000)", 'unrelated.log')
    const parent = launch(`
      const { spawn } = require('node:child_process');
      const child = spawn(process.execPath, ['-e', "process.on('SIGTERM', () => {}); console.log('READY'); setInterval(() => {}, 1000)"],
        { detached: true, stdio: ['ignore', 'pipe', 'ignore'] });
      child.stdout.on('data', () => console.log('DESCENDANT ' + child.pid));
      setInterval(() => {}, 1000);
    `, 'parent.log')
    await Promise.all(owned.map(child => child.ready))
    const pid = await waitFor(() => Number(parent.output.match(/DESCENDANT (\d+)/)?.[1]), 5000, 'descendant readiness')
    await waitFor(() => parent.groups.has(pid), 5000, 'detached group ownership')
    parent.child.kill('SIGKILL')
    const cleanup = await stopProcess(parent, 100)
    assert.equal(cleanup.confirmed, true)
    assert.equal(cleanup.forced, true)
    assert.ok(cleanup.groups.includes(pid))
    assert.equal(unrelated.child.exitCode, null)
    assert.equal(unrelated.child.signalCode, null)
    assert.ok(process.kill(unrelated.child.pid, 0))
    const missing = startProcess(path.join(temp, 'no-executable'), [], {
      cwd: temp, env: {}, logPath: path.join(temp, 'missing.log')
    })
    owned.push(missing)
    await assert.rejects(missing.ready, /ENOENT/)
    assert.equal((await stopProcess(missing, 100)).confirmed, true)
  } finally {
    await Promise.all(owned.map(child => stopProcess(child, 100)))
    await rm(temp, { recursive: true, force: true })
  }
})
