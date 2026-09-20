import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { constants } from 'node:fs'
import { access, lstat, mkdir, readFile, readdir, realpath, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'
import { startProcess, stopProcess, waitFor } from './processes.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const groups = ['recovery', 'smoke', 'usability', 'profile-isolation', 'transcript-freshness']
export const usage = 'npm run test:browser-compat -- --backend-root /ABS/stock --python /ABS/test-venv/bin/python --chrome /ABS/chromium --web-dist /ABS/dist-browser --evidence /ABS/new-private-directory'

export function options(argv) {
  const names = ['backend-root', 'python', 'chrome', 'web-dist', 'evidence']
  const { values } = parseArgs({ args: argv, options: Object.fromEntries(names.map(name => [name, { type: 'string' }])) })
  for (const name of names) assert.ok(values[name] && path.isAbsolute(values[name]), `Explicit absolute --${name} required\n${usage}`)
  assert.equal(process.platform, 'linux', 'This harness currently requires Linux /proc process ownership checks')
  return values
}

export function childEnvironment(home, inputs) {
  return {
    PATH: '/usr/bin:/bin', HOME: home, LANG: 'C.UTF-8', TZ: 'UTC',
    PYTHONDONTWRITEBYTECODE: '1', PYTHONNOUSERSITE: '1',
    SPIKE_PYTHON: inputs.python, SPIKE_BACKEND_ROOT: inputs['backend-root'], CHROME_PATH: inputs.chrome,
    // Browser routing and this failing proxy are guardrails, not an OS egress sandbox.
    HTTP_PROXY: 'http://127.0.0.1:1', HTTPS_PROXY: 'http://127.0.0.1:1', ALL_PROXY: 'http://127.0.0.1:1',
    NO_PROXY: '127.0.0.1,localhost,::1,recovery.localhost', HF_HUB_OFFLINE: '1'
  }
}

async function bundleIdentity(root) {
  const files = []
  async function walk(dir) {
    for (const name of (await readdir(dir)).sort()) {
      const file = path.join(dir, name)
      const stat = await lstat(file)
      assert.ok(!stat.isSymbolicLink(), `Frozen bundle must not contain symlinks: ${file}`)
      if (stat.isDirectory()) await walk(file)
      else {
        assert.ok(stat.isFile(), `Not a bundle file: ${file}`)
        files.push([path.relative(root, file), createHash('sha256').update(await readFile(file)).digest('hex')])
      }
    }
  }
  await walk(root)
  return { sha256: createHash('sha256').update(JSON.stringify(files)).digest('hex'), files }
}

export async function run(inputs) {
  process.umask(0o077)
  // Refuse reuse, symlinks and stale receipts; never chmod or overwrite a caller's existing directory.
  await mkdir(inputs.evidence, { mode: 0o700 })
  const evidence = await realpath(inputs.evidence)
  const summary = { inputs, started: new Date().toISOString(), status: 'FAIL',
    limitations: ['Existing five scripts only; no additional boundary/auth matrix coverage',
      'Not an OS network sandbox; use an externally supplied loopback-only egress sandbox for strict offline execution',
      'UI upstream revision is not inferred from a built bundle; local git identity and bundle digest are recorded'],
    groups: groups.map(name => ({ name, status: 'SKIP' })) }
  let interrupted
  const onSignal = signal => { interrupted = signal }
  process.on('SIGINT', onSignal)
  process.on('SIGTERM', onSignal)
  const checkSignal = () => { if (interrupted) throw new Error(`Interrupted by ${interrupted}`) }
  const fixtureArgs = [path.join(here, 'backend.py'), '--backend-root', inputs['backend-root'],
    '--python', inputs.python, '--web-dist', inputs['web-dist']]
  try {
    for (const file of [inputs.python, inputs.chrome]) await access(file, constants.X_OK)
    await access(path.join(inputs['web-dist'], 'index.html'), constants.R_OK)
    const env = childEnvironment(evidence, inputs)
    summary.stock = JSON.parse(execFileSync(inputs.python, [...fixtureArgs, '--check-source'], {
      cwd: evidence, env, encoding: 'utf8', timeout: 30000
    }))
    summary.bundle = await bundleIdentity(inputs['web-dist'])
    const git = (...args) => execFileSync('git', ['-C', here, ...args], { env, encoding: 'utf8', timeout: 15000 }).trim()
    summary.ui = { revision: git('rev-parse', 'HEAD'), changes: git('status', '--porcelain'), upstream: null }
    for (const result of summary.groups) {
      checkSignal()
      const dir = path.join(evidence, result.name)
      await mkdir(dir, { mode: 0o700 })
      const home = path.join(dir, 'home')
      await mkdir(home, { mode: 0o700 })
      const env = childEnvironment(home, inputs)
      const owned = []
      result.status = 'FAIL'
      result.cleanup = []
      try {
        let args = [inputs['web-dist'], dir]
        let fixture
        if (result.name !== 'recovery') {
          fixture = startProcess(inputs.python, fixtureArgs, { cwd: home, env, logPath: path.join(dir, 'fixture.log') })
          owned.push(fixture)
          await fixture.ready
          const ready = await waitFor(() => {
            checkSignal()
            if (fixture.failure) throw fixture.failure
            assert.ok(!fixture.closed && fixture.child.exitCode === null && !fixture.child.signalCode, 'Fixture exited before READY; inspect fixture.log')
            const line = fixture.output.match(/^READY (.+)$/m)?.[1]
            return line && JSON.parse(line)
          }, 100000, `${result.name} fixture READY`)
          const runtimePath = path.join(ready.run_dir, 'runtime.json')
          const runtime = JSON.parse(await readFile(runtimePath, 'utf8'))
          assert.equal(runtime.harness_pid, fixture.child.pid)
          assert.equal(runtime.stock.revision, summary.stock.revision)
          assert.equal(runtime.stock.root, summary.stock.root)
          assert.equal(runtime.public_url, null)
          for (const url of [runtime.url, runtime.model_url]) {
            assert.equal(new URL(url).hostname, '127.0.0.1')
            assert.equal(new URL(url).protocol, 'http:')
          }
          result.fixture = runtime.run_dir
          await writeFile(path.join(dir, 'runtime.json'), JSON.stringify(runtime, null, 2))
          args = [runtimePath, dir]
        }
        const script = startProcess(process.execPath, [path.join(here, `${result.name}.mjs`), ...args], {
          cwd: home, env, logPath: path.join(dir, 'scenario.log')
        })
        owned.push(script)
        await script.ready
        await waitFor(() => {
          checkSignal()
          if (script.failure) throw script.failure
          if (fixture) {
            if (fixture.failure) throw fixture.failure
            assert.ok(!fixture.closed && fixture.child.exitCode === null && !fixture.child.signalCode, 'Fixture exited during scenario')
          }
          return script.closed
        }, 20 * 60 * 1000, `${result.name} scenario`)
        assert.equal(script.child.exitCode, 0, `Scenario exited ${script.child.exitCode ?? script.child.signalCode}; inspect ${result.name}/scenario.log`)
        result.status = 'PASS'
      } catch (error) {
        result.error = error.stack
      } finally {
        for (const child of owned.reverse()) {
          try { result.cleanup.push(await stopProcess(child)) }
          catch (error) { result.cleanup.push({ confirmed: false, error: error.stack }); result.status = 'FAIL' }
        }
      }
      console.log(`${result.status}: ${result.name}`)
      await writeFile(path.join(evidence, 'summary.json'), JSON.stringify(summary, null, 2))
      if (result.status !== 'PASS') throw new Error(`Failed ${result.name}; later groups were not run`)
    }
    assert.deepEqual(await bundleIdentity(inputs['web-dist']), summary.bundle, 'Bundle changed during gate')
    const stockAfter = JSON.parse(execFileSync(inputs.python, [...fixtureArgs, '--check-source'], {
      cwd: evidence, env: childEnvironment(evidence, inputs), encoding: 'utf8', timeout: 30000
    }))
    assert.deepEqual(stockAfter, summary.stock, 'Stock source changed during gate')
    checkSignal()
    summary.status = 'PASS'
  } catch (error) {
    summary.error = error.stack
  } finally {
    process.off('SIGINT', onSignal)
    process.off('SIGTERM', onSignal)
    summary.finished = new Date().toISOString()
    await writeFile(path.join(evidence, 'summary.json'), JSON.stringify(summary, null, 2))
  }
  console.log(`${summary.status}; private evidence: ${evidence}`)
  return summary
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.slice(2).includes('--help')) console.log(usage)
  else {
    try { process.exitCode = (await run(options(process.argv.slice(2)))).status === 'PASS' ? 0 : 1 }
    catch (error) { console.error(error.message); process.exitCode = 1 }
  }
}
