import assert from 'node:assert/strict'
import { execFileSync, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { constants } from 'node:fs'
import { access, lstat, mkdir, mkdtemp, readFile, readdir, realpath, writeFile } from 'node:fs/promises'
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
  assert.equal(process.platform, 'linux', 'This harness requires Linux and working unprivileged Bubblewrap PID/network namespaces')
  return values
}

export function childEnvironment(home, inputs) {
  return {
    PATH: '/usr/bin:/bin', HOME: home, LANG: 'C.UTF-8', TZ: 'UTC',
    PYTHONDONTWRITEBYTECODE: '1', PYTHONNOUSERSITE: '1',
    SPIKE_PYTHON: inputs.python, SPIKE_BACKEND_ROOT: inputs['backend-root'], CHROME_PATH: inputs.chrome,
    // Additional guardrails inside the whole gate's loopback-only network namespace.
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

function initialSummary(inputs) {
  return { inputs, started: new Date().toISOString(), status: 'FAIL',
    limitations: ['Existing five scripts only; no additional boundary/auth matrix coverage',
      'Group receipts confirm direct-child termination only; detached descendants are killed at whole-gate namespace exit',
      'Read-only host files remain visible; this is not a confidentiality sandbox',
      'UI upstream revision is not inferred from a built bundle; local git identity and bundle digest are recorded'],
    groups: groups.map(name => ({ name, status: 'SKIP' })) }
}

export async function namespaceArguments(inputs, temporary) {
  // /tmp is private, but explicit inputs (including a symlinked venv interpreter)
  // may themselves live in the host /tmp. Restore them read-only at their original paths.
  const readOnly = [...new Set(await Promise.all([
    path.resolve(here, '../../..'), inputs['backend-root'], inputs['web-dist'],
    path.dirname(path.dirname(inputs.python)), inputs.python, inputs.chrome, process.execPath
  ].map(file => realpath(file))))]
  const evidence = await realpath(inputs.evidence)
  for (const root of readOnly) {
    assert.ok(!['/', '/tmp', '/proc', '/dev'].includes(root), `Input must not replace a namespace mount: ${root}`)
    assert.ok(evidence !== root && !evidence.startsWith(root + '/') && !root.startsWith(evidence + '/'),
      `Evidence must not overlap read-only input: ${root}`)
  }
  return ['--unshare-pid', '--unshare-net', '--unshare-ipc', '--die-with-parent',
    '--ro-bind', '/', '/', '--bind', temporary, '/tmp', '--dev', '/dev', '--proc', '/proc',
    ...readOnly.flatMap(root => ['--ro-bind', root, root]), '--bind', evidence, evidence,
    '--chdir', evidence, '--cap-drop', 'ALL', '--json-status-fd', '3']
}

export async function run(inputs) {
  process.umask(0o077)
  // Refuse reuse, symlinks and stale receipts; never chmod or overwrite a caller's existing directory.
  await mkdir(inputs.evidence, { mode: 0o700 })
  inputs = { ...inputs, evidence: await realpath(inputs.evidence) }
  let summary = initialSummary(inputs)
  const receipt = path.join(inputs.evidence, 'summary.json')
  await writeFile(receipt, JSON.stringify(summary, null, 2))
  let interrupted, namespacePid, child, timer
  const stop = signal => {
    interrupted = signal
    // Kill namespace init, not the host bwrap supervisor: its wait is the cleanup proof.
    // Linux kills/reaps every member, including fast double-fork/setsid descendants.
    if (namespacePid && child?.exitCode === null && child.signalCode === null) {
      try { process.kill(namespacePid, 'SIGKILL') }
      catch (error) { if (error.code !== 'ESRCH') throw error }
    } else if (!namespacePid) child?.kill('SIGKILL')
  }
  process.on('SIGINT', stop)
  process.on('SIGTERM', stop)
  try {
    inputs = { ...inputs,
      'backend-root': await realpath(inputs['backend-root']), 'web-dist': await realpath(inputs['web-dist']),
      chrome: await realpath(inputs.chrome),
      // Resolve directory aliases without replacing a venv's symlinked executable with bare Python.
      python: path.join(await realpath(path.dirname(inputs.python)), path.basename(inputs.python)) }
    const temporary = await mkdtemp(path.join(inputs.evidence, 'tmp-'))
    const args = await namespaceArguments(inputs, temporary)
    const entry = `import { runGate } from ${JSON.stringify(import.meta.url)}; process.exitCode = (await runGate(JSON.parse(process.argv[1]))).status === 'AWAITING_NAMESPACE_EXIT' ? 0 : 1`
    assert.ok(!interrupted, `Interrupted by ${interrupted}`)
    child = spawn('/usr/bin/bwrap', [...args, process.execPath, '--input-type=module', '-e', entry, JSON.stringify(inputs)], {
      cwd: inputs.evidence, env: childEnvironment(inputs.evidence, inputs), detached: true,
      stdio: ['ignore', 'inherit', 'inherit', 'pipe']
    })
    let pending = ''
    child.stdio[3].on('data', bytes => {
      pending += bytes
      const lines = pending.split('\n')
      pending = lines.pop()
      for (const line of lines.filter(Boolean)) {
        const status = JSON.parse(line)
        if (status['child-pid']) {
          namespacePid = status['child-pid']
          if (interrupted) stop(interrupted)
        }
      }
    })
    // Five sequential groups have a 20-minute bound each; leave room for fixture setup/teardown.
    timer = setTimeout(() => stop('whole-gate timeout'), 110 * 60 * 1000)
    const exit = await new Promise((resolve, reject) => {
      child.once('error', reject)
      child.once('exit', (code, signal) => resolve({ code, signal }))
    })
    summary = JSON.parse(await readFile(receipt, 'utf8'))
    summary.isolation = { network: 'private-loopback', temporaryRoot: temporary, namespaceTmp: '/tmp' }
    summary.cleanup = { scope: 'whole-gate-pid-namespace', confirmed: Boolean(namespacePid) && exit.signal === null,
      forced: Boolean(interrupted), ...exit }
    assert.ok(summary.cleanup.confirmed, 'Namespace exit could not be confirmed')
    assert.ok(!interrupted, `Interrupted by ${interrupted}`)
    assert.equal(exit.code, 0, `Namespaced gate exited ${exit.code}`)
    assert.equal(summary.status, 'AWAITING_NAMESPACE_EXIT', 'Missing successful scenario receipt')
    summary.status = 'PASS'
  } catch (error) {
    summary.status = 'FAIL'
    summary.error = [summary.error, error.stack].filter(Boolean).join('\n')
  } finally {
    clearTimeout(timer)
    process.off('SIGINT', stop)
    process.off('SIGTERM', stop)
    summary.finished = new Date().toISOString()
    await writeFile(receipt, JSON.stringify(summary, null, 2))
  }
  console.log(`${summary.status}; private evidence: ${inputs.evidence}`)
  return summary
}

// Only the outer run() publishes whole-gate cleanup after the kernel namespace has exited.
export async function runGate(inputs) {
  process.umask(0o077)
  const evidence = inputs.evidence
  const summary = initialSummary(inputs)
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
    summary.status = 'AWAITING_NAMESPACE_EXIT'
  } catch (error) {
    summary.error = error.stack
  } finally {
    process.off('SIGINT', onSignal)
    process.off('SIGTERM', onSignal)
    summary.finished = new Date().toISOString()
    await writeFile(path.join(evidence, 'summary.json'), JSON.stringify(summary, null, 2))
  }
  return summary
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.slice(2).includes('--help')) console.log(usage)
  else {
    try { process.exitCode = (await run(options(process.argv.slice(2)))).status === 'PASS' ? 0 : 1 }
    catch (error) { console.error(error.message); process.exitCode = 1 }
  }
}
