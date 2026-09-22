import assert from 'node:assert/strict'
import { execFileSync, spawn, spawnSync } from 'node:child_process'
import { copyFile, mkdir, mkdtemp, readFile, rm, stat, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { childEnvironment, namespaceArguments, options, run } from './compat.mjs'
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

test('direct-child teardown escalates honestly and leaves unrelated children alone', async () => {
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
    const stubborn = launch("process.on('SIGTERM', () => {}); console.log('READY'); setInterval(() => {}, 1000)", 'stubborn.log')
    await Promise.all(owned.map(child => child.ready))
    await waitFor(() => stubborn.output.includes('READY'), 5000, 'signal handler readiness')
    const cleanup = await stopProcess(stubborn, 100)
    assert.equal(cleanup.confirmed, true)
    assert.equal(cleanup.scope, 'direct-child')
    assert.match(cleanup.descendants, /not-confirmed/)
    assert.deepEqual(cleanup.groupSignals, ['SIGTERM', 'SIGKILL'])
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

test('whole-gate namespace owns a fast double-fork child, shares only private loopback, and protects inputs', async () => {
  const temp = await mkdtemp(path.join(os.tmpdir(), 'compat-namespace-test-'))
  const inputs = { 'backend-root': path.join(temp, 'stock'), 'web-dist': path.join(temp, 'bundle'),
    python: path.join(temp, 'venv/bin/python'), chrome: process.execPath, evidence: path.join(temp, 'evidence') }
  const scratch = path.join(temp, 'scratch')
  let child, namespacePid
  try {
    for (const dir of [inputs['backend-root'], inputs['web-dist'], path.dirname(inputs.python), inputs.evidence, scratch]) {
      await mkdir(dir, { recursive: true })
    }
    // Keep venv symlinks intact while both the venv and actual interpreter stay read-only.
    const testInterpreter = path.join(temp, 'test-python')
    await copyFile('/usr/bin/python3', testInterpreter)
    await symlink(testInterpreter, inputs.python)
    const protectedFiles = [path.join(inputs['backend-root'], 'stock.py'), path.join(inputs['web-dist'], 'index.html'),
      path.join(temp, 'venv/pyvenv.cfg')]
    for (const file of protectedFiles) await writeFile(file, 'fixture')
    const args = await namespaceArguments(inputs, scratch)
    const hostNetwork = execFileSync('readlink', ['/proc/self/ns/net'], { encoding: 'utf8' }).trim()
    for (const force of [false, true]) {
      const lock = path.join(inputs.evidence, `lock-${force}`)
      const release = path.join(inputs.evidence, `release-${force}`)
      // Neither intermediate parent waits for a sampler, readiness, or its child.
      const detached = `import os,fcntl,time,signal
if os.fork(): os._exit(0)
os.setsid()
if os.fork(): os._exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
f=open(${JSON.stringify(lock)}, 'w')
fcntl.flock(f, fcntl.LOCK_EX)
f.write('locked'); f.flush()
time.sleep(120)
`
      const code = `
        import assert from 'node:assert/strict';
        import { spawn, spawnSync } from 'node:child_process';
        import { openSync, writeFileSync, existsSync, readlinkSync } from 'node:fs';
        import { networkInterfaces } from 'node:os';
        import net from 'node:net';
        assert.notEqual(readlinkSync('/proc/self/ns/net'), ${JSON.stringify(hostNetwork)});
        assert.deepEqual(Object.keys(networkInterfaces()), ['lo']);
        for (const file of ${JSON.stringify([...protectedFiles, inputs.python])}) {
          assert.throws(() => openSync(file, 'r+'), { code: 'EROFS' });
        }
        writeFileSync('/tmp/disposable', 'private');
        const server = net.createServer(socket => socket.end('shared-loopback'));
        await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
        const client = spawn(${JSON.stringify(inputs.python)}, ['-c',
          'import socket,sys; s=socket.create_connection(("127.0.0.1",int(sys.argv[1]))); assert s.recv(1024)==b"shared-loopback"', String(server.address().port)], { stdio: 'inherit' });
        assert.equal(await new Promise(resolve => client.on('exit', resolve)), 0);
        server.close();
        const launcher = spawnSync(${JSON.stringify(inputs.python)}, ['-c', ${JSON.stringify(detached)}], { stdio: 'ignore' });
        assert.equal(launcher.status, 0);
        while (!existsSync(${JSON.stringify(lock)})) await new Promise(resolve => setTimeout(resolve, 5));
        console.log('DETACHED');
        while (!existsSync(${JSON.stringify(release)})) await new Promise(resolve => setTimeout(resolve, 5));
        process.exit(0);
      `
      namespacePid = undefined
      child = spawn('/usr/bin/bwrap', [...args, process.execPath, '--input-type=module', '-e', code], {
        env: childEnvironment(inputs.evidence, inputs), stdio: ['ignore', 'pipe', 'pipe', 'pipe']
      })
      let output = '', errors = '', status = ''
      child.stdout.on('data', bytes => { output += bytes })
      child.stderr.on('data', bytes => { errors += bytes })
      child.stdio[3].on('data', bytes => {
        status += bytes
        const line = status.split('\n')[0]
        if (status.includes('\n')) namespacePid = JSON.parse(line)['child-pid']
      })
      await waitFor(async () => {
        assert.equal(child.exitCode, null, errors)
        return output.includes('DETACHED') && (await readFile(lock, 'utf8')) === 'locked'
      }, 10000, 'fast detached child holds kernel lock after both parents exited')
      const probe = () => spawnSync('/usr/bin/python3', ['-c',
        'import fcntl,sys; f=open(sys.argv[1]); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)', lock])
      assert.notEqual(probe().status, 0, 'descendant must still be alive before gate exit')
      if (force) process.kill(namespacePid, 'SIGKILL')
      else await writeFile(release, 'exit gate')
      await waitFor(() => child.exitCode !== null, 10000, 'bwrap waits for namespace exit')
      assert.equal(child.exitCode, force ? 137 : 0, errors)
      assert.equal(probe().status, 0, 'namespace exit must release the detached child\'s kernel lock')
    }
    assert.equal(await readFile(path.join(scratch, 'disposable'), 'utf8'), 'private')
    for (const file of protectedFiles) assert.equal(await readFile(file, 'utf8'), 'fixture')
    // Exercise the actual outer wrapper too, stopping at stdlib-only source preflight.
    await writeFile(path.join(inputs['backend-root'], '.env'), 'NOT_A_REAL_SECRET=test\n')
    await symlink(path.join(temp, 'venv'), path.join(temp, 'venv-alias'))
    const failed = await run({ ...inputs, python: path.join(temp, 'venv-alias/bin/python'), evidence: path.join(temp, 'failed-gate') })
    assert.equal(failed.status, 'FAIL')
    assert.match(failed.error, /\.env/)
    assert.equal(failed.inputs.python, inputs.python)
    assert.equal(failed.cleanup.confirmed, true)
    assert.equal(failed.cleanup.scope, 'whole-gate-pid-namespace')
    assert.ok(failed.groups.every(group => group.status === 'SKIP'))
  } finally {
    if (child && child.exitCode === null && namespacePid) {
      process.kill(namespacePid, 'SIGKILL')
      await waitFor(() => child.exitCode !== null, 10000, 'failed test namespace cleanup')
    }
    await rm(temp, { recursive: true, force: true })
  }
})
