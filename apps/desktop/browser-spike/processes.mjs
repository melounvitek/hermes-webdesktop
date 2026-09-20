import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { readdir, readFile } from 'node:fs/promises'
import { setTimeout as delay } from 'node:timers/promises'

// Linux process groups keep fixture/backend ownership even if the launcher dies.
async function processes() {
  const rows = await Promise.all((await readdir('/proc')).filter(name => /^\d+$/.test(name)).map(async pid => {
    try {
      const text = await readFile(`/proc/${pid}/stat`, 'utf8')
      const fields = text.slice(text.lastIndexOf(')') + 2).split(' ')
      return { pid: Number(pid), state: fields[0], parent: Number(fields[1]), group: Number(fields[2]), born: fields[19] }
    } catch (error) {
      if (['ENOENT', 'ESRCH'].includes(error.code)) return null
      throw error
    }
  }))
  return rows.filter(Boolean)
}

export async function waitFor(probe, timeout, label) {
  const deadline = Date.now() + timeout
  do {
    const value = await probe()
    if (value) return value
    await delay(50)
  } while (Date.now() < deadline)
  throw new Error(`Timed out: ${label}`)
}

async function trackGroups(owned) {
  const rows = await processes()
  const descendants = new Set(rows.filter(row => owned.groups.has(row.group)).map(row => row.pid))
  let size
  do {
    size = descendants.size
    for (const row of rows) if (descendants.has(row.parent)) descendants.add(row.pid)
  } while (descendants.size !== size)
  // Playwright starts Chromium in a separate group; capture it while its parent is alive.
  for (const row of rows) {
    if (descendants.has(row.pid) && row.group === row.pid && !owned.groups.has(row.pid)) {
      owned.groups.set(row.pid, row.born)
    }
  }
  return rows
}

export function startProcess(command, args, { cwd, env, logPath, announce = false }) {
  const child = spawn(command, args, { cwd, env, detached: true, stdio: ['ignore', 'pipe', 'pipe'] })
  const log = createWriteStream(logPath, { flags: 'wx', mode: 0o600 })
  const owned = { child, log, groups: new Map(), output: '', failure: null, closed: false }
  let tracking = false
  owned.tracker = setInterval(() => {
    if (tracking) return
    tracking = true
    void trackGroups(owned).catch(error => { owned.failure = error }).finally(() => { tracking = false })
  }, 250)
  owned.tracker.unref()
  child.stdout.pipe(log, { end: false })
  child.stderr.pipe(log, { end: false })
  log.on('error', error => { owned.failure = error })
  child.on('error', error => { owned.failure = error })
  child.on('close', () => { owned.closed = true; log.end() })
  let pending = ''
  child.stdout.on('data', bytes => {
    owned.output = (owned.output + bytes).slice(-1024 * 1024)
    pending += bytes
    const lines = pending.split('\n')
    pending = lines.pop()
    for (const line of lines) {
      if (!line.startsWith('OWNED ')) continue
      try {
        const record = JSON.parse(line.slice(6))
        // Recovery announces its own directly spawned fixture groups, never arbitrary PIDs.
        assert.equal(record.parent, child.pid)
        assert.ok(Number.isSafeInteger(record.pid) && record.pid > 1)
        owned.groups.set(record.pid, record.born)
      } catch (error) { owned.failure = error }
    }
  })
  owned.ready = (async () => {
    const row = await waitFor(async () => {
      if (owned.failure) throw owned.failure
      if (owned.closed) throw new Error(`Child exited before ownership was recorded: ${command}`)
      return (await processes()).find(row => row.pid === child.pid)
    }, 5000, 'child ownership')
    assert.equal(row.parent, process.pid)
    assert.equal(row.group, child.pid)
    owned.groups.set(row.pid, row.born)
    if (announce) console.log('OWNED ' + JSON.stringify(row))
  })()
  // Install the rejection handler immediately; callers can still await the original promise.
  owned.ready.catch(error => { owned.failure = error })
  return owned
}

export async function stopProcess(owned, grace = 20000) {
  await owned.ready.catch(() => {})
  const live = async () => {
    const rows = await trackGroups(owned)
    for (const [pid, born] of owned.groups) {
      const leader = rows.find(row => row.pid === pid)
      assert.ok(!leader || leader.born === born, `Refusing reused process group ${pid}`)
    }
    return rows.filter(row => row.state !== 'Z' && owned.groups.has(row.group))
  }
  const signal = async name => {
    for (const group of new Set((await live()).map(row => row.group))) {
      try { process.kill(-group, name) } catch (error) { if (error.code !== 'ESRCH') throw error }
    }
  }
  let confirmed = false
  try {
    await signal('SIGTERM')
    let forced = false
    try {
      await waitFor(async () => !(await live()).length && owned.closed, grace, 'owned children to stop')
    } catch {
      forced = true
      await signal('SIGKILL')
      await waitFor(async () => !(await live()).length && owned.closed, 5000, 'SIGKILL cleanup confirmation')
    }
    confirmed = true
    return { confirmed, forced, groups: [...owned.groups.keys()] }
  } finally {
    clearInterval(owned.tracker)
    if (!confirmed) {
      // An unconfirmed cleanup must produce a failed receipt, not hang the runner's event loop.
      owned.child.unref()
      owned.child.stdout.destroy()
      owned.child.stderr.destroy()
      owned.log.end()
    }
  }
}
