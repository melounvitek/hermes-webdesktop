import { spawn } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { setTimeout as delay } from 'node:timers/promises'

export async function waitFor(probe, timeout, label) {
  const deadline = Date.now() + timeout
  do {
    const value = await probe()
    if (value) return value
    await delay(50)
  } while (Date.now() < deadline)
  throw new Error(`Timed out: ${label}`)
}

export function startProcess(command, args, { cwd, env, logPath }) {
  const child = spawn(command, args, { cwd, env, detached: true, stdio: ['ignore', 'pipe', 'pipe'] })
  const log = createWriteStream(logPath, { flags: 'wx', mode: 0o600 })
  const owned = { child, log, output: '', failure: null, closed: false }
  child.stdout.pipe(log, { end: false })
  child.stderr.pipe(log, { end: false })
  log.on('error', error => { owned.failure = error })
  child.on('error', error => { owned.failure = error })
  child.on('close', () => { owned.closed = true; log.end() })
  child.stdout.on('data', bytes => { owned.output = (owned.output + bytes).slice(-1024 * 1024) })
  owned.ready = new Promise((resolve, reject) => {
    child.once('spawn', resolve)
    child.once('error', reject)
  })
  owned.ready.catch(error => { owned.failure = error })
  return owned
}

export async function stopProcess(owned, grace = 20000) {
  await owned.ready.catch(() => {})
  const exited = () => owned.closed || owned.child.exitCode !== null || owned.child.signalCode !== null
  const signals = []
  const signal = name => {
    // Only address a group while its directly owned leader is still running.
    // Detached descendants are contained by the whole gate's PID namespace,
    // not by sampling ancestry or trusting an exited leader's reusable PID.
    if (exited() || !owned.child.pid) return
    try { process.kill(-owned.child.pid, name); signals.push(name) }
    catch (error) { if (error.code !== 'ESRCH') throw error }
  }
  try {
    signal('SIGTERM')
    try { await waitFor(exited, grace, 'direct child to stop') }
    catch {
      signal('SIGKILL')
      await waitFor(exited, 5000, 'direct child SIGKILL confirmation')
    }
    return { confirmed: true, scope: 'direct-child', pid: owned.child.pid ?? null,
      groupSignals: signals, descendants: 'not-confirmed; deferred to whole-gate PID namespace exit' }
  } finally {
    // A detached descendant can retain these pipes after the leader exits.
    owned.child.unref()
    owned.child.stdout.destroy()
    owned.child.stderr.destroy()
    owned.log.end()
  }
}
