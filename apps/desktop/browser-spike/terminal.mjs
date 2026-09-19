import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdir, readFile, realpath, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium, expect } from '@playwright/test'

// Build first, then: node browser-spike/terminal.mjs <runtime.json> [artifact directory]
// backend.py --terminal-plugin supplies stock Hermes plus the optional plugin.
const runtimePath = process.argv[2]
assert.ok(runtimePath, 'Supply a disposable backend.py runtime.json')
const base = '/api/plugins/browser-terminal'
const results = [], frames = [], requests = [], errors = [], blocked = [], processes = []
const delay = ms => new Promise(resolve => setTimeout(resolve, ms))
const started = Date.now()
let browser, page, missingFixture, artifacts, currentCheck, sequence = 0

async function fixture(file, plugin) {
  const r = JSON.parse(await readFile(file, 'utf8'))
  assert.equal(new URL(r.url).hostname, '127.0.0.1')
  assert.equal(new URL(r.model_url).hostname, '127.0.0.1')
  assert.equal(r.public_url, null)
  assert.equal(r.terminal_plugin, plugin)
  assert.equal(path.dirname(r.run_dir), os.tmpdir())
  assert.ok(path.basename(r.run_dir).startsWith('hermes-browser-spike-'))
  assert.equal(await realpath(r.run_dir), r.run_dir)
  assert.equal(path.resolve(file), path.join(r.run_dir, 'runtime.json'))
  assert.equal(r.home, path.join(r.run_dir, 'home'))
  assert.equal(r.hermes_home, path.join(r.home, '.hermes'))
  assert.equal(await realpath(r.home), r.home)
  assert.equal(await realpath(r.hermes_home), r.hermes_home)
  const config = JSON.parse(await readFile(path.join(r.hermes_home, 'config.yaml'), 'utf8'))
  assert.equal(config.model.default, 'browser-spike-local')
  assert.equal(config.model.provider, 'custom')
  assert.equal(config.model.base_url, r.model_url)
  const env = await processEnv(r.backend_pid)
  assert.equal(env.HOME, r.home)
  assert.equal(env.HERMES_HOME, r.hermes_home)
  assert.equal(await realpath(`/proc/${r.backend_pid}/cwd`), r.home)
  return { ...r, config }
}

async function processEnv(pid) {
  return Object.fromEntries((await readFile(`/proc/${pid}/environ`, 'utf8')).split('\0').filter(Boolean).map(s => {
    const i = s.indexOf('=')
    return [s.slice(0, i), s.slice(i + 1)]
  }))
}

async function alive(pid) {
  try {
    const stat = await readFile(`/proc/${pid}/stat`, 'utf8')
    return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[0] !== 'Z'
  } catch (error) {
    if (error.code === 'ENOENT') return false
    throw error
  }
}

function safeUrl(value) {
  const url = new URL(value)
  for (const key of ['ticket', 'token']) if (url.searchParams.has(key)) url.searchParams.set(key, '[redacted]')
  return url.href
}

async function snapshot(name) {
  await page.screenshot({ path: path.join(artifacts, `${name}.png`) })
  await writeFile(path.join(artifacts, `${name}.aria.txt`), await page.locator('body').ariaSnapshot())
}

async function check(name, body) {
  currentCheck = name
  await body()
  results.push({ name, status: 'PASS', elapsedMs: Date.now() - started })
  console.log(`PASS: ${name}`)
  await snapshot(name)
}

async function newPage(r) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
  const sockets = [], tickets = new Map(), sessions = []
  await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: r.url })
  await context.route('**/*', route => {
    const url = new URL(route.request().url())
    if (url.origin === r.url || ['data:', 'blob:'].includes(url.protocol)) return route.continue()
    blocked.push({ url: safeUrl(url.href), resourceType: route.request().resourceType() })
    return route.abort('blockedbyclient')
  })
  await context.routeWebSocket('**/*', async ws => {
    const url = new URL(ws.url())
    if (url.origin !== r.url.replace('http:', 'ws:')) {
      blocked.push({ url: safeUrl(ws.url()), resourceType: 'websocket' })
      return ws.close()
    }
    if (url.pathname === `${base}/ws`) {
      // Response-body observation races the renderer's next WebSocket dial.
      await expect.poll(() => tickets.has(url.searchParams.get('ticket'))).toBe(true)
    }
    const server = ws.connectToServer()
    if (url.pathname !== `${base}/ws`) {
      ws.onMessage(message => {
        const rpc = JSON.parse(String(message))
        frames.push({ at: Date.now(), kind: 'rpc', direction: 'sent', method: rpc.method })
        server.send(message)
      })
      return
    }
    const socket = { number: sockets.length + 1, sid: tickets.get(url.searchParams.get('ticket')), output: '', decoder: new TextDecoder(), ws, server }
    assert.ok(socket.sid, 'Terminal WebSocket must consume a freshly minted ticket')
    sockets.push(socket)
    frames.push({ at: Date.now(), socket: socket.number, sid: socket.sid, kind: 'open', url: safeUrl(ws.url()) })
    for (const [source, destination, direction] of [[ws, server, 'sent'], [server, ws, 'received']]) {
      source.onMessage(message => {
        const data = Buffer.isBuffer(message)
          ? direction === 'received' ? socket.decoder.decode(message, { stream: true }) : message.toString('utf8')
          : message
        frames.push({ at: Date.now(), socket: socket.number, sid: socket.sid, direction, binary: Buffer.isBuffer(message), bytes: Buffer.byteLength(message), data })
        if (direction === 'received') socket.output += data
        destination.send(message)
      })
    }
  })
  const p = await context.newPage()
  p.setDefaultTimeout(15000)
  p.on('pageerror', error => errors.push(error.stack))
  p.on('response', async response => {
    const url = new URL(response.url())
    const record = { at: Date.now(), fixture: r.terminal_plugin ? 'plugin' : 'missing', method: response.request().method(), url: safeUrl(url.href), status: response.status() }
    requests.push(record)
    if (!url.pathname.startsWith(base) || !response.ok()) return
    const body = await response.json().catch(() => null)
    if (url.pathname.endsWith('/ticket')) {
      const sid = url.pathname.split('/').at(-2)
      tickets.set(body.ticket, sid)
      record.body = { ticket: '[redacted]' }
    } else if (url.pathname === `${base}/sessions` && body?.id) {
      record.body = body
      sessions.push({ ...body, profile: url.searchParams.get('profile') })
    }
  })
  await p.goto(r.url)
  await expect(p.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible({ timeout: 60000 })
  return { page: p, sockets, sessions }
}

const terminal = () => page.locator('[data-terminal]:visible')
const input = () => terminal().locator('.xterm-helper-textarea')
const tabs = () => page.getByRole('tablist', { name: 'Terminals', exact: true }).getByRole('tab')
const composer = () => page.getByRole('textbox', { name: 'Message', exact: true })

async function selectProfile(profile) {
  const button = page.getByRole('button', { name: profile, exact: true })
  if (await button.isVisible()) {
    await button.click()
    await expect(button).toHaveAttribute('aria-pressed', 'true')
  } else {
    const dropdown = page.getByRole('button', { name: 'Profiles', exact: true })
    await dropdown.click()
    await page.getByRole('menuitemradio', { name: profile, exact: true }).click()
    await expect(dropdown).toContainText(profile)
  }
  await expect(composer()).toBeEditable()
}

async function openTerminal(state) {
  const count = state.sessions.length
  await page.getByRole('button', { name: 'New terminal', exact: true }).first().click()
  await expect.poll(() => state.sessions.length).toBe(count + 1)
  const session = state.sessions.at(-1)
  await expect.poll(() => state.sockets.some(s => s.sid === session.id && /[%#$>]\s/.test(s.output))).toBe(true)
  await expect(input()).toBeVisible()
  return session
}

async function command(state, sid, text) {
  const socket = state.sockets.findLast(s => s.sid === sid)
  const mark = ++sequence
  const start = socket.output.length
  // Markers are assembled by printf, so an echoed command cannot satisfy them.
  await terminal().locator('.xterm-screen').click()
  await expect(input()).toBeFocused()
  await page.keyboard.insertText(`printf '\\n__begin_%s__\\n' '${mark}'; ${text}; printf '\\n__done_%s__\\n' '${mark}'`)
  await input().press('Enter')
  await expect.poll(() => socket.output.slice(start).includes(`__done_${mark}__`)).toBe(true)
  return socket.output.slice(start).split(`__begin_${mark}__`).at(-1).split(`__done_${mark}__`)[0]
}

async function assertInitialPromptVisible(name) {
  // Output arrival precedes xterm's paint frame (the helper starts at 1×1).
  await expect.poll(() => input().evaluate(element => element.getBoundingClientRect().height)).toBeGreaterThan(2)
  const geometry = await terminal().locator('.xterm-screen').evaluate(screen => {
    const box = screen.getBoundingClientRect()
    const caret = screen.querySelector('.xterm-helper-textarea').getBoundingClientRect()
    const height = caret.height
    const ancestors = []
    for (let node = screen; node; node = node.parentElement) {
      const rect = node.getBoundingClientRect(), css = getComputedStyle(node)
      ancestors.push({ rect: rect.toJSON(), scrollTop: node.scrollTop, overflow: css.overflow })
    }
    // Hit-test the first row, not merely its outer terminal pane.
    const uncovered = [2, height / 2, height - 2].every(y =>
      [2, box.width / 2, box.width - 2].every(x =>
        screen.contains(document.elementFromPoint(box.left + x, box.top + y))))
    return { x: box.x, y: box.y, width: box.width, height, caret: caret.toJSON(), uncovered, ancestors }
  })
  await writeFile(path.join(artifacts, `${name}-geometry.json`), JSON.stringify(geometry, null, 2))
  assert.ok(geometry.uncovered, 'The first row must not be covered by chrome')
  for (const ancestor of geometry.ancestors) {
    if (!/hidden|clip|auto|scroll/.test(ancestor.overflow)) continue
    assert.ok(geometry.y >= ancestor.rect.top - 1 && geometry.y + geometry.height <= ancestor.rect.bottom + 1,
      'The first row must fit inside every clipping ancestor')
  }
  // Inspect the actual composed screenshot, not canvas/buffer text: a selectable
  // prompt can still be clipped by a mismatched WebGL viewport.
  const png = await page.screenshot({ path: path.join(artifacts, `${name}.png`), animations: 'disabled' })
  const ink = await page.evaluate(async ({ image, row }) => {
    const bitmap = await createImageBitmap(await (await fetch(image)).blob())
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width; canvas.height = bitmap.height
    const ctx = canvas.getContext('2d'); ctx.drawImage(bitmap, 0, 0)
    const x = Math.ceil(row.x), y = Math.ceil(row.y)
    const width = Math.floor(row.width) - 2, height = Math.floor(row.height)
    const { data } = ctx.getImageData(x, y, width, height)
    const background = data.slice((width - 1) * 4, width * 4)
    let left = width, right = -1, top = height, bottom = -1
    // Exclude the cursor: its full-height block must not disguise clipped text.
    for (let py = 0; py < height; py++) for (let px = 0; px < Math.min(width, Math.floor(row.caret.left) - x); px++) {
      const i = (py * width + px) * 4
      if (Math.max(...[0, 1, 2].map(c => Math.abs(data[i + c] - background[c]))) < 60) continue
      left = Math.min(left, px); right = Math.max(right, px)
      top = Math.min(top, py); bottom = Math.max(bottom, py)
    }
    bitmap.close()
    return { left: x + left, top: y + top, right: x + right, bottom: y + bottom,
      width: Math.max(0, right - left + 1), height: Math.max(0, bottom - top + 1) }
  }, { image: `data:image/png;base64,${png.toString('base64')}`, row: geometry })
  await writeFile(path.join(artifacts, `${name}-visibility.json`), JSON.stringify({ geometry, ink }, null, 2))
  assert.ok(ink.width >= geometry.height * 3 && ink.height >= geometry.height * 0.4,
    `Initial prompt glyphs must paint across the first row, not just a cursor sliver: ${JSON.stringify(ink)}`)
}

async function visibleBuffer() {
  // Mouse selection reads xterm buffer cells via its public textarea, NOT proof
  // that glyphs are painted. assertInitialPromptVisible checks the screen separately.
  const screen = terminal().locator('.xterm-screen')
  const box = await screen.boundingBox()
  await page.mouse.move(box.x + 1, box.y + 1)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width - 2, box.y + box.height - 2, { steps: 10 })
  await page.mouse.up()
  await expect.poll(() => input().inputValue()).not.toBe('')
  const value = await input().inputValue()
  await writeFile(path.join(artifacts, `selection-${sequence}.txt`), value)
  await screen.click()
  return value
}

async function identity(state, session, r, home, cwd, variable) {
  const output = await command(state, session.id, `printf 'PID=%s\\nHOME=%s\\nHERMES_HOME=%s\\nPWD=%s\\nKEEP=%s\\n' "$$" "$HOME" "$HERMES_HOME" "$PWD" "$SPIKE_KEEP"`)
  const pid = Number(output.match(/PID=(\d+)/)?.[1])
  assert.ok(pid > 1, output)
  assert.ok(output.includes(`HOME=${r.home}\r\n`), output)
  assert.ok(output.includes(`HERMES_HOME=${home}\r\n`), output)
  assert.ok(output.includes(`PWD=${cwd}\r\n`), output)
  if (variable !== undefined) assert.ok(output.includes(`KEEP=${variable}\r\n`), output)
  const env = await processEnv(pid)
  assert.equal(env.HOME, r.home)
  assert.equal(env.HERMES_HOME, home)
  assert.equal(await realpath(`/proc/${pid}/cwd`), cwd)
  processes.push({ pid, profile: session.profile, home, cwd, at: Date.now() })
  return pid
}

async function launchMissing(r) {
  const script = fileURLToPath(new URL('./backend.py', import.meta.url))
  const child = spawn(r.command[0], [script, '--web-dist', path.resolve(path.dirname(script), '../dist-browser'), '--python', r.command[0]], { stdio: ['ignore', 'pipe', 'pipe'] })
  missingFixture = child
  let log = ''
  child.stdout.on('data', chunk => { log += chunk })
  child.stderr.on('data', chunk => { log += chunk })
  await expect.poll(() => /READY (.+)/.test(log) || child.exitCode !== null, { timeout: 100000 }).toBe(true)
  assert.equal(child.exitCode, null, log)
  const ready = JSON.parse(log.match(/READY (.+)/)[1])
  return fixture(path.join(ready.run_dir, 'runtime.json'), false)
}

const runtime = await fixture(runtimePath, true)
artifacts = path.resolve(process.argv[3] || path.join(runtime.run_dir, `terminal-evidence-${started}`))
assert.ok(artifacts.startsWith(runtime.run_dir + path.sep), 'Keep evidence inside the disposable fixture')
await mkdir(artifacts, { recursive: true })
assert.equal(await realpath(artifacts), artifacts)
const profiles = [`terminal-a-${started}`, `terminal-b-${started}`]
const homes = profiles.map(profile => path.join(runtime.hermes_home, 'profiles', profile))
for (const home of homes) {
  await mkdir(path.join(home, 'workspace'), { recursive: true })
  await writeFile(path.join(home, 'config.yaml'), JSON.stringify({ ...runtime.config, terminal: { backend: 'local', cwd: path.join(home, 'workspace') } }), { flag: 'wx' })
}
const modelLog = async r => readFile(path.join(r.run_dir, 'model-requests.jsonl'), 'utf8').catch(error => {
  if (error.code === 'ENOENT') return ''
  throw error
})
const modelBefore = await modelLog(runtime)
let state, a, a2, aPid, a2Pid
const baselineHome = runtime.hermes_home
const baselineCwd = runtime.home
try {
  // Chromium's Unix socket path must fit sockaddr_un (108 bytes).
  const chromeTemp = path.join(runtime.run_dir, `chrome-${process.pid}`)
  await mkdir(chromeTemp)
  const previousTmp = process.env.TMPDIR
  process.env.TMPDIR = chromeTemp
  browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: true,
    env: { PATH: process.env.PATH, HOME: runtime.home, TMPDIR: chromeTemp,
      XDG_CONFIG_HOME: path.join(chromeTemp, 'config'), XDG_CACHE_HOME: path.join(chromeTemp, 'cache') }
  }).finally(() => {
    if (previousTmp === undefined) delete process.env.TMPDIR
    else process.env.TMPDIR = previousTmp
  })
  state = await newPage(runtime)
  page = state.page
  await check('first-prompt-and-unicode', async () => {
    await page.keyboard.press('Control+Backquote')
    await expect.poll(() => state.sessions.length).toBe(1)
    a = state.sessions[0]
    await expect.poll(() => state.sockets.find(s => s.sid === a.id)?.output || '').toMatch(/[%#$>]\s/)
    await expect(input()).toBeVisible()
    await assertInitialPromptVisible('first-prompt')
    const original = await page.evaluate(() => window.hermesDesktop.zoom.get())
    try {
      for (const percent of [90, 100, 125]) {
        await page.evaluate(value => window.hermesDesktop.zoom.setPercent(value), percent)
        await openTerminal(state)
        await assertInitialPromptVisible(`fresh-first-prompt-${percent}`)
        await tabs().last().click({ button: 'right' })
        await page.getByRole('menuitem', { name: 'Close', exact: true }).click()
        await expect(tabs()).toHaveCount(1)
      }
    } finally {
      await page.evaluate(value => window.hermesDesktop.zoom.setPercent(value), original.percent)
    }
    assert.ok(!frames.some(f => f.direction === 'sent' && f.binary), 'First-prompt visibility must pass without shell input')
    assert.match(await visibleBuffer(), /[%#$>]\s/)
    aPid = await identity(state, a, runtime, baselineHome, baselineCwd, '')
    assert.match(await command(state, a.id, "stty -echo; SPIKE_KEEP='A-owned'; printf 'Unicode: žluťoučký 東京 🦀\\n'"), /Unicode: žluťoučký 東京 🦀/)
    assert.ok((await visibleBuffer()).includes('Unicode: žluťoučký 東京 🦀'))
  })
  try {
    await check('terminal-overlay-follows-pane', async () => {
      const original = await page.evaluate(() => window.hermesDesktop.zoom.get())
      const samples = []
      try {
        for (const percent of [90, 100, 125, original.percent]) {
          // Use the real bridge's UI scale implementation, not a transport stub.
          await page.evaluate(value => window.hermesDesktop.zoom.setPercent(value), percent)
          let geometry
          await expect.poll(async () => {
            geometry = await page.locator('[data-persistent-terminal]').evaluate(overlay => {
              const slot = document.querySelector('[data-terminal-slot]')
              return { zoom: Number(getComputedStyle(document.documentElement).zoom), overlay: overlay.getBoundingClientRect().toJSON(), slot: slot.getBoundingClientRect().toJSON(), screen: overlay.querySelector('.xterm-screen').getBoundingClientRect().toJSON() }
            })
            return geometry.zoom === percent / 100 && geometry.slot.width > 0 && geometry.slot.height > 0 &&
              ['left', 'top', 'right', 'bottom'].every(edge => Math.abs(geometry.overlay[edge] - geometry.slot[edge]) <= 2) &&
              geometry.screen.width > 0 && geometry.screen.height > 0 &&
              ['left', 'top'].every(edge => geometry.screen[edge] >= geometry.slot[edge] - 2) &&
              ['right', 'bottom'].every(edge => geometry.screen[edge] <= geometry.slot[edge] + 2)
          }, { message: `Terminal overlay and painted screen must follow the pane at ${percent}% UI scale` }).toBe(true)
          samples.push({ percent, ...geometry })
          assert.match(await command(state, a.id, `printf 'SCALE-%s-responsive\\n' '${percent}'`), new RegExp(`SCALE-${percent}-responsive`))
          await snapshot(`terminal-scale-${percent}-${samples.length}`)
        }
      } finally {
        await page.evaluate(value => window.hermesDesktop.zoom.setPercent(value), original.percent)
        await writeFile(path.join(artifacts, 'terminal-geometry.json'), JSON.stringify(samples, null, 2))
      }
    })
  } catch (error) {
    results.push({ name: 'terminal-overlay-follows-pane', status: 'FAIL', error: error.stack })
    console.error(error.stack)
    process.exitCode = 1
    await snapshot('geometry-failure')
  }
  await check('viewport-resize-and-ctrl-c', async () => {
    const size = async () => (await command(state, a.id, 'stty size')).match(/(?:^|\n)(\d+) (\d+)\r?\n/).slice(1).map(Number)
    const before = await size()
    await page.setViewportSize({ width: 1120, height: 760 })
    await expect.poll(async () => JSON.stringify(await size())).not.toBe(JSON.stringify(before))
    const after = await size()
    assert.ok(after[0] < before[0] || after[1] < before[1], `${before} -> ${after}`)
    await page.setViewportSize({ width: 1440, height: 1000 })
    await expect.poll(async () => (await size())[1]).toBe(before[1])
    const separators = await page.getByRole('separator').evaluateAll(elements => elements.map(el => el.getBoundingClientRect().toJSON()))
    const divider = separators.find(box => box.x > 720 && box.height > box.width)
    assert.ok(divider, 'Terminal pane must have a real draggable divider')
    await page.mouse.move(divider.x + divider.width / 2, divider.y + divider.height / 2)
    await page.mouse.down()
    await page.mouse.move(divider.x - 160, divider.y + divider.height / 2, { steps: 15 })
    await page.mouse.up()
    await expect.poll(async () => (await size())[1]).toBeGreaterThan(before[1])
    const pane = await size()
    await writeFile(path.join(artifacts, 'resize.json'), JSON.stringify({ before, viewport: after, pane, separators }))
    const pidFile = path.join(artifacts, 'interrupt.pid')
    await terminal().locator('.xterm-screen').click()
    await page.keyboard.insertText(`sh -c 'echo $$ > "${pidFile}"; exec sleep 120'`)
    await input().press('Enter')
    await expect.poll(async () => readFile(pidFile, 'utf8').catch(() => '')).toMatch(/^\d+\n$/)
    const job = Number(await readFile(pidFile, 'utf8'))
    assert.equal((await processEnv(job)).HERMES_HOME, baselineHome)
    processes.push({ pid: job, kind: 'interrupted-job', home: baselineHome })
    await input().press('Control+c')
    await expect.poll(() => alive(job)).toBe(false)
    assert.match(await command(state, a.id, "printf 'INTERRUPTED-and-responsive\\n'"), /INTERRUPTED-and-responsive/)
    assert.equal(await identity(state, a, runtime, baselineHome, baselineCwd, 'A-owned'), aPid)
  })
  await check('large-clipboard-paste-real-pty', async () => {
    const payload = 'žluťoučký東京🦀'.repeat(12000)
    const bytes = Buffer.byteLength(payload)
    const digest = createHash('sha256').update(payload).digest('hex')
    // A bounded raw-stdin reader consumes inert text, never a giant shell command.
    const code = [
      'import os, sys, tty, termios, hashlib',
      'fd = sys.stdin.fileno()',
      'before = termios.tcgetattr(fd)',
      'data = bytearray()',
      'try:',
      ' tty.setraw(fd)',
      ' print("__paste_ready__", flush=True)',
      ` while len(data) < ${bytes}:`,
      `  data.extend(os.read(fd, min(65536, ${bytes} - len(data))))`,
      'finally:',
      ' termios.tcsetattr(fd, termios.TCSADRAIN, before)',
      'print("\\nPASTE_BYTES=%s SHA256=%s" % (len(data), hashlib.sha256(data).hexdigest()), flush=True)',
    ].join('\n')
    const quote = value => `'${value.replaceAll("'", "'\\''")}'`
    const socket = state.sockets.findLast(s => s.sid === a.id)
    const start = socket.output.length
    await terminal().locator('.xterm-screen').click()
    await page.keyboard.insertText(`${quote(runtime.command[0])} -c ${quote(`exec(${JSON.stringify(code)})`)}`)
    await input().press('Enter')
    await expect.poll(() => socket.output.slice(start).includes('__paste_ready__')).toBe(true)
    const frameStart = frames.length
    await page.evaluate(text => navigator.clipboard.writeText(text), payload)
    await input().press('Control+Shift+v')
    await expect.poll(() => socket.output.slice(start), { timeout: 30000 }).toContain(`PASTE_BYTES=${bytes} SHA256=${digest}`)
    const sent = frames.slice(frameStart).filter(f => f.sid === a.id && f.direction === 'sent' && f.binary)
    assert.ok(sent.length > 1, 'Large clipboard paste must traverse multiple real binary WebSocket frames')
    assert.equal(sent.reduce((sum, f) => sum + f.bytes, 0), bytes)
    await writeFile(path.join(artifacts, 'large-paste.json'), JSON.stringify({ bytes, digest, frames: sent.map(f => f.bytes) }, null, 2))
    assert.equal(await identity(state, a, runtime, baselineHome, baselineCwd, 'A-owned'), aPid)
  })
  await check('tabs-preserved-through-hide-unhide', async () => {
    a2 = await openTerminal(state)
    a2Pid = await identity(state, a2, runtime, baselineHome, baselineCwd, '')
    assert.notEqual(a2Pid, aPid)
    await command(state, a2.id, "stty -echo; SPIKE_KEEP='A-second'; printf 'ONLY-A-SECOND\\n'")
    await expect(tabs()).toHaveCount(2)
    await tabs().nth(0).click()
    assert.equal(await identity(state, a, runtime, baselineHome, baselineCwd, 'A-owned'), aPid)
    assert.ok(!(await visibleBuffer()).includes('ONLY-A-SECOND'))
    await page.getByRole('button', { name: 'Hide terminal', exact: true }).click()
    await expect(page.getByRole('tablist', { name: 'Terminals', exact: true })).not.toBeVisible()
    await snapshot('terminal-hidden')
    await composer().fill('unsent composer draft')
    await page.keyboard.press('Control+Backquote')
    assert.equal(await identity(state, a, runtime, baselineHome, baselineCwd, 'A-owned'), aPid)
    await tabs().nth(1).click()
    assert.equal(await identity(state, a2, runtime, baselineHome, baselineCwd, 'A-second'), a2Pid)
    await tabs().nth(0).click()
  })
  await check('network-reconnect-same-shell-composer-focus', async () => {
    const old = state.sockets.findLast(s => s.sid === a.id)
    const count = state.sessions.length
    await composer().click()
    await composer().fill('do not send: reconnect focus')
    await expect(composer()).toBeFocused()
    // Drop only this real socket, on both proxy legs. No bridge replacement.
    await old.server.close({ code: 1012, reason: 'E2E transient network loss' })
    await old.ws.close({ code: 1012, reason: 'E2E transient network loss' })
    await expect(page.getByRole('status').filter({ hasText: 'Reconnecting to terminal' })).toBeVisible()
    await expect.poll(() => state.sockets.filter(s => s.sid === a.id).length).toBe(2)
    await expect.poll(() => state.sockets.findLast(s => s.sid === a.id).output.includes('Unicode:')).toBe(true)
    await expect(composer()).toBeFocused()
    await expect(composer()).toHaveText('do not send: reconnect focus')
    assert.equal(state.sessions.length, count, 'Reconnect must not create a replacement shell')
    assert.equal(await identity(state, a, runtime, baselineHome, baselineCwd, 'A-owned'), aPid)
    await command(state, a.id, "printf 'RECONNECTED: žluťoučký 東京 🦀\\n'")
    assert.ok((await visibleBuffer()).includes('RECONNECTED: žluťoučký 東京 🦀'))
  })
  await check('heartbeat-beyond-60-seconds', async () => {
    const since = Date.now()
    await composer().click()
    await delay(65000)
    assert.ok(Date.now() - since > 60000)
    for (const session of [a, a2]) {
      const beats = requests.filter(r => r.at >= since && r.url.includes(`/sessions/${session.id}/heartbeat`) && r.status === 200)
      assert.ok(beats.length >= 3, `Expected sustained heartbeat for ${session.profile}: ${beats.length}`)
      assert.ok(beats.at(-1).at - beats[0].at >= 35000)
    }
    await expect(composer()).toBeFocused()
    assert.equal(await identity(state, a, runtime, baselineHome, baselineCwd, 'A-owned'), aPid)
    await tabs().nth(1).click()
    assert.equal(await identity(state, a2, runtime, baselineHome, baselineCwd, 'A-second'), a2Pid)
    await tabs().nth(0).click()
  })
  await check('explicit-close-kills-job-and-shell', async () => {
    const pidFile = path.join(artifacts, 'foreground.pid')
    const cmd = `sh -c 'echo $$ > "${pidFile}"; exec sleep 120'`
    await terminal().locator('.xterm-screen').click()
    await page.keyboard.insertText(cmd)
    await input().press('Enter')
    await expect.poll(async () => readFile(pidFile, 'utf8').catch(() => '')).toMatch(/^\d+\n$/)
    const job = Number(await readFile(pidFile, 'utf8'))
    assert.equal((await processEnv(job)).HERMES_HOME, baselineHome)
    assert.equal(await realpath(`/proc/${job}/cwd`), baselineCwd)
    processes.push({ pid: job, kind: 'foreground-job', home: baselineHome })
    await tabs().nth(0).click({ button: 'right' })
    await page.getByRole('menuitem', { name: 'Close', exact: true }).click()
    await expect(tabs()).toHaveCount(1)
    await expect.poll(() => alive(job)).toBe(false)
    await expect.poll(() => alive(aPid)).toBe(false)
    assert.equal(await identity(state, a2, runtime, baselineHome, baselineCwd, 'A-second'), a2Pid)
    await tabs().nth(0).click({ button: 'right' })
    await page.getByRole('menuitem', { name: 'Close', exact: true }).click()
    await expect.poll(() => alive(a2Pid)).toBe(false)

  })
  try {
    await check('fresh-profiles-a-b-a-no-cross-profile-display', async () => {
      await selectProfile(profiles[0])
      const count = state.sessions.length
      await page.keyboard.press('Control+Backquote')
      await expect.poll(() => state.sessions.length).toBe(count + 1)
      const first = state.sessions.at(-1)
      assert.equal(first.profile, profiles[0], 'Fresh profile selection must own the shell, not the launch profile')
      const firstPid = await identity(state, first, runtime, homes[0], path.join(homes[0], 'workspace'), '')
      await command(state, first.id, "stty -echo; SPIKE_KEEP='profile-A'; printf 'ONLY-PROFILE-A\\n'")
      const second = await openTerminal(state)
      const secondPid = await identity(state, second, runtime, homes[0], path.join(homes[0], 'workspace'), '')
      await command(state, second.id, "stty -echo; SPIKE_KEEP='profile-A-second'")
      await tabs().nth(0).click()
      await selectProfile(profiles[1])
      await expect(tabs()).toHaveCount(0)
      await expect(terminal()).toHaveCount(0)
      const other = await openTerminal(state)
      assert.equal(other.profile, profiles[1])
      const otherPid = await identity(state, other, runtime, homes[1], path.join(homes[1], 'workspace'), '')
      assert.notEqual(otherPid, firstPid)
      await command(state, other.id, "stty -echo; SPIKE_KEEP='profile-B'; printf 'ONLY-PROFILE-B\\n'")
      const otherBuffer = await visibleBuffer()
      assert.ok(!otherBuffer.includes('ONLY-PROFILE-A') && !otherBuffer.includes(homes[0]))
      await snapshot('profile-b-isolated')
      await selectProfile(profiles[0])
      await expect(tabs()).toHaveCount(2)
      const firstBuffer = await visibleBuffer()
      assert.ok(firstBuffer.includes('ONLY-PROFILE-A') && !firstBuffer.includes('ONLY-PROFILE-B') && !firstBuffer.includes(homes[1]))
      assert.equal(await identity(state, first, runtime, homes[0], path.join(homes[0], 'workspace'), 'profile-A'), firstPid)
      await tabs().nth(1).click()
      assert.equal(await identity(state, second, runtime, homes[0], path.join(homes[0], 'workspace'), 'profile-A-second'), secondPid)
      const fresh = await openTerminal(state)
      assert.equal(fresh.profile, profiles[0])
      const freshPid = await identity(state, fresh, runtime, homes[0], path.join(homes[0], 'workspace'), '')
      assert.notEqual(freshPid, firstPid, 'A new tab on return must start a fresh A shell')
    })
  } catch (error) {
    results.push({ name: 'fresh-profiles-a-b-a-no-cross-profile-display', status: 'FAIL', error: error.stack })
    console.error(error.stack)
    process.exitCode = 1
    await snapshot('profile-failure')
  }
  await check('missing-plugin-helpful-error-chat-still-usable', async () => {
    const r = await launchMissing(runtime)
    await writeFile(path.join(artifacts, 'missing-fixture.json'), JSON.stringify({ run_dir: r.run_dir, backend_pid: r.backend_pid }))
    const missing = await newPage(r)
    page = missing.page
    await expect(composer()).toBeEditable()
    await composer().fill('unsent without plugin')
    await page.keyboard.press('Control+Backquote')
    const help = page.getByText('Install and enable browser-terminal plugin on the Hermes server, then restart the dashboard.', { exact: true })
    const retry = page.getByRole('button', { name: 'Retry', exact: true })
    const intersections = []
    for (const [name, target] of [['help', help], ['retry', retry]]) {
      await target.scrollIntoViewIfNeeded()
      await expect(target).toBeVisible()
      await expect(target).toBeInViewport()
      const geometry = await target.evaluate(element => {
        const viewport = element.closest('[data-slot="scroll-area-viewport"]')
        if (!viewport) throw new Error('Missing-plugin recovery must have a scroll viewport')
        const box = element.getBoundingClientRect(), clip = viewport.getBoundingClientRect()
        const width = Math.max(0, Math.min(box.right, clip.right) - Math.max(box.left, clip.left))
        const height = Math.max(0, Math.min(box.bottom, clip.bottom) - Math.max(box.top, clip.top))
        return { target: box.toJSON(), viewport: clip.toJSON(), ratio: width * height / (box.width * box.height), scrollTop: viewport.scrollTop, scrollHeight: viewport.scrollHeight, clientHeight: viewport.clientHeight }
      })
      assert.ok(geometry.ratio > 0, `${name} must intersect its nearest scroll viewport`)
      if (name === 'retry') assert.ok(geometry.ratio >= 0.99, 'Retry must be fully reachable')
      intersections.push({ name, ...geometry })
      await snapshot(`missing-plugin-${name}-scrolled`)
    }
    assert.ok(intersections.some(g => g.scrollHeight > g.clientHeight && g.scrollTop > 0), 'Exercise actual small-pane scrolling')
    await writeFile(path.join(artifacts, 'missing-plugin-scroll.json'), JSON.stringify(intersections, null, 2))
    const retried = page.waitForResponse(response => new URL(response.url()).pathname === `${base}/sessions` && response.request().method() === 'POST')
    await retry.click()
    assert.ok([404, 405].includes((await retried).status()), 'Retry must make a real missing-plugin request')
    await help.scrollIntoViewIfNeeded()
    await expect(help).toBeVisible()
    await expect(composer()).toHaveText('unsent without plugin')
    await composer().click()
    await composer().fill('still unsent after retry')
    await expect(composer()).toBeEditable()
    await expect(composer()).toBeFocused()
    await expect(composer()).toHaveText('still unsent after retry')
    assert.equal(missing.sockets.length, 0)
    assert.equal(await modelLog(r), '')
    await snapshot('missing-plugin-composer-loaded')
  })
} catch (error) {
  results.push({ name: currentCheck || 'startup', status: 'FAIL', error: error.stack })
  console.error(error.stack)
  process.exitCode = 1
  if (page && !page.isClosed()) await snapshot('failure')
} finally {
  try {
    assert.equal(await modelLog(runtime), modelBefore, 'No model requests are permitted')
    assert.ok(!frames.some(f => f.kind === 'rpc' && f.method?.startsWith('prompt.')))
    assert.deepEqual(errors, [], 'Chromium page errors')
    assert.deepEqual(requests.filter(r => r.status >= 400 && !(r.fixture === 'missing' && new URL(r.url).pathname === `${base}/sessions` && [404, 405].includes(r.status))), [], 'Unexpected HTTP/asset failures')
    assert.deepEqual(blocked.filter(r => !['font', 'stylesheet'].includes(r.resourceType)), [], 'Unexpected outbound requests')
    results.push({ name: 'no-model-no-unexpected-network-or-assets', status: 'PASS' })
  } catch (error) {
    results.push({ name: 'no-model-no-unexpected-network-or-assets', status: 'FAIL', error: error.stack })
    process.exitCode = 1
  }
  // Cleanup only sessions minted by this browser; never terminate the supplied fixture.
  if (state) for (const session of state.sessions) {
    try {
      const response = await state.page.request.delete(`${runtime.url}${base}/sessions/${encodeURIComponent(session.id)}?profile=${encodeURIComponent(session.profile)}`, { headers: { 'X-Hermes-Session-Token': runtime.token, Origin: runtime.url } })
      assert.ok([200, 204, 404].includes(response.status()), `Cleanup returned ${response.status()}`)
    } catch (error) {
      results.push({ name: `cleanup-${session.id}`, status: 'FAIL', error: error.stack })
      process.exitCode = 1
    }
  }
  await browser?.close()
  for (const pid of new Set(processes.map(p => p.pid))) {
    const running = await alive(pid)
    if (running) process.exitCode = 1
    results.push({ name: `process-${pid}-reaped`, status: running ? 'FAIL' : 'PASS' })
  }
  if (missingFixture && missingFixture.exitCode === null) {
    const exited = new Promise(resolve => missingFixture.once('exit', resolve))
    missingFixture.kill('SIGTERM')
    await exited
  }
  await writeFile(path.join(artifacts, 'results.json'), JSON.stringify({ results, errors, blocked, processes, elapsedMs: Date.now() - started }, null, 2))
  await writeFile(path.join(artifacts, 'frames.json'), JSON.stringify(frames, null, 2))
  await writeFile(path.join(artifacts, 'http.json'), JSON.stringify(requests, null, 2))
  console.log(`Evidence: ${artifacts}`)
}
