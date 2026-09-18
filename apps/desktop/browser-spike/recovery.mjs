import assert from 'node:assert/strict'
import { spawn, execFileSync } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from 'node:fs/promises'
import http from 'node:http'
import https from 'node:https'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium, expect } from '@playwright/test'

// node browser-spike/recovery.mjs <web-dist> [evidence directory] [case-name regex]
// Owns disposable fixture processes, not an existing dashboard. No live services.
const selected = new RegExp(process.argv[4] || '.')
const dist = path.resolve(process.argv[2] || 'dist-browser')
const artifacts = path.resolve(process.argv[3] || (await mkdtemp(path.join(os.tmpdir(), 'hermes-recovery-'))))
const python = process.env.SPIKE_PYTHON || path.join(os.homedir(), '.hermes/hermes-agent/venv/bin/python')
const here = path.dirname(fileURLToPath(import.meta.url))
await mkdir(artifacts, { recursive: true })
const browserHome = await mkdtemp(path.join(artifacts, 'chromium-home-'))
await readFile(path.join(dist, 'index.html'))
const results = []
const fixtures = []
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true,
  env: { PATH: '/usr/bin:/bin', HOME: browserHome, LANG: 'C.UTF-8', TZ: 'UTC' }
})

async function startFixture(cookie = false) {
  const name = cookie ? 'cookie' : 'token'
  let target
  const sockets = new Set()
  const handler = (request, response) => {
    if (!target) {
      response.writeHead(503).end()
      return
    }
    const upstream = http.request(
      target,
      {
        path: request.url,
        method: request.method,
        // Do not turn a stale pooled upstream socket into a synthetic 503.
        agent: false,
        headers: { ...request.headers, 'x-forwarded-proto': cookie ? 'https' : 'http' }
      },
      reply => {
        response.writeHead(reply.statusCode, reply.headers)
        reply.pipe(response)
      }
    )
    upstream.on('error', () => {
      if (!response.headersSent) response.writeHead(503)
      response.end()
    })
    request.pipe(upstream)
  }
  let server
  if (cookie) {
    const key = path.join(artifacts, 'localhost.key')
    const cert = path.join(artifacts, 'localhost.crt')
    execFileSync(
      'openssl',
      [
        'req',
        '-x509',
        '-newkey',
        'rsa:2048',
        '-nodes',
        '-days',
        '1',
        '-keyout',
        key,
        '-out',
        cert,
        '-subj',
        '/CN=recovery.localhost',
        '-addext',
        'subjectAltName=DNS:recovery.localhost'
      ],
      { stdio: 'ignore' }
    )
    server = https.createServer({ key: await readFile(key), cert: await readFile(cert) }, handler)
  } else server = http.createServer(handler)
  server.on('upgrade', (request, socket, head) => {
    if (!target) {
      socket.destroy()
      return
    }
    const upstream = net.connect(Number(new URL(target).port), '127.0.0.1', () => {
      upstream.write(
        `${request.method} ${request.url} HTTP/${request.httpVersion}\r\n` +
          Object.entries(request.headers)
            .map(([key, value]) => `${key}: ${value}\r\n`)
            .join('') +
          `X-Forwarded-Proto: ${cookie ? 'https' : 'http'}\r\n\r\n`
      )
      upstream.write(head)
      upstream.pipe(socket)
      socket.pipe(upstream)
    })
    sockets.add(socket)
    socket.on('close', () => {
      sockets.delete(socket)
      upstream.destroy()
    })
    socket.on('error', () => upstream.destroy())
    upstream.on('error', () => socket.destroy())
    upstream.on('close', () => socket.destroy())
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const origin = `${cookie ? 'https://recovery.localhost' : 'http://127.0.0.1'}:${server.address().port}`
  const args = [path.join(here, 'backend.py'), '--web-dist', dist, '--python', python, '--restartable']
  if (cookie) args.push('--public-url', origin)
  const child = spawn(python, args, {
    cwd: artifacts,
    env: { PATH: '/usr/bin:/bin', HOME: artifacts, LANG: 'C.UTF-8', PYTHONDONTWRITEBYTECODE: '1' },
    stdio: ['ignore', 'pipe', 'pipe']
  })
  const log = createWriteStream(path.join(artifacts, `${name}-fixture.log`))
  child.stdout.pipe(log)
  child.stderr.pipe(log, { end: false })
  let output = ''
  child.stdout.on('data', bytes => {
    output += bytes
  })
  const fixture = { child, server, sockets, cookie, origin }
  fixtures.push(fixture)
  await expect
    .poll(
      () => {
        assert.equal(child.exitCode, null, output)
        return output.match(/^READY (.+)$/m)?.[1]
      },
      { timeout: 100000 }
    )
    .toBeTruthy()
  const ready = JSON.parse(output.match(/^READY (.+)$/m)[1])
  fixture.runtimePath = path.join(ready.run_dir, 'runtime.json')
  fixture.runtime = JSON.parse(await readFile(fixture.runtimePath, 'utf8'))
  const runtime = fixture.runtime
  assert.equal(new URL(runtime.url).hostname, '127.0.0.1')
  assert.equal(new URL(runtime.model_url).hostname, '127.0.0.1')
  assert.equal(path.dirname(runtime.run_dir), os.tmpdir())
  assert.ok(path.basename(runtime.run_dir).startsWith('hermes-browser-spike-'))
  assert.equal(await realpath(runtime.run_dir), runtime.run_dir)
  assert.equal(runtime.home, path.join(runtime.run_dir, 'home'))
  assert.equal(runtime.hermes_home, path.join(runtime.home, '.hermes'))
  const config = JSON.parse(await readFile(path.join(runtime.hermes_home, 'config.yaml'), 'utf8'))
  assert.equal(config.model.default, 'browser-spike-local')
  assert.equal(config.model.base_url, runtime.model_url)
  assert.equal(config.terminal.cwd, runtime.home)
  target = runtime.url
  return fixture
}

async function restart(fixture) {
  const old = fixture.runtime
  fixture.child.kill('SIGUSR1')
  await expect
    .poll(
      async () => {
        fixture.runtime = JSON.parse(await readFile(fixture.runtimePath, 'utf8'))
        return fixture.runtime.generation
      },
      { timeout: 100000 }
    )
    .toBe(old.generation + 1)
  assert.equal(fixture.runtime.url, old.url)
  assert.equal(fixture.runtime.home, old.home)
  assert.notEqual(fixture.runtime.token, old.token)
  assert.notEqual(fixture.runtime.backend_pid, old.backend_pid)
}

// Read-only DB evidence is independent of the renderer's optimistic message cache.
function messageRows(fixture, text, role = 'user') {
  return JSON.parse(
    execFileSync(
      '/usr/bin/python3',
      [
        '-c',
        'import sqlite3,json,sys; c=sqlite3.connect("file:"+sys.argv[1]+"?mode=ro",uri=True); print(json.dumps(c.execute("select session_id,content from messages where role=? and instr(content,?)>0", (sys.argv[3],sys.argv[2])).fetchall()))',
        path.join(fixture.runtime.hermes_home, 'state.db'),
        text,
        role
      ],
      { encoding: 'utf8' }
    )
  )
}

const occurrences = (text, original) => text.split(original).length - 1
const normalize = text => text.replace(/\s+/g, ' ').trim()

async function check(fixture, name, body, { ticket503 = false, shiki503 = false } = {}) {
  if (!selected.test(name)) return
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, ignoreHTTPSErrors: true })
  const frames = [],
    errors = [],
    consoleErrors = [],
    blocked = [],
    responses = [],
    navigations = [],
    socketEvents = [],
    turns = []
  await context.route('**/*', route => {
    const url = new URL(route.request().url())
    if (url.origin === fixture.origin || ['data:', 'blob:'].includes(url.protocol)) return route.continue()
    blocked.push({ url: url.href, type: route.request().resourceType() })
    return route.abort('blockedbyclient')
  })
  // Let Chromium own the real sockets; proxy only loopback, drop them to simulate link loss.
  await context.routeWebSocket('**/*', ws => {
    if (new URL(ws.url()).origin === fixture.origin.replace(/^http/, 'ws')) ws.connectToServer()
    else {
      blocked.push({ url: ws.url(), type: 'websocket' })
      ws.close()
    }
  })
  const page = await context.newPage()
  page.setDefaultTimeout(15000)
  page.on('framenavigated', frame => {
    if (frame === page.mainFrame()) navigations.push(frame.url())
  })
  page.on('pageerror', error => errors.push({ name: error.name, message: error.message, stack: error.stack }))
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push(message.text())
  })
  page.on('response', response => {
    if (response.status() >= 400 || new URL(response.url()).pathname === '/api/auth/ws-ticket') {
      responses.push({ url: response.url(), status: response.status() })
    }
  })
  let socketId = 0
  page.on('websocket', ws => {
    const socket = ++socketId
    socketEvents.push({ socket, event: 'open' })
    ws.on('close', () => socketEvents.push({ socket, event: 'close' }))
    for (const [event, direction] of [
      ['framesent', 'sent'],
      ['framereceived', 'received']
    ]) {
      ws.on(event, ({ payload }) => frames.push({ socket, direction, ...JSON.parse(String(payload)) }))
    }
  })
  const input = () => page.getByRole('textbox', { name: 'Message', exact: true })
  const events = start =>
    frames
      .slice(start)
      .filter(f => f.direction === 'received' && f.method === 'event')
      .map(f => f.params)
  const holdFile = path.join(fixture.runtime.run_dir, 'hold-stream')
  const release = () => rm(holdFile, { force: true })
  const ready = () =>
    expect(page.getByRole('button', { name: 'Gateway ready', exact: true })).toBeVisible({ timeout: 30000 })
  const completion = turn =>
    events(turn.start).find(e => e.type === 'message.complete' && e.payload.text?.includes(turn.text))
  async function visibleReply(text) {
    // Markdown paragraphs are separate nodes. Compare the entire rendered reply,
    // normalizing whitespace only — never just its prefix or final paragraph.
    const replies = page.locator('[data-slot="aui_assistant-message-content"]')
    await expect
      .poll(async () => (await replies.allInnerTexts()).filter(t => normalize(t) === normalize(text)).length)
      .toBe(1)
    const index = (await replies.allInnerTexts()).findIndex(t => normalize(t) === normalize(text))
    await expect(replies.nth(index)).toBeVisible()
  }
  async function completed(turn) {
    await expect.poll(() => Boolean(completion(turn)), { timeout: 60000 }).toBe(true)
    const result = completion(turn).payload
    assert.equal(result.status, 'complete', JSON.stringify(result))
    const requests = (await readFile(path.join(fixture.runtime.run_dir, 'model-requests.jsonl'), 'utf8'))
      .trim()
      .split('\n')
      .map(JSON.parse)
    const users = requests
      .findLast(r => JSON.stringify(r.messages.at(-1)?.content).includes(turn.text))
      .messages.filter(m => m.role === 'user')
    const prompt = users.at(-1).content
    const expected = `Spike turn ${users.length}: ${typeof prompt === 'string' ? prompt : prompt.map(p => p.text || '').join('\n')}`
    assert.equal(result.text, expected, 'The completed stream must deliver every byte of the fixture response')
    await visibleReply(result.text)
    turn.reply = result.text
    await expect.poll(() => messageRows(fixture, result.text, 'assistant')).toEqual([[turn.stored, result.text]])
  }
  async function resumed(turn, running) {
    await ready()
    await expect
      .poll(
        () => {
          const request = frames
            .slice(turn.start)
            .findLast(
              f => f.direction === 'sent' && f.method === 'session.resume' && f.params.session_id === turn.stored
            )
          return (
            request &&
            frames.find(f => f.direction === 'received' && f.socket === request.socket && f.id === request.id)?.result
              ?.running
          )
        },
        { timeout: 30000 }
      )
      .toBe(running)
  }
  async function finishHeld(turn) {
    await resumed(turn, true)
    await release()
    await completed(turn)
    assert.equal(turn.reply, `Spike turn 1: ${turn.text}`, 'A transport interruption must not truncate the held reply')
    assert.equal(
      frames.slice(turn.start).filter(f => f.direction === 'sent' && /interrupt|cancel/.test(f.method || '')).length,
      0
    )
  }
  async function submit(text, hold = false) {
    if (hold) await writeFile(holdFile, name)
    const start = frames.length
    await expect(page.getByRole('button', { name: 'Gateway ready', exact: true })).toBeVisible({ timeout: 30000 })
    await input().fill(text)
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect.poll(() => messageRows(fixture, text).length).toBe(1)
    const turn = { text, stored: messageRows(fixture, text)[0][0], start }
    turns.push(turn)
    if (hold) {
      await expect.poll(() => events(start).some(e => e.type === 'message.delta'), { timeout: 60000 }).toBe(true)
    } else await completed(turn)
    return turn
  }
  async function retained(turn) {
    const users = page.locator('[data-slot="aui_user-message-root"]')
    await expect.poll(async () => occurrences((await users.allInnerTexts()).join('\n'), turn.text)).toBe(1)
    const rows = messageRows(fixture, turn.text)
    assert.equal(rows.length, 1, 'Original user turn must occur in exactly one DB row')
    assert.equal(rows[0][0], turn.stored, 'Original user turn must retain its durable identity')
    assert.equal(occurrences(rows[0][1], turn.text), 1, 'A merged DB row must not duplicate an original turn')
    if (turn.reply) await visibleReply(turn.reply)
    assert.equal(
      frames.filter(f => f.direction === 'sent' && f.method === 'prompt.submit' && f.params?.text === turn.text).length,
      1,
      'Recovery must never replay prompt.submit'
    )
  }
  async function followup(turn) {
    await ready()
    await release()
    await expect(page.getByRole('button', { name: 'Stop', exact: true }).first()).not.toBeVisible({ timeout: 30000 })
    const next = await submit(`spike: followup-${name}`)
    assert.equal(next.stored, turn.stored, 'Recovery must keep the durable conversation identity')
    await retained(turn)
    await retained(next)
    const requests = (await readFile(path.join(fixture.runtime.run_dir, 'model-requests.jsonl'), 'utf8'))
      .trim()
      .split('\n')
      .map(JSON.parse)
    const last = requests.findLast(r => JSON.stringify(r.messages).includes(next.text))
    assert.ok(last, 'Follow-up must reach the isolated model')
    // Consecutive interrupted user rows may be merged for model role alternation.
    // Count occurrences, not rows: one merged message could contain a duplicate.
    const history = last.messages
      .filter(m => m.role === 'user')
      .map(m => JSON.stringify(m.content))
      .join('\n')
    for (const original of [turn, next]) {
      assert.equal(
        occurrences(history, original.text),
        1,
        'Model history must contain each original user occurrence once'
      )
    }
    if (turn.reply) {
      assert.equal(
        last.messages.filter(m => m.role === 'assistant' && m.content === turn.reply).length,
        1,
        'The follow-up model history must retain the complete original assistant reply'
      )
    }
  }
  async function login() {
    const credentials = JSON.parse(await readFile(path.join(fixture.runtime.run_dir, 'login.json'), 'utf8'))
    await page.getByRole('textbox', { name: 'Username', exact: true }).fill(credentials.username)
    await page.getByLabel('Password', { exact: true }).fill(credentials.password)
    await page.getByRole('button', { name: 'Sign in', exact: true }).click()
    await expect(input()).toBeEditable({ timeout: 60000 })
  }
  async function expireCookies() {
    const cookies = await context.cookies()
    assert.ok(
      cookies.some(c => c.name.endsWith('hermes_session_at') && c.httpOnly && c.secure),
      'Exercise real secure HttpOnly auth'
    )
    await context.addCookies(
      cookies.filter(c => /hermes_session_(at|rt)$/.test(c.name)).map(c => ({ ...c, expires: 1 }))
    )
    assert.equal((await context.cookies()).filter(c => /hermes_session_(at|rt)$/.test(c.name)).length, 0)
  }
  async function signInAgain() {
    const before = new URL(page.url())
    assert.ok(before.hash.length > 2, 'Exercise a non-empty SPA hash route')
    await expect(page.getByRole('button', { name: /^(Sign in again|Sign in)$/ }).first()).toBeVisible({
      timeout: 30000
    })
    await page.screenshot({ path: path.join(artifacts, `${name}-visible-recovery.png`) })
    await page
      .getByRole('button', { name: /^(Sign in again|Sign in)$/ })
      .first()
      .click()
    await expect(page.getByRole('textbox', { name: 'Username', exact: true })).toBeVisible()
    assert.equal(
      new URL(page.url()).searchParams.get('next'),
      before.pathname + before.search + before.hash,
      'Sign-in must carry the full original path, query and hash'
    )
    await login()
    await expect.poll(() => page.url()).toBe(before.href)
  }
  let shikiRequests = 0
  if (shiki503) {
    // Fail only the optional plugin bundle, not the renderer/highlighter modules.
    // A fresh browser context makes the initial import deterministic.
    await page.route(/\/assets\/shiki-(?!block-|highlighter-|plain-)[\w-]+\.js$/, async route => {
      shikiRequests++
      await route.fulfill({ status: 503, contentType: 'text/javascript', body: 'temporary fixture outage' })
    })
  }
  let injected503 = 0
  if (ticket503) {
    await page.route('**/api/auth/ws-ticket', async route => {
      if (injected503 === 0) {
        injected503++
        await route.fulfill({
          status: 503,
          contentType: 'application/json',
          body: '{"error":"temporary fixture outage"}'
        })
      } else await route.fallback() // Recovery must obtain a real backend ticket.
    })
  }
  try {
    await page.goto(fixture.origin)
    if (fixture.cookie) await login()
    await expect(input()).toBeEditable({ timeout: 60000 })
    await body({
      page,
      context,
      submit,
      retained,
      followup,
      resumed,
      finishHeld,
      ready,
      login,
      signInAgain,
      expireCookies,
      responses,
      socketEvents
    })
    if (ticket503) {
      assert.equal(injected503, 1)
      assert.deepEqual(
        responses.filter(r => new URL(r.url).pathname === '/api/auth/ws-ticket').map(r => r.status),
        [503, 200]
      )
    }
    if (shiki503) {
      assert.equal(shikiRequests, 1, 'Exercise the failed initial import without network retries on remount')
      assert.equal(
        responses.filter(
          r => /\/assets\/shiki-(?!block-|highlighter-|plain-)[\w-]+\.js$/.test(r.url) && r.status === 503
        ).length,
        1
      )
    }
    assert.deepEqual(errors, [], 'Renderer must not throw')
    assert.deepEqual(
      blocked.filter(r => !['font', 'stylesheet'].includes(r.type)),
      [],
      'Unexpected external requests'
    )
    results.push({ name, status: 'PASS', run_dir: fixture.runtime.run_dir })
    console.log(`PASS: ${name}`)
  } catch (error) {
    const diagnostic = { name: error.name, message: error.message, stack: error.stack }
    results.push({ name, status: 'FAIL', error: diagnostic, run_dir: fixture.runtime.run_dir })
    process.exitCode = 1
    console.error(`FAIL: ${name}\n${error.name}: ${error.message}\n${error.stack || ''}`)
  } finally {
    await page.screenshot({ path: path.join(artifacts, `${name}.png`) })
    await writeFile(path.join(artifacts, `${name}.aria.txt`), await page.locator('body').ariaSnapshot())
    await writeFile(
      path.join(artifacts, `${name}.json`),
      JSON.stringify(
        {
          url: page.url(),
          navigations,
          injected503,
          shikiRequests,
          frames,
          socketEvents,
          turns,
          errors,
          consoleErrors,
          blocked,
          responses
        },
        null,
        2
      )
    )
    await context.setOffline(false)
    await release()
    await context.close()
  }
}

try {
  const token = [
    'shiki503',
    'reload-during-stream',
    'network-loss-during-stream',
    'backend-restart-interrupted-turn',
    'backend-restart-token-rotation',
    'user-interrupted-turn'
  ].some(name => selected.test(name))
    ? await startFixture()
    : null
  await check(
    token,
    'shiki503',
    async ({ page, submit, retained, followup }) => {
      const turn = await submit(
        'spike: shiki503 keeps this entire reply readable through its final sentence after the optional syntax highlighting import fails.'
      )
      const reply = await page.locator('[data-slot="aui_assistant-message-content"]').elementHandle()
      // Navigate within the same document: reloading would clear the failed
      // module cache and would not test a subsequent markdown mount.
      await page.getByRole('button', { name: 'New session', exact: true }).click()
      const tab = page.getByRole('tab', { name: /spike: shiki503/ })
      await tab.hover()
      await tab.getByRole('button', { name: 'Close', exact: true }).click()
      await expect.poll(() => reply.evaluate(element => element.isConnected)).toBe(false)
      await page
        .getByRole('button', { name: /^(Reorder )?spike: shiki503/ })
        .first()
        .click()
      await retained(turn)
      await followup(turn)
    },
    { shiki503: true }
  )
  await check(token, 'reload-during-stream', async ({ page, submit, retained, finishHeld, followup }) => {
    const turn = await submit('spike: hold reload-during-stream', true)
    await page.reload()
    await retained(turn)
    await finishHeld(turn)
    await followup(turn)
  })
  await check(token, 'network-loss-during-stream', async ({ page, context, submit, finishHeld, followup }) => {
    const turn = await submit('spike: hold network-loss-during-stream', true)
    await context.setOffline(true)
    for (const socket of token.sockets) socket.destroy()
    await expect(page.getByText(/reconnect|connection lost|offline/i).first()).toBeVisible({ timeout: 30000 })
    await expect(page.getByRole('button', { name: /^(Sign in again|Sign in)$/ })).not.toBeVisible()
    await page.screenshot({ path: path.join(artifacts, 'network-loss-visible-recovery.png') })
    await context.setOffline(false)
    await finishHeld(turn)
    await followup(turn)
  })
  for (const held of [false, true]) {
    const name = held ? 'backend-restart-interrupted-turn' : 'backend-restart-token-rotation'
    await check(token, name, async ({ page, submit, resumed, followup }) => {
      const turn = await submit(`spike: ${held ? 'hold ' : ''}${name}`, held)
      await restart(token)
      // Automatic reconnect OR a visible recovery action; never manually reload as a workaround.
      const reconnect = page.getByRole('button', { name: /^(Reconnect|Retry|Reload)/ }).first()
      await expect
        .poll(
          async () =>
            (await reconnect.isVisible()) ||
            (await page.getByRole('button', { name: 'Gateway ready', exact: true }).isVisible()),
          { timeout: 30000 }
        )
        .toBe(true)
      if (await reconnect.isVisible()) await reconnect.click()
      await resumed(turn, false)
      await followup(turn)
    })
  }
  await check(token, 'user-interrupted-turn', async ({ page, submit, followup }) => {
    const turn = await submit('spike: hold user-interrupted-turn', true)
    await page.getByRole('button', { name: 'Stop', exact: true }).first().click()
    await followup(turn)
  })
  const cookie = [
    'fresh-cookie-login-hash',
    'expired-cookie-relogin',
    'cookie-rest401-connected-ws',
    'cookie-backend-restart',
    'cold-boot-ticket503'
  ].some(name => selected.test(name))
    ? await startFixture(true)
    : null
  await check(
    cookie,
    'fresh-cookie-login-hash',
    async ({ page, context, submit, retained, followup, expireCookies, login }) => {
      const turn = await submit('spike: fresh-cookie-login-hash')
      const target = page.url()
      assert.ok(new URL(target).hash.includes(turn.stored), 'Start from the durable session deep link')
      await page.goto('about:blank') // Avoid a same-document hash navigation.
      await expireCookies()
      // Reopen the deep link without any remembered route/session. A cache restore
      // must not conceal a login redirect that drops the hash.
      await context.addInitScript(origin => {
        if (location.origin !== origin) return
        localStorage.clear()
        sessionStorage.clear()
      }, cookie.origin)
      await page.goto(target)
      await login()
      await expect.poll(() => page.url()).toBe(target)
      await retained(turn)
      await followup(turn)
    }
  )
  await check(
    cookie,
    'expired-cookie-relogin',
    async ({ submit, retained, followup, expireCookies, signInAgain, responses }) => {
      const turn = await submit('spike: expired-cookie-relogin')
      await expireCookies()
      for (const socket of cookie.sockets) socket.destroy()
      // A background REST request can observe expiry before the reconnect
      // mints its ticket; either confirmed rejection must stop retries.
      await expect
        .poll(() => responses.some(r => new URL(r.url).pathname.startsWith('/api/') && r.status === 401), {
          timeout: 30000,
          message: 'Expired cookie must be rejected by the real backend before signing in'
        })
        .toBe(true)
      await signInAgain()
      await retained(turn)
      await followup(turn)
    }
  )
  await check(
    cookie,
    'cookie-rest401-connected-ws',
    async ({ page, submit, retained, followup, expireCookies, signInAgain, responses, socketEvents }) => {
      const turn = await submit('spike: cookie-rest401-connected-ws')
      const sockets = [...cookie.sockets]
      assert.equal(sockets.length, 1, 'The authenticated socket must already be connected')
      const before = socketEvents.length
      await expireCookies()
      // Invoke the shipped bridge, not fetch or a mocked error. Keep the real WS
      // open so this proves REST alone can publish the reauthentication state.
      const error = await page.evaluate(async () => {
        try {
          await window.hermesDesktop.api({ path: '/api/config' })
          return null
        } catch (error) {
          return String(error)
        }
      })
      assert.match(error, /401/)
      assert.ok(responses.some(r => new URL(r.url).pathname === '/api/config' && r.status === 401))
      await expect(page.getByRole('button', { name: /^(Sign in again|Sign in)$/ }).first()).toBeVisible({
        timeout: 30000
      })
      assert.deepEqual(socketEvents.slice(before), [], 'REST 401 must be exercised without a socket reconnect')
      assert.deepEqual([...cookie.sockets], sockets)
      assert.ok(sockets.every(socket => !socket.destroyed))
      assert.equal(
        responses.filter(r => new URL(r.url).pathname === '/api/auth/ws-ticket' && r.status === 401).length,
        0
      )
      await signInAgain()
      await retained(turn)
      await followup(turn)
    }
  )
  await check(cookie, 'cookie-backend-restart', async ({ page, context, submit, resumed, followup, responses }) => {
    const turn = await submit('spike: cookie-backend-restart')
    const cookies = (await context.cookies()).filter(c => /hermes_session_(at|rt)$/.test(c.name))
    assert.ok(cookies.length)
    const before = responses.length
    await restart(cookie)
    await resumed(turn, false)
    assert.deepEqual(
      (await context.cookies()).filter(c => /hermes_session_(at|rt)$/.test(c.name)),
      cookies,
      'Backend restart must reuse the persisted cookie login'
    )
    assert.ok(
      responses.slice(before).some(r => new URL(r.url).pathname === '/api/auth/ws-ticket' && r.status === 200),
      'Reconnect must acquire a fresh ticket from the restarted backend'
    )
    await expect(page.getByRole('button', { name: /^(Sign in again|Sign in)$/ })).not.toBeVisible()
    await followup(turn)
  })
  await check(
    cookie,
    'cold-boot-ticket503',
    async ({ page, submit, followup, ready }) => {
      await ready()
      await expect(page.getByRole('button', { name: /^(Sign in again|Sign in)$/ })).not.toBeVisible()
      const turn = await submit('spike: cold-boot-ticket503')
      await followup(turn)
    },
    { ticket503: true }
  )
  assert.ok(results.length, 'No recovery cases matched the filter')
} finally {
  await browser.close()
  for (const fixture of fixtures) {
    fixture.child.kill('SIGTERM')
    await new Promise(resolve => {
      if (fixture.child.exitCode !== null) {
        resolve()
        return
      }
      const timer = setTimeout(() => {
        fixture.child.kill('SIGKILL')
        resolve()
      }, 20000)
      fixture.child.once('exit', () => {
        clearTimeout(timer)
        resolve()
      })
    })
    for (const socket of fixture.sockets) socket.destroy()
    fixture.server.closeAllConnections()
    await new Promise(resolve => fixture.server.close(resolve))
  }
  await writeFile(path.join(artifacts, 'results.json'), JSON.stringify({ dist, results }, null, 2))
  console.log(`Evidence: ${artifacts}`)
}
