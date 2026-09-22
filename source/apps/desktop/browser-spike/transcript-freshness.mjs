import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { chromium, expect } from '@playwright/test'

// Use only an owned backend.py fixture, never an existing/live dashboard.
// node browser-spike/transcript-freshness.mjs <runtime.json> <evidence-directory>
const runtime = JSON.parse(await readFile(process.argv[2], 'utf8'))
const artifacts = path.resolve(process.argv[3])
assert.equal(new URL(runtime.url).hostname, '127.0.0.1')
assert.equal(new URL(runtime.model_url).hostname, '127.0.0.1')
assert.equal(path.dirname(runtime.run_dir), os.tmpdir())
assert.ok(path.basename(runtime.run_dir).startsWith('hermes-browser-spike-'))
assert.equal(runtime.home, path.join(runtime.run_dir, 'home'))
assert.equal(runtime.hermes_home, path.join(runtime.home, '.hermes'))
const config = JSON.parse(await readFile(path.join(runtime.hermes_home, 'config.yaml'), 'utf8'))
assert.equal(config.model.default, 'browser-spike-local')
assert.equal(config.model.base_url, runtime.model_url)
await mkdir(artifacts, { recursive: true })
const home = await mkdtemp(path.join(artifacts, 'chromium-home-'))
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true,
  env: { PATH: '/usr/bin:/bin', HOME: home, LANG: 'C.UTF-8', TZ: 'UTC' }
})
try {
  const frames = [],
    http = [],
    errors = [],
    blocked = []
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
  await context.route('**/*', route => {
    const url = new URL(route.request().url())
    if (url.origin === runtime.url || ['data:', 'blob:'].includes(url.protocol)) return route.continue()
    blocked.push({ url: url.href, type: route.request().resourceType() })
    return route.abort('blockedbyclient')
  })
  await context.routeWebSocket('**/*', ws => {
    if (new URL(ws.url()).origin === runtime.url.replace(/^http/, 'ws')) ws.connectToServer()
    else {
      blocked.push({ url: ws.url(), type: 'websocket' })
      ws.close()
    }
  })
  const page = await context.newPage()
  const cdp = await context.newCDPSession(page)
  await cdp.send('Network.enable')
  cdp.on('Network.requestWillBeSent', e => {
    if (e.request.url.includes('/messages?'))
      http.push({ time: Date.now(), event: 'initiator', url: e.request.url, initiator: e.initiator })
  })
  page.on('pageerror', error => errors.push(String(error)))
  page.on('websocket', ws => {
    for (const [event, direction] of [
      ['framesent', 'sent'],
      ['framereceived', 'received']
    ]) {
      ws.on(event, ({ payload }) => frames.push({ time: Date.now(), direction, ...JSON.parse(String(payload)) }))
    }
  })
  let stopping = false,
    held,
    deliver,
    delivered
  const gate = new Promise(resolve => {
    deliver = resolve
  })
  const deliveredGate = new Promise(resolve => {
    delivered = resolve
  })
  await page.route('**/api/sessions/*/messages?*', async route => {
    if (!stopping || held) return route.fallback()
    held = { url: route.request().url() }
    const response = await route.fetch()
    held.body = await response.json()
    http.push({ time: Date.now(), event: 'held', ...held })
    await gate
    await route.fulfill({ response })
    http.push({ time: Date.now(), event: 'released', url: held.url })
    delivered()
  })
  const holdFile = path.join(runtime.run_dir, 'hold-stream')
  const original = 'spike: hold transcript-freshness'
  const followup = 'spike: followup-transcript-freshness'
  const events = () => frames.filter(f => f.direction === 'received' && f.method === 'event').map(f => f.params)
  const completion = text => events().find(e => e.type === 'message.complete' && e.payload.text?.includes(text))
  const users = page.locator('[data-slot="aui_user-message-root"]')
  const replies = page.locator('[data-slot="aui_assistant-message-content"]')
  const normalize = text => text.replace(/\s+/g, ' ').trim()
  async function visibleReply(text) {
    await expect
      .poll(async () => (await replies.allInnerTexts()).filter(t => normalize(t) === normalize(text)).length)
      .toBe(1)
  }
  function rows() {
    return JSON.parse(
      execFileSync(
        '/usr/bin/python3',
        [
          '-c',
          'import sqlite3,json,sys; c=sqlite3.connect("file:"+sys.argv[1]+"?mode=ro",uri=True); print(json.dumps(c.execute("select session_id,role,content from messages order by id").fetchall()))',
          path.join(runtime.hermes_home, 'state.db')
        ],
        { encoding: 'utf8' }
      )
    )
  }
  async function send(text) {
    await page.getByRole('textbox', { name: 'Message', exact: true }).fill(text)
    await page.getByRole('button', { name: 'Send', exact: true }).click()
  }
  let result
  try {
    await page.goto(runtime.url)
    await expect(page.getByRole('button', { name: 'Gateway ready', exact: true })).toBeVisible({ timeout: 60000 })
    await writeFile(holdFile, original)
    await send(original)
    await expect.poll(() => events().some(e => e.type === 'message.delta'), { timeout: 60000 }).toBe(true)
    stopping = true
    await page.getByRole('button', { name: 'Stop', exact: true }).first().click()
    await expect.poll(() => held?.body, { timeout: 15000 }).toBeTruthy()
    // Stock versions differ on whether the interrupted partial is persisted
    // before this read. The held snapshot must still predate the follow-up turn.
    assert.deepEqual(held.body.messages.filter(m => m.role === 'user').map(m => m.content), [original])
    const partials = held.body.messages.filter(m => m.role === 'assistant')
    assert.ok(partials.length <= 1)
    assert.ok(held.body.messages.every(m => m.role === 'user' || (m.role === 'assistant' && m.content === 'Spike tu')))
    await rm(holdFile, { force: true })
    // Isolate freshness from the separate old-completion/new-submit race.
    await expect
      .poll(() => events().some(e => e.type === 'message.complete' && e.payload.status === 'interrupted'))
      .toBe(true)
    await expect.poll(() => events().some(e => e.type === 'session.info' && e.payload.running === false)).toBe(true)
    await send(followup)
    await expect.poll(() => Boolean(completion(followup)), { timeout: 60000 }).toBe(true)
    const reply = completion(followup).payload
    assert.equal(reply.status, 'complete')
    assert.equal(reply.text, `Spike turn 2: ${followup}`)
    await visibleReply(reply.text)
    const expectedRows = [
      [original, 'user'],
      ['Spike tu', 'assistant'],
      [followup, 'user'],
      [reply.text, 'assistant']
    ]
    await expect.poll(() => rows().map(([, role, text]) => [text, role])).toEqual(expectedRows)
    const durable = rows()[0][0]
    assert.ok(rows().every(([sid]) => sid === durable))
    const finished = page.waitForEvent('requestfinished', request => request.url() === held.url)
    deliver()
    await deliveredGate
    await finished
    // Flush response publication through browser paint, not an arbitrary delay.
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
    await expect.poll(async () => (await users.allInnerTexts()).map(normalize)).toEqual([original, followup])
    await visibleReply('Spike tu')
    await visibleReply(reply.text)
    assert.deepEqual(
      frames.filter(f => f.direction === 'sent' && f.method === 'prompt.submit').map(f => f.params.text),
      [original, followup]
    )
    const requests = (await readFile(path.join(runtime.run_dir, 'model-requests.jsonl'), 'utf8'))
      .trim()
      .split('\n')
      .map(JSON.parse)
    const history = JSON.stringify(requests.at(-1).messages.filter(m => m.role === 'user'))
    for (const text of [original, followup]) assert.equal(history.split(text).length - 1, 1)
    assert.deepEqual(errors, [])
    assert.deepEqual(
      blocked.filter(r => !['font', 'stylesheet'].includes(r.type)),
      []
    )
    result = { status: 'PASS', durable, rows: rows() }
  } catch (error) {
    result = { status: 'FAIL', error: { message: error.message, stack: error.stack }, rows: rows() }
    process.exitCode = 1
  } finally {
    deliver()
    await rm(holdFile, { force: true })
    await writeFile(
      path.join(artifacts, 'results.json'),
      JSON.stringify({ result, frames, http, errors, blocked }, null, 2)
    )
    await page.screenshot({ path: path.join(artifacts, 'transcript.png') })
    await writeFile(path.join(artifacts, 'transcript.aria.txt'), await page.locator('body').ariaSnapshot())
  }
  console.log(JSON.stringify(result, null, 2))
} finally {
  await browser.close()
}
