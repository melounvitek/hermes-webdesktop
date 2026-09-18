import assert from 'node:assert/strict'
import { access, mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium, expect } from '@playwright/test'

const runtimePath = process.argv[2]
if (!runtimePath) throw new Error('Usage: node browser-spike/smoke.mjs <harness runtime.json> [artifact directory]')
const runtime = JSON.parse(await readFile(runtimePath, 'utf8'))
assert.equal(new URL(runtime.url).hostname, '127.0.0.1')
const url = runtime.public_url || runtime.url
assert.equal(new URL(runtime.model_url).hostname, '127.0.0.1')
assert.equal(path.resolve(runtimePath), path.join(runtime.run_dir, 'runtime.json'))
assert.equal(runtime.hermes_home, path.join(runtime.run_dir, 'home', '.hermes'))
const config = JSON.parse(await readFile(path.join(runtime.hermes_home, 'config.yaml'), 'utf8'))
assert.equal(config.model.default, 'browser-spike-local')
assert.equal(config.model.provider, 'custom')
assert.equal(config.model.base_url, runtime.model_url)
const target = path.join(runtime.home, 'approval-target')
const artifacts = process.argv[3] || '/tmp/hermes-browser-evidence'
await mkdir(artifacts, { recursive: true })
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true
})
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
const frames = [],
  errors = [],
  httpErrors = [],
  assetErrors = []
page.on('pageerror', error => errors.push(error.stack))
page.on('response', response => {
  if (response.status() >= 400) httpErrors.push({ status: response.status(), url: response.url() })
  if (
    ['script', 'stylesheet', 'font', 'image'].includes(response.request().resourceType()) &&
    (response.status() >= 400 || response.headers()['content-type']?.includes('text/html'))
  ) {
    assetErrors.push(response.url())
  }
})
page.on('websocket', ws => {
  for (const [event, direction] of [
    ['framereceived', 'received'],
    ['framesent', 'sent']
  ]) {
    ws.on(event, ({ payload }) => frames.push({ direction, ...JSON.parse(String(payload)) }))
  }
})
const eventsSince = start =>
  frames
    .slice(start)
    .filter(f => f.direction === 'received' && f.method === 'event')
    .map(f => f.params)
async function send(text) {
  const start = frames.length
  const input = page.getByRole('textbox', { name: 'Message', exact: true })
  await input.fill(text)
  await input.press('Enter')
  return start
}
async function complete(start, text) {
  await expect.poll(() => eventsSince(start).some(e => e.type === 'message.complete'), { timeout: 60000 }).toBe(true)
  const events = eventsSince(start)
  const result = events.find(e => e.type === 'message.complete').payload
  assert.notEqual(result.status, 'error')
  assert.equal(
    events.some(e => e.type === 'error'),
    false
  )
  assert.ok(
    events.filter(e => e.type === 'message.delta').length > 1,
    'Real backend must deliver multiple streaming deltas'
  )
  assert.ok(result.text.includes(text), result.text)
  await expect(page.getByText(result.text, { exact: true })).toBeVisible({ timeout: 10000 })
  return result.text
}
try {
  // Verify fixture credentials and the live model before opening the renderer,
  // which can itself submit background prompts.
  const headers = runtime.public_url ? {} : { 'X-Hermes-Session-Token': runtime.token }
  if (runtime.public_url) {
    assert.equal(runtime.public_url, config.dashboard.public_url)
    assert.equal((await page.request.get(new URL('/api/config', url).href)).status(), 401)
    const login = JSON.parse(await readFile(path.join(runtime.run_dir, 'login.json'), 'utf8'))
    const response = await page.request.post(new URL('/auth/password-login', url).href, {
      data: { provider: 'basic', ...login },
      headers: { Origin: url }
    })
    assert.equal(response.status(), 200)
  }
  const response = await page.request.get(new URL('/api/config', url).href, { headers })
  assert.equal(response.status(), 200)
  assert.equal((await response.json()).model, config.model.default)
  await page.goto(url)
  const first = await send('spike: hello from Chromium')
  await complete(first, 'Spike turn 1: spike: hello from Chromium')
  await page.reload()
  await expect(page.getByText('Spike turn 1: spike: hello from Chromium', { exact: true })).toBeVisible({
    timeout: 60000
  })
  await complete(await send('spike: after reload'), 'Spike turn 2: spike: after reload')

  const clarification = await send('spike: clarify')
  await page.getByRole('button', { name: 'B Green', exact: true }).click({ timeout: 60000 })
  await page.getByRole('button', { name: /Confirm and continue/ }).click()
  assert.match(await complete(clarification, 'Spike turn 3: clarify result:'), /Green/)

  const denied = await send('spike: approval')
  await page.getByRole('button', { name: 'Reject', exact: true }).click({ timeout: 60000 })
  assert.match(await complete(denied, 'Spike turn 4: approval result:'), /denied/i)
  await access(target)
  const allowed = await send('spike: approval')
  await page.getByRole('button', { name: /^Run\s*↵$/ }).click({ timeout: 60000 })
  assert.match(await complete(allowed, 'Spike turn 5: approval result:'), /"exit_code": 0/)
  await assert.rejects(access(target), { code: 'ENOENT' })

  // Exercise a non-assets/ resource emitted by the existing Vite configuration.
  const emoji = await page.request.get(new URL('/emojibase/en/data.json', url).href)
  assert.equal(emoji.status(), 200)
  assert.match(emoji.headers()['content-type'], /application\/json/)
  assert.ok(Array.isArray(await emoji.json()))
  assert.deepEqual(assetErrors, [])
  assert.deepEqual(errors, [])
  console.log(
    'PASS: official renderer cold boot, streaming, reload/resume, second turn, clarification, approval reject/run, static assets; no page errors'
  )
  console.log(`HTTP errors: ${JSON.stringify(httpErrors)}`)
} finally {
  await page.screenshot({ path: `${artifacts}/browser.png`, fullPage: true })
  await writeFile(`${artifacts}/frames.json`, JSON.stringify(frames, null, 2))
  await writeFile(`${artifacts}/errors.json`, JSON.stringify({ errors, httpErrors, assetErrors }, null, 2))
  await browser.close()
}
