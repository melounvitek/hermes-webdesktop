import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { access, mkdir, readFile, realpath, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { chromium, expect } from '@playwright/test'

// Run against an owned backend.py fixture; the caller owns its shutdown.
const runtimePath = process.argv[2]
if (!runtimePath) throw new Error('Usage: node profile-isolation.mjs <runtime.json> <evidence directory>')
const runtime = JSON.parse(await readFile(runtimePath, 'utf8'))
assert.equal(new URL(runtime.url).hostname, '127.0.0.1')
assert.equal(new URL(runtime.model_url).hostname, '127.0.0.1')
assert.equal(runtime.public_url, null)
assert.equal(path.dirname(runtime.run_dir), os.tmpdir())
assert.ok(path.basename(runtime.run_dir).startsWith('hermes-browser-spike-'))
assert.equal(await realpath(runtime.run_dir), runtime.run_dir)
assert.equal(path.resolve(runtimePath), path.join(runtime.run_dir, 'runtime.json'))
assert.equal(runtime.home, path.join(runtime.run_dir, 'home'))
assert.equal(runtime.hermes_home, path.join(runtime.home, '.hermes'))
assert.equal(await realpath(runtime.hermes_home), runtime.hermes_home)
const config = JSON.parse(await readFile(path.join(runtime.hermes_home, 'config.yaml'), 'utf8'))
assert.equal(config.model.default, 'browser-spike-local')
assert.equal(config.model.provider, 'custom')
assert.equal(config.model.base_url, runtime.model_url)
const artifacts = process.argv[3]
assert.ok(artifacts, 'An evidence directory is required')
await mkdir(artifacts, { recursive: true })
const profiles = ['isolation-a', 'isolation-b']
const initial = [7, 11]
await mkdir(path.join(runtime.hermes_home, 'profiles'), { recursive: true })
for (const [i, profile] of profiles.entries()) {
  const home = path.join(runtime.hermes_home, 'profiles', profile)
  // Refuse to overwrite an earlier run, even in a disposable fixture.
  await mkdir(home, { recursive: false })
  await writeFile(path.join(home, 'config.yaml'), JSON.stringify({
    ...config, agent: { ...config.agent, max_turns: initial[i] }
  }))
}
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: true,
  env: { PATH: '/usr/bin:/bin', HOME: artifacts, LANG: 'C.UTF-8', TZ: 'UTC' }
})
const results = []
const headers = { 'X-Hermes-Session-Token': runtime.token }
const holdFile = path.join(runtime.run_dir, 'hold-stream')
const input = page => page.getByRole('textbox', { name: 'Message', exact: true })
const users = page => page.locator('[data-slot="aui_user-message-root"]')
const replies = page => page.locator('[data-slot="aui_assistant-message-content"]')
const row = (page, marker) => page.locator('[data-tree-group="grp-sessions"]')
  .getByRole('button', { name: new RegExp(marker) }).first()
const normalize = text => text.replace(/\s+/g, ' ').trim()

// Independent durable evidence, not the renderer's optimistic cache.
function rows(profile, text, role = 'user') {
  const home = profile === 'default' ? runtime.hermes_home : path.join(runtime.hermes_home, 'profiles', profile)
  const database = path.join(home, 'state.db')
  if (!existsSync(database)) return []
  return JSON.parse(execFileSync('/usr/bin/python3', ['-c',
    'import sqlite3,json,sys; c=sqlite3.connect("file:"+sys.argv[1]+"?mode=ro",uri=True); print(json.dumps(c.execute("select session_id,content from messages where role=? and instr(content,?)>0",(sys.argv[3],sys.argv[2])).fetchall()))',
    database, text, role
  ], { encoding: 'utf8' }))
}

async function selectProfile(page, profile) {
  // The rail, unlike a settings scope chip or status-bar button, declares pressed state.
  const rail = page.locator('button[aria-pressed]').and(page.getByRole('button', { name: profile, exact: true }))
  await rail.click()
  await expect(rail).toHaveAttribute('aria-pressed', 'true')
  await expect(input(page)).toBeEditable()
}

async function check(name, body) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
  const frames = [], errors = [], blocked = [], http = [], observations = []
  const started = Date.now()
  let socketId = 0
  await context.route('**/*', route => {
    const url = new URL(route.request().url())
    if (url.origin === runtime.url || ['data:', 'blob:'].includes(url.protocol)) return route.continue()
    blocked.push({ url: url.href, type: route.request().resourceType() })
    return route.abort('blockedbyclient')
  })
  await context.routeWebSocket('**/*', ws => {
    if (new URL(ws.url()).origin === runtime.url.replace('http:', 'ws:')) ws.connectToServer()
    else { blocked.push({ url: ws.url(), type: 'websocket' }); ws.close() }
  })
  const page = await context.newPage()
  page.setDefaultTimeout(15000)
  page.on('pageerror', error => errors.push(error.stack))
  const pendingHttp = []
  page.on('response', response => {
    if (new URL(response.url()).pathname !== '/api/config') return
    pendingHttp.push((async () => {
      http.push({ at: Date.now(), url: response.url(), method: response.request().method(),
        body: response.request().postData(), status: response.status(), response: await response.json() })
    })().catch(error => http.push({ url: response.url(), error: String(error) })))
  })
  page.on('websocket', ws => {
    const socket = ++socketId
    for (const [event, direction] of [['framesent', 'sent'], ['framereceived', 'received']]) {
      ws.on(event, ({ payload }) => frames.push({ at: Date.now(), socket, direction,
        transportProfile: new URL(ws.url()).searchParams.get('profile'), ...JSON.parse(String(payload)) }))
    }
  })
  const sent = method => frames.filter(f => f.direction === 'sent' && f.method === method)
  const events = (turn, type) => frames.slice(turn.start).filter(f => f.direction === 'received' && f.method === 'event' &&
    f.params.session_id === turn.live && f.params.type === type)
  async function submit(profile, text) {
    const start = frames.length
    await input(page).fill(text)
    await input(page).press('Enter')
    await expect.poll(() => sent('prompt.submit').findLast(f => f.params.text === text), { timeout: 60000 }).toBeTruthy()
    const request = sent('prompt.submit').findLast(f => f.params.text === text)
    await expect.poll(() => frames.find(f => f.direction === 'received' && f.socket === request.socket &&
      f.id === request.id)?.result?.status, { timeout: 60000 }).toBe('streaming')
    const live = request.params.session_id
    const owner = frames.findLast(f => f.direction === 'received' && f.result?.session_id === live)?.result
    assert.equal(owner?.info?.profile_name, profile, 'Correlate owner by runtime id, never shared socket profile')
    await expect.poll(() => rows(profile, text).length).toBe(1)
    const turn = { profile, text, live, stored: rows(profile, text)[0][0], start }
    observations.push({ turn })
    return turn
  }
  async function complete(turn) {
    await expect.poll(() => events(turn, 'message.complete').length, { timeout: 60000 }).toBeGreaterThan(0)
    const result = events(turn, 'message.complete').at(-1).params.payload
    assert.equal(result.status, 'complete', JSON.stringify(result))
    turn.reply = result.text
    await expect.poll(() => rows(turn.profile, result.text, 'assistant')).toEqual([[turn.stored, result.text]])
    return result.text
  }
  async function retained(turn) {
    await expect.poll(async () => (await users(page).allInnerTexts()).join('\n').split(turn.text).length - 1).toBe(1)
    await expect.poll(async () => (await replies(page).allInnerTexts())
      .filter(text => normalize(text) === normalize(turn.reply)).length).toBe(1)
    assert.deepEqual(rows(turn.profile, turn.text), [[turn.stored, turn.text]])
    assert.equal(sent('prompt.submit').filter(f => f.params.text === turn.text).length, 1)
  }
  try {
    const probe = await context.request.get(`${runtime.url}/api/config`, { headers })
    assert.equal(probe.status(), 200)
    assert.equal((await probe.json()).model, config.model.default)
    await page.goto(runtime.url)
    await expect(input(page)).toBeEditable({ timeout: 60000 })
    await body({ page, frames, sent, events, observations, submit, complete, retained, context })
    for (const { turn } of observations.filter(item => item.turn)) {
      for (const profile of [...profiles, 'default'].filter(profile => profile !== turn.profile)) {
        for (const role of ['user', 'assistant']) {
          const text = role === 'assistant' ? turn.reply : turn.text
          assert.deepEqual(rows(profile, text, role), [], `${turn.profile}'s message leaked into ${profile}`)
        }
      }
    }
    assert.deepEqual(errors, [])
    assert.deepEqual(blocked.filter(r => !['font', 'stylesheet'].includes(r.type)), [])
    results.push({ name, status: 'PASS', elapsedMs: Date.now() - started })
    console.log(`PASS: ${name}`)
  } catch (error) {
    results.push({ name, status: 'FAIL', elapsedMs: Date.now() - started, error: error.stack })
    console.error(`FAIL: ${name}\n${error.stack}`)
    process.exitCode = 1
  } finally {
    try {
      await page.screenshot({ path: path.join(artifacts, `${name}.png`) })
      await writeFile(path.join(artifacts, `${name}.aria.txt`), await page.locator('body').ariaSnapshot())
    } finally {
      await Promise.all(pendingHttp)
      await writeFile(path.join(artifacts, `${name}.json`), JSON.stringify({
        url: page.url(), frames, http, observations, errors, blocked
      }, null, 2))
      await rm(holdFile, { force: true })
      await context.close()
    }
  }
}

async function readConfig(context, profile) {
  const response = await context.request.get(`${runtime.url}/api/config?profile=${profile}`, { headers })
  assert.equal(response.status(), 200)
  return response.json()
}
const maxTurns = page => page.locator('[data-tour="field-agent.max_turns"] input')
async function openSettings(page) {
  await page.getByRole('button', { name: 'Open settings', exact: true }).click()
  await page.getByRole('button', { name: 'Advanced', exact: true }).click()
  await expect(maxTurns(page)).toBeVisible()
}

try {
  await check('approvals-a-b-a', async ({ page, frames, submit, complete, observations }) => {
    await selectProfile(page, profiles[0])
    const a = await submit(profiles[0], 'spike: approval isolation-a')
    await expect(page.getByRole('button', { name: 'Reject', exact: true })).toBeVisible({ timeout: 60000 })
    await selectProfile(page, profiles[1])
    await expect(page.getByRole('button', { name: 'Reject', exact: true })).not.toBeVisible()
    const b = await submit(profiles[1], 'spike: approval isolation-b')
    await expect(page.getByRole('button', { name: 'Reject', exact: true })).toBeVisible({ timeout: 60000 })
    const approval = turn => frames.findLast(f => f.direction === 'received' && f.method === 'approval' &&
      f.params.session_id === turn.live)
    assert.ok(approval(a)); assert.ok(approval(b))
    assert.notEqual(approval(a).params.request_id, approval(b).params.request_id)
    await selectProfile(page, profiles[0])
    // Selecting a profile intentionally opens a fresh draft; reopen its pending session.
    await row(page, 'approval isolation-a').click()
    await expect(page.getByRole('button', { name: 'Reject', exact: true })).toBeVisible()
    await page.screenshot({ path: path.join(artifacts, 'approvals-both-pending-return-a.png') })
    // Both commands share the fixture target. Deny B before allowing A to remove it.
    for (const [turn, choice, label] of [[b, 'deny', 'Reject'], [a, 'once', /^Run\s*↵$/]]) {
      await selectProfile(page, turn.profile)
      await row(page, `approval ${turn.profile}`).click()
      await expect(page.getByRole('button', { name: 'Reject', exact: true })).toBeVisible()
      const pending = approval(turn)
      await page.getByRole('button', { name: label, exact: true }).click()
      await expect.poll(() => frames.find(f => f.direction === 'sent' && f.id === pending.id &&
        f.socket === pending.socket)?.result?.choice).toBe(choice)
      const reply = await complete(turn)
      assert.match(reply, choice === 'deny' ? /denied/i : /"exit_code": 0/)
      if (choice === 'deny') {
        await access(path.join(runtime.home, 'approval-target'))
        assert.equal(frames.some(f => f.direction === 'sent' && f.id === approval(a).id && f.result), false)
      } else await assert.rejects(access(path.join(runtime.home, 'approval-target')), { code: 'ENOENT' })
      observations.push({ owner: turn.profile, live: turn.live, requestId: pending.params.request_id, choice, reply })
    }
  })

  await check('config-api-a-b-a', async ({ context, observations }) => {
    for (const [i, expected, value] of [[0, 7, 8], [1, 11, 12], [0, 8, 9]]) {
      const profile = profiles[i]
      const before = await readConfig(context, profile)
      assert.equal(before.agent.max_turns, expected)
      const sibling = profiles[1 - i]
      const siblingBefore = (await readConfig(context, sibling)).agent.max_turns
      const response = await context.request.put(`${runtime.url}/api/config?profile=${profile}`, {
        headers, data: { config: { agent: { max_turns: value } } }
      })
      assert.equal(response.status(), 200)
      assert.equal((await response.json()).ok, true)
      const after = await readConfig(context, profile)
      assert.equal(after.agent.max_turns, value)
      assert.equal((await readConfig(context, sibling)).agent.max_turns, siblingBefore)
      observations.push({ profile, before: before.agent.max_turns, after: after.agent.max_turns })
    }
    assert.equal((await readConfig(context, 'default')).agent.max_turns, config.agent.max_turns)
  })

  await check('settings-delayed-b-response', async ({ page, context, observations }) => {
    const [a, b] = await Promise.all(profiles.map(async profile =>
      String((await readConfig(context, profile)).agent.max_turns)))
    assert.notEqual(a, b)
    await selectProfile(page, profiles[0])
    await openSettings(page)
    await expect(maxTurns(page)).toHaveValue(a)
    await page.getByRole('button', { name: 'Close settings', exact: true }).click()
    await expect(maxTurns(page)).toHaveCount(0)

    const release = Promise.withResolvers()
    const held = []
    const mismatches = []
    // Fetch stock B data now, but deliver none of it until Settings has mounted.
    await page.route(url => url.origin === runtime.url && url.pathname === '/api/config' &&
      url.searchParams.get('profile') === profiles[1], async route => {
      if (route.request().method() !== 'GET') return route.fallback()
      const response = await route.fetch()
      held.push({ url: route.request().url(), status: response.status(), body: await response.json() })
      await release.promise
      await route.fulfill({ response })
    })
    // Observe from before mounting, so even a transient editable A seed fails.
    const probe = await page.evaluateHandle(() => {
      const values = []
      const sample = () => {
        const field = document.querySelector('[data-tour="field-agent.max_turns"] input')
        if (field && !field.matches(':disabled') && !field.readOnly && !values.includes(field.value)) {
          values.push(field.value)
        }
      }
      const observer = new MutationObserver(sample)
      observer.observe(document.body, { childList: true, subtree: true, attributes: true })
      return { values, sample, observer }
    })
    try {
      await selectProfile(page, profiles[1])
      await page.getByRole('button', { name: 'Open settings', exact: true }).click()
      await page.getByRole('button', { name: 'Advanced', exact: true }).click()
      await expect.poll(() => new URLSearchParams(new URL(page.url()).hash.split('?')[1]).get('tab')).toBe('config:advanced')
      await expect(maxTurns(page).or(page.locator('[data-slot="skeleton"]')).first()).toBeVisible()
      await expect.poll(() => held.length).toBeGreaterThan(0)
      for (const response of held) {
        assert.equal(response.status, 200)
        assert.equal(String(response.body.agent.max_turns), b)
      }
      if (await maxTurns(page).count()) await maxTurns(page).scrollIntoViewIfNeeded()
      await page.screenshot({ path: path.join(artifacts, 'settings-b-response-held.png') })
      await writeFile(path.join(artifacts, 'settings-b-response-held.aria.txt'), await page.locator('body').ariaSnapshot())
      const values = await probe.evaluate(({ values, sample }) => { sample(); return values })
      observations.push({ phase: 'B response held', cachedA: a, expectedB: b, editableValues: values, held })
      try { assert.equal(values.includes(a), false, 'A must never seed an editable B settings field while B is pending') }
      catch (error) { mismatches.push(error.stack) }
    } finally {
      await probe.evaluate(({ observer }) => observer.disconnect())
      await probe.dispose()
      release.resolve()
      await page.unrouteAll({ behavior: 'wait' })
    }
    try { await expect(maxTurns(page)).toHaveValue(b) }
    catch (error) { mismatches.push(error.stack) }
    observations.push({ phase: 'B response released', expected: b, actual: await maxTurns(page).inputValue() })
    await page.getByRole('button', { name: 'Close settings', exact: true }).click()
    await selectProfile(page, profiles[0])
    await openSettings(page)
    await expect(maxTurns(page)).toHaveValue(a)
    observations.push({ phase: 'return A', expected: a, actual: await maxTurns(page).inputValue() })
    assert.deepEqual(mismatches, [], 'Settings drafts must belong to the active profile before and after its response')
  })

  await check('settings-active-a-b-a', async ({ page, context, observations }) => {
    const mismatches = []
    for (const profile of [...profiles, profiles[0]]) {
      await selectProfile(page, profile)
      await openSettings(page)
      const expected = (await readConfig(context, profile)).agent.max_turns
      try { await expect(maxTurns(page)).toHaveValue(String(expected), { timeout: 15000 }) }
      catch (error) { mismatches.push({ profile, expected, actual: await maxTurns(page).inputValue(), error: error.stack }) }
      observations.push({ at: Date.now(), profile, expected, actual: await maxTurns(page).inputValue() })
      await maxTurns(page).scrollIntoViewIfNeeded()
      await writeFile(path.join(artifacts, `settings-active-${observations.length}.aria.txt`), await page.locator('body').ariaSnapshot())
      await page.screenshot({ path: path.join(artifacts, `settings-active-${observations.length}.png`) })
      await page.getByRole('button', { name: 'Close settings', exact: true }).click()
    }
    assert.deepEqual(mismatches, [], 'Settings must show the active profile, not the previous cached draft')
  })

  await check('settings-explicit-scope', async ({ page, context, observations }) => {
    await selectProfile(page, profiles[0])
    await openSettings(page)
    // Scope chips have no aria-pressed; rail and status-bar controls are outside this group.
    const scope = page.getByText('Applies to', { exact: true }).locator('..')
    for (const profile of [...profiles, profiles[0]]) {
      await scope.getByRole('button', { name: profile, exact: true }).click()
      const expected = Number((await readConfig(context, profile)).agent.max_turns)
      const sibling = profiles.find(p => p !== profile)
      const siblingBefore = (await readConfig(context, sibling)).agent.max_turns
      await expect(maxTurns(page)).toHaveValue(String(expected))
      const value = expected + 1
      await maxTurns(page).fill(String(value))
      await expect.poll(async () => Number((await readConfig(context, profile)).agent.max_turns)).toBe(value)
      assert.equal((await readConfig(context, sibling)).agent.max_turns, siblingBefore, 'Scope save must not write sibling')
      observations.push({ scope: profile, saved: value,
        all: await Promise.all(profiles.map(async p => [p, (await readConfig(context, p)).agent.max_turns])) })
    }
    await page.getByRole('button', { name: 'Close settings', exact: true }).click()
    await expect(page.locator('button[aria-pressed="true"]').and(page.getByRole('button', {
      name: profiles[0], exact: true
    }))).toBeVisible()
    assert.equal((await readConfig(context, 'default')).agent.max_turns, config.agent.max_turns)
  })

  await check('background-stream-a-b-a', async ({ page, sent, events, submit, complete, retained, observations }) => {
    await selectProfile(page, profiles[1])
    const b = await submit(profiles[1], 'spike: foreground-b-marker')
    await complete(b)
    await selectProfile(page, profiles[0])
    await writeFile(holdFile, 'background-stream-a-b-a')
    const a = await submit(profiles[0], 'spike: hold background-a-marker complete tail')
    await expect.poll(() => events(a, 'message.delta').length, { timeout: 60000 }).toBeGreaterThan(0)
    await selectProfile(page, profiles[1])
    await row(page, 'foreground-b-marker').click()
    await retained(b)
    const draft = 'UNSENT private B draft'
    await input(page).fill(draft)
    const route = page.url()
    const transcript = { users: await users(page).allInnerTexts(), replies: await replies(page).allInnerTexts() }
    const deltas = events(a, 'message.delta').length
    await rm(holdFile)
    await complete(a)
    assert.ok(events(a, 'message.delta').length > deltas, 'A must stream while B is active')
    assert.equal(a.reply, `Spike turn 1: ${a.text}`)
    assert.equal(page.url(), route, 'Background completion must not steal B route')
    await expect(input(page)).toHaveText(draft)
    assert.deepEqual({ users: await users(page).allInnerTexts(), replies: await replies(page).allInnerTexts() }, transcript)
    assert.equal(sent('prompt.submit').filter(f => f.params.text === draft).length, 0)
    observations.push({ route, draft, transcript, backgroundDeltas: events(a, 'message.delta').length - deltas })
    await page.screenshot({ path: path.join(artifacts, 'background-a-complete-b-still-active.png') })
    await selectProfile(page, profiles[0])
    await row(page, 'background-a-marker').click()
    await retained(a)
    const resumeStart = sent('session.resume').length
    await page.reload()
    await retained(a)
    await expect.poll(() => sent('session.resume').slice(resumeStart).some(f => f.params.session_id === a.stored)).toBe(true)
    const next = await submit(profiles[0], 'spike: resumed-a-followup')
    await complete(next)
    assert.equal(next.stored, a.stored)
    await retained(a)
    await retained(next)
    const requests = (await readFile(path.join(runtime.run_dir, 'model-requests.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse)
    const history = requests.findLast(r => JSON.stringify(r.messages.at(-1)?.content).includes(next.text)).messages
    assert.equal(JSON.stringify(history).includes(b.text), false)
    assert.equal(JSON.stringify(history).includes(draft), false)
    for (const turn of [a, next]) assert.equal(history.filter(m => m.role === 'user' && m.content === turn.text).length, 1)
    assert.equal(history.filter(m => m.role === 'assistant' && m.content === a.reply).length, 1)
  })
} finally {
  await browser.close()
  await writeFile(path.join(artifacts, 'results.json'), JSON.stringify({ runtimePath, results }, null, 2))
  console.log(`Evidence: ${artifacts}`)
}
