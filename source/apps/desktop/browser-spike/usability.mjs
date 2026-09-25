import assert from 'node:assert/strict'
import { chmod, mkdir, readFile, realpath, stat, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { chromium, expect } from '@playwright/test'

// Run after build:browser and backend.py. Only this fixture's loopback origin is allowed.
const runtimePath = process.argv[2]
if (!runtimePath) throw new Error('Usage: node browser-spike/usability.mjs <runtime.json> [artifact directory]')
const runtime = JSON.parse(await readFile(runtimePath, 'utf8'))
assert.equal(new URL(runtime.url).hostname, '127.0.0.1')
assert.equal(new URL(runtime.model_url).hostname, '127.0.0.1')
assert.equal(runtime.public_url, null, 'Use a private loopback fixture, not a public or live service')
assert.equal(path.dirname(runtime.run_dir), os.tmpdir())
assert.ok(path.basename(runtime.run_dir).startsWith('hermes-browser-spike-'))
assert.equal(await realpath(runtime.run_dir), runtime.run_dir)
assert.equal(path.resolve(runtimePath), path.join(runtime.run_dir, 'runtime.json'))
assert.equal(runtime.home, path.join(runtime.run_dir, 'home'))
assert.equal(runtime.hermes_home, path.join(runtime.home, '.hermes'))
const config = JSON.parse(await readFile(path.join(runtime.hermes_home, 'config.yaml'), 'utf8'))
assert.equal(config.model.default, 'browser-spike-local')
assert.equal(config.model.provider, 'custom')
assert.equal(config.model.base_url, runtime.model_url)
const artifacts = process.argv[3] || path.join(runtime.run_dir, 'usability-evidence')
await mkdir(artifacts, { recursive: true })
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true
})
const frames = [],
  errors = [],
  blocked = [],
  results = [],
  downloads = [],
  boundaryAttempts = [],
  boundaryHttpFailures = [],
  boundaryRequestFailures = []
let nextSocketId = 0
const headers = { 'X-Hermes-Session-Token': runtime.token }
const profiles = ['usability-a', 'usability-b']
const png = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAEUlEQVR4nGMQmvDuPwgzwBgAVeQKPT6g3A0AAAAASUVORK5CYII=',
  'base64'
)
const requestFrames = (start = 0) => frames.slice(start).filter(f => f.direction === 'sent')
const attachFrames = start =>
  requestFrames(start).filter(f => ['image.attach_bytes', 'file.attach', 'image.attach'].includes(f.method))
const eventsSince = start =>
  frames
    .slice(start)
    .filter(f => f.direction === 'received' && f.method === 'event')
    .map(f => f.params)

// These are renderer-policy tripwires, not a claim that stock APIs deny writes.
// Agent tools and ordinary project/config RPCs remain stock and are not mocked.
function forbiddenBoundaryPath(url, method) {
  const pathname = decodeURIComponent(new URL(url).pathname)
  return (
    /^\/api\/hermes\/update(?:\/|$)/.test(pathname) ||
    /^\/api\/plugins\/browser-terminal(?:\/|$)/.test(pathname) ||
    /^\/api\/fs\/write-text(?:\/|$)/.test(pathname) ||
    (pathname.startsWith('/api/fs/') && !['GET', 'HEAD'].includes(method))
  )
}

async function newPage(boundary = false) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true })
  await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: runtime.url })
  await context.route('**/*', route => {
    const url = new URL(route.request().url())
    if (boundary && forbiddenBoundaryPath(url.href, route.request().method())) {
      boundaryAttempts.push({ url: url.href, method: route.request().method() })
      return route.abort('blockedbyclient')
    }
    if (url.origin === runtime.url || ['data:', 'blob:'].includes(url.protocol)) return route.continue()
    blocked.push({ url: url.href, resourceType: route.request().resourceType() })
    return route.abort('blockedbyclient')
  })
  await context.routeWebSocket('**/*', ws => {
    const url = new URL(ws.url())
    if (boundary && forbiddenBoundaryPath(url.href, 'GET')) {
      boundaryAttempts.push({ url: url.href, method: 'websocket' })
      ws.close()
    } else if (url.origin === runtime.url.replace('http:', 'ws:')) {
      const server = ws.connectToServer()
      if (boundary)
        ws.onMessage(payload => {
          const message = JSON.parse(String(payload))
          if (/^(?:terminal|pty)\.(?:create|open|start|spawn)$/.test(message.method || '')) {
            boundaryAttempts.push({ url: url.href, method: message.method })
            ws.close()
          } else server.send(payload)
        })
    } else {
      blocked.push({ url: ws.url(), resourceType: 'websocket' })
      ws.close()
    }
  })
  const page = await context.newPage()
  page.setDefaultTimeout(15000)
  page.on('pageerror', error => errors.push(error.stack))
  if (boundary)
    page.on('requestfailed', request => {
      boundaryRequestFailures.push({ url: request.url(), error: request.failure()?.errorText })
    })
  page.on('response', response => {
    if (boundary && response.status() >= 400) {
      boundaryHttpFailures.push({ url: response.url(), status: response.status() })
    }
    if (new URL(response.url()).pathname === '/api/fs/download') {
      downloads.push({
        url: response.url(),
        status: response.status(),
        headers: response.headers(),
        requestHeaders: response.request().headers()
      })
    }
  })
  page.on('websocket', ws => {
    const socket = ++nextSocketId
    ws.on('close', () => frames.push({ direction: 'closed', socket }))
    const transportProfile = new URL(ws.url()).searchParams.get('profile') || 'default'
    for (const [event, direction] of [
      ['framereceived', 'received'],
      ['framesent', 'sent']
    ]) {
      ws.on(event, ({ payload }) => {
        const message = JSON.parse(String(payload))
        // Tile RPCs may omit profile: the real backend binds their runtime
        // session to the owner returned by session.create/session.resume.
        const sessionOwner = message.params?.session_id
          ? frames.findLast(f => f.direction === 'received' && f.result?.session_id === message.params.session_id)
              ?.result?.info?.profile_name
          : undefined
        frames.push({
          direction,
          socket,
          transportProfile,
          profile: message.params?.profile || sessionOwner || transportProfile,
          ...message
        })
      })
    }
  })
  await page.goto(runtime.url)
  await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible({ timeout: 60000 })
  return page
}

async function check(name, body, { boundary = false } = {}) {
  let page
  const attemptsStart = boundaryAttempts.length
  const failuresStart = boundaryHttpFailures.length
  try {
    page = await newPage(boundary)
    await body(page)
    if (boundary) {
      assert.deepEqual(boundaryAttempts.slice(attemptsStart), [], 'Forbidden browser-boundary requests (blocked)')
      assert.deepEqual(
        boundaryHttpFailures.slice(failuresStart).filter(f => new URL(f.url).pathname.startsWith('/api/fs/')),
        [],
        'Read-only filesystem HTTP failures (other optional API failures are recorded separately)'
      )
    }
    results.push({ name, status: 'PASS' })
    console.log(`PASS: ${name}`)
  } catch (error) {
    results.push({ name, status: 'FAIL', error: error.stack })
    console.error(`FAIL: ${name}\n${error.stack}`)
    process.exitCode = 1
  } finally {
    if (page) {
      await page.screenshot({ path: path.join(artifacts, `${name}.png`) })
      await writeFile(path.join(artifacts, `${name}.aria.txt`), await page.locator('body').ariaSnapshot())
      await page.context().close()
    }
  }
}

async function send(page, text) {
  const start = frames.length
  const input = page.getByRole('textbox', { name: 'Message', exact: true })
  await input.fill(text)
  await input.press('Enter')
  await expect.poll(() => eventsSince(start).some(e => e.type === 'message.complete'), { timeout: 60000 }).toBe(true)
  const events = eventsSince(start)
  const completed = events.find(e => e.type === 'message.complete').payload
  assert.notEqual(completed.status, 'error', JSON.stringify(completed))
  assert.equal(
    events.some(e => e.type === 'error'),
    false
  )
  assert.ok(completed.text.includes(text), completed.text)
  return start
}

async function selectProfile(page, profile) {
  const button = page.getByRole('button', { name: profile, exact: true })
  await button.click()
  await expect(button).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeEditable()
}

function sessionRow(page, marker) {
  // Single-session lists have no reorder wrapper; target either row shape
  // inside the sidebar, never a matching transcript or tab caption.
  return page
    .locator('[data-tree-group="grp-sessions"]')
    .getByRole('button', { name: new RegExp(marker) })
    .first()
}

async function openPalette(page) {
  const palette = page.getByRole('dialog', { name: 'Command palette', exact: true })
  // Toggle actions can keep it open. The shortcut works on both layouts;
  // the newer status-bar context menu only configures visibility.
  if (!(await palette.isVisible())) await page.keyboard.press('Control+k')
  await expect(palette.getByRole('combobox')).toBeVisible()
  // Keyboard-first lists ignore pointers until mouse movement. Enter through
  // the input before Playwright checks option hit targets for its clicks.
  await palette.getByRole('combobox').hover()
  return palette
}

async function openSettingsSection(page, section, subpage) {
  await page.getByRole('button', { name: 'Open settings', exact: true }).click()
  await page.getByRole('complementary').getByRole('button', { name: section, exact: true }).click()
  // Upstream splits some settings into subpages; do not encode route URLs.
  if (subpage) {
    const button = page.getByRole('complementary').getByRole('button', { name: subpage, exact: true })
    if (await button.isVisible()) await button.click()
  }
}

async function menuClick(page) {
  await page.getByRole('button', { name: 'Add context', exact: true }).click()
  await page.getByRole('menuitem', { name: /Prompt snippets/ }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog).toBeVisible()
  await dialog.getByRole('button', { name: /Code review/ }).click()
  await expect(dialog).not.toBeVisible()
  await expect(page.getByRole('textbox', { name: 'Message', exact: true })).not.toBeEmpty()
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('')
}

async function geometry(page, percent) {
  const button = page.getByRole('button', { name: 'Open settings', exact: true })
  await expect
    .poll(() => button.evaluate(el => el.getBoundingClientRect().width / el.offsetWidth))
    .toBeCloseTo(percent / 100, 2)
  const control = await button.evaluate(el => ({
    width: el.getBoundingClientRect().width,
    height: el.getBoundingClientRect().height,
    layoutWidth: el.offsetWidth
  }))
  const layout = await page.getByRole('contentinfo').evaluate(el => ({
    shell: el.getBoundingClientRect().toJSON(),
    viewport: { width: innerWidth, height: innerHeight }
  }))
  return { ...control, ...layout }
}

async function pick(page, files, images = false) {
  await page.getByRole('button', { name: 'Add context', exact: true }).click()
  const [chooser] = await Promise.all([
    page.waitForEvent('filechooser'),
    page.getByRole('menuitem', { name: images ? 'Images…' : 'Files…', exact: true }).click()
  ])
  await chooser.setFiles(files)
}

async function dropFiles(page, files) {
  const input = page.getByRole('textbox', { name: 'Message', exact: true })
  // DOM dispatch alone skips the pointer entry that selects the receiving pane.
  await input.hover()
  // Browser DataTransfer carries real Files, not fabricated server paths.
  // This exercises DOM drag handlers, not OS file-manager drag integration.
  await input.evaluate(
    (el, files) => {
      const transfer = new DataTransfer()
      for (const file of files)
        transfer.items.add(
          new File([Uint8Array.from(atob(file.base64), c => c.charCodeAt(0))], file.name, { type: file.mimeType })
        )
      for (const name of ['dragenter', 'dragover', 'drop'])
        el.dispatchEvent(new DragEvent(name, { bubbles: true, cancelable: true, dataTransfer: transfer }))
    },
    files.map(file => ({ name: file.name, mimeType: file.mimeType, base64: file.buffer.toString('base64') }))
  )
}

async function modelRequests() {
  return (await readFile(path.join(runtime.run_dir, 'model-requests.jsonl'), 'utf8'))
    .trim()
    .split('\n')
    .map(line => JSON.parse(line))
}

try {
  const probe = await browser.newContext()
  const response = await probe.request.get(`${runtime.url}/api/config`, { headers })
  assert.equal(response.status(), 200)
  assert.equal((await response.json()).model, config.model.default)
  // This disposable fixture has no credentials to clone and no external models.
  for (const profile of profiles) {
    const home = path.join(runtime.hermes_home, 'profiles', profile)
    await mkdir(path.join(home, 'workspace'), { recursive: true })
    // The loopback model accepts image_url parts. Keep the profile's attachment
    // directory inside its workspace so @file expansion exercises the real bytes.
    await writeFile(
      path.join(home, 'config.yaml'),
      JSON.stringify({
        ...config,
        model: { ...config.model, supports_vision: true },
        agent: { ...config.agent, image_input_mode: 'native' },
        terminal: { ...config.terminal, cwd: home }
      })
    )
    await writeFile(
      path.join(home, 'workspace', 'résumé & report.bin'),
      Buffer.concat([Buffer.from(`${profile}\n`), Buffer.from([0, 255, 1, 128])])
    )
  }
  await probe.close()

  await check('scale-and-menus', async page => {
    await menuClick(page)
    const baseline = await geometry(page, 90)
    const measurements = [{ percent: 90, ...baseline }]
    for (const percent of [100, 125]) {
      await openSettingsSection(page, 'Appearance', 'Typography')
      await page.getByRole('button', { name: `${percent}%`, exact: true }).click()
      await page.getByRole('button', { name: 'Close settings', exact: true }).click()
      const measured = await geometry(page, percent)
      measurements.push({ percent, ...measured })
      assert.ok(Math.abs(measured.width / baseline.width - percent / 90) < 0.02)
      await menuClick(page)
    }
    await page.reload()
    await geometry(page, 125)
    await menuClick(page)
    measurements.push({ percent: 125, persisted: true, ...(await geometry(page, 125)) })
    await writeFile(path.join(artifacts, 'scale-geometry.json'), JSON.stringify(measurements, null, 2))
    for (const { percent, shell, viewport } of measurements) {
      assert.ok(
        Math.abs(shell.left) <= 2 &&
          Math.abs(shell.width - viewport.width) <= 2 &&
          Math.abs(shell.bottom - viewport.height) <= 2,
        `At ${percent}%, shell is x=${shell.left}, width=${shell.width}, bottom=${shell.bottom}; viewport is ${viewport.width}×${viewport.height}`
      )
    }
  })

  await check('attachments-a-b-a', async page => {
    const completedMarkers = []
    for (const [index, method] of ['picker', 'drop', 'paste'].entries()) {
      const profile = profiles[index % 2]
      await selectProfile(page, profile)
      const marker = `usability-${method}-${Date.now()}`
      const text = Buffer.from(`${marker}: nonimage bytes\n`)
      const files = [
        { name: `${marker}.png`, mimeType: 'image/png', buffer: png },
        { name: `${marker}.txt`, mimeType: 'text/plain', buffer: text }
      ]
      // Dragging New session onto the composer creates a fresh tile, rather
      // than testing the primary pane's unbound draft as if it were a tile.
      await page
        .getByRole('button', { name: /^New session Ctrl/ })
        .dragTo(page.getByRole('textbox', { name: 'Message', exact: true }))
      await expect(
        page.locator('[data-session-anchor^="session-tile:"]').getByRole('textbox', { name: 'Message', exact: true })
      ).toBeEditable()
      const staged = frames.length
      if (method === 'picker') {
        await pick(page, [files[0]], true)
        await pick(page, [files[1]])
      } else if (method === 'drop') await dropFiles(page, files)
      else {
        // A genuine Chromium clipboard image paste. OS nonimage file clipboard
        // paste is not a supported web ClipboardItem type and is not covered.
        await page.evaluate(async base64 => {
          const blob = new Blob([Uint8Array.from(atob(base64), c => c.charCodeAt(0))], { type: 'image/png' })
          await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
        }, png.toString('base64'))
        await page.getByRole('textbox', { name: 'Message', exact: true }).press('Control+v')
      }
      const chips = page.locator('[data-slot="composer-attachments"]')
      await expect(chips.getByRole('button', { name: /^Remove / })).toHaveCount(method === 'paste' ? 1 : 2)
      assert.equal(
        attachFrames(staged).length,
        0,
        'FRESH tile drafts keep files local until submit; main existing-session files deliberately eager-upload'
      )
      const submitted = await send(page, `spike: ${marker}`)
      const attaches = attachFrames(submitted)
      assert.deepEqual(
        attaches.map(f => f.method).sort(),
        method === 'paste' ? ['image.attach_bytes'] : ['file.attach', 'image.attach_bytes']
      )
      assert.ok(
        attaches.every(f => f.profile === profile),
        'Attachment RPC must target the selected profile'
      )
      const image = attaches.find(f => f.method === 'image.attach_bytes')
      const file = attaches.find(f => f.method === 'file.attach')
      if (file) assert.deepEqual(Buffer.from(file.params.data_url.split(',')[1], 'base64'), text)
      // Clipboard serialization can re-encode PNG; other entry paths must preserve bytes.
      if (method !== 'paste') assert.deepEqual(Buffer.from(image.params.content_base64, 'base64'), png)
      let imagePath
      for (const attach of attaches) {
        const reply = frames.find(f => f.direction === 'received' && f.socket === attach.socket && f.id === attach.id)
        assert.equal(reply?.result?.attached, true, JSON.stringify(reply))
        const storedPath = await realpath(reply.result.path)
        if (attach === image) imagePath = storedPath
        assert.ok(storedPath.startsWith(path.join(runtime.hermes_home, 'profiles', profile) + path.sep), storedPath)
        const bytes = await readFile(storedPath)
        assert.deepEqual(bytes, attach === file ? text : Buffer.from(image.params.content_base64, 'base64'))
      }
      await expect(chips).not.toBeVisible()
      const resumeStart = frames.length
      await page.reload()
      // Reopen the saved tile session explicitly; cold boot can land on the
      // primary workspace rather than this profile's tile tab.
      await selectProfile(page, profile)
      await sessionRow(page, marker).click()
      await expect
        .poll(() => requestFrames(resumeStart).some(f => f.method === 'session.resume'), { timeout: 60000 })
        .toBe(true)
      const resumedUser = page
        .locator('[data-slot="aui_user-message-root"]')
        .filter({ hasText: `spike: ${marker}` })
        .last()
      await expect(resumedUser).toBeVisible({ timeout: 60000 })
      // Attachment thumbnails are flow siblings of the sticky user bubble.
      const resumedImage = page.getByRole('img', { name: imagePath, exact: true })
      await expect(resumedImage).toBeVisible()
      await expect.poll(() => resumedImage.evaluate(img => img.naturalWidth)).toBeGreaterThan(0)
      await send(page, `spike: resumed-${marker}`)
      const requests = await modelRequests()
      const request = requests.findLast(r => JSON.stringify(r.messages).includes(`resumed-${marker}`))
      assert.ok(request, 'The local model must receive the resumed turn')
      for (const previous of completedMarkers.filter(previous => previous.profile !== profile)) {
        assert.ok(
          !JSON.stringify(request.messages).includes(previous.marker),
          'A sibling profile transcript must not leak into model history'
        )
      }
      completedMarkers.push({ profile, marker })
      if (file)
        assert.ok(
          JSON.stringify(request.messages).includes(text.toString().trim()),
          'Nonimage content must survive resume'
        )
      assert.ok(
        request.messages.some(
          m =>
            m.role === 'user' &&
            Array.isArray(m.content) &&
            m.content.some(
              p => p.type === 'image_url' && p.image_url.url === `data:image/png;base64,${image.params.content_base64}`
            )
        ),
        'Image content must survive resume into real model history'
      )
      results.push({ name: `${profile}-${method}-submit-resume`, status: 'PASS' })
      console.log(`PASS: ${profile} ${method} submit/resume`)
    }
  })

  await check('existing-main-attachments-a-b-a', async page => {
    for (const [index, method] of ['picker', 'drop', 'paste'].entries()) {
      const profile = profiles[index % 2]
      await selectProfile(page, profile)
      const seed = profile === profiles[0] ? 'picker' : 'drop'
      // Cached transcript paint can precede resume. Existing-session attachment
      // checks need the live recipient, not a still-hydrating draft.
      const resumeStart = frames.length
      await sessionRow(page, `usability-${seed}-`).click()
      await expect
        .poll(
          () => {
            const request = requestFrames(resumeStart).find(f => f.method === 'session.resume' && f.profile === profile)
            if (!request) return null
            return frames.find(
              f => f.direction === 'received' && f.socket === request.socket && f.id === request.id
            )?.result?.session_id
          },
          { timeout: 60000 }
        )
        .toBeTruthy()
      const main = page.locator('[data-session-anchor="workspace"]')
      await expect(main.getByRole('button', { name: 'Edit message', exact: true }).first()).toBeVisible()
      await expect(main.getByRole('textbox', { name: 'Message', exact: true })).toBeEditable()
      const marker = `existing-${method}-${Date.now()}`
      const text = Buffer.from(`${marker}: eager file bytes\n`)
      const files = [
        { name: `${marker}.png`, mimeType: 'image/png', buffer: png },
        { name: `${marker}.txt`, mimeType: 'text/plain', buffer: text }
      ]
      const staged = frames.length
      if (method === 'picker') {
        await pick(page, [files[0]], true)
        await pick(page, [files[1]])
      } else if (method === 'drop') await dropFiles(page, files)
      else {
        await page.evaluate(async base64 => {
          const blob = new Blob([Uint8Array.from(atob(base64), c => c.charCodeAt(0))], { type: 'image/png' })
          await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
        }, png.toString('base64'))
        await main.getByRole('textbox', { name: 'Message', exact: true }).press('Control+v')
      }
      const chips = main.locator('[data-slot="composer-attachments"]')
      await expect(chips.getByRole('button', { name: /^Remove / })).toHaveCount(method === 'paste' ? 1 : 2)
      if (method !== 'paste') {
        await expect.poll(() => attachFrames(staged).filter(f => f.method === 'file.attach').length).toBe(1)
        const eager = attachFrames(staged)[0]
        await expect
          .poll(
            () =>
              frames.find(f => f.direction === 'received' && f.socket === eager.socket && f.id === eager.id)?.result
                ?.attached
          )
          .toBe(true)
      }
      assert.deepEqual(
        attachFrames(staged).map(f => f.method),
        method === 'paste' ? [] : ['file.attach']
      )
      const submitted = await send(page, `spike: ${marker}`)
      assert.deepEqual(
        attachFrames(submitted).map(f => f.method),
        ['image.attach_bytes'],
        'Submit must not upload the eager file twice'
      )
      for (const attach of attachFrames(staged)) {
        assert.equal(attach.profile, profile)
        const reply = frames.find(f => f.direction === 'received' && f.socket === attach.socket && f.id === attach.id)
        assert.equal(reply?.result?.attached, true, JSON.stringify(reply))
        const storedPath = await realpath(reply.result.path)
        assert.ok(storedPath.startsWith(path.join(runtime.hermes_home, 'profiles', profile) + path.sep), storedPath)
        assert.deepEqual(
          await readFile(storedPath),
          attach.method === 'file.attach' ? text : Buffer.from(attach.params.content_base64, 'base64')
        )
      }
      await expect(chips).not.toBeVisible()
      await page.reload()
      await expect(
        page
          .locator('[data-session-anchor="workspace"]')
          .getByRole('button', { name: 'Edit message', exact: true })
          .filter({ hasText: marker })
      ).toBeVisible({ timeout: 60000 })
      await send(page, `spike: resumed-${marker}`)
      const request = (await modelRequests()).findLast(r => JSON.stringify(r.messages).includes(`resumed-${marker}`))
      assert.ok(request, 'Existing-session attachments must reach the real model after resume')
      if (method !== 'paste') assert.ok(JSON.stringify(request.messages).includes(text.toString().trim()))
      const image = attachFrames(staged).find(f => f.method === 'image.attach_bytes')
      assert.ok(
        request.messages.some(
          m =>
            m.role === 'user' &&
            Array.isArray(m.content) &&
            m.content.some(
              p => p.type === 'image_url' && p.image_url.url === `data:image/png;base64,${image.params.content_base64}`
            )
        )
      )
      results.push({ name: `${profile}-existing-main-${method}-submit-resume`, status: 'PASS' })
    }
  })

  await check('downloads-a-b-a', async page => {
    for (const profile of [...profiles, profiles[0]]) {
      await selectProfile(page, profile)
      const filename = 'résumé & report.bin'
      const file = path.join(runtime.hermes_home, 'profiles', profile, 'workspace', filename)
      const unauthorized = await page.request.get(
        `${runtime.url}/api/fs/download?path=${encodeURIComponent(file)}&profile=${profile}`
      )
      assert.equal(unauthorized.status(), 401, 'Downloads must require authentication')
      await send(page, `spike: [fixture download](#media:${encodeURIComponent(file)})`)
      const start = downloads.length
      const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.getByRole('button', { name: 'Download', exact: true }).last().click()
      ])
      assert.equal(download.suggestedFilename(), filename)
      const destination = path.join(artifacts, `${profile}-${start}.bin`)
      await download.saveAs(destination)
      assert.deepEqual(await readFile(destination), await readFile(file))
      const request = downloads[start]
      assert.equal(request.status, 200)
      assert.equal(new URL(request.url).searchParams.get('profile'), profile)
      assert.equal(request.requestHeaders['x-hermes-session-token'], runtime.token)
      assert.match(request.headers['content-disposition'], /^attachment;/)
      results.push({ name: `${profile}-download-bytes-filename`, status: 'PASS' })
    }
  })
  await check('relative-download-no-workspace-visible-error', async page => {
    await selectProfile(page, profiles[0])
    const start = frames.length
    // Keep a fresh tile connected: orphan reap + cold resume can persist the
    // terminal default cwd, so an earlier session is not a no-workspace fixture.
    await page
      .getByRole('button', { name: /^New session Ctrl/ })
      .dragTo(page.getByRole('textbox', { name: 'Message', exact: true }))
    const tile = page.locator('[data-session-anchor^="session-tile:"]')
    await expect(tile.getByRole('textbox', { name: 'Message', exact: true })).toBeEditable()
    const relative = './workspace/résumé & report.bin'
    await send(page, `spike: [no workspace download](#media:${encodeURIComponent(relative)})`)
    const storedId = (await tile.getAttribute('data-session-anchor')).slice('session-tile:'.length)
    const sessionUrl = `${runtime.url}/api/sessions/${encodeURIComponent(storedId)}?profile=${profiles[0]}`
    const persisted = await page.request.get(sessionUrl, { headers })
    assert.equal(persisted.status(), 200)
    const session = await persisted.json()
    assert.equal(session.profile, profiles[0])
    assert.equal(session.cwd, null, 'Negative fixture must have no persisted workspace')
    const downloadButton = tile.getByRole('button', { name: 'Download', exact: true }).last()
    const [unavailable] = await Promise.all([
      page.waitForResponse(response => new URL(response.url()).pathname === '/api/fs/download'),
      downloadButton.click()
    ])
    assert.equal(unavailable.status(), 400)
    const query = new URL(unavailable.url()).searchParams
    assert.equal(query.get('path'), relative)
    assert.equal(query.get('profile'), profiles[0])
    assert.equal(query.get('session_id'), storedId)
    const detail = (await unavailable.json()).detail
    assert.equal(detail, 'Session working directory is unavailable')
    const alert = page.getByRole('alert').filter({ hasText: 'Download failed' })
    await expect(alert).toBeVisible()
    await expect(alert).toContainText(detail)
    await expect(downloadButton).toBeEnabled()
    const after = await page.request.get(sessionUrl, { headers })
    assert.equal(after.status(), 200)
    assert.equal((await after.json()).cwd, null)
    assert.equal(
      frames.slice(start).some(f => f.direction === 'closed'),
      false,
      'Negative fixture must stay connected'
    )
    assert.equal(
      requestFrames(start).some(f => f.method === 'session.resume'),
      false
    )
    await page.screenshot({ path: path.join(artifacts, 'relative-download-no-workspace.png') })
    await writeFile(
      path.join(artifacts, 'relative-download-no-workspace.json'),
      JSON.stringify({ ...downloads.at(-1), session, detail, alert: await alert.innerText() }, null, 2)
    )
    await alert.getByRole('button', { name: 'Dismiss notification', exact: true }).click()
    await expect(alert).not.toBeVisible()
  })
  await check('relative-download-a-tile-b-foreground', async page => {
    const filename = 'résumé & report.bin'
    const relative = `./workspace/${filename}`
    await selectProfile(page, profiles[0])
    await sessionRow(page, 'usability-picker-').click()
    await expect(
      page
        .locator('[data-session-anchor="workspace"]')
        .getByRole('button', { name: 'Edit message', exact: true })
        .first()
    ).toBeVisible()
    await send(page, `spike: [owner relative download](#media:${encodeURIComponent(relative)})`)
    await selectProfile(page, profiles[1])
    await sessionRow(page, 'usability-drop-').click()
    const main = page.locator('[data-session-anchor="workspace"]')
    await expect(main.getByRole('button', { name: 'Edit message', exact: true }).first()).toBeVisible()
    await page.getByRole('button', { name: 'Filters', exact: true }).click()
    await page.getByRole('menuitemcheckbox', { name: 'All profiles', exact: true }).click()
    await page.keyboard.press('Escape')
    await sessionRow(page, 'usability-picker-').click({ button: 'right' })
    await page.getByRole('menuitem', { name: 'Open in new tab', exact: true }).click()
    const tile = page.locator('[data-session-anchor^="session-tile:"]')
    await expect(tile.getByRole('button', { name: 'Download', exact: true }).last()).toBeVisible()
    const storedId = (await tile.getAttribute('data-session-anchor')).slice('session-tile:'.length)
    const tabBox = await page.locator(`[data-tree-tab="session-tile:${storedId}"]`).boundingBox()
    const paneBox = await tile.boundingBox()
    // Real tab drag: keep both transcripts visible, then focus B's main composer.
    await page.mouse.move(tabBox.x + tabBox.width / 3, tabBox.y + tabBox.height / 2)
    await page.mouse.down()
    await page.mouse.move(paneBox.x + paneBox.width - 20, paneBox.y + paneBox.height / 2, { steps: 20 })
    await page.mouse.up()
    await expect(main).toBeVisible()
    await expect(tile).toBeVisible()
    await main.getByRole('textbox', { name: 'Message', exact: true }).click()
    const foreground = page.getByRole('contentinfo').getByRole('button', { name: profiles[1], exact: true })
    await expect(foreground).toBeVisible()
    const downloadButton = tile.getByRole('button', { name: 'Download', exact: true }).last()
    // Projects' folder picker is native in local browser mode. Use the public
    // RPC instead, with A's actual live id (not the stored id used by downloads).
    const owner = frames.findLast(
      f =>
        f.direction === 'received' &&
        (f.result?.session_key === storedId ||
          f.result?.resumed === storedId ||
          f.result?.stored_session_id === storedId)
    )?.result
    assert.ok(owner?.session_id, 'A must have a live session from the real resume frames')
    assert.equal(owner.info.profile_name, profiles[0])
    const cwd = path.join(runtime.hermes_home, 'profiles', profiles[0])
    const wsUrl = new URL('/api/ws', runtime.url.replace('http:', 'ws:'))
    wsUrl.searchParams.set('token', runtime.token)
    wsUrl.searchParams.set('profile', profiles[0])
    const reply = await page.evaluate(
      ({ url, sessionId, cwd, profile }) =>
        new Promise((resolve, reject) => {
          const ws = new WebSocket(url)
          const timer = setTimeout(() => {
            ws.close()
            reject(new Error('session.cwd.set timed out'))
          }, 15000)
          ws.onopen = () =>
            ws.send(
              JSON.stringify({
                jsonrpc: '2.0',
                id: 'fixture-set-workspace',
                method: 'session.cwd.set',
                params: { session_id: sessionId, cwd, profile }
              })
            )
          ws.onmessage = event => {
            const reply = JSON.parse(event.data)
            if (reply.id !== 'fixture-set-workspace') return
            clearTimeout(timer)
            resolve(reply)
            ws.close()
          }
          ws.onclose = () => {
            clearTimeout(timer)
            reject(new Error('Workspace socket closed before reply'))
          }
        }),
      { url: wsUrl.href, sessionId: owner.session_id, cwd, profile: profiles[0] }
    )
    assert.equal(reply.error, undefined, JSON.stringify(reply))
    assert.equal(reply.result.cwd, cwd)
    await expect(foreground).toBeVisible()
    const start = downloads.length
    // Invoke the real tile button without pointer hover/focus-follow switching
    // the foreground to A first. No bridge, resolver, or transport is mocked.
    // Capture HTTP failure bodies rather than hiding them behind a download
    // timeout. Closing the page cancels the pending event on that failure path.
    const downloadEvent = page.waitForEvent('download').catch(() => null)
    const [response] = await Promise.all([
      page.waitForResponse(response => new URL(response.url()).pathname === '/api/fs/download'),
      downloadButton.evaluate(button => button.click())
    ])
    await expect(foreground).toBeVisible()
    const request = downloads[start]
    const query = new URL(request.url).searchParams
    assert.equal(query.get('path'), relative)
    assert.equal(query.get('profile'), profiles[0])
    assert.equal(query.get('session_id'), storedId, 'Relative path must resolve in A’s owning session')
    assert.equal(request.requestHeaders['x-hermes-session-token'], runtime.token)
    if (!response.ok()) {
      const detail = await response.text()
      await writeFile(
        path.join(artifacts, 'relative-download-error.json'),
        JSON.stringify({ ...request, detail }, null, 2)
      )
      assert.fail(`Owning A session resolved, but relative download returned HTTP ${response.status()}: ${detail}`)
    }
    assert.equal(request.status, 200)
    assert.match(request.headers['content-disposition'], /^attachment;/)
    const download = await downloadEvent
    assert.ok(download, 'A successful response must produce a browser download')
    assert.equal(download.suggestedFilename(), filename)
    const destination = path.join(artifacts, 'a-tile-b-foreground.bin')
    await download.saveAs(destination)
    const source = path.join(runtime.hermes_home, 'profiles', profiles[0], 'workspace', filename)
    assert.deepEqual(await readFile(destination), await readFile(source))
    assert.notDeepEqual(
      await readFile(destination),
      await readFile(path.join(runtime.hermes_home, 'profiles', profiles[1], 'workspace', filename))
    )
  })
  await check(
    'server-file-preview-and-downloads',
    async page => {
      // Run after the attachment/download cases: creating a project establishes a
      // workspace and must not change their deliberately workspace-less drafts.
      const profile = profiles[1]
      const workspace = path.join(runtime.hermes_home, 'profiles', profile, 'workspace')
      const files = [
        { name: 'boundary résumé.txt', bytes: Buffer.from('Read-only boundary — žluťoučký 🐚\n'), mode: 0o640 },
        { name: 'boundary raw.bin', bytes: Buffer.from([0, 255, 128, 1, 13, 10, 254]), mode: 0o444 }
      ]
      for (const file of files) {
        file.path = path.join(workspace, file.name)
        await writeFile(file.path, file.bytes)
        await chmod(file.path, file.mode)
      }
      try {
        await selectProfile(page, profile)
        await page.getByRole('button', { name: 'Filters', exact: true }).click()
        await page.getByRole('menuitem', { name: /^Grouping/ }).hover()
        await page.getByRole('menuitemradio', { name: 'Project', exact: true }).click()
        await page.keyboard.press('Escape')
        await page.keyboard.press('Escape')
        await page.getByRole('button', { name: 'New project', exact: true }).click()
        const dialog = page.getByRole('dialog', { name: 'New project', exact: true })
        const projectName = `Read-only boundary ${Date.now()}`
        await dialog.getByRole('textbox').fill(projectName)
        await expect(dialog.getByPlaceholder(/saved to IDEA\.md/)).toHaveCount(0)
        await expect(dialog.getByRole('button', { name: /Generate idea|Shuffle/ })).toHaveCount(0)
        await dialog.getByRole('button', { name: 'Add folder', exact: true }).click()
        const picker = page.getByRole('dialog', { name: 'Choose remote folder', exact: true })
        await expect(picker).toBeVisible()
        // Start from a known breadcrumb, not an assumed profile working directory.
        await picker.getByRole('button', { name: '/', exact: true }).click()
        for (const part of workspace.split(path.sep).filter(Boolean)) {
          await picker.getByRole('button', { name: part, exact: true }).click()
        }
        await picker.getByRole('button', { name: 'Select folder', exact: true }).click()
        await expect(dialog).toContainText(workspace)
        const created = frames.length
        await dialog.getByRole('button', { name: 'Create', exact: true }).click()
        await expect(dialog).not.toBeVisible()
        const createRequest = requestFrames(created).find(f => f.method === 'projects.create')
        assert.ok(createRequest, 'Ordinary project creation must reach the stock API')
        await expect
          .poll(
            () =>
              frames.find(
                f => f.direction === 'received' && f.socket === createRequest.socket && f.id === createRequest.id
              )?.result
          )
          .toBeTruthy()
        await page.reload()
        await selectProfile(page, profile)
        const showProjects = page.getByRole('button', { name: 'Show projects', exact: true })
        const projectRow = page
          .locator('[data-tree-group="grp-sessions"]')
          .getByRole('button', { name: new RegExp(projectName) })
          .first()
        // The status-bar project button opens a menu, not the sidebar project.
        await expect(showProjects.or(projectRow).first()).toBeVisible()
        if (await showProjects.isVisible()) await showProjects.click()
        await projectRow.click()
        const tree = page.getByRole('complementary', { name: 'Right sidebar', exact: true })
        if (!(await tree.isVisible())) {
          await page.getByRole('button', { name: 'Show right sidebar', exact: true }).click()
        }
        await expect(tree).toBeVisible()
        for (const [index, file] of files.entries()) {
          const row = tree.getByText(file.name, { exact: true })
          await expect(row).toBeVisible()
          await row.dblclick()
          const tab = page.getByRole('tab', { name: new RegExp(file.name) })
          await expect(tab).toBeVisible()
          const preview = page.locator('[data-tree-group]').filter({ has: tab })
          await expect(preview).toBeVisible()
          if (index === 0) {
            const content = page.getByText(file.bytes.toString().trim(), { exact: true })
            await expect(content).toBeVisible()
            // Enter the real editor, but never change text or explicitly save.
            await content.click()
            await page.keyboard.press('e')
            const editor = preview.getByRole('textbox')
            await expect(editor).toBeVisible()
            await expect(editor).toBeEditable()
            await expect(editor).toContainText(file.bytes.toString().trim())
            const warning = preview.getByRole('note')
            await expect(warning).toBeVisible()
            await expect(warning).toContainText('No autosave.')
            await expect(warning).toContainText('may reset permissions and other metadata')
            await expect(preview.getByRole('button', { name: 'Save to server', exact: true })).toBeDisabled()
            await preview.getByRole('button', { name: 'Cancel', exact: true }).click()
            await expect(content).toBeVisible()
          } else {
            await expect(page.getByText('This looks like a binary file', { exact: true })).toBeVisible()
            await expect(preview.getByRole('button', { name: /^(Edit|Save|Save to server|Overwrite)$/ })).toHaveCount(0)
          }
          // After cancel (or for binary files), only the read preview remains.
          await expect(preview.getByRole('textbox')).toHaveCount(0)
          await row.click({ button: 'right' })
          await expect(
            page.getByRole('menuitem', {
              name: /^(Rename|Delete|Reveal in Finder|Reveal in File Explorer|Open containing folder)$/
            })
          ).toHaveCount(0)
          await expect(page.getByRole('menuitem', { name: 'Copy path', exact: true })).toBeVisible()
          const [download] = await Promise.all([
            page.waitForEvent('download'),
            page.getByRole('menuitem', { name: /^Download/ }).click()
          ])
          assert.equal(download.suggestedFilename(), file.name)
          const destination = path.join(artifacts, file.name)
          await download.saveAs(destination)
          assert.deepEqual(await readFile(destination), file.bytes)
          await page.screenshot({ path: path.join(artifacts, `read-only-boundary-${index}.png`) })
        }
        assert.equal(
          requestFrames(created).some(f => f.method === 'prompt.submit'),
          false,
          'An empty-idea project must not generate an idea or start a model turn'
        )
      } finally {
        for (const file of files) {
          assert.deepEqual(await readFile(file.path), file.bytes, `${file.name}: source bytes changed`)
          assert.equal((await stat(file.path)).mode & 0o7777, file.mode, `${file.name}: source mode changed`)
        }
      }
    },
    { boundary: true }
  )

  await check(
    'unsupported-native-actions',
    async page => {
      await openSettingsSection(page, 'Appearance', /^Themes?$/)
      const themes = page.getByPlaceholder('Search built-in themes…', { exact: true })
      await expect(themes).toBeVisible()
      await themes.fill('boundary-no-such-theme')
      await expect(page.getByText(/No installed themes match/)).toBeVisible()
      await expect(page.getByRole('button', { name: /^Install/ })).toHaveCount(0)
      await themes.fill('')
      await page.getByRole('button', { name: 'Close settings', exact: true }).click()
      for (const section of ['Advanced', 'Sessions', 'About']) {
        await openSettingsSection(page, section === 'Sessions' ? /^(Sessions|Archived Chats)$/ : section)
        // Negative assertions only count after the real section has loaded.
        if (section === 'Advanced') await expect(page.getByText('Applies to', { exact: true })).toBeVisible()
        if (section === 'Sessions') await expect(page.getByText('Archived sessions', { exact: true })).toBeVisible()
        if (section === 'About')
          await expect(page.getByRole('link', { name: 'Release notes', exact: true })).toBeVisible()
        for (const label of ['Keep computer awake', 'Disable F12 DevTools', 'Default project directory']) {
          await expect(page.getByText(label, { exact: true })).toHaveCount(0)
        }
        await expect(
          page.getByRole('button', {
            name: /^(Check now|Check for updates|Update now|Update Hermes|Install update|Restart Hermes|Restart backend|Uninstall Hermes)$/i
          })
        ).toHaveCount(0)
        await expect(page.getByRole('link', { name: /^(Get the installer|Install Hermes locally)$/ })).toHaveCount(0)
        await page.getByRole('button', { name: 'Close settings', exact: true }).click()
      }
      const palette = await openPalette(page)
      await expect(palette.getByRole('option', { name: /^Reload window/ })).toBeVisible()
      await expect(
        palette.getByRole('option', {
          name: /^(Update Hermes|Install theme|Open browser|New terminal|Restart backend|Uninstall Hermes)/
        })
      ).toHaveCount(0)
      await palette.getByRole('combobox').fill('Toggle terminal')
      await palette.getByRole('option', { name: /^Toggle terminal/ }).click()
      await expect(page.getByText('No terminal in this profile', { exact: true })).toBeVisible()
      await expect(page.getByRole('button', { name: 'New terminal', exact: true })).toHaveCount(0)
      await expect(page.getByRole('textbox', { name: /Terminal input/i })).toHaveCount(0)
      await expect(
        page.getByRole('contentinfo').getByRole('button', { name: /Update Hermes|Install update/ })
      ).toHaveCount(0)
      await page.screenshot({ path: path.join(artifacts, 'native-boundary-terminal.png') })
      const reloadPalette = await openPalette(page)
      await reloadPalette.getByRole('combobox').fill('Reload window')
      await Promise.all([
        page.waitForEvent('domcontentloaded'),
        reloadPalette.getByRole('option', { name: /^Reload window/ }).click()
      ])
      await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible({ timeout: 60000 })
    },
    { boundary: true }
  )
  await check(
    'native-link-context-menu',
    async page => {
      await openSettingsSection(page, 'About')
      const link = page.getByRole('link', { name: 'Release notes', exact: true })
      await expect(link).toBeVisible()
      await link.click({ button: 'right' })
      await expect(page.getByRole('menuitem', { name: 'Open in external browser', exact: true })).toBeVisible()
      await expect(page.getByRole('menuitem', { name: /Open in in-app browser|Inspect element/ })).toHaveCount(0)
      await page.keyboard.press('Escape')
      await page.screenshot({ path: path.join(artifacts, 'native-boundary-about.png') })
    },
    { boundary: true }
  )
  assert.deepEqual(errors, [], 'Chromium page errors')
  // Optional remote font styles are deliberately unavailable in this isolated run.
  assert.deepEqual(
    blocked.filter(r => !['font', 'stylesheet'].includes(r.resourceType)),
    [],
    'Unexpected outbound requests were blocked'
  )
} finally {
  await writeFile(
    path.join(artifacts, 'results.json'),
    JSON.stringify(
      { results, errors, blocked, downloads, boundaryAttempts, boundaryHttpFailures, boundaryRequestFailures },
      null,
      2
    )
  )
  await writeFile(path.join(artifacts, 'frames.json'), JSON.stringify(frames, null, 2))
  await browser.close()
  console.log(`Evidence: ${artifacts}`)
}
