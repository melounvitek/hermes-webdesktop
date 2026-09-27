import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { createHash, randomBytes } from 'node:crypto'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import http from 'node:http'
import https from 'node:https'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { chromium, expect } from '@playwright/test'
import { startProcess, stopProcess, waitFor } from './processes.mjs'

// Build first: cd apps/desktop && npm run build:browser
// SPIKE_BACKEND_ROOT=/absolute/clean/stock SPIKE_PYTHON=/absolute/venv/bin/python \
//   node browser-spike/updates.mjs [dist-browser] [evidence-parent]
// FULL browser app, real HTTP updater client, scripted protocol/auth responses.
// NOT stock updater compatibility, real login, service restart, or deployment proof.
const here = path.dirname(fileURLToPath(import.meta.url))
const desktop = path.dirname(here)
const dist = path.resolve(process.argv[2] || path.join(desktop, 'dist-browser'))
const evidence = path.resolve(process.argv[3] || path.join(os.tmpdir(), 'hermes-updater-evidence'))
const python = process.env.SPIKE_PYTHON
const backendRoot = process.env.SPIKE_BACKEND_ROOT
assert.ok(python && path.isAbsolute(python), 'SPIKE_PYTHON must name an absolute test interpreter')
assert.ok(backendRoot && path.isAbsolute(backendRoot), 'SPIKE_BACKEND_ROOT must name a clean, credential-free stock checkout')
process.umask(0o077)
await mkdir(evidence, { recursive: true })
const artifacts = await mkdtemp(path.join(evidence, 'run-'))
const temporary = await mkdtemp(path.join(os.tmpdir(), 'hermes-updater-chromium-'))
const json = value => JSON.stringify(value, null, 2)
const requests = [], violations = [], errors = [], consoleErrors = [], blocked = [], results = [], assertions = []
const report = {
  scope: 'Full browser app with scripted HTTPS updater/auth protocol fixture; not stock compatibility or service integration',
  started: new Date().toISOString(), artifacts, dist, requests, violations, errors, consoleErrors, blocked, results, assertions
}
const revision = execFileSync('git', ['-C', desktop, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim()
report.sourceRevision = revision
report.sourceDiffSha256 = createHash('sha256').update(execFileSync('git', ['-C', desktop, 'diff', 'HEAD'])).digest('hex')
report.indexSha256 = createHash('sha256').update(await readFile(path.join(dist, 'index.html'))).digest('hex')
const cookie = `__Host-updater-fixture=${randomBytes(16).toString('hex')}`
let browser, server, fixture, runtime, origin, state, activeCase, activePage
const sockets = new Set(), upstreamSockets = new Set()
let cleanupPromise
const helperRequests = () => requests.filter(r => r.path.startsWith('/browser-updater/'))
const count = endpoint => helperRequests().filter(r => r.path.endsWith(`/${endpoint}`)).length
const offer = (seconds = 600) => ({
  id: randomBytes(16).toString('hex'), current_release: `browser-${revision.slice(0, 12)}`,
  target_release: `browser-fixture-future-${randomBytes(6).toString('hex')}`,
  expires_at: Date.now() / 1000 + seconds,
  warning: 'Protocol fixture only. All users disconnect; chats and terminals may stop; rollback can restart; login may expire. Hermes is not upgraded.',
  tested_backend: runtime.stock.revision, compatibility: 'not-exercised'
})
const snapshot = (phase, selected = state.offer) => ({
  capabilities: ['install'], phase,
  ...(!['idle', 'preparing'].includes(phase) ? { offer: selected } : {}),
  ...(['stopping', 'switching', 'starting', 'rollback_stopping', 'rollback_switching', 'rollback_starting',
    'succeeded', 'rolled_back', 'recovery_required', 'failed'].includes(phase) ? { job: { id: selected.id, phase } } : {})
})
function reset() { state = { offer: offer(), phase: 'idle', statusCode: 200, sessionCode: 200, applyCode: 200 } }
async function prove(name, body) {
  await body()
  assertions.push({ case: activeCase, name, status: 'PASS' })
  console.log(`PASS: ${name}`)
}
async function shot(page, name) {
  await page.screenshot({ path: path.join(artifacts, `${name}.png`) })
  await writeFile(path.join(artifacts, `${name}.aria.txt`), await page.locator('body').ariaSnapshot())
}
function reply(response, code, value) {
  response.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }).end(JSON.stringify(value))
}
async function handle(request, response) {
  const pathname = new URL(request.url, origin).pathname
  if (pathname.startsWith('/browser-updater/') || pathname === '/api/auth/me') {
    let body = ''
    for await (const bytes of request) body += bytes
    const row = { case: activeCase, time: new Date().toISOString(), method: request.method,
      path: pathname, headers: request.headers, body }
    requests.push(row)
    try {
      assert.equal(request.headers.host, new URL(origin).host)
      assert.ok(request.headers.cookie?.split('; ').includes(cookie), 'Secure HttpOnly fixture cookie required')
      assert.equal(request.headers.accept, 'application/json')
      assert.equal(request.headers['sec-fetch-site'], 'same-origin')
      assert.equal(request.headers.authorization, undefined)
      assert.equal(request.headers['x-hermes-session-token'], undefined, 'Updater must not borrow the dashboard bootstrap token')
      if (pathname === '/api/auth/me') {
        assert.equal(request.method, 'GET')
        assert.equal(body, '')
        row.status = state.sessionCode
        reply(response, row.status, row.status === 200
          ? { provider: 'fixture', user_id: 'fixture-admin', org_id: null } : { error: 'fixture expired session' })
        return
      }
      assert.equal(request.method, 'POST')
      assert.equal(request.headers.origin, origin, 'Exact current HTTPS origin required')
      assert.equal(request.headers['content-type'], 'application/json')
      assert.equal(request.headers['sec-fetch-mode'], 'same-origin')
      const endpoint = pathname.slice('/browser-updater/api/'.length)
      assert.ok(['status', 'offer', 'apply'].includes(endpoint))
      if (endpoint === 'apply') {
        assert.deepEqual(JSON.parse(body), { offer_id: state.offer.id, confirm_restart_and_rollback: true })
        row.status = state.applyCode
        if (row.status === 200) state.phase = 'stopping'
      } else {
        assert.equal(body, '{}')
        row.status = endpoint === 'status' ? state.statusCode : 200
        if (endpoint === 'offer') state.phase = 'preparing'
      }
      row.response = row.status === 200 ? snapshot(state.phase) : { error: `scripted ${row.status}` }
      reply(response, row.status, row.response)
    } catch (error) {
      violations.push({ ...row, error: error.stack })
      row.status = 400
      reply(response, 400, { error: 'fixture request contract violated' })
    }
    return
  }
  if (pathname === '/login') {
    requests.push({ case: activeCase, method: request.method, path: request.url, status: 200 })
    response.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' })
      .end('<!doctype html><title>Fixture sign in</title><h1>Disposable protocol fixture sign in</h1><p>No real authentication is performed.</p>')
    return
  }
  if (!runtime) { response.writeHead(503).end(); return }
  const upstream = http.request(runtime.url, {
    path: request.url, method: request.method, agent: false,
    headers: { ...request.headers, 'x-forwarded-proto': 'https' }
  }, remote => {
    const headers = { ...remote.headers }
    if (pathname === '/') headers['set-cookie'] = `${cookie}; Path=/; Secure; HttpOnly; SameSite=Strict`
    response.writeHead(remote.statusCode, headers)
    remote.pipe(response)
  })
  upstream.on('error', () => { if (!response.headersSent) response.writeHead(503); response.end() })
  request.pipe(upstream)
}
async function start() {
  const ca = path.join(temporary, 'ca'), leaf = path.join(temporary, 'leaf')
  const openssl = args => execFileSync('openssl', args, { stdio: 'ignore' })
  openssl(['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1', '-keyout', `${ca}.key`,
    '-out', `${ca}.crt`, '-subj', '/CN=Disposable updater test CA'])
  openssl(['req', '-new', '-newkey', 'rsa:2048', '-nodes', '-keyout', `${leaf}.key`,
    '-out', `${leaf}.csr`, '-subj', '/CN=127.0.0.1'])
  await writeFile(`${leaf}.ext`, 'subjectAltName=IP:127.0.0.1\nbasicConstraints=CA:FALSE\nextendedKeyUsage=serverAuth\n')
  openssl(['x509', '-req', '-in', `${leaf}.csr`, '-CA', `${ca}.crt`, '-CAkey', `${ca}.key`,
    '-CAcreateserial', '-out', `${leaf}.crt`, '-days', '1', '-extfile', `${leaf}.ext`])
  server = https.createServer({ key: await readFile(`${leaf}.key`), cert: await readFile(`${leaf}.crt`) },
    (request, response) => { void handle(request, response).catch(error => {
      violations.push({ error: error.stack }); response.destroy()
    }) })
  server.on('connection', socket => { sockets.add(socket); socket.on('close', () => sockets.delete(socket)) })
  server.on('upgrade', (request, socket, head) => {
    if (!runtime) { socket.destroy(); return }
    const upstream = net.connect(Number(new URL(runtime.url).port), '127.0.0.1', () => {
      upstream.write(`${request.method} ${request.url} HTTP/${request.httpVersion}\r\n` +
        Object.entries({ ...request.headers, 'x-forwarded-proto': 'https' })
          .map(([key, value]) => `${key}: ${value}\r\n`).join('') + '\r\n')
      upstream.write(head); upstream.pipe(socket); socket.pipe(upstream)
    })
    upstreamSockets.add(upstream)
    upstream.on('close', () => { upstreamSockets.delete(upstream); socket.destroy() })
    socket.on('close', () => upstream.destroy())
    socket.on('error', () => upstream.destroy())
    upstream.on('error', () => socket.destroy())
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  origin = `https://127.0.0.1:${server.address().port}`
  report.origin = origin
  report.tls = 'Temporary CA-signed IP SAN leaf; ignoreHTTPSErrors in test browser context only; no system trust changes'
  const env = { PATH: '/usr/bin:/bin', HOME: temporary, LANG: 'C.UTF-8', TZ: 'UTC', PYTHONDONTWRITEBYTECODE: '1' }
  fixture = startProcess(python, [path.join(here, 'backend.py'), '--backend-root', backendRoot,
    '--python', python, '--web-dist', dist], { cwd: temporary, env, logPath: path.join(artifacts, 'fixture.log') })
  await fixture.ready
  const ready = await waitFor(() => {
    if (fixture.failure) throw fixture.failure
    assert.ok(!fixture.closed, `Fixture exited: ${fixture.output}`)
    const line = fixture.output.match(/^READY (.+)$/m)?.[1]
    return line && JSON.parse(line)
  }, 100000, 'disposable full-app backend')
  runtime = JSON.parse(await readFile(path.join(ready.run_dir, 'runtime.json'), 'utf8'))
  assert.equal(runtime.harness_pid, fixture.child.pid)
  assert.equal(new URL(runtime.url).hostname, '127.0.0.1')
  assert.equal(new URL(runtime.model_url).hostname, '127.0.0.1')
  assert.equal(runtime.public_url, null)
  assert.equal(path.dirname(runtime.run_dir), os.tmpdir())
  assert.ok(path.basename(runtime.run_dir).startsWith('hermes-browser-spike-'))
  report.runtime = { run_dir: runtime.run_dir, stock: runtime.stock, backend_pid: runtime.backend_pid,
    url: runtime.url, model_url: runtime.model_url }
  browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
    headless: true, env: { ...env, HOME: temporary } })
  report.chromium = browser.version()
}
const panel = page => page.getByRole('dialog', { name: 'Browser updates', exact: true })
const confirmation = page => page.getByRole('dialog', { name: 'Update and restart web desktop?', exact: true })
const button = (scope, name) => scope.getByRole('button', { name, exact: true })
const update = page => button(panel(page), 'Update and restart')
const retry = page => button(panel(page), 'Retry status check')
const loaded = page => panel(page).locator('dl > div').filter({ has: page.getByText('Browser build loaded in this tab', { exact: true }) }).locator('dd')
const beforeRelease = page => panel(page).locator('dl > div').filter({ has: page.getByText('Release before this update', { exact: true }) }).locator('dd')
const offeredRelease = page => panel(page).locator('dl > div').filter({ has: page.getByText('Release offered for this update', { exact: true }) }).locator('dd')
async function open(page) {
  await page.getByRole('contentinfo').getByRole('button', { name: /^Browser / }).click()
  await expect(panel(page)).toBeVisible()
  await expect(retry(page)).toBeEnabled()
}
async function prepare(page) {
  await button(panel(page), 'Check and download').click()
  await expect(panel(page)).toContainText('Downloading and verifying the release…')
  state.phase = 'offered'
  await expect(update(page)).toBeEnabled({ timeout: 15000 })
}
async function run(name, body, { mobile = false, file = false } = {}) {
  activeCase = name
  reset()
  const before = { requests: requests.length, applies: count('apply'), assertions: assertions.length, errors: errors.length }
  const context = await browser.newContext({ viewport: mobile ? { width: 390, height: 844 } : { width: 1440, height: 1000 },
    ignoreHTTPSErrors: true, reducedMotion: 'reduce' })
  await context.route('**/*', route => {
    const url = new URL(route.request().url())
    if (url.origin === origin || ['data:', 'blob:'].includes(url.protocol)) return route.continue()
    blocked.push({ case: name, url: url.href, type: route.request().resourceType() })
    return route.abort('blockedbyclient')
  })
  await context.routeWebSocket('**/*', ws => {
    if (new URL(ws.url()).origin === origin.replace('https:', 'wss:')) ws.connectToServer()
    else { blocked.push({ case: name, url: ws.url(), type: 'websocket' }); ws.close() }
  })
  await context.tracing.start({ screenshots: true, snapshots: true })
  let filePath
  if (file) {
    filePath = path.join(runtime.home, 'updater-draft.txt')
    await writeFile(filePath, 'Original fixture file; never save this acceptance draft.\n')
    await context.addInitScript(({ file, url }) => {
      localStorage.setItem('hermes.desktop.previewTabs.v2', JSON.stringify([{ id: `file:${file}`, target: {
        kind: 'file', label: 'updater-draft.txt', path: file, source: file, url, previewKind: 'text'
      } }]))
      localStorage.setItem('hermes.desktop.rightRailActiveTab', `file:${file}`)
    }, { file: filePath, url: pathToFileURL(filePath).href })
  }
  const page = await context.newPage()
  activePage = page
  page.setDefaultTimeout(15000)
  const navigations = []
  page.on('pageerror', error => errors.push({ case: name, error: error.stack }))
  page.on('console', message => { if (message.type() === 'error') consoleErrors.push({ case: name, text: message.text() }) })
  page.on('framenavigated', frame => { if (frame === page.mainFrame()) navigations.push(frame.url()) })
  try {
    await page.goto(origin)
    await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible({ timeout: 60000 })
    await body(page, { filePath, navigations, before })
    assert.equal(errors.length, before.errors, 'Unexpected UI JavaScript errors')
    results.push({ name, status: 'PASS', assertions: assertions.length - before.assertions,
      requests: requests.length - before.requests, applies: count('apply') - before.applies, navigations })
  } catch (error) {
    results.push({ name, status: 'FAIL', error: error.stack, assertions: assertions.length - before.assertions,
      requests: requests.length - before.requests, applies: count('apply') - before.applies, navigations })
    console.error(`FAIL: ${name}\n${error.stack}`)
    process.exitCode = 1
  } finally {
    await shot(page, name)
    await context.tracing.stop({ path: path.join(artifacts, `${name}.trace.zip`) })
    await context.close()
    activePage = null
  }
}
async function cleanup() {
  if (cleanupPromise) return cleanupPromise
  cleanupPromise = (async () => {
    await browser?.close()
    for (const socket of [...sockets, ...upstreamSockets]) socket.destroy()
    if (server) await new Promise(resolve => server.close(resolve))
    if (fixture) report.fixtureCleanup = await stopProcess(fixture)
    if (runtime) {
      for (const [label, url] of [['backend', runtime.url], ['model', runtime.model_url], ['https', origin]]) {
        await waitFor(() => new Promise(resolve => {
          const socket = net.connect(Number(new URL(url).port), '127.0.0.1')
          socket.on('connect', () => { socket.destroy(); resolve(false) })
          socket.on('error', error => resolve(error.code === 'ECONNREFUSED'))
        }), 5000, `${label} port closed`)
      }
      await writeFile(path.join(artifacts, 'backend.log'), await readFile(runtime.backend_log))
      await rm(runtime.run_dir, { recursive: true, force: true })
    }
    await rm(temporary, { recursive: true, force: true })
    report.cleanup = 'Owned browser, fixture and HTTPS proxy stopped; all three listening ports refused; temporary homes and TLS keys removed'
  })()
  return cleanupPromise
}
const interrupt = () => { process.exitCode = 1; void cleanup() }
process.on('SIGINT', interrupt); process.on('SIGTERM', interrupt)
try {
  await start()
  await run('success-preserves-drafts', async (page, { filePath, navigations, before }) => {
    const footer = page.getByRole('contentinfo').getByRole('button', { name: /^Browser / })
    const input = page.getByRole('textbox', { name: 'Message', exact: true })
    const draft = 'Round3: keep my unsent composer draft.'
    await input.fill(draft)
    await button(page, 'Edit').click()
    const editor = page.locator('.cm-editor .cm-content[contenteditable="true"]')
    await editor.click(); await editor.press('Control+a'); await page.keyboard.insertText('Round3: keep my unsaved file draft.')
    await prove('Git footer is visible before any updater request', async () => {
      await expect(footer).toBeVisible()
      const shortRevision = (await footer.innerText()).match(/^Browser ([a-f0-9]{7,40})(?:\s|$)/)?.[1]
      assert.ok(shortRevision && revision.startsWith(shortRevision), 'Footer must identify the loaded Git build')
      assert.equal(helperRequests().length, 0)
      await expect(button(page, 'Save to server')).toBeEnabled()
    })
    const footerText = await footer.innerText()
    await shot(page, '01-footer-and-real-drafts')
    const initialURL = page.url(), initialNavigations = navigations.length
    await open(page)
    await prove('Opening footer only reads POST status; no download or apply', async () => {
      assert.equal(count('status'), 1); assert.equal(count('offer'), 0); assert.equal(count('apply'), 0)
      await expect(loaded(page)).toContainText(revision)
      await expect(update(page)).toBeDisabled()
    })
    const loadedText = await loaded(page).innerText()
    await prepare(page)
    await prove('Explicit Check and download prepares a future offered release', async () => {
      assert.equal(count('offer'), 1)
      assert.notEqual(state.offer.current_release, state.offer.target_release)
      await expect(panel(page)).toContainText(state.offer.target_release)
      await expect(panel(page)).toContainText('This does not certify compatibility with your current backend.')
    })
    await update(page).click()
    await prove('Confirmation names exact releases and all interruption/recovery warnings', async () => {
      await expect(confirmation(page)).toContainText(`Update from ${state.offer.current_release} to ${state.offer.target_release}?`)
      for (const warning of ['All browser users will be disconnected.', 'Running dashboard chats may be interrupted and terminals may close.',
        'Finish your work and save it first.', 'If activation fails, the previous release will be restored and restarted.',
        'Hermes itself will not be upgraded.', 'You may need to sign in again.']) await expect(confirmation(page)).toContainText(warning)
    })
    await shot(page, '02-exact-confirmation')
    await button(confirmation(page), 'Cancel').focus()
    await page.keyboard.press('Enter')
    await prove('Keyboard Enter on Cancel never applies', async () => {
      await expect(confirmation(page)).not.toBeVisible(); await expect(panel(page)).toBeVisible()
      assert.equal(count('apply'), before.applies)
    })
    await update(page).click(); await page.keyboard.press('Escape')
    await prove('Escape closes only top confirmation, not parent', async () => {
      await expect(confirmation(page)).not.toBeVisible(); await expect(panel(page)).toBeVisible()
      assert.equal(count('apply'), before.applies)
    })
    await update(page).click()
    await button(confirmation(page), 'Update and restart').focus()
    await page.keyboard.press('Enter')
    await prove('Exact confirmation applies once through the real narrow HTTP client', async () => {
      await expect(confirmation(page)).not.toBeVisible()
      await expect(panel(page)).toContainText('Stopping web desktop…')
      assert.equal(count('apply'), before.applies + 1)
      assert.deepEqual(JSON.parse(helperRequests().findLast(r => r.path.endsWith('/apply')).body),
        { offer_id: state.offer.id, confirm_restart_and_rollback: true })
    })
    await shot(page, '03-stopping-progress')
    state.statusCode = 503
    await expect(panel(page)).toContainText('Update status is unavailable.', { timeout: 15000 })
    await shot(page, '04-network-503')
    state.statusCode = 403
    await expect(panel(page)).toContainText('Your session may have expired or lack permission.', { timeout: 15000 })
    await prove('503 then 403 distinguishes connection loss from rejected authorization; polling stops', async () => {
      const checks = count('status')
      await page.waitForTimeout(5500)
      assert.equal(count('status'), checks)
      assert.equal(count('apply'), before.applies + 1)
      assert.equal(requests.filter(r => r.path === '/api/auth/me').length, 0)
      await expect(update(page)).toBeDisabled()
    })
    await shot(page, '05-expired-session-403')
    state.sessionCode = 403
    await button(panel(page), 'Check session and retry').click()
    await expect(button(panel(page), 'Check session and retry')).toBeEnabled()
    await prove('Session check is explicit and uses GET /api/auth/me without applying', async () => {
      assert.equal(requests.filter(r => r.path === '/api/auth/me').length, 1)
      assert.equal(count('apply'), before.applies + 1)
    })
    const newTab = page.context().waitForEvent('page')
    await button(panel(page), 'Sign in in another tab').click()
    const login = await newTab
    await login.waitForLoadState('domcontentloaded')
    await prove('Sign in opens the exact local login path in a new tab without navigation or apply', async () => {
      assert.equal(login.url(), `${origin}/login?next=/`)
      // eslint-disable-next-line no-undef -- evaluated inside Chromium
      assert.equal(await login.evaluate(() => window.opener), null)
      assert.equal(page.url(), initialURL)
      assert.equal(count('apply'), before.applies + 1)
    })
    await login.close(); await page.bringToFront()
    state.sessionCode = 200; state.statusCode = 200; state.phase = 'succeeded'
    await button(panel(page), 'Check session and retry').click()
    await expect(panel(page)).toContainText('The last update completed successfully.')
    await prove('Successful session recheck shows update history without claiming a live installed release', async () => {
      await expect(beforeRelease(page)).toHaveText(state.offer.current_release)
      await expect(offeredRelease(page)).toHaveText(state.offer.target_release)
      await expect(loaded(page)).toHaveText(loadedText)
      await expect(panel(page)).not.toContainText(/release is running|tab still has|Installed release/)
      await expect(panel(page)).toContainText('Compare the browser build loaded in this tab with the releases above.')
      await expect(panel(page)).toContainText('If needed, reload only after saving your work.')
      await expect(panel(page)).toContainText('This tab will not reload automatically.')
      assert.equal(count('apply'), before.applies + 1)
    })
    await shot(page, '06-success-history-loaded-build')
    await button(panel(page), 'Reload this tab').click()
    const reload = page.getByRole('dialog', { name: 'Reload this tab?', exact: true })
    await expect(reload).toContainText('Reloading may lose drafts and unsaved file changes')
    await button(reload, 'Cancel').focus(); await page.keyboard.press('Enter')
    await expect(reload).not.toBeVisible()
    await page.keyboard.press('Escape')
    await prove('No auto reload/route change; real composer and unsaved CodeMirror drafts survive', async () => {
      await expect(panel(page)).not.toBeVisible()
      await expect(input).toHaveText(draft)
      await expect(editor).toContainText('Round3: keep my unsaved file draft.')
      await expect(button(page, 'Save to server')).toBeEnabled()
      assert.equal(await readFile(filePath, 'utf8'), 'Original fixture file; never save this acceptance draft.\n')
      await expect(footer).toHaveText(footerText)
      assert.equal(page.url(), initialURL); assert.equal(navigations.length, initialNavigations)
    })
    await shot(page, '07-preserved-drafts-after-success')
  }, { file: true })

  await run('retained-success-after-reload', async page => {
    state.phase = 'succeeded'
    const history = snapshot(state.phase)
    const applies = count('apply')
    await open(page)
    const loadedText = await loaded(page).innerText()
    await expect(panel(page)).toContainText('The last update completed successfully.')
    await page.reload()
    await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible()
    await open(page)
    await prove('The same retained job remains history after reload, not evidence of server or tab freshness', async () => {
      assert.deepEqual(helperRequests().findLast(r => r.path.endsWith('/status')).response, history)
      await expect(panel(page)).toContainText('The last update completed successfully.')
      await expect(beforeRelease(page)).toHaveText(state.offer.current_release)
      await expect(offeredRelease(page)).toHaveText(state.offer.target_release)
      await expect(loaded(page)).toHaveText(loadedText)
      await expect(panel(page)).not.toContainText(/release is running|tab still has|Installed release/)
      await expect(update(page)).toBeDisabled()
      assert.equal(count('apply'), applies)
    })
  })

  await run('rollback-and-manual-recovery', async page => {
    state.phase = 'rolled_back'
    await open(page)
    await prove('Rollback reports the last update result, not the current server release', async () => {
      await expect(panel(page)).toContainText('The last update did not activate. The previous release was restored and restarted.')
      await expect(beforeRelease(page)).toHaveText(state.offer.current_release)
      await expect(offeredRelease(page)).toHaveText(state.offer.target_release)
      await expect(panel(page)).not.toContainText(/release is running|tab still has|Installed release/)
      await expect(button(panel(page), 'Reload this tab')).toBeVisible()
    })
    await shot(page, '08-rolled-back')
    state.phase = 'recovery_required'; await retry(page).click()
    await prove('Manual recovery requires operator; no install, download or reload affordance', async () => {
      await expect(panel(page)).toContainText('Recovery requires a server operator to inspect the update journal.')
      await expect(panel(page)).toContainText('Do not retry the update automatically.')
      await expect(update(page)).toBeDisabled()
      await expect(button(panel(page), 'Check and download')).toBeDisabled()
      await expect(button(panel(page), 'Reload this tab')).toHaveCount(0)
    })
    await shot(page, '09-manual-recovery')
  })
  await run('helper-404-unsupported', async page => {
    state.statusCode = 404
    const before = count('apply')
    await open(page)
    await prove('Missing helper 404 is unsupported, not an update prompt', async () => {
      await expect(panel(page)).toContainText('update helper is not configured')
      await expect(update(page)).toBeDisabled()
      await expect(button(panel(page), 'Check and download')).toBeDisabled()
      assert.equal(count('apply'), before)
    })
  })
  await run('expired-and-stale-offers', async page => {
    state.offer = offer(-1); state.phase = 'offered'
    await open(page)
    await prove('Expired offer disables install but allows a new explicit download', async () => {
      await expect(panel(page)).toContainText('The downloaded update offer has expired.')
      await expect(update(page)).toBeDisabled()
      await expect(button(panel(page), 'Check and download')).toBeEnabled()
    })
    state.offer = offer(); await retry(page).click()
    await expect(update(page)).toBeEnabled()
    state.statusCode = 409; await retry(page).click()
    await prove('A conflicting status invalidates the stale offered action', async () => {
      await expect(panel(page)).toContainText('Update state has changed.')
      await expect(update(page)).toBeDisabled()
    })
    state.statusCode = 200; state.offer = offer(4); await retry(page).click()
    await update(page).click()
    await page.waitForTimeout(4500)
    const before = count('apply')
    await button(confirmation(page), 'Update and restart').click()
    await prove('Offer expiry while confirmation is open cannot send apply', async () => {
      await expect(confirmation(page)).not.toBeVisible()
      assert.equal(count('apply'), before)
      await expect(update(page)).toBeDisabled()
    })
  })
  await run('rejected-apply-no-resend', async page => {
    await open(page); await prepare(page)
    state.applyCode = 403
    const before = count('apply')
    await update(page).click(); await button(confirmation(page), 'Update and restart').click()
    await prove('Rejected apply closes confirmation; no blind resend button or automatic apply', async () => {
      await expect(confirmation(page)).not.toBeVisible()
      await expect(panel(page)).toContainText('Your session may have expired or lack permission.')
      await expect(update(page)).toBeDisabled()
      await page.waitForTimeout(2500)
      assert.equal(count('apply'), before + 1)
    })
  })
  await run('narrow-mobile', async page => {
    await open(page); await prepare(page); await update(page).click()
    await prove('390px mobile confirmation fits without horizontal overflow', async () => {
      // eslint-disable-next-line no-undef -- evaluated inside Chromium
      const dimensions = await page.evaluate(() => ({ viewport: innerWidth, width: document.documentElement.scrollWidth }))
      assert.ok(dimensions.width <= dimensions.viewport, json(dimensions))
      const box = await confirmation(page).boundingBox()
      assert.ok(box.x >= 0 && box.x + box.width <= dimensions.viewport + 1, json(box))
      for (const label of ['Cancel', 'Update and restart']) {
        const control = await button(confirmation(page), label).boundingBox()
        assert.ok(control.x >= 0 && control.x + control.width <= dimensions.viewport + 1)
      }
    })
    await shot(page, '10-mobile-confirmation')
    await button(confirmation(page), 'Cancel').click()
    await shot(page, '11-mobile-status')
  }, { mobile: true })
  await prove('All updater requests pin method, exact origin, cookie, header and bodies', async () => assert.deepEqual(violations, []))
  await prove('No uncaught UI JavaScript errors or unexpected external requests', async () => {
    assert.deepEqual(errors, [])
    assert.deepEqual(blocked.filter(r => !['font', 'stylesheet'].includes(r.type)), [])
  })
} catch (error) {
  report.error = error.stack
  console.error(error.stack)
  process.exitCode = 1
  if (activePage && !activePage.isClosed()) await shot(activePage, 'fatal')
} finally {
  try { await cleanup() } catch (error) { report.cleanupError = error.stack; process.exitCode = 1 }
  process.off('SIGINT', interrupt); process.off('SIGTERM', interrupt)
  report.status = process.exitCode ? 'FAIL' : 'PASS'
  report.finished = new Date().toISOString()
  report.counts = { cases: results.length, passedCases: results.filter(r => r.status === 'PASS').length,
    passedAssertionGroups: assertions.length, requests: requests.length,
    status: count('status'), offer: count('offer'), apply: count('apply'), uiJsErrors: errors.length,
    consoleErrors: consoleErrors.length, protocolViolations: violations.length }
  await writeFile(path.join(artifacts, 'results.json'), json(report))
  await writeFile(path.join(evidence, 'latest.json'), json({ status: report.status, results: path.join(artifacts, 'results.json') }))
  console.log(`${report.status}: ${path.join(artifacts, 'results.json')}`)
}
