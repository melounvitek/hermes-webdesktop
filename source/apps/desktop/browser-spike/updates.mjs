import assert from 'node:assert/strict'
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium, expect } from '@playwright/test'
import { startProcess, stopProcess, waitFor } from './processes.mjs'

// Build first: cd apps/desktop && npm run build:browser
// SPIKE_BACKEND_ROOT=/absolute/clean/stock SPIKE_PYTHON=/absolute/venv/bin/python CHROME_PATH=/absolute/chrome \
//   node browser-spike/updates.mjs <dist-browser> <new evidence directory>
// The real browser app against a stock backend. This script plays the launcher:
// it creates the updates directory in the fixture's disposable home and answers
// the page's request.json with status.json. No launcher or release is involved.
const here = path.dirname(fileURLToPath(import.meta.url))
const [dist, evidence] = process.argv.slice(2).map(argument => path.resolve(argument))
const python = process.env.SPIKE_PYTHON
const backendRoot = process.env.SPIKE_BACKEND_ROOT
assert.ok(dist && evidence, 'Usage: node browser-spike/updates.mjs <dist-browser> <new evidence directory>')
assert.ok(python && path.isAbsolute(python), 'SPIKE_PYTHON must name an absolute test interpreter')
assert.ok(backendRoot && path.isAbsolute(backendRoot), 'SPIKE_BACKEND_ROOT must name a clean, credential-free stock checkout')
process.umask(0o077)
await mkdir(evidence) // Refuse existing evidence; never overwrite another run.
const temporary = await mkdtemp(path.join(os.tmpdir(), 'hermes-updates-chromium-'))
const env = { PATH: '/usr/bin:/bin', HOME: temporary, LANG: 'C.UTF-8', TZ: 'UTC', PYTHONDONTWRITEBYTECODE: '1' }
const errors = []
let fixture, browser, page, runtime
// The fixture is its own process group, so an interrupt has to stop it here.
const interrupt = () => {
  process.exitCode = 1
  void browser?.close()
  if (fixture) void stopProcess(fixture)
}
process.on('SIGINT', interrupt)
process.on('SIGTERM', interrupt)

try {
  fixture = startProcess(python, [path.join(here, 'backend.py'), '--backend-root', backendRoot,
    '--python', python, '--web-dist', dist], { cwd: temporary, env, logPath: path.join(evidence, 'fixture.log') })
  await fixture.ready
  const ready = await waitFor(() => {
    if (fixture.failure) throw fixture.failure
    assert.ok(!fixture.closed, `Fixture exited: ${fixture.output}`)
    const line = fixture.output.match(/^READY (.+)$/m)?.[1]
    return line && JSON.parse(line)
  }, 100000, 'stock backend READY')
  runtime = JSON.parse(await readFile(path.join(ready.run_dir, 'runtime.json'), 'utf8'))
  assert.equal(runtime.home, path.join(runtime.run_dir, 'home'))
  const updates = path.join(runtime.home, '.local/lib/hermes-browser/installation.updates')
  const requestFile = path.join(updates, 'request.json')
  const report = status => writeFile(path.join(updates, 'status.json'), JSON.stringify(status))

  // Playwright's own SIGINT handler would exit before the cleanup below.
  browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: true,
    handleSIGINT: false, env })
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, reducedMotion: 'reduce' })
  await context.route(url => url.protocol.startsWith('http') && url.origin !== runtime.url, route => route.abort())
  page = await context.newPage()
  page.setDefaultTimeout(15000)
  page.on('pageerror', error => errors.push(error.stack))
  const panel = page.getByRole('dialog', { name: 'Browser updates', exact: true })
  const update = panel.getByRole('button', { name: 'Update', exact: true })
  const reload = panel.getByRole('button', { name: 'Reload this tab', exact: true })
  const shot = name => page.screenshot({ path: path.join(evidence, `${name}.png`) })
  const passed = name => console.log(`PASS: ${name}`)
  const open = async () => {
    await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible({ timeout: 60000 })
    await page.getByRole('contentinfo').getByRole('button', { name: /^Browser / }).click()
    await expect(panel).toContainText('Browser build loaded in this tab')
  }
  // Each press must make stock write a new, valid request; an earlier status must not answer it.
  let id
  const request = async () => {
    const previous = id
    await update.click()
    await expect.poll(async () => {
      id = (await readFile(requestFile, 'utf8').catch(() => '')).match(/^\{"id":"([0-9a-f]{32})"\}$/)?.[1]
      return id && id !== previous
    }).toBe(true)
    await expect(panel).toContainText('Checking for a newer release and installing it…')
    await expect(update).toBeDisabled()
  }

  await page.goto(runtime.url)
  await open()
  await expect(panel).toContainText('installs it while Hermes keeps running')
  await expect(update).toBeEnabled()
  const loaded = (await panel.locator('dd').first().textContent()).match(/^[0-9a-f]{40}/)?.[0]
  assert.ok(loaded, 'The dialog must name the commit this build was made from')
  await shot('idle')
  passed('Idle dialog shows the loaded build and offers Update')

  await update.click()
  await expect(panel).toContainText('Updating from the browser is not available for this installation.')
  assert.equal(await panel.locator('pre').textContent(), 'hermes-browser stop\nhermes-browser update\nhermes-browser start')
  await expect(update).toBeEnabled()
  await assert.rejects(access(updates), { code: 'ENOENT' })
  await shot('unavailable')
  passed('Without the launcher directory stock refuses the request; the dialog shows the terminal commands')

  await mkdir(updates, { recursive: true })
  await request()
  await shot('working')
  await report({ id, state: 'running' })
  await page.waitForTimeout(16000) // Longer than the page waits for a launcher that never answers.
  await expect(update).toBeDisabled()
  await report({ id, state: 'updated', release: 'browser-fixture-2', commit: 'b'.repeat(40) })
  await expect(panel).toContainText('Updated to browser-fixture-2.')
  await expect(panel).toContainText('This tab still runs the previous build.')
  await shot('updated')
  passed('Update writes a valid request.json through stock; running then updated shows the release and Reload')

  await reload.click()
  await page.getByRole('dialog', { name: 'Reload this tab?', exact: true })
    .getByRole('button', { name: 'Reload this tab', exact: true }).click()
  await expect(panel).toBeHidden()
  await open()
  await expect(update).toBeEnabled()
  passed('Confirmed Reload reloads the tab')

  await request()
  await page.waitForTimeout(3000)
  await expect(update).toBeDisabled()
  await report({ id, state: 'current', release: 'browser-fixture-2', commit: loaded })
  await expect(panel).toContainText('browser-fixture-2 is already the newest release.')
  await expect(panel).not.toContainText('This tab still runs the previous build.')
  await expect(update).toBeEnabled()
  await shot('current')
  passed('A new request ignores the earlier result; current names the release and Update stays available')

  const message = `<img src=x onerror="document.title='markup'"> Download <b>failed</b>: https://releases.invalid/${'a'.repeat(120)}`
  await request()
  await report({ id, state: 'failed', error: message })
  await expect(panel.getByText(message, { exact: true })).toBeVisible()
  await expect(panel.locator('img, b')).toHaveCount(0)
  assert.notEqual(await page.title(), 'markup')
  await expect(update).toBeEnabled()
  await shot('failed')
  passed('A failure shows the launcher message as plain text')

  await request()
  await report({ id, state: 'current', release: 'browser-fixture-3', commit: 'c'.repeat(40) })
  await expect(panel).toContainText('browser-fixture-3 is already the newest release.')
  await expect(panel).toContainText('This tab still runs the previous build.')
  await expect(reload).toBeVisible()
  await expect(update).toBeHidden()
  await shot('outdated')
  passed('A release installed from another tab is current, and this tab is offered a reload')

  assert.deepEqual(errors, [], 'Uncaught browser errors')
} catch (error) {
  console.error(error.stack)
  process.exitCode = 1
  if (page) await page.screenshot({ path: path.join(evidence, 'failure.png') }).catch(() => {})
} finally {
  await browser?.close()
  if (fixture) await stopProcess(fixture)
  if (runtime) await rm(runtime.run_dir, { recursive: true, force: true })
  await rm(temporary, { recursive: true, force: true })
  console.log(`${process.exitCode ? 'FAIL' : 'PASS'}: ${evidence}`)
}
