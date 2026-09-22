import assert from 'node:assert/strict'
import { mkdir, readFile, realpath } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { chromium, expect } from '@playwright/test'

// Run against backend.py's isolated stock-backend fixture after build:browser.
const runtimePath = process.argv[2]
if (!runtimePath) throw new Error('Usage: node browser-spike/loading.mjs <runtime.json> [artifact directory]')
const runtime = JSON.parse(await readFile(runtimePath, 'utf8'))
assert.equal(new URL(runtime.url).hostname, '127.0.0.1')
assert.equal(runtime.public_url, null)
assert.equal(path.dirname(runtime.run_dir), os.tmpdir())
assert.ok(path.basename(runtime.run_dir).startsWith('hermes-browser-spike-'))
assert.equal(await realpath(runtime.run_dir), runtime.run_dir)
assert.equal(path.resolve(runtimePath), path.join(runtime.run_dir, 'runtime.json'))
const artifacts = process.argv[3] || path.join(runtime.run_dir, 'loading-evidence')
await mkdir(artifacts, { recursive: true })
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true
})

try {
  for (const reducedMotion of ['no-preference', 'reduce']) {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion })
    const page = await context.newPage()
    await page.clock.install()
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    let releaseScripts
    let releaseConfig
    const scripts = new Promise(resolve => {
      releaseScripts = resolve
    })
    const config = new Promise(resolve => {
      releaseConfig = resolve
    })
    let configRequested = false
    await context.route('**/*', async route => {
      const request = route.request()
      const url = new URL(request.url())
      if (url.origin !== runtime.url && !['data:', 'blob:'].includes(url.protocol)) return route.abort()
      if (request.resourceType() === 'script') await scripts
      if (url.pathname === '/api/config') {
        configRequested = true
        await config
      }
      return route.continue()
    })
    await context.routeWebSocket('**/*', ws => {
      if (new URL(ws.url()).origin === runtime.url.replace('http:', 'ws:')) ws.connectToServer()
      else ws.close()
    })

    try {
      await page.goto(runtime.url, { waitUntil: 'commit' })
      const loading = page.getByRole('status', { name: 'Loading Hermes…', exact: true })
      await expect(loading).toBeVisible()
      assert.deepEqual(await loading.boundingBox(), { x: 0, y: 0, width: 390, height: 844 })
      await expect(loading.locator('.spinner')).toHaveCSS(
        'animation-name',
        reducedMotion === 'reduce' ? 'none' : 'hermes-startup-spin'
      )
      await page.screenshot({ path: path.join(artifacts, `before-javascript-${reducedMotion}.png`) })
      await expect(page.locator('#hermes-startup-retry')).toBeHidden()
      await page.clock.fastForward(15000)
      await expect(page.getByRole('link', { name: 'Reload', exact: true })).toBeVisible()

      releaseScripts()
      await expect.poll(() => configRequested, { timeout: 30000 }).toBe(true)
      // React owns the screen now, but the app is not ready until config settles.
      await expect(page.locator('#hermes-startup')).toHaveCount(0)
      await expect(loading).toBeVisible()
      assert.deepEqual(await loading.boundingBox(), { x: 0, y: 0, width: 390, height: 844 })
      await expect(loading.locator('svg')).toHaveCount(reducedMotion === 'reduce' ? 0 : 1)
      await page.screenshot({ path: path.join(artifacts, `initializing-${reducedMotion}.png`) })

      releaseConfig()
      await expect(loading).toHaveCount(0, { timeout: 60000 })
      await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible()
      await page.screenshot({ path: path.join(artifacts, `ready-${reducedMotion}.png`) })
      assert.deepEqual(errors, [])
      console.log(`PASS: mobile loading before JavaScript, during initialization, and ready (${reducedMotion})`)
    } finally {
      releaseScripts()
      releaseConfig()
      await context.close()
    }
  }
} finally {
  await browser.close()
}
