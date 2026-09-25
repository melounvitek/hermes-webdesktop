import assert from 'node:assert/strict'
import { execFileSync, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { access, mkdir, mkdtemp, readFile, readlink, realpath, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { childEnvironment, options } from './compat.mjs'
import { startProcess, stopProcess, waitFor } from './processes.mjs'

// node browser-spike/file-editor.mjs --backend-root /ABS/stock --python /ABS/venv/bin/python
//   --chrome /ABS/chromium --web-dist /ABS/dist-browser --evidence /ABS/new-directory
// Only the disposable fixture is edited. No model prompts, backend patches, or live services.
const here = path.dirname(fileURLToPath(import.meta.url))
const receipt = inputs => path.join(inputs.evidence, 'file-editor.json')
const json = value => JSON.stringify(value, null, 2)
const git = (root, ...args) => execFileSync('/usr/bin/git', ['-C', root, ...args], {
  env: { PATH: '/usr/bin:/bin', HOME: '/nonexistent', GIT_CONFIG_NOSYSTEM: '1' }, encoding: 'utf8'
}).trim()

async function outer(inputs) {
  process.umask(0o077)
  await mkdir(inputs.evidence, { mode: 0o700 }) // Refuse existing evidence; never overwrite another run.
  inputs = { ...inputs, evidence: await realpath(inputs.evidence), chrome: await realpath(inputs.chrome),
    'backend-root': await realpath(inputs['backend-root']), 'web-dist': await realpath(inputs['web-dist']) }
  let summary = { status: 'FAIL', inputs, started: new Date().toISOString() }
  await writeFile(receipt(inputs), json(summary))
  let child, namespacePid, interrupted, timer
  const stop = signal => {
    interrupted = signal
    if (child?.exitCode !== null || child?.signalCode !== null) return
    try {
      if (namespacePid) process.kill(namespacePid, 'SIGKILL')
      else child?.kill('SIGKILL')
    } catch (error) { if (error.code !== 'ESRCH') throw error }
  }
  process.on('SIGINT', stop)
  process.on('SIGTERM', stop)
  try {
    const temporary = await mkdtemp(path.join(inputs.evidence, 'tmp-'))
    const home = path.join(inputs.evidence, 'home')
    await mkdir(home)
    const commonGit = git(inputs['backend-root'], 'rev-parse', '--path-format=absolute', '--git-common-dir')
    const pythonRuntime = path.dirname(path.dirname(await realpath(inputs.python)))
    let pythonAlias = pythonRuntime
    try { pythonAlias = path.dirname(path.dirname(path.resolve(path.dirname(inputs.python), await readlink(inputs.python)))) }
    catch (error) { if (error.code !== 'EINVAL') throw error }
    const mounts = [here, inputs['backend-root'], inputs['web-dist'], commonGit, pythonRuntime, pythonAlias,
      path.dirname(path.dirname(inputs.python)), inputs.chrome, process.execPath,
      path.resolve(here, '../node_modules'), path.resolve(here, '../../../node_modules')]
    const readOnly = [...new Set((await Promise.all(mounts.map(async file => [file, await realpath(file)]))).flat())]
    for (const file of readOnly) {
      assert.ok(!['/', '/tmp', '/home', '/root', '/run', '/proc', '/dev'].includes(file))
      assert.ok(inputs.evidence !== file && !inputs.evidence.startsWith(file + '/') && !file.startsWith(inputs.evidence + '/'),
        `Evidence overlaps input: ${file}`)
    }
    const args = ['--unshare-pid', '--unshare-net', '--unshare-ipc', '--die-with-parent',
      '--ro-bind', '/', '/', '--tmpfs', '/home', '--tmpfs', '/root', '--tmpfs', '/run',
      '--bind', temporary, '/tmp', '--dev', '/dev', '--proc', '/proc',
      ...readOnly.flatMap(file => ['--ro-bind', file, file]),
      '--bind', inputs.evidence, inputs.evidence, '--chdir', home, '--cap-drop', 'ALL', '--json-status-fd', '3']
    // The shared git config is not needed for identity and may contain user-specific credentials.
    try {
      await access(path.join(commonGit, 'config'))
      const emptyConfig = path.join(inputs.evidence, 'empty-git-config')
      await writeFile(emptyConfig, '', { flag: 'wx' })
      args.push('--ro-bind', emptyConfig, path.join(commonGit, 'config'))
    }
    catch (error) { if (error.code !== 'ENOENT') throw error }
    const entry = `import { scenario } from ${JSON.stringify(import.meta.url)}; process.exitCode = await scenario(JSON.parse(process.argv[1]))`
    child = spawn('/usr/bin/bwrap', [...args, process.execPath, '--input-type=module', '-e', entry, JSON.stringify(inputs)], {
      env: childEnvironment(home, inputs), cwd: inputs.evidence, stdio: ['ignore', 'inherit', 'inherit', 'pipe']
    })
    let pending = ''
    child.stdio[3].on('data', bytes => {
      pending += bytes
      const lines = pending.split('\n'); pending = lines.pop()
      for (const line of lines.filter(Boolean)) {
        const status = JSON.parse(line)
        if (status['child-pid']) { namespacePid = status['child-pid']; if (interrupted) stop(interrupted) }
      }
    })
    timer = setTimeout(() => stop('five-minute timeout'), 5 * 60 * 1000)
    const exit = await new Promise((resolve, reject) => {
      child.once('error', reject)
      child.once('exit', (code, signal) => resolve({ code, signal }))
    })
    summary = JSON.parse(await readFile(receipt(inputs), 'utf8'))
    summary.isolation = { network: 'private loopback', pid: 'private namespace', ipc: 'private namespace',
      hidden: ['/home (except explicit inputs)', '/root', '/run', 'shared git config'],
      host: 'read-only', writable: [inputs.evidence, '/tmp (backed by evidence)', '/dev'], temporary }
    summary.cleanup = { scope: 'whole PID namespace', confirmed: Boolean(namespacePid) && exit.signal === null, ...exit }
    assert.ok(summary.cleanup.confirmed, 'Namespace cleanup was not confirmed')
    assert.ok(!interrupted, `Interrupted: ${interrupted}`)
    assert.equal(exit.code, 0, 'Namespaced probe failed')
    assert.equal(summary.status, 'AWAITING_NAMESPACE_EXIT')
    summary.status = 'PASS'
  } catch (error) {
    summary.status = 'FAIL'
    summary.error = [summary.error, error.stack].filter(Boolean).join('\n')
  } finally {
    clearTimeout(timer)
    process.off('SIGINT', stop); process.off('SIGTERM', stop)
    summary.finished = new Date().toISOString()
    await writeFile(receipt(inputs), json(summary))
  }
  console.log(`${summary.status}: ${receipt(inputs)}`)
  return summary.status === 'PASS' ? 0 : 1
}

export async function scenario(inputs) {
  const { chromium, expect } = await import('@playwright/test')
  const summary = JSON.parse(await readFile(receipt(inputs), 'utf8'))
  const env = childEnvironment(path.join(inputs.evidence, 'home'), inputs)
  const fixtureArgs = [path.join(here, 'backend.py'), '--backend-root', inputs['backend-root'],
    '--python', inputs.python, '--web-dist', inputs['web-dist']]
  let fixture, browser, page, runtime
  const frames = [], requests = [], errors = []
  summary.checks = []
  const passed = name => { summary.checks.push(name); console.log(`PASS: ${name}`) }
  const screenshot = async name => {
    await page.screenshot({ path: path.join(inputs.evidence, `${name}.png`) })
    await writeFile(path.join(inputs.evidence, `${name}.aria.txt`), await page.locator('body').ariaSnapshot())
  }
  try {
    summary.stock = JSON.parse(execFileSync(inputs.python, [...fixtureArgs, '--check-source'], { env, encoding: 'utf8' }))
    const bundleIndex = await readFile(path.join(inputs['web-dist'], 'index.html'))
    summary.bundleIndexSha256 = createHash('sha256').update(bundleIndex).digest('hex')
    fixture = startProcess(inputs.python, fixtureArgs, { cwd: env.HOME, env, logPath: path.join(inputs.evidence, 'fixture.log') })
    await fixture.ready
    const ready = await waitFor(() => {
      if (fixture.failure) throw fixture.failure
      assert.ok(!fixture.closed, 'Fixture exited; inspect fixture.log')
      const line = fixture.output.match(/^READY (.+)$/m)?.[1]
      return line && JSON.parse(line)
    }, 100000, 'stock backend READY')
    runtime = JSON.parse(await readFile(path.join(ready.run_dir, 'runtime.json'), 'utf8'))
    assert.deepEqual(runtime.stock, summary.stock)
    assert.equal(runtime.harness_pid, fixture.child.pid)
    assert.equal(new URL(runtime.url).hostname, '127.0.0.1')
    assert.equal(runtime.public_url, null)
    assert.equal(path.dirname(runtime.run_dir), '/tmp')
    assert.equal(runtime.home, path.join(runtime.run_dir, 'home'))
    summary.runtime = runtime
    const file = path.join(runtime.home, 'file-editor-fixture.txt')
    const initial = 'Temporary browser editor fixture.\nOriginal UTF-8: café.\n'
    const clicked = 'Saved through the real Save to server button.\nUTF-8: café.\n'
    const keyboard = 'Saved through real Control+s.\nSecond edit.\n'
    const draft = 'Keep this unsaved browser draft after a conflict.\n'
    const external = 'External fixture writer changed the server file.\n'
    await writeFile(file, initial, { flag: 'wx' })
    summary.fixture = file
    summary.fileVersions = { initial, clicked, keyboard, draft, external }
    browser = await chromium.launch({ executablePath: inputs.chrome, headless: true })
    summary.chromium = browser.version()
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
    await context.addInitScript(({ file, url }) => {
      localStorage.setItem('hermes.desktop.previewTabs.v2', JSON.stringify([{ id: `file:${file}`, target: {
        kind: 'file', label: 'file-editor-fixture.txt', path: file, source: file, url, previewKind: 'text'
      } }]))
      localStorage.setItem('hermes.desktop.rightRailActiveTab', `file:${file}`)
    }, { file, url: pathToFileURL(file).href })
    page = await context.newPage()
    page.setDefaultTimeout(15000)
    page.on('pageerror', error => errors.push(error.stack))
    page.on('request', request => {
      const url = new URL(request.url())
      if (url.pathname.startsWith('/api/')) requests.push({ method: request.method(), path: url.pathname,
        query: [...url.searchParams].filter(([key]) => !/token/i.test(key)), body: request.postData() })
    })
    page.on('websocket', ws => {
      for (const [event, direction] of [['framesent', 'sent'], ['framereceived', 'received']]) {
        ws.on(event, ({ payload }) => frames.push({ direction, ...JSON.parse(String(payload)) }))
      }
    })
    await page.goto(runtime.url)
    const edit = page.getByRole('button', { name: 'Edit', exact: true })
    await expect(edit).toBeVisible({ timeout: 60000 })
    await edit.click()
    const editor = page.locator('.cm-editor .cm-content[contenteditable="true"]')
    await expect(editor).toBeVisible()
    const note = page.getByRole('note')
    await expect(note).toContainText('Editing a server file. No autosave.')
    await expect(note).toContainText('Saving replaces the server file')
    await expect(note).toContainText('Concurrent edits can still be overwritten.')
    await expect(note).toContainText('Save before reloading or closing the browser.')
    const save = page.getByRole('button', { name: 'Save to server', exact: true })
    await expect(save).toBeDisabled()
    const editorEquals = text => expect(editor.locator('.cm-line')).toHaveText(text.split('\n'))
    const type = async text => {
      await editor.click()
      await editor.press('Control+a')
      await page.keyboard.insertText(text)
      await editorEquals(text)
      await expect(save).toBeEnabled()
    }
    const fileEquals = async text => {
      await expect.poll(() => readFile(file, 'utf8')).toBe(text)
      assert.deepEqual(await readFile(file), Buffer.from(text), 'Real fixture bytes differ')
    }
    await type(clicked)
    // A bounded observation, not a claim to prove absence of every possible delayed autosave.
    await page.waitForTimeout(2000)
    await fileEquals(initial)
    await expect(note).toContainText('Unsaved changes')
    await screenshot('desktop-editor-warning')
    passed('CodeMirror typing, warning visible, no autosave during two-second observation')
    await save.click()
    await fileEquals(clicked)
    await expect(note).toContainText('Saved to server')
    await expect(save).toBeDisabled()
    passed('Save to server updates exact fixture bytes')
    await type(keyboard)
    await editor.press('Control+s')
    await fileEquals(keyboard)
    await expect(note).toContainText('Saved to server')
    passed('Control+s updates exact fixture bytes')
    await type(draft)
    await writeFile(file, external)
    await save.click()
    await expect(page.getByText('The server file changed. Your draft was kept.', { exact: false })).toBeVisible()
    await editorEquals(draft)
    await fileEquals(external)
    await expect(page.getByRole('button', { name: 'Overwrite', exact: true })).toHaveCount(0)
    await screenshot('desktop-conflict')
    passed('External change blocks save, retains draft, no overwrite action')
    await page.getByRole('button', { name: 'Cancel', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'Discard unsaved edits?' })
    await expect(dialog).toBeVisible()
    await expect(dialog).toContainText('The server file will not be changed.')
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(dialog).toBeHidden()
    await editorEquals(draft)
    await fileEquals(external)
    passed('Cancel edit opens confirmation; cancelling confirmation retains draft and server bytes')
    await page.setViewportSize({ width: 600, height: 900 })
    await expect(editor).toBeVisible()
    await expect(note).toContainText('No autosave.')
    await screenshot('narrow-editor-warning')
    summary.narrow = { viewport: page.viewportSize(), editor: await editor.boundingBox(), warning: await note.boundingBox() }
    passed('Narrow viewport editor and warning visible (600x900)')
    await page.setViewportSize({ width: 1440, height: 1000 })
    await page.getByRole('button', { name: 'Discard & reload', exact: true }).click()
    await expect(dialog).toBeVisible()
    await screenshot('discard-confirmation')
    await dialog.getByRole('button', { name: 'Discard & reload', exact: true }).click()
    await expect(dialog).toBeHidden()
    await expect(editor).toHaveCount(0)
    await expect(edit).toBeVisible()
    await fileEquals(external)
    await edit.click()
    await editorEquals(external)
    await expect(save).toBeDisabled()
    await page.getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(editor).toHaveCount(0)
    await expect(dialog).toHaveCount(0)
    passed('Confirmed discard reloads external bytes; clean cancel needs no confirmation')
    assert.ok(!frames.some(frame => frame.direction === 'sent' && frame.method === 'prompt.submit'), 'Unexpected model prompt')
    let modelRequests = ''
    try { modelRequests = await readFile(path.join(runtime.run_dir, 'model-requests.jsonl'), 'utf8') }
    catch (error) { if (error.code !== 'ENOENT') throw error }
    assert.equal(modelRequests, '', 'Unexpected model requests')
    assert.deepEqual(requests.filter(request => request.path === '/api/fs/write-text').map(request => ({
      method: request.method, ...JSON.parse(request.body)
    })), [clicked, keyboard].map(content => ({ method: 'POST', content, path: file })),
    'Only the two explicit fixture saves may issue writes')
    assert.deepEqual(await readFile(path.join(inputs['web-dist'], 'index.html')), bundleIndex, 'Bundle index changed during probe')
    const stockAfter = JSON.parse(execFileSync(inputs.python, [...fixtureArgs, '--check-source'], { env, encoding: 'utf8' }))
    assert.deepEqual(stockAfter, summary.stock)
    assert.deepEqual(errors, [], 'Uncaught browser errors')
    summary.finalBytesSha256 = createHash('sha256').update(await readFile(file)).digest('hex')
    passed('No prompts/model requests, no page errors, stock checkout remains clean')
    summary.status = 'AWAITING_NAMESPACE_EXIT'
  } catch (error) {
    summary.status = 'FAIL'; summary.error = error.stack
    console.error(error.stack)
    if (page) {
      try { await screenshot('failure') }
      catch (captureError) { summary.captureError = captureError.stack }
    }
  } finally {
    if (browser) await browser.close()
    if (fixture) summary.fixtureCleanup = await stopProcess(fixture)
    await writeFile(path.join(inputs.evidence, 'network.json'), json({ requests, frames, errors }))
    await writeFile(receipt(inputs), json(summary))
  }
  return summary.status === 'AWAITING_NAMESPACE_EXIT' ? 0 : 1
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = await outer(options(process.argv.slice(2)))
}
