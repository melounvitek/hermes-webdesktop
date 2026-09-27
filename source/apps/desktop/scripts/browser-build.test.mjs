import { execFileSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { afterEach, expect, it } from 'vitest'

import { getBrowserBuild } from './browser-build.mjs'

const roots = []
const git = (root, ...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim()
const put = (root, file, content = 'fixture') => {
  const target = path.join(root, file)
  mkdirSync(path.dirname(target), { recursive: true })
  writeFileSync(target, content)
}
const fixture = () => {
  const root = mkdtempSync(path.join(tmpdir(), 'hermes-browser-build-'))
  roots.push(root)
  return root
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

it.each(['.', 'source'])('stamps HEAD and scopes dirt to UI inputs with workspace at %s', workspace => {
  const repository = fixture()
  git(repository, 'init', '--quiet')
  const root = path.join(repository, workspace)
  mkdirSync(root, { recursive: true })
  const inputs = ['apps/desktop/src/view.tsx', 'apps/shared/src/client.ts', 'package.json', 'package-lock.json']
  for (const file of [...inputs, 'backend.py']) put(root, file)
  put(root, '.gitignore', 'apps/desktop/dist/\n')
  git(root, 'add', '.')
  git(
    root,
    '-c',
    'user.name=Fixture',
    '-c',
    'user.email=fixture@example.invalid',
    '-c',
    'commit.gpgsign=false',
    'commit',
    '--quiet',
    '-m',
    'Fixture'
  )
  const revision = git(root, 'rev-parse', 'HEAD')
  expect(getBrowserBuild(root)).toEqual({ revision, dirty: false })

  put(root, 'backend.py', 'backend edit')
  put(root, 'backend-new.py')
  put(root, 'apps/desktop/dist/generated.js')
  expect(getBrowserBuild(root)).toEqual({ revision, dirty: false })

  for (const file of inputs) {
    put(root, file, 'UI edit')
    expect(getBrowserBuild(root)).toEqual({ revision, dirty: true })
    git(root, 'restore', '--', file)
  }
  for (const file of ['apps/desktop/src/new.tsx', 'apps/shared/src/new.ts', 'package-extra.json']) {
    put(root, file)
    expect(getBrowserBuild(root)).toEqual({ revision, dirty: true })
    rmSync(path.join(root, file))
  }
  git(root, 'checkout', '--quiet', '--detach')
  expect(getBrowserBuild(root)).toEqual({ revision, dirty: false })
})

it('reports unknown provenance when Git cannot read the revision or worktree status', () => {
  const root = fixture()
  expect(getBrowserBuild(root)).toEqual({ revision: null, dirty: null })
  git(root, 'init', '--quiet')
  expect(getBrowserBuild(root)).toEqual({ revision: null, dirty: null })
  put(root, 'package.json')
  git(root, 'add', '.')
  git(
    root,
    '-c',
    'user.name=Fixture',
    '-c',
    'user.email=fixture@example.invalid',
    '-c',
    'commit.gpgsign=false',
    'commit',
    '--quiet',
    '-m',
    'Fixture'
  )
  put(root, '.git/index', 'invalid index')
  expect(getBrowserBuild(root)).toEqual({ revision: null, dirty: null })
})
