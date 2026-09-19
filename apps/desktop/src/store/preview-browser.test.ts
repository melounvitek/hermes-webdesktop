import { afterAll, afterEach, beforeAll, expect, it, vi } from 'vitest'

import type * as Layout from './layout'
import type * as Notifications from './notifications'
import type * as Preview from './preview'

const url = { kind: 'url' as const, label: 'Page', source: 'https://example.invalid', url: 'https://example.invalid' }
const file = { kind: 'file' as const, label: 'notes', source: '/work/notes.txt', url: 'file:///work/notes.txt' }
let preview: typeof Preview
let layout: typeof Layout
let notifications: typeof Notifications
const openExternal = vi.fn()

beforeAll(async () => {
  vi.stubGlobal('hermesDesktop', { browser: { authRequired: false, signIn: vi.fn() }, openExternal })
  localStorage.setItem(
    'hermes.desktop.previewTabs.v2',
    JSON.stringify([
      { id: 'url:restored', target: url },
      { id: 'file:restored', target: file }
    ])
  )
  layout = await import('./layout')
  layout.$rightRailActiveTabId.set('url:restored')
  preview = await import('./preview')
  notifications = await import('./notifications')
})

afterEach(() => {
  preview.closeRightRail()
  notifications.$notifications.set([])
  openExternal.mockClear()
})
afterAll(() => {
  localStorage.clear()
  vi.unstubAllGlobals()
})

it('restores only supported previews and selects the surviving file instead of the removed URL', () => {
  expect(preview.$previewTabs.get()).toEqual([{ id: 'file:restored', target: file }])
  expect(layout.$rightRailActiveTabId.get()).toBe('file:restored')
})

it('refuses every URL opening path visibly without replacing a supported preview or opening externally', () => {
  preview.openPreview(file)
  const before = preview.$previewTabs.get()
  const active = layout.$rightRailActiveTabId.get()
  preview.openPreview(url, 'tool-result')
  preview.openBrowserTab()
  preview.newBrowserTab()
  expect(preview.$previewTabs.get()).toEqual(before)
  expect(layout.$rightRailActiveTabId.get()).toBe(active)
  expect(notifications.$notifications.get().at(-1)?.message).toMatch(/unavailable.*browser/i)
  expect(openExternal).not.toHaveBeenCalled()
})

it('keeps file, artifact and static HTML targets usable in browser mode', () => {
  preview.openPreview(file)
  preview.openPreview({ kind: 'artifact', label: 'Artifact', source: 'artifact:a', url: 'artifact:a' })
  preview.openPreview(
    {
      ...file,
      url: 'file:///work/page.html',
      previewKind: 'html',
      dataUrl: 'data:text/html,<h1>hello</h1>',
      transient: true
    },
    'tool-result'
  )
  expect(preview.$previewTabs.get().map(tab => tab.target.kind)).toEqual(['file', 'artifact', 'file'])
  expect(preview.$previewTarget.get()?.renderMode).toBe('preview')
})

it('preserves Electron URL restore, open and new-tab dispatch', () => {
  vi.stubGlobal('hermesDesktop', { openExternal })
  expect(preview.decodePreviewTabs(JSON.stringify([{ id: 'url:restored', target: url }]))).toHaveLength(1)
  preview.openPreview(url)
  preview.openBrowserTab()
  preview.newBrowserTab()
  expect(preview.$previewTabs.get()).toHaveLength(2)
  expect(preview.$previewTabs.get()[0].target).toEqual(url)
  expect(openExternal).not.toHaveBeenCalled()
})
