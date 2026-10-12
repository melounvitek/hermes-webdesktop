import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createBrowserBridge } from './bridge'

beforeEach(() => localStorage.clear())
afterEach(() => {
  localStorage.clear()
  document.documentElement.removeAttribute('style')
  document.getElementById('hermes-browser-zoom')?.remove()
  vi.restoreAllMocks()
})

// jsdom resolves neither calc() nor var(), so read the scale the stylesheet multiplies by.
const rootScale = () => Number(document.documentElement.style.getPropertyValue('--hermes-ui-scale'))

it('applies document-wide UI scale, reports changes and restores it independently of profiles', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response('<script>window.__HERMES_SESSION_TOKEN__="test";</script>')
  )
  const bridge = createBrowserBridge({ token: '', authRequired: false })
  expect((await bridge.zoom.get()).percent).toBe(100)
  const changed = vi.fn()
  const unsubscribe = bridge.zoom.onChanged(changed)
  bridge.zoom.setPercent(125)
  expect(rootScale()).toBeCloseTo(1.25)
  expect(document.documentElement.style.zoom).toBe('')
  expect(changed).toHaveBeenLastCalledWith(await bridge.zoom.get())
  await bridge.getConnection('another-profile')
  const reloaded = createBrowserBridge({ token: '', authRequired: false })
  expect((await reloaded.zoom.get()).percent).toBe(125)
  expect((await reloaded.zoom.get()).level).toBeCloseTo(Math.log(1.25) / Math.log(1.2))
  unsubscribe()
  bridge.zoom.setPercent(100)
  expect(changed).toHaveBeenCalledTimes(1)

  for (const modifier of [{ ctrlKey: true }, { metaKey: true }]) {
    for (const key of ['+', '-', '0']) {
      const event = new KeyboardEvent('keydown', { key, ...modifier, cancelable: true })
      window.dispatchEvent(event)
      expect(event.defaultPrevented).toBe(false)
      expect(rootScale()).toBe(1)
    }
  }
})

it('uses the default for corrupt values and works when persistent storage is unavailable', async () => {
  const bridge = createBrowserBridge({ token: '', authRequired: false })
  bridge.zoom.setPercent(120)
  const key = localStorage.key(0)!
  expect(key).toContain('global')
  localStorage.setItem(key, 'not a scale')
  const reloaded = createBrowserBridge({ token: '', authRequired: false })
  expect((await reloaded.zoom.get()).percent).toBe(100)
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
    throw new DOMException('blocked')
  })
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new DOMException('blocked')
  })
  const blocked = createBrowserBridge({ token: '', authRequired: false })
  blocked.zoom.setPercent(110)
  expect((await blocked.zoom.get()).percent).toBe(110)
  blocked.zoom.setPercent(NaN)
  expect((await blocked.zoom.get()).percent).toBe(100)
})
