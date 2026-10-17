import { afterEach, expect, it, vi } from 'vitest'

import { stopIosTypingZoom } from './ios-typing-zoom'

const VIEWPORT = 'width=device-width, initial-scale=1.0'

const IPHONE =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1'

const MAC =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15'

const ANDROID =
  'Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36'

afterEach(() => {
  document.head.innerHTML = ''
  vi.unstubAllGlobals()
})

function viewportAfterStart(userAgent: string, maxTouchPoints: number) {
  document.head.innerHTML = `<meta name="viewport" content="${VIEWPORT}">`
  vi.stubGlobal('navigator', { maxTouchPoints, userAgent })
  stopIosTypingZoom()

  return document.querySelector<HTMLMetaElement>('meta[name="viewport"]')!.content
}

it.each([
  ['an iPhone', IPHONE, 5],
  ['an iPad that reports itself as a Mac', MAC, 5]
])('stops the typing zoom on %s', (_, userAgent, maxTouchPoints) => {
  expect(viewportAfterStart(userAgent, maxTouchPoints)).toBe(`${VIEWPORT}, maximum-scale=1`)
})

it.each([
  ['a Mac', MAC, 0],
  ['an Android phone', ANDROID, 5]
])('leaves zoom alone on %s', (_, userAgent, maxTouchPoints) => {
  expect(viewportAfterStart(userAgent, maxTouchPoints)).toBe(VIEWPORT)
})
