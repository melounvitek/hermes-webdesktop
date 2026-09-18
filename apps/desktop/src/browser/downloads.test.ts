import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createBrowserBridge } from './bridge'

let downloads: Array<{ href: string; name: string }>
const createObjectURL = vi.fn((_blob: Blob) => 'blob:download-fixture')
const revokeObjectURL = vi.fn()

beforeEach(() => {
  vi.useFakeTimers()
  downloads = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response('file bytes', { headers: { 'Content-Type': 'image/png' } }))
  )
  vi.stubGlobal(
    'URL',
    class extends URL {
      static createObjectURL = createObjectURL
      static revokeObjectURL = revokeObjectURL
    }
  )
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push({ href: this.href, name: this.download })
  })
})

afterEach(() => {
  vi.runAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

it('downloads bytes with explicit profile/session and header or cookie auth, never a credential URL', async () => {
  for (const token of ['fixture-secret', '']) {
    const bridge = createBrowserBridge({ token, authRequired: !token })
    await expect(
      bridge.saveGatewayFile({
        path: '../output/report.txt',
        profile: 'background',
        sessionId: 'origin-session',
        suggestedName: 'report.txt'
      })
    ).resolves.toEqual({ saved: true })
    const [url, init] = vi.mocked(fetch).mock.calls.at(-1)!
    const target = new URL(String(url))
    expect(target.origin).toBe(window.location.origin)
    expect(target.pathname).toBe('/api/fs/download')
    expect(Object.fromEntries(target.searchParams)).toEqual({
      path: '../output/report.txt',
      profile: 'background',
      session_id: 'origin-session'
    })
    expect(new Headers(init?.headers).get('X-Hermes-Session-Token')).toBe(token || null)
    expect(init?.credentials).toBe('same-origin')
    expect(init?.redirect).toBe('error')
  }

  expect(downloads).toEqual(Array(2).fill({ href: 'blob:download-fixture', name: 'report.txt' }))
  expect(await createObjectURL.mock.calls[0][0].text()).toBe('file bytes')
  expect(document.querySelector('a[download]')).toBeNull()
  expect(revokeObjectURL).not.toHaveBeenCalled()
  vi.runAllTimers()
  expect(revokeObjectURL).toHaveBeenCalledWith('blob:download-fixture')
})

it('rejects failures and unsupported connections without claiming a download; external images never receive gateway credentials', async () => {
  const bridge = createBrowserBridge({ token: 'fixture-secret', authRequired: true })
  await expect(bridge.saveGatewayFile({ path: '/report', connectionId: 'other' })).rejects.toThrow('connection')
  expect(fetch).not.toHaveBeenCalled()
  vi.mocked(fetch).mockResolvedValueOnce(new Response('expired', { status: 401 }))
  await expect(bridge.saveGatewayFile({ path: '/report' })).rejects.toMatchObject({ needsOauthLogin: true })
  vi.mocked(fetch).mockRejectedValueOnce(new TypeError('Failed to fetch'))
  await expect(bridge.saveImageFromUrl('https://images.example/picture')).rejects.toThrow('Failed to fetch')
  expect(downloads).toEqual([])

  await bridge.saveImageFromUrl('https://images.example/picture')
  const [, init] = vi.mocked(fetch).mock.calls.at(-1)!
  expect(new Headers(init?.headers).has('X-Hermes-Session-Token')).toBe(false)
  expect(init?.credentials).toBe('omit')
  expect(downloads.at(-1)?.name).toBe('picture.png')

  await bridge.saveImageFromUrl('/api/fs/download?path=%2Fpicture.png&profile=background&session_id=origin')
  const [url, authenticated] = vi.mocked(fetch).mock.calls.at(-1)!
  expect(new URL(String(url)).searchParams.get('profile')).toBe('background')
  expect(new Headers(authenticated?.headers).get('X-Hermes-Session-Token')).toBe('fixture-secret')
  await expect(bridge.saveImageFromUrl('file:///private/image.png')).rejects.toThrow('Unsupported')
})
