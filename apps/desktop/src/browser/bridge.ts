import { buildHermesWebSocketUrl, GatewayReauthRequiredError } from '@hermes/shared'

import type { DesktopBootProgress, HermesApiRequest, HermesConnection } from '../global'

import { setBrowserAttentionConnected } from './attention-notifications'
import { createBrowserDownloads } from './downloads'
import { createBrowserZoom } from './zoom'

interface BrowserConfig {
  token: string
  authRequired: boolean
}

export function createBrowserBridge({ token, authRequired }: BrowserConfig) {
  // main.tsx replaces writeText with the desktop clipboard shim after we load.
  const writeText = navigator.clipboard?.writeText.bind(navigator.clipboard)
  const readText = navigator.clipboard?.readText.bind(navigator.clipboard)

  const bootListeners = new Set<(progress: DesktopBootProgress) => void>()
  let authError: string | null = null

  const bootProgress = (): DesktopBootProgress => ({
    error: authError,
    fakeMode: false,
    message: authError ?? '',
    phase: authError ? 'backend.error' : 'backend.ready',
    progress: 100,
    running: false,
    retryable: !authError,
    timestamp: Date.now()
  })

  const wsUrl = (profile?: string | null, ticket?: string) =>
    buildHermesWebSocketUrl({
      path: '/api/ws',
      authParam: ticket ? ['ticket', ticket] : ['token', token],
      params: profile ? { profile } : undefined
    })

  async function fetchResponse(request: HermesApiRequest, redirect?: RequestRedirect): Promise<Response> {
    if (request.connectionId) {
      throw new Error('The browser spike supports only its same-origin connection')
    }

    const url = new URL(request.path, window.location.origin)

    if (url.origin !== window.location.origin || !url.pathname.startsWith('/api/')) {
      throw new Error('Only same-origin /api/ requests are supported')
    }

    // Some renderer actions call REST directly instead of the update store.
    if (decodeURIComponent(url.pathname).replace(/\/+$/, '') === '/api/hermes/update') {
      throw new Error('Backend updates are unavailable in the browser.')
    }

    if (request.profile && !url.searchParams.has('profile')) {
      url.searchParams.set('profile', request.profile)
    }

    const headers = new Headers()

    if (token) {
      headers.set('X-Hermes-Session-Token', token)
    }

    let body: BodyInit | undefined

    if (request.upload) {
      const form = new FormData()
      form.append(
        'file',
        new Blob([request.upload.bytes], { type: request.upload.contentType }),
        request.upload.filename
      )
      body = form
    } else if (request.body !== undefined) {
      headers.set('Content-Type', 'application/json')
      body = JSON.stringify(request.body)
    }

    const response = await fetch(url.toString(), {
      method: request.method ?? 'GET',
      body,
      headers,
      credentials: 'same-origin',
      redirect,
      signal: AbortSignal.timeout(request.timeoutMs ?? 30_000)
    })

    if (!response.ok) {
      const message = `HTTP ${response.status}: ${await response.text()}`

      // A resource-level 403 is not proof that the login expired. The ticket
      // endpoint, unlike a file/settings operation, is an authentication probe.
      if (
        authRequired &&
        (response.status === 401 || (response.status === 403 && url.pathname === '/api/auth/ws-ticket'))
      ) {
        setBrowserAttentionConnected(false)
        authError = `Gateway sign-in required. ${message}`

        for (const listener of bootListeners) {
          listener(bootProgress())
        }

        throw new GatewayReauthRequiredError(authError)
      }

      throw new Error(message)
    }

    return response
  }

  async function api<T>(request: HermesApiRequest): Promise<T> {
    const response = await fetchResponse(request)

    return response.status === 204 ? (undefined as T) : response.json()
  }

  return {
    browser: {
      authRequired,
      signIn() {
        setBrowserAttentionConnected(false)
        const { pathname, search, hash } = window.location
        window.location.assign(`/login?${new URLSearchParams({ next: pathname + search + hash })}`)
      }
    },
    // Never forward the gateway's custom auth header through a redirect.
    ...createBrowserDownloads(request => fetchResponse(request, 'error')),
    zoom: createBrowserZoom(),
    glassSupported: false,
    translucencySupported: false,
    guestOnboardingEnabled: false,
    localModelsEnabled: false,
    api,
    async getConnection(profile?: string | null): Promise<HermesConnection> {
      if (!authRequired) {
        // Refresh before publishing the descriptor: both sockets and media URLs
        // need the token from the current backend process, without a page reload.
        const response = await fetch('/', {
          cache: 'no-store',
          credentials: 'same-origin',
          redirect: 'error',
          signal: AbortSignal.timeout(15_000)
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: could not refresh browser credentials`)
        }

        const document = new DOMParser().parseFromString(await response.text(), 'text/html')

        const assignment = Array.from(document.scripts)
          .map(
            script => script.textContent?.match(/window\.__HERMES_SESSION_TOKEN__\s*=\s*("(?:[^"\\]|\\.)*")\s*;/)?.[1]
          )
          .find(Boolean)

        if (!assignment) {
          throw new Error('Backend returned no browser session token. Reload the page.')
        }

        token = JSON.parse(assignment) as string
      }

      return {
        baseUrl: window.location.origin,
        token,
        wsUrl: wsUrl(profile),
        mode: 'remote',
        remoteKind: 'url',
        authMode: authRequired ? 'oauth' : 'token',
        profile: profile || 'default',
        sharedPrimary: true,
        isFullscreen: false,
        isMaximized: false,
        customWindowControls: false,
        nativeOverlayWidth: 0,
        windowButtonPosition: null,
        logs: []
      }
    },
    async getGatewayWsUrl(profile?: string | null): Promise<string> {
      if (!authRequired) {
        return wsUrl(profile)
      }

      const { ticket } = await api<{ ticket: string }>({ path: '/api/auth/ws-ticket', method: 'POST' })

      if (!ticket) {
        throw new Error('Backend returned no WebSocket ticket')
      }

      return wsUrl(profile, ticket)
    },
    async getBootProgress(): Promise<DesktopBootProgress> {
      return bootProgress()
    },
    onBootProgress(listener: (progress: DesktopBootProgress) => void) {
      bootListeners.add(listener)

      return () => {
        bootListeners.delete(listener)
      }
    },
    // Browser transport observes disconnects; there is no child-process lifecycle.
    onBackendExit: () => () => {},
    notify: async () => false,
    async writeClipboard(text: string) {
      if (!writeText) {
        throw new Error('Browser clipboard is unavailable')
      }

      await writeText(text)

      return true
    },
    async readClipboard() {
      if (!readText) {
        throw new Error('Browser clipboard is unavailable')
      }

      return readText()
    },
    async openExternal(value: string) {
      const url = new URL(value)

      if (!['https:', 'http:', 'mailto:'].includes(url.protocol)) {
        throw new Error('Unsupported external URL scheme')
      }

      window.open(url.href, '_blank', 'noopener,noreferrer')
    }
  } satisfies Partial<Window['hermesDesktop']>
}
