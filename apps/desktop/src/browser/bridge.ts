import { buildHermesWebSocketUrl, GatewayReauthRequiredError } from '@hermes/shared'

import type { DesktopBootProgress, HermesApiRequest, HermesConnection } from '../global'

import { createBrowserDownloads } from './downloads'
import { createBrowserTerminal } from './terminal'
import { createBrowserZoom } from './zoom'

interface BrowserConfig {
  token: string
  authRequired: boolean
}

export function createBrowserBridge({ token, authRequired }: BrowserConfig) {
  // main.tsx replaces writeText with the desktop clipboard shim after we load.
  const writeText = navigator.clipboard?.writeText.bind(navigator.clipboard)
  const readText = navigator.clipboard?.readText.bind(navigator.clipboard)

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

      if (authRequired && (response.status === 401 || response.status === 403)) {
        throw new GatewayReauthRequiredError(message)
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
    // Never forward the gateway's custom auth header through a redirect.
    ...createBrowserDownloads(request => fetchResponse(request, 'error')),
    zoom: createBrowserZoom(),
    terminal: createBrowserTerminal(api),
    glassSupported: false,
    translucencySupported: false,
    guestOnboardingEnabled: false,
    localModelsEnabled: false,
    api,
    async getConnection(profile?: string | null): Promise<HermesConnection> {
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
      return {
        error: null,
        fakeMode: false,
        message: '',
        phase: 'backend.ready',
        progress: 100,
        running: false,
        timestamp: Date.now()
      }
    },
    // Browser transport observes disconnects; there is no child-process lifecycle.
    onBootProgress: () => () => {},
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
