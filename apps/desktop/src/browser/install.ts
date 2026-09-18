import { createBrowserBridge } from './bridge'

declare global {
  interface Window {
    __HERMES_SESSION_TOKEN__?: string
    __HERMES_AUTH_REQUIRED__?: boolean
    __HERMES_BASE_PATH__?: string
  }
}

if (import.meta.env.VITE_BROWSER === '1' && !window.hermesDesktop) {
  if (window.__HERMES_BASE_PATH__) throw new Error('The browser spike requires an origin-root deployment')
  // Spike only: the native contract marks many optional-at-runtime methods required.
  // Leave unsupported capabilities absent instead of returning fabricated success.
  window.hermesDesktop = createBrowserBridge({
    token: window.__HERMES_SESSION_TOKEN__ ?? '',
    authRequired: window.__HERMES_AUTH_REQUIRED__ === true
  }) as unknown as Window['hermesDesktop']
}
