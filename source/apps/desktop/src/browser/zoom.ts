const STORAGE_KEY = 'hermes:browser:global:uiScalePercent'
const DEFAULT_PERCENT = 90

function normalizePercent(percent: number): number {
  if (!Number.isFinite(percent) || percent <= 0) {
    return DEFAULT_PERCENT
  }

  // Match the native bridge's Chromium zoom-level range without importing Electron.
  return Math.min(Math.max(percent, 100 * 1.2 ** -9), 100 * 1.2 ** 9)
}

export function createBrowserZoom() {
  let percent = DEFAULT_PERCENT

  try {
    percent = normalizePercent(Number(localStorage.getItem(STORAGE_KEY)))
  } catch {
    // Storage can be blocked; UI scale still works for the current page.
  }

  const listeners = new Set<(payload: { level: number; percent: number }) => void>()
  const snapshot = () => ({ level: Math.log(percent / 100) / Math.log(1.2), percent: Math.round(percent) })

  // Chromium compensates the html/body/#root percentage dimensions for root zoom,
  // but not viewport units. Let the full-window shells inherit that sizing.
  const style = document.getElementById('hermes-browser-zoom') ?? document.createElement('style')
  style.id = 'hermes-browser-zoom'
  style.textContent = `
    #root .h-screen { height: 100%; }
    #root .w-screen { width: 100%; }
    [data-radix-popper-content-wrapper] { zoom: calc(1 / var(--hermes-browser-zoom)); }
    [data-radix-popper-content-wrapper] > * { zoom: var(--hermes-browser-zoom); }
  `
  document.head.appendChild(style)

  const apply = () => {
    // Floating UI positions fixed portals in viewport pixels. Keep their wrappers
    // unscaled while restoring the document scale on the portaled content.
    document.documentElement.style.setProperty('--hermes-browser-zoom', String(percent / 100))
    document.documentElement.style.zoom = String(percent / 100)
  }

  apply()

  return {
    async get() {
      return snapshot()
    },
    factor: () => percent / 100,
    setPercent(value: number) {
      percent = normalizePercent(value)
      apply()

      try {
        localStorage.setItem(STORAGE_KEY, String(percent))
      } catch {
        // Keep the current-page setting usable when persistence is unavailable.
      }

      for (const listener of listeners) {
        listener(snapshot())
      }
    },
    onChanged(callback: (payload: { level: number; percent: number }) => void) {
      listeners.add(callback)

      return () => {
        listeners.delete(callback)
      }
    }
  }
}
