const STORAGE_KEY = 'hermes:browser:global:uiScalePercent'
const DEFAULT_PERCENT = 100

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

  // A page cannot set the browser's own zoom, and CSS `zoom` reports measured
  // positions in different units than it lays out, which misplaces anything
  // positioned from a measurement. Scaling the root font size grows every
  // rem-based size instead and keeps one coordinate space. Codicons default to
  // a fixed 16px, so give them 1rem; `html` outranks the icon font's own rule.
  const style = document.getElementById('hermes-browser-zoom') ?? document.createElement('style')
  style.id = 'hermes-browser-zoom'
  style.textContent = `
    html { font-size: calc(var(--dt-base-size, 0.875rem) * var(--hermes-ui-scale, 1)); }
    html .codicon[class*='codicon-'] { font-size: 1rem; }
  `
  document.head.appendChild(style)

  const apply = () => {
    document.documentElement.style.setProperty('--hermes-ui-scale', String(percent / 100))
  }

  apply()

  return {
    async get() {
      return snapshot()
    },
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
