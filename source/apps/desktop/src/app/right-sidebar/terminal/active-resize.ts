interface ActiveTerminalResizeOptions {
  onActivate: () => void
  onFit: () => void
}

/**
 * Observe one visible xterm host.
 *
 * Inactive terminals never call this helper, so their preserved DOM/PTY stays
 * mounted without paying for ResizeObserver delivery or FitAddon work. The
 * first frame reconciles the current box, which may have changed since mount.
 * Observer deliveries after activation coalesce to one fit per animation frame.
 */
export function observeActiveTerminalResize(
  host: HTMLElement,
  { onActivate, onFit }: ActiveTerminalResizeOptions
): () => void {
  let activated = false
  let frame = 0
  let stopped = false

  const scheduleFit = () => {
    if (!activated || stopped || frame !== 0) {
      return
    }

    frame = window.requestAnimationFrame(() => {
      frame = 0

      if (!stopped) {
        onFit()
      }
    })
  }

  // Even the first delivery can describe a change since the activation frame.
  const observer = new ResizeObserver(scheduleFit)

  observer.observe(host)

  frame = window.requestAnimationFrame(() => {
    frame = 0

    if (stopped) {
      return
    }

    activated = true

    onFit()
    onActivate()
  })

  return () => {
    stopped = true
    observer.disconnect()

    if (frame !== 0) {
      window.cancelAnimationFrame(frame)
      frame = 0
    }
  }
}
