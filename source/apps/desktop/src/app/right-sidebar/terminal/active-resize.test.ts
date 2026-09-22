import { afterEach, describe, expect, it, vi } from 'vitest'

import { observeActiveTerminalResize } from './active-resize'

afterEach(() => {
  vi.unstubAllGlobals()
})

function installRaf() {
  let nextId = 1
  const frames = new Map<number, FrameRequestCallback>()

  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    const id = nextId++
    frames.set(id, callback)

    return id
  })
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))

  return {
    flush() {
      const pending = [...frames.entries()]
      frames.clear()
      pending.forEach(([, callback]) => callback(0))
    },
    pending: () => frames.size
  }
}

describe('observeActiveTerminalResize', () => {
  it('fits once on activation and coalesces later resize bursts', () => {
    const raf = installRaf()
    const resize = { current: null as ResizeObserverCallback | null }
    const disconnect = vi.fn()

    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: ResizeObserverCallback) {
          resize.current = callback
        }

        disconnect = disconnect
        observe = vi.fn((target: Element) => {
          resize.current?.([{ target } as ResizeObserverEntry], this as unknown as ResizeObserver)
        })
        unobserve = vi.fn()
      } as unknown as typeof ResizeObserver
    )

    const onFit = vi.fn()
    const onActivate = vi.fn()
    const host = document.createElement('div')
    const dispose = observeActiveTerminalResize(host, { onActivate, onFit })

    // ResizeObserver's initial delivery is absorbed by the activation frame.
    expect(raf.pending()).toBe(1)
    raf.flush()
    expect(onFit).toHaveBeenCalledTimes(1)
    expect(onActivate).toHaveBeenCalledTimes(1)

    resize.current?.([], {} as ResizeObserver)
    resize.current?.([], {} as ResizeObserver)
    resize.current?.([], {} as ResizeObserver)
    expect(raf.pending()).toBe(1)
    raf.flush()
    expect(onFit).toHaveBeenCalledTimes(2)

    dispose()
    expect(disconnect).toHaveBeenCalledTimes(1)
  })

  it('cancels activation without fitting when hidden before the first frame', () => {
    const raf = installRaf()

    vi.stubGlobal(
      'ResizeObserver',
      class {
        disconnect = vi.fn()
        observe = vi.fn()
        unobserve = vi.fn()
      } as unknown as typeof ResizeObserver
    )

    const onFit = vi.fn()

    const dispose = observeActiveTerminalResize(document.createElement('div'), {
      onActivate: vi.fn(),
      onFit
    })

    dispose()

    raf.flush()
    expect(onFit).not.toHaveBeenCalled()
  })

  it('fits a changed host on the first resize delivery after activation', () => {
    const raf = installRaf()
    const resize = { current: null as ResizeObserverCallback | null }

    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: ResizeObserverCallback) {
          resize.current = callback
        }

        disconnect = vi.fn()
        observe = vi.fn()
        unobserve = vi.fn()
      } as unknown as typeof ResizeObserver
    )

    const host = document.createElement('div')
    let width = 300
    let fittedWidth = 0
    Object.defineProperty(host, 'clientWidth', { get: () => width })
    const onFit = vi.fn(() => { fittedWidth = host.clientWidth })
    observeActiveTerminalResize(host, { onActivate: vi.fn(), onFit })

    raf.flush()
    expect(fittedWidth).toBe(300)

    // The first browser delivery may describe a different box than activation.
    width = 200
    resize.current?.([], {} as ResizeObserver)
    expect(raf.pending()).toBe(1)
    raf.flush()
    expect(fittedWidth).toBe(200)
    expect(onFit).toHaveBeenCalledTimes(2)
  })

  it('cancels a queued resize and ignores deliveries after disposal', () => {
    const raf = installRaf()
    let deliverResize!: () => void
    vi.stubGlobal('ResizeObserver', class {
      constructor(callback: ResizeObserverCallback) {
        deliverResize = () => callback([], this as unknown as ResizeObserver)
      }
      disconnect = vi.fn()
      observe = vi.fn()
    })
    const onFit = vi.fn()
    const dispose = observeActiveTerminalResize(document.createElement('div'), { onActivate: vi.fn(), onFit })
    raf.flush()
    expect(onFit).toHaveBeenCalledOnce()
    deliverResize()
    expect(raf.pending()).toBe(1)
    dispose()
    deliverResize()
    raf.flush()
    expect(onFit).toHaveBeenCalledOnce()
    expect(raf.pending()).toBe(0)
  })
})
