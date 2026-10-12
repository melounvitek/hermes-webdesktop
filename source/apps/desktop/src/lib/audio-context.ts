// One lazily-created AudioContext for the app's synthesized UI cues (wake
// chime, turn-completion sound, thinking blips) and spoken replies. Browsers
// cap how many contexts a page may open, and iOS WebKit lets each context start
// only from a user gesture, so they share one instead of each holding its own.
// Nothing closes it — it lives for the window.

let ctx: AudioContext | null = null

/** The shared AudioContext, or null where WebAudio is unavailable. */
export function getAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    if (!ctx || ctx.state === 'closed') {
      const Ctor =
        window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext

      if (!Ctor) {
        return null
      }

      ctx = new Ctor()
    }

    // Autoplay policies can leave the context suspended until a gesture, and
    // iOS interrupts it (calls, backgrounding); a resume() here recovers it once
    // the user has interacted with the window.
    if (ctx.state !== 'running') {
      void ctx.resume().catch(() => undefined)
    }

    return ctx
  } catch {
    return null
  }
}
