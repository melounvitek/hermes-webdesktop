// Safari on iPhone and iPad zooms the page in when a text field with text
// under 16px takes focus, and stays zoomed after the keyboard closes, so the
// composer's Send button ends up off screen. maximum-scale stops that zoom,
// while iOS still lets the user pinch-zoom. Other browsers would block pinch
// zoom too, so only iOS gets it; iPads can report themselves as a Mac.
export function stopIosTypingZoom() {
  const { maxTouchPoints, userAgent } = navigator
  const ios = /iPhone|iPad|iPod/.test(userAgent) || (/Macintosh/.test(userAgent) && maxTouchPoints > 1)
  const viewport = document.querySelector<HTMLMetaElement>('meta[name="viewport"]')

  if (ios && viewport && !viewport.content.includes('maximum-scale')) {
    viewport.content += ', maximum-scale=1'
  }
}
