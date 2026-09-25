import { translateNow } from '@/i18n'

import { readDesktopFileDataUrl } from './desktop-fs'

function decodeEditorBytes(bytes: Uint8Array): string {
  if (
    bytes.byteLength > 512 * 1024 ||
    bytes.some(byte => (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) || byte === 127)
  ) {
    throw new Error(translateNow('preview.browserInvalidText'))
  }

  try {
    return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes)
  } catch {
    throw new Error(translateNow('preview.browserInvalidText'))
  }
}

export async function readBrowserEditorText(path: string): Promise<string> {
  const dataUrl = await readDesktopFileDataUrl(path)
  const header = /^data:[^,]*;base64,/.exec(dataUrl)

  if (!header) {
    throw new Error(translateNow('preview.browserInvalidText'))
  }

  const base64 = dataUrl.slice(header[0].length)
  let binary: string

  try {
    binary = atob(base64)
  } catch {
    throw new Error(translateNow('preview.browserInvalidText'))
  }

  if (btoa(binary) !== base64) {
    throw new Error(translateNow('preview.browserInvalidText'))
  }

  return decodeEditorBytes(Uint8Array.from(binary, character => character.charCodeAt(0)))
}

export function validateBrowserEditorText(text: string): void {
  // TextEncoder replaces lone surrogates; roundtripping must not change the text.
  if (decodeEditorBytes(new TextEncoder().encode(text)) !== text) {
    throw new Error(translateNow('preview.browserInvalidText'))
  }
}
