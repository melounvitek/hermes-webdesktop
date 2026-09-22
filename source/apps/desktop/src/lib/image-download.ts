const MIME_EXTENSIONS: Record<string, string> = {
  'image/bmp': '.bmp',
  'image/gif': '.gif',
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/svg+xml': '.svg',
  'image/webp': '.webp'
}

const KNOWN_IMAGE_EXTENSION_RE = /\.(?:apng|avif|bmp|gif|ico|jpe?g|png|svg|tiff?|webp)$/i

export function imageFilename(src?: string): string {
  if (!src) {
    return 'image'
  }

  try {
    const url = new URL(src, window.location.href)

    if (url.protocol === 'data:' || url.protocol === 'blob:') {
      return 'image'
    }

    return url.pathname.split('/').filter(Boolean).pop() || 'image'
  } catch {
    return src.split(/[\\/]/).filter(Boolean).pop() || 'image'
  }
}

/** Extensionless generated-image URLs need the blob's MIME extension to open locally. */
export function downloadFilename(src: string, mimeType?: string): string {
  const base = imageFilename(src)

  if (KNOWN_IMAGE_EXTENSION_RE.test(base)) {
    return base
  }

  const type = String(mimeType || '')
    .split(';')[0]
    .trim()
    .toLowerCase()

  return `${base}${MIME_EXTENSIONS[type] || '.png'}`
}

export function saveBlobDownload(blob: Blob, filename: string): void {
  const blobUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = blobUrl
  link.download = filename
  link.rel = 'noopener noreferrer'
  document.body.appendChild(link)

  try {
    link.click()
  } finally {
    link.remove()
    // Revoking immediately can race the browser consuming the download URL.
    window.setTimeout(() => URL.revokeObjectURL(blobUrl), 30_000)
  }
}
