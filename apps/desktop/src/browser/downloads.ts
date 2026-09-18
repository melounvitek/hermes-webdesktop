import type { HermesApiRequest } from '../global'
import { downloadFilename, saveBlobDownload } from '../lib/image-download'

export function createBrowserDownloads(request: (payload: HermesApiRequest) => Promise<Response>) {
  return {
    async saveGatewayFile(payload: Parameters<NonNullable<Window['hermesDesktop']['saveGatewayFile']>>[0]) {
      const params = new URLSearchParams({ path: payload.path })

      if (payload.sessionId !== undefined) {
        params.set('session_id', payload.sessionId)
      }

      const response = await request({
        path: `/api/fs/download?${params}`,
        profile: payload.profile,
        connectionId: payload.connectionId
      })

      const filename = payload.suggestedName || payload.path.split(/[\\/]/).pop() || 'download'
      saveBlobDownload(await response.blob(), filename)

      // Browsers acknowledge dispatch, not the user's eventual save/cancel decision.
      return { saved: true }
    },
    async saveImageFromUrl(value: string) {
      const url = new URL(value, window.location.href)

      if (!['http:', 'https:', 'blob:', 'data:'].includes(url.protocol)) {
        throw new Error('Unsupported image URL scheme')
      }

      const gatewayImage = url.origin === window.location.origin && url.pathname.startsWith('/api/')

      const response = gatewayImage
        ? await request({ path: url.href })
        : await fetch(url.href, { credentials: 'omit', signal: AbortSignal.timeout(30_000) })

      if (!response.ok) {
        throw new Error(`Could not fetch image: HTTP ${response.status}`)
      }

      const blob = await response.blob()
      const source = gatewayImage ? url.searchParams.get('path') || value : value
      saveBlobDownload(blob, downloadFilename(source, blob.type))

      return true
    }
  }
}
