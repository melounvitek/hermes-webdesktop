import type { ComposerAttachment } from '@/store/composer'
import { createComposerAttachmentOccurrenceId } from '@/store/composer'

/** The File stays renderer-local; its name is never a gateway filesystem path. */
export function browserAttachment(blob: Blob, kind: 'file' | 'image'): ComposerAttachment {
  const occurrenceId = createComposerAttachmentOccurrenceId()

  return {
    blob,
    id: `upload:${occurrenceId}`,
    occurrenceId,
    kind,
    label: blob instanceof File && blob.name ? blob.name : 'image.png'
  }
}

export function pickBrowserFiles(accept = ''): Promise<File[]> {
  return new Promise(resolve => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    input.accept = accept
    input.hidden = true

    const finish = (files: File[]) => {
      input.remove()
      resolve(files)
    }

    input.addEventListener('change', () => finish(Array.from(input.files ?? [])), { once: true })
    input.addEventListener('cancel', () => finish([]), { once: true })
    document.body.append(input)
    input.click()
  })
}
