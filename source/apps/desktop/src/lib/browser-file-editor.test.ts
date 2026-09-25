import { Buffer } from 'node:buffer'

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { translateNow } from '@/i18n'

import { readBrowserEditorText, validateBrowserEditorText } from './browser-file-editor'
import { readDesktopFileDataUrl } from './desktop-fs'

vi.mock('./desktop-fs', () => ({ readDesktopFileDataUrl: vi.fn() }))
vi.mock('@/i18n', () => ({ translateNow: vi.fn() }))

const invalidTextMessage = 'Only complete UTF-8 text files up to 512 KB, without binary control bytes, can be edited.'
const byteLimit = 512 * 1024
const multibyteText = '\uFEFFPříliš žluťoučký 🐎 日本語\t\r\n'
const atByteLimit = 'é'.repeat(byteLimit / 2)

function dataUrl(bytes: Uint8Array): string {
  return `data:application/octet-stream;base64,${Buffer.from(bytes).toString('base64')}`
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(translateNow).mockReturnValue(invalidTextMessage)
})

describe('browser file editor text', () => {
  it('reads original bytes losslessly and rejects malformed, binary, or oversized responses', async () => {
    for (const text of ['', multibyteText, '\uFFFD', atByteLimit]) {
      const bytes = new TextEncoder().encode(text)
      vi.mocked(readDesktopFileDataUrl).mockResolvedValue(dataUrl(bytes))

      const result = await readBrowserEditorText('/work/example.txt')

      expect(readDesktopFileDataUrl).toHaveBeenLastCalledWith('/work/example.txt')
      expect(result).toBe(text)
      expect(new TextEncoder().encode(result)).toEqual(bytes)
      expect(() => validateBrowserEditorText(result)).not.toThrow()
    }

    const invalidBytes = [
      [0xc3, 0x28], // Invalid continuation.
      [0xe2, 0x82], // Truncated character.
      [0xc0, 0xaf], // Overlong encoding.
      [0xed, 0xa0, 0x80], // Encoded surrogate.
      [0xf4, 0x90, 0x80, 0x80], // Beyond Unicode.
      ...Array.from({ length: 32 }, (_, byte) => byte)
        .filter(byte => ![9, 10, 13].includes(byte))
        .map(byte => [65, byte, 66]),
      [0x7f]
    ]

    const invalidResponses = [
      ...invalidBytes.map(bytes => dataUrl(Uint8Array.from(bytes))),
      dataUrl(new TextEncoder().encode(`${atByteLimit}a`)),
      '',
      'data:text/plain,hello',
      'https://example.com/file.txt',
      'data:text/plain;base64,!!!!',
      'data:text/plain;base64,YQ=',
      'data:text/plain;base64,YQ===',
      'data:text/plain;base64,YQ==trailing',
      'data:text/plain;base64,YQ==\n'
    ]

    for (const response of invalidResponses) {
      vi.mocked(readDesktopFileDataUrl).mockResolvedValue(response)
      await expect(readBrowserEditorText('/work/example.txt')).rejects.toThrow(invalidTextMessage)
      expect(translateNow).toHaveBeenLastCalledWith('preview.browserInvalidText')
    }
  })

  it('validates outgoing text without allowing replacement, binary controls, or byte-limit overflow', () => {
    for (const text of ['', multibyteText, '\uFFFD', atByteLimit]) {
      expect(() => validateBrowserEditorText(text)).not.toThrow()
    }

    const invalidTexts = [
      '\uD800',
      '\uDC00',
      'a\uD800b',
      '\uD800\uD800',
      '\uDC00\uD800',
      `${atByteLimit}a`,
      '😀'.repeat(byteLimit / 4 + 1),
      ...Array.from({ length: 32 }, (_, byte) => byte)
        .filter(byte => ![9, 10, 13].includes(byte))
        .map(byte => `a${String.fromCharCode(byte)}b`),
      '\u007F'
    ]

    for (const text of invalidTexts) {
      expect(() => validateBrowserEditorText(text)).toThrow(invalidTextMessage)
      expect(translateNow).toHaveBeenLastCalledWith('preview.browserInvalidText')
    }
  })
})
