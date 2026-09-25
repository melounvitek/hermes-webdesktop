export interface BrowserFileDraft {
  baseline: string
  text: string
}

// Memory only: file contents must not be copied into persistent browser storage.
// Keep drafts across pane/profile switches, but warn before losing the browser tab.
const drafts = new Map<string, BrowserFileDraft>()

interface BrowserFileSave {
  text: string
  promise: Promise<string>
}

const saves = new Map<string, BrowserFileSave>()

export function getBrowserFileSave(key: string): BrowserFileSave | undefined {
  return saves.get(key)
}

// A write belongs to the file, not the mounted editor. A reopened editor must
// wait for it and adopt its new baseline (or display its error).
export function saveBrowserFileDraft(key: string, text: string, write: () => Promise<unknown>): Promise<string> {
  const pending = saves.get(key)

  if (pending) {
    return pending.promise
  }

  const promise = write()
    .then(() => {
      const latest = drafts.get(key)

      if (latest) {
        setBrowserFileDraft(key, { baseline: text, text: latest.text })
      }

      return text
    })
    .finally(() => {
      saves.delete(key)
    })

  saves.set(key, { text, promise })

  return promise
}

function warnBeforeUnload(event: BeforeUnloadEvent) {
  event.preventDefault()
  event.returnValue = ''
}

export function getBrowserFileDraft(key: string): BrowserFileDraft | undefined {
  return drafts.get(key)
}

export function setBrowserFileDraft(key: string, draft: BrowserFileDraft): void {
  if (draft.text === draft.baseline) {
    clearBrowserFileDraft(key)

    return
  }

  if (!drafts.size) {
    window.addEventListener('beforeunload', warnBeforeUnload)
  }

  drafts.set(key, draft)
}

export function clearBrowserFileDraft(key: string): void {
  drafts.delete(key)

  if (!drafts.size) {
    window.removeEventListener('beforeunload', warnBeforeUnload)
  }
}
