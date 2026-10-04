import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { BROWSER_BUILD } from '@/browser/build'
import { $browserUpdate, $browserUpdatesOpen, closeBrowserUpdates, requestBrowserUpdate } from '@/browser/updates'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { ErrorState } from '@/components/ui/error-state'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'
import { isBrowserClient } from '@/lib/platform'

export function BrowserUpdatesDialog() {
  const open = useStore($browserUpdatesOpen)

  return isBrowserClient() && open ? <BrowserUpdatesPanel /> : null
}

function BrowserUpdatesPanel() {
  const { t } = useI18n()
  const copy = t.browserUpdates
  const update = useStore($browserUpdate)
  const [confirmingReload, setConfirmingReload] = useState(false)

  return (
    <>
      <Dialog
        onOpenChange={open => {
          if (!open) {
            closeBrowserUpdates()
          }
        }}
        open
      >
        <DialogContent bodyClassName="grid gap-5 p-6" className="max-w-xl">
          <DialogHeader>
            <DialogTitle>{copy.title}</DialogTitle>
            <DialogDescription>{copy.description}</DialogDescription>
          </DialogHeader>
          <dl className="grid gap-3 text-sm">
            <div>
              <dt className="text-muted-foreground">{copy.loadedBuild}</dt>
              <dd className="break-all font-mono">
                {BROWSER_BUILD.revision ?? copy.unknownBuild}
                {BROWSER_BUILD.dirty ? ` · ${copy.localChanges}` : ''}
              </dd>
            </div>
          </dl>
          {update.state === 'working' && (
            <div className="flex items-start gap-2 text-sm" role="status">
              <GlyphSpinner />
              <p>{copy.working}</p>
            </div>
          )}
          {update.state === 'updated' && (
            <div className="grid gap-2 text-sm" role="status">
              <p>{copy.updated(update.release)}</p>
              <p className="text-muted-foreground">{copy.reloadHint}</p>
            </div>
          )}
          {update.state === 'current' && (
            <p className="text-sm" role="status">
              {copy.current(update.release)}
            </p>
          )}
          {update.state === 'failed' && (
            <ErrorState className="wrap-anywhere" description={update.error ?? copy.timedOut} title={t.common.error} />
          )}
          {update.state === 'unavailable' && (
            <div className="grid gap-2 text-sm" role="status">
              <p>{copy.unavailable}</p>
              <pre className="overflow-x-auto rounded-md border border-(--stroke-nous) px-3 py-2.5 font-mono text-[12px]">
                <code>{'hermes-browser stop\nhermes-browser update\nhermes-browser start'}</code>
              </pre>
            </div>
          )}
          <DialogFooter>
            {update.state === 'updated' ? (
              <Button onClick={() => setConfirmingReload(true)}>{copy.reload}</Button>
            ) : (
              <Button disabled={update.state === 'working'} onClick={() => void requestBrowserUpdate()}>
                {copy.update}
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <ConfirmDialog
        confirmLabel={copy.reload}
        description={copy.reloadDescription}
        destructive
        dismissOnConfirm
        onClose={() => setConfirmingReload(false)}
        onConfirm={() => window.location.reload()}
        open={confirmingReload}
        title={copy.reloadTitle}
      />
    </>
  )
}
