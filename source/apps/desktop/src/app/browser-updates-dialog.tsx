import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { BROWSER_BUILD } from '@/browser/build'
import type { BrowserUpdateOffer, BrowserUpdatePhase } from '@/browser/update-client'
import {
  $browserUpdatesOpen,
  $browserUpdateView,
  checkBrowserSessionAndRetry,
  closeBrowserUpdates,
  confirmBrowserUpdateOffer,
  forgetBrowserUpdateTracking,
  observeBrowserUpdates,
  prepareBrowserUpdateOffer,
  refreshBrowserUpdates
} from '@/browser/updates'
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
import { reconnectGateway } from '@/store/gateway-reconnect'

export function BrowserUpdatesDialog() {
  const open = useStore($browserUpdatesOpen)

  return isBrowserClient() && open ? <BrowserUpdatesPanel /> : null
}

function BrowserUpdatesPanel() {
  const { t } = useI18n()
  const copy = t.browserUpdates
  const view = useStore($browserUpdateView)
  const [confirmation, setConfirmation] = useState<BrowserUpdateOffer | 'reload' | null>(null)
  const [now, setNow] = useState(Date.now)
  const { snapshot, busy, error } = view
  const offer = snapshot?.offer
  const phase = snapshot?.phase
  const expired = offer ? offer.expires_at * 1000 <= now : false
  const active = phase === 'preparing' || (phase !== undefined && phase in copy.phases)
  const capable = snapshot?.capabilities.includes('install') === true
  const canInstall = capable && phase === 'offered' && !expired && !error && !busy
  const canPrepare = capable && !active && !error && !busy && !view.pendingId && phase !== 'recovery_required'
  const completed = phase === 'succeeded' || phase === 'rolled_back'

  const message: Partial<Record<BrowserUpdatePhase, string>> = {
    ...copy.phases,
    idle: copy.checkHint,
    preparing: copy.preparing,
    offered: expired ? copy.expired : copy.checkHint,
    current: copy.current,
    succeeded: copy.succeeded,
    rolled_back: copy.rolledBack,
    failed: copy.failed,
    recovery_required: copy.recoveryRequired
  }

  useEffect(observeBrowserUpdates, [])
  useEffect(() => {
    if (!offer) {
      return
    }

    const timer = setTimeout(
      () => setNow(Date.now()),
      Math.max(0, Math.min(offer.expires_at * 1000 - Date.now() + 1, 2_147_483_647))
    )

    return () => clearTimeout(timer)
  }, [offer])

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
            <DialogDescription>{copy.disconnectNotice}</DialogDescription>
          </DialogHeader>
          <dl className="grid gap-3 text-sm">
            <div>
              <dt className="text-muted-foreground">{copy.loadedBuild}</dt>
              <dd className="break-all font-mono">
                {BROWSER_BUILD.revision ?? copy.unknownBuild}
                {BROWSER_BUILD.dirty ? ` · ${copy.localChanges}` : ''}
              </dd>
            </div>
            {offer && (
              <>
                <div>
                  <dt className="text-muted-foreground">{copy.beforeRelease}</dt>
                  <dd className="break-all font-mono">{offer.current_release}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">{copy.offeredRelease}</dt>
                  <dd className="break-all font-mono">{offer.target_release}</dd>
                </div>
              </>
            )}
          </dl>
          {error || phase === 'failed' || phase === 'recovery_required' ? (
            <ErrorState description={error ? copy[error] : phase ? message[phase] : undefined} title={t.common.error} />
          ) : (
            <div className="flex items-start gap-2 text-sm" role="status">
              {(busy || active) && <GlyphSpinner />}
              <p>
                {busy === 'apply'
                  ? copy.sending
                  : busy === 'offer'
                    ? copy.preparing
                    : phase
                      ? message[phase]
                      : busy
                        ? copy.checking
                        : copy.unsupported}
              </p>
            </div>
          )}
          {view.unknownOutcome && (
            <p className="text-sm text-muted-foreground" role="status">
              {copy.unknownOutcome}
            </p>
          )}
          {offer && <p className="text-sm text-muted-foreground">{copy.compatibility(offer.tested_backend)}</p>}
          {(error === 'forbidden' || error === 'network' || error === 'paused') && (
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => window.open('/login?next=/', '_blank', 'noopener,noreferrer')} variant="outline">
                {copy.signIn}
              </Button>
              <Button
                disabled={Boolean(busy)}
                onClick={() =>
                  void checkBrowserSessionAndRetry()
                    .then(() => reconnectGateway())
                    .catch(() => undefined)
                }
                variant="outline"
              >
                {copy.checkSession}
              </Button>
            </div>
          )}
          {view.pendingId && (error === 'differentJob' || view.unknownOutcome) && (
            <div className="grid gap-2">
              <p className="text-sm text-muted-foreground">{copy.forgetTrackingHint}</p>
              <Button disabled={Boolean(busy)} onClick={forgetBrowserUpdateTracking} variant="outline">
                {copy.forgetTracking}
              </Button>
            </div>
          )}
          {completed && !error && (
            <div className="grid gap-2">
              <p className="text-sm text-muted-foreground">{copy.reloadHint}</p>
              <Button onClick={() => setConfirmation('reload')} variant="outline">
                {copy.reload}
              </Button>
            </div>
          )}
          <DialogFooter className="flex-wrap">
            <Button
              disabled={Boolean(busy)}
              onClick={() => void refreshBrowserUpdates().catch(() => undefined)}
              variant="outline"
            >
              {copy.retry}
            </Button>
            <Button
              disabled={!canPrepare}
              onClick={() => void prepareBrowserUpdateOffer().catch(() => undefined)}
              variant="outline"
            >
              {copy.check}
            </Button>
            <Button
              disabled={!canInstall}
              onClick={() => {
                if (offer) {
                  setConfirmation(offer)
                }
              }}
            >
              {copy.update}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <ConfirmDialog
        busyLabel={copy.sending}
        confirmLabel={confirmation === 'reload' ? copy.reload : copy.update}
        description={
          confirmation === 'reload'
            ? copy.reloadDescription
            : confirmation
              ? copy.confirmDescription(confirmation.current_release, confirmation.target_release)
              : ''
        }
        destructive
        dismissOnConfirm
        onClose={() => setConfirmation(null)}
        onConfirm={async () => {
          if (confirmation === 'reload') {
            window.location.reload()
          } else if (confirmation) {
            // All outcomes return to the status surface. An uncertain response
            // must never leave a confirmation button inviting a blind resend.
            await confirmBrowserUpdateOffer(confirmation).catch(() => undefined)
          }
        }}
        open={confirmation !== null}
        title={confirmation === 'reload' ? copy.reloadTitle : copy.confirmTitle}
      />
    </>
  )
}
