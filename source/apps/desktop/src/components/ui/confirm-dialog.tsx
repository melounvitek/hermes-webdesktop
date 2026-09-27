import type { ReactNode } from 'react'
import { useEffect, useRef, useState } from 'react'

import { ActionStatus } from '@/components/ui/action-status'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'

interface ConfirmDialogProps {
  open: boolean
  onClose: () => void
  // Does the work. Throw to surface an inline error and keep the dialog open.
  onConfirm: () => Promise<void> | void
  title: ReactNode
  description?: ReactNode
  confirmLabel?: string
  busyLabel?: string
  doneLabel?: string
  cancelLabel?: string
  destructive?: boolean
  /** Close as soon as onConfirm resolves — for optimistic actions that finish in the background. */
  dismissOnConfirm?: boolean
  /** A third, non-destructive way out, shown between Cancel and Confirm (e.g.
   *  "Remove from sidebar" beside "Delete worktree"). Closes on click. */
  secondaryAction?: ConfirmSecondaryAction
}

interface ConfirmSecondaryAction {
  label: string
  onClick: () => void
}

// Shared confirmation dialog: opens focused on Confirm; Enter/Space activate
// the focused button. Esc/Cancel/backdrop dismiss when idle. Owns the pending
// → done → close beat and inline error, so callers pass only an async onConfirm.
export function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  description,
  confirmLabel,
  busyLabel,
  doneLabel,
  cancelLabel,
  destructive = false,
  dismissOnConfirm = false,
  secondaryAction
}: ConfirmDialogProps) {
  const { t } = useI18n()
  const confirmRef = useRef<HTMLButtonElement>(null)
  const inFlightRef = useRef(false)
  const mountedRef = useRef(false)
  const closeTimerRef = useRef<null | number>(null)
  const [status, setStatus] = useState<'done' | 'idle' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)
  const busy = status === 'saving' || status === 'done'
  const resolvedConfirmLabel = confirmLabel ?? t.common.confirm
  const resolvedBusyLabel = busyLabel ?? t.common.loading
  const resolvedDoneLabel = doneLabel ?? t.common.done
  const resolvedCancelLabel = cancelLabel ?? t.common.cancel

  useEffect(() => {
    if (open) {
      setStatus('idle')
      setError(null)
    }
  }, [open])

  // Unmount retires the UI, not the work: late completions must not close a
  // replacement dialog or install a timer after cleanup has already run.
  // eslint-disable-next-line no-restricted-syntax -- lifecycle and timer ownership, not mirrored state
  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false

      if (closeTimerRef.current !== null) {
        window.clearTimeout(closeTimerRef.current)
        closeTimerRef.current = null
      }
    }
  }, [])

  async function run() {
    if (busy || inFlightRef.current) {
      return
    }

    inFlightRef.current = true
    setError(null)
    setStatus('saving')

    try {
      await onConfirm()

      if (!mountedRef.current) {
        return
      }

      if (dismissOnConfirm) {
        onClose()
      } else {
        setStatus('done')
        closeTimerRef.current = window.setTimeout(() => {
          closeTimerRef.current = null
          onClose()
        }, 600)
      }
    } catch (err) {
      if (!mountedRef.current) {
        return
      }

      setStatus('idle')
      setError(err instanceof Error ? err.message : t.errors.genericFailure)
    } finally {
      inFlightRef.current = false
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !busy && onClose()} open={open}>
      <DialogContent
        className="max-w-md"
        onOpenAutoFocus={event => {
          // Radix defaults to the X; start on Confirm and let native button
          // activation follow focus from there.
          event.preventDefault()
          confirmRef.current?.focus()
        }}
      >
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {/* pre-line: a backend-composed description keeps its paragraph
              breaks instead of collapsing into one run-on line (#112458). */}
          {description ? <DialogDescription className="whitespace-pre-line">{description}</DialogDescription> : null}
        </DialogHeader>

        {error && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <DialogFooter>
          <Button disabled={busy} onClick={onClose} type="button" variant="ghost">
            {resolvedCancelLabel}
          </Button>
          {secondaryAction && (
            <Button
              disabled={busy}
              onClick={() => {
                secondaryAction.onClick()
                onClose()
              }}
              type="button"
              variant="secondary"
            >
              {secondaryAction.label}
            </Button>
          )}
          <Button
            disabled={busy}
            onClick={() => void run()}
            ref={confirmRef}
            variant={destructive ? 'destructive' : 'default'}
          >
            <ActionStatus
              busy={resolvedBusyLabel}
              done={resolvedDoneLabel}
              idle={resolvedConfirmLabel}
              state={status}
            />
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
