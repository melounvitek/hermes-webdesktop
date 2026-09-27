import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { $confirmRequest, settleConfirm } from '@/store/confirm'

// The one mount point for `confirm()` from @/store/confirm. Mounted once at the
// shell, the way NotificationStack backs notify().
export function ConfirmHost() {
  const request = useStore($confirmRequest)
  // The atom clears the moment the question is answered, but Radix still has a
  // close animation to play — hold the copy so the dialog doesn't blank mid-fade.
  const [{ request: shown, generation }, setShown] = useState({ request, generation: 0 })

  useEffect(() => {
    if (request) {
      setShown(previous => (previous.request === request ? previous : { request, generation: previous.generation + 1 }))
    }
  }, [request])

  if (!shown) {
    return null
  }

  // A's confirmation can open B before A's awaited onClose continuation runs.
  function settle(confirmed: boolean) {
    if ($confirmRequest.get() === shown) {
      settleConfirm(confirmed)
    }
  }

  return (
    <ConfirmDialog
      cancelLabel={shown.cancelLabel}
      confirmLabel={shown.confirmLabel}
      description={shown.description}
      destructive={shown.destructive}
      // The caller does the work once it has its answer, so there is nothing
      // here to keep the dialog open for.
      dismissOnConfirm
      // A replacement must not inherit the previous request's pending state.
      key={generation}
      onClose={() => settle(false)}
      onConfirm={() => settle(true)}
      open={request !== null}
      title={shown.title}
    />
  )
}
