import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo } from 'react'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import {
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { Zap, ZapFilled } from '@/lib/icons'
import { isBrowserClient } from '@/lib/platform'
import { requestGatewayForProfile } from '@/store/gateway'
import {
  $approvalModes,
  type ApprovalMode,
  type ApprovalModeRequester,
  setApprovalModeForProfile,
  syncApprovalModeForProfile
} from '@/store/approval-mode'
import { notifyError } from '@/store/notifications'

export function useApprovalModeStatusbarItem(profile: string, requestGateway: ApprovalModeRequester): StatusbarItem {
  const { t } = useI18n()
  const copy = t.shell.approvalMode
  const modes = useStore($approvalModes)
  const mode = modes[profile.trim() || 'default']
  // The browser shares one backend; Electron must retain its exact device/session route.
  const requestApprovalMode = useCallback<ApprovalModeRequester>(
    (method, params) => isBrowserClient()
      ? requestGatewayForProfile(profile, method, params)
      : requestGateway(method, params),
    [profile, requestGateway]
  )

  const labels = useMemo<Record<ApprovalMode, string>>(
    () => ({ manual: copy.manual, smart: copy.smart, off: copy.off }),
    [copy.manual, copy.off, copy.smart]
  )

  const descriptions = useMemo<Record<ApprovalMode, string>>(
    () => ({
      manual: copy.manualDescription,
      smart: copy.smartDescription,
      off: copy.offDescription
    }),
    [copy.manualDescription, copy.offDescription, copy.smartDescription]
  )

  useEffect(() => {
    void syncApprovalModeForProfile(requestApprovalMode, profile).catch(error => notifyError(error, copy.loadFailed))
  }, [profile, requestApprovalMode, copy.loadFailed])

  return {
    className: mode === 'off' ? 'bg-(--chrome-action-hover) text-foreground' : undefined,
    icon: mode === 'off' ? <ZapFilled className="size-3.5" /> : <Zap className="size-3.5 opacity-70" />,
    id: 'approval-mode',
    label: mode ? labels[mode] : t.shell.statusbar.unknown,
    menuAlign: 'end',
    menuClassName: 'w-72 p-1',
    menuContent: (
      <>
        <DropdownMenuLabel>{copy.title}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {!mode && (
          <DropdownMenuItem
            onSelect={() =>
              void syncApprovalModeForProfile(requestApprovalMode, profile).catch(error =>
                notifyError(error, copy.loadFailed)
              )
            }
          >
            {t.common.retry}
          </DropdownMenuItem>
        )}
        <DropdownMenuRadioGroup
          onValueChange={value => {
            void setApprovalModeForProfile(requestApprovalMode, profile, value as ApprovalMode).catch(error =>
              notifyError(error, copy.saveFailed)
            )
          }}
          value={mode ?? ''}
        >
          {(['manual', 'smart', 'off'] as const).map(value => (
            <DropdownMenuRadioItem className="items-start gap-2" disabled={!mode} key={value} value={value}>
              <span className="flex min-w-0 flex-col gap-0.5">
                <span className="text-xs text-foreground">{labels[value]}</span>
                <span className="text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">{descriptions[value]}</span>
              </span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </>
    ),
    title: copy.ariaLabel(mode ? labels[mode] : t.shell.statusbar.unknown),
    variant: 'menu'
  }
}
