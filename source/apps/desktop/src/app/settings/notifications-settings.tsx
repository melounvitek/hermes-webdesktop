import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import {
  $browserAttentionStatus,
  disableBrowserAttention,
  enableBrowserAttention,
  testBrowserAttention
} from '@/browser/attention-notifications'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { COMPLETION_SOUND_VARIANTS, previewCompletionSound } from '@/lib/completion-sound'
import { triggerHaptic } from '@/lib/haptics'
import { Bell, Play } from '@/lib/icons'
import { isBrowserClient } from '@/lib/platform'
import { cn } from '@/lib/utils'
import { $completionSoundVariantId, setCompletionSoundVariantId } from '@/store/completion-sound'
import {
  $nativeNotifyPrefs,
  NATIVE_NOTIFICATION_KINDS,
  sendTestNativeNotification,
  setNativeNotifyEnabled,
  setNativeNotifyKind
} from '@/store/native-notifications'
import { notify } from '@/store/notifications'

import { CONTROL_TEXT } from './constants'
import { ListRow, SectionHeading, SettingsContent, ToggleRow } from './primitives'

const CAPTION = 'text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)'

function Caption({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn(CAPTION, className)}>{children}</p>
}

function BrowserAttentionSettings() {
  const { t } = useI18n()
  const copy = t.settings.notifications.browser
  const status = useStore($browserAttentionStatus)
  const enabled = status === 'enabled' || status === 'requested'

  return (
    <div className="flex flex-col gap-2">
      <Caption>{copy.description}</Caption>
      <div className="flex flex-wrap gap-2">
        <Button
          disabled={enabled || status === 'enabling'}
          onClick={() => void enableBrowserAttention()}
          size="sm"
          variant="outline"
        >
          {copy.enable}
        </Button>
        {(enabled || status === 'enabling') && (
          <Button onClick={() => disableBrowserAttention()} size="sm" variant="outline">
            {copy.disable}
          </Button>
        )}
        <Button disabled={!enabled} onClick={testBrowserAttention} size="sm" variant="outline">
          {copy.test}
        </Button>
      </div>
      <p className={CAPTION} role="status">
        {copy.status[status]}
      </p>
      <Caption>{copy.limits}</Caption>
    </div>
  )
}

export function NotificationsSettings() {
  const { t } = useI18n()
  const prefs = useStore($nativeNotifyPrefs)
  const completionSoundVariantId = useStore($completionSoundVariantId)
  const copy = t.settings.notifications

  const runTest = async () => {
    triggerHaptic('open')
    const ok = await sendTestNativeNotification(copy.testTitle, copy.testBody)
    notify({ kind: ok ? 'info' : 'error', message: ok ? copy.testSent : copy.testUnsupported })
  }

  return (
    <SettingsContent>
      <SectionHeading icon={Bell} title={copy.title} />
      {isBrowserClient() ? (
        <BrowserAttentionSettings />
      ) : (
        <>
          <Caption className="mb-2 leading-(--conversation-caption-line-height)">{copy.intro}</Caption>

          <ToggleRow
            checked={prefs.enabled}
            description={copy.enableAllDesc}
            label={copy.enableAll}
            onChange={setNativeNotifyEnabled}
          />

          {NATIVE_NOTIFICATION_KINDS.map(kind => (
            <ToggleRow
              checked={prefs.enabled && prefs.kinds[kind]}
              description={copy.kinds[kind].description}
              disabled={!prefs.enabled}
              key={kind}
              label={copy.kinds[kind].label}
              onChange={on => setNativeNotifyKind(kind, on)}
            />
          ))}
        </>
      )}

      <ListRow
        action={
          <div className="flex flex-wrap items-center justify-end gap-2">
            <Select
              onValueChange={value => {
                const variantId = Number.parseInt(value, 10)

                setCompletionSoundVariantId(variantId)
                previewCompletionSound(variantId)
                triggerHaptic('selection')
              }}
              value={String(completionSoundVariantId)}
            >
              <SelectTrigger className={cn('min-w-56', CONTROL_TEXT)}>
                <SelectValue />
              </SelectTrigger>

              <SelectContent>
                {COMPLETION_SOUND_VARIANTS.map(variant => (
                  <SelectItem key={variant.id} value={String(variant.id)}>
                    {variant.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Button
              className="gap-1.5"
              onClick={() => {
                previewCompletionSound()
                triggerHaptic('crisp')
              }}
              size="sm"
              type="button"
              variant="outline"
            >
              <Play className="size-3.5" />
              {copy.completionSoundPreview}
            </Button>
          </div>
        }
        description={copy.completionSoundDesc}
        title={copy.completionSoundTitle}
      />

      {!isBrowserClient() && (
        <div className="mt-4 flex flex-col gap-2">
          <Button className="self-start" onClick={() => void runTest()} size="sm" type="button" variant="outline">
            <Bell />
            {copy.test}
          </Button>
          <Caption>{copy.focusedHint}</Caption>
        </div>
      )}
    </SettingsContent>
  )
}
