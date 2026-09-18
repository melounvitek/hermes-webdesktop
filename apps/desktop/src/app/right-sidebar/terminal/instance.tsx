import '@xterm/xterm/css/xterm.css'

import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { KbdCombo } from '@/components/ui/kbd'
import { Loader } from '@/components/ui/loader'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { closeTerminal, reportTerminalShell } from './terminals'
import { useAgentTerminal } from './use-agent-terminal'
import { useTerminalSession } from './use-terminal-session'

// Absolute-stacked so inactive tabs keep layout size (a display:none host goes
// 0×0 and renders garbled on re-show); visibility toggles which one is seen.
const INSTANCE_CLASS = 'absolute inset-0 flex flex-col bg-(--ui-terminal-surface-background) px-2 pb-2 pt-0'

// xterm host. The screen/viewport overrides matter for the DOM renderer (the
// WebGL fast-path paints the canvas from ITheme.background instead) — both
// resolve to the same token, so the two renderers can't disagree.
const HOST_CLASS =
  'h-full min-h-0 overflow-hidden text-(--ui-text-secondary) [&_.xterm]:h-full [&_.xterm-screen]:bg-(--ui-terminal-surface-background)! [&_.xterm-viewport]:bg-(--ui-terminal-surface-background)!'

interface TerminalInstanceProps {
  id: string
  cwd: string
  profile?: string
  active: boolean
  onAddSelectionToChat: (text: string, label?: string) => void
  restoreCwd?: string
  reviveBuffer?: string
}

/** One persistent xterm+PTY. Every open tab stays mounted (so its shell and
 *  scrollback survive tab switches); only the active one is shown. */
export function TerminalInstance({
  id,
  active,
  cwd,
  onAddSelectionToChat,
  restoreCwd,
  reviveBuffer,
  profile
}: TerminalInstanceProps) {
  const { t } = useI18n()

  const { addSelectionToChat, hostRef, selection, selectionStyle, status, connectionStatus, retry } =
    useTerminalSession({
      id,
      cwd,
      profile,
      active,
      onAddSelectionToChat,
      restoreCwd,
      reviveBuffer,
      onShell: shell => reportTerminalShell(id, shell)
    })

  return (
    <div
      className={cn(INSTANCE_CLASS, active ? 'visible' : 'invisible pointer-events-none')}
      // Focus-scope marker so isFocusWithin('[data-terminal]') can route ⌘W here.
      data-terminal=""
    >
      {status === 'starting' && (
        <div className="pointer-events-none absolute inset-0 z-10 grid place-items-center">
          <Loader
            className="size-8 text-(--ui-text-tertiary)"
            pathSteps={180}
            strokeScale={0.68}
            type="spiral-search"
          />
          {import.meta.env.VITE_BROWSER === '1' && connectionStatus.state === 'reconnecting' && (
            <span className="text-sm text-(--ui-text-secondary)" role="status">
              {t.rightSidebar.terminalReconnecting}
            </span>
          )}
        </div>
      )}
      {import.meta.env.VITE_BROWSER === '1' && connectionStatus.state === 'disconnected' && (
        <ScrollArea className="absolute inset-0 z-10 bg-(--ui-terminal-surface-background)" role="status">
          <div className="p-4">
            <ErrorState
              description={t.rightSidebar.terminalErrors[connectionStatus.reason ?? 'connection']}
              title={t.rightSidebar.terminalDisconnected}
            >
              {connectionStatus.reason === 'missing-session' ? (
                <Button onClick={() => closeTerminal(id)} size="sm" variant="secondary">
                  {t.common.close}
                </Button>
              ) : (
                <Button onClick={retry} size="sm" variant="secondary">
                  {t.common.retry}
                </Button>
              )}
            </ErrorState>
          </div>
        </ScrollArea>
      )}
      {selection.trim() && (
        <div className="absolute z-50 flex items-center gap-1" style={selectionStyle ?? { right: 12, top: 8 }}>
          <Button
            className="h-6 rounded-md px-2 text-[0.68rem] shadow-md backdrop-blur-md"
            onClick={event => event.preventDefault()}
            onMouseDown={event => {
              event.preventDefault()
              event.stopPropagation()
              addSelectionToChat()
            }}
            type="button"
            variant="secondary"
          >
            {t.rightSidebar.addToChat}
            <KbdCombo className="ml-1 opacity-70" combo="mod+l" size="sm" />
          </Button>
        </div>
      )}
      {/* Outer div paints the terminal inset; inner div is the xterm host so the
          canvas sizes to the content area and p-2 stays as terminal padding. */}
      <div className={HOST_CLASS} ref={hostRef} />
    </div>
  )
}

interface AgentTerminalInstanceProps {
  active: boolean
  id: string
  procId: string
  profile?: string
}

/** Read-only mirror of an agent background process — a write-only xterm streamed
 *  live from the backend output (no PTY, no input). */
export function AgentTerminalInstance({ active, id, procId, profile }: AgentTerminalInstanceProps) {
  const { hostRef } = useAgentTerminal({ active, id, procId, profile })

  return (
    <div
      className={cn(INSTANCE_CLASS, active ? 'visible' : 'invisible pointer-events-none')}
      // Same focus-scope marker as the user terminal so isFocusWithin('[data-terminal]')
      // routes ⌘W here and closes the focused agent tab (not a preview).
      data-terminal=""
    >
      <div className={HOST_CLASS} ref={hostRef} />
    </div>
  )
}
