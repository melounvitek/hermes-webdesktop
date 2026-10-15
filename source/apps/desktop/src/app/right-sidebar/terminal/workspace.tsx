import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/ui/empty-state'
import { useI18n } from '@/i18n'
import { $backgroundStatusBySession } from '@/store/composer-status'
import { knownOwnerForSession } from '@/store/session-states'

import { seedAgentTerminalCommand, syncAgentTerminalSnapshot } from './agent-terminal-stream'
import { setActiveTerminalId } from './buffer'
import { AgentTerminalInstance, TerminalInstance } from './instance'
import { $terminals, $visibleActiveTerminalId, createTerminal, ensureAgentTerminal } from './terminals'

interface TerminalWorkspaceProps {
  onAddSelectionToChat: (text: string, label?: string) => void
}

/** The persistent-overlay layer: the stack of live xterm instances (only these
 *  must stay in the fixed overlay, for the WebGL host). Mount/visibility is owned
 *  by PersistentTerminal (latched so shells survive hiding); the tab rail and
 *  new-terminal control live in the pane DOM — see TerminalPaneChrome. */
export function TerminalWorkspace({ onAddSelectionToChat }: TerminalWorkspaceProps) {
  const terminals = useStore($terminals)
  const activeId = useStore($visibleActiveTerminalId)
  const { t } = useI18n()
  const background = useStore($backgroundStatusBySession)

  // Mirror the tab selection into the agent reader (read_terminal reads it).
  useEffect(() => {
    const unsubscribe = $visibleActiveTerminalId.subscribe(setActiveTerminalId)

    return () => {
      unsubscribe()
      setActiveTerminalId(null)
    }
  }, [])

  // Surface the agent's background processes as read-only tabs (once each).
  // Live chunks stream via agent.terminal.output; the process-list snapshot also
  // seeds/falls back so the tab never stays blank if the stream races startup.
  useEffect(() => {
    for (const [sessionId, list] of Object.entries(background)) {
      const owner = knownOwnerForSession(sessionId)
      const profile = typeof owner === 'string' ? owner : owner?.profile

      for (const item of list) {
        ensureAgentTerminal(item.id, item.title, profile)
        seedAgentTerminalCommand(item.id, item.title, profile)
        syncAgentTerminalSnapshot(item.id, item.output ?? '', profile)
      }
    }
  }, [background])

  return (
    <>
      {import.meta.env.VITE_BROWSER === '1' && !activeId && (
        <div className="grid place-items-center p-4">
          <EmptyState title={t.rightSidebar.terminalEmpty} />
          <Button onClick={() => createTerminal()} size="sm" variant="secondary">
            {t.rightSidebar.terminalNew}
          </Button>
        </div>
      )}
      {terminals.map(term =>
        term.kind === 'agent' ? (
          <AgentTerminalInstance
            active={term.id === activeId}
            id={term.id}
            key={term.id}
            procId={term.procId!}
            profile={term.profile}
          />
        ) : (
          <TerminalInstance
            active={term.id === activeId}
            cwd={term.cwd}
            id={term.id}
            key={term.id}
            onAddSelectionToChat={onAddSelectionToChat}
            profile={term.profile}
            restoreCwd={term.restoreCwd}
            reviveBuffer={term.reviveBuffer}
          />
        )
      )}
    </>
  )
}
