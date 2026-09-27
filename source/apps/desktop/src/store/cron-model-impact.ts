import { getApiRequestProfile } from '@/hermes'
import { translateNow } from '@/i18n'
import type { CronModelImpact, CronModelImpactJob } from '@/types/hermes'

import { requestCronReview } from './cron'
import { dismissNotification, notify } from './notifications'
import { $activeGatewayProfile } from './profile'
import { $connection } from './session'

const NOTIFICATION_ID = 'cron-model-impact'
const MAX_JOBS = 50
let assignmentGeneration = 0
let scopeGeneration = 0

const profileIdentity = () => getApiRequestProfile()?.trim() || 'default'

function connectionIdentity(): string {
  const connection = $connection.get()

  return JSON.stringify([
    connection?.connectionId,
    connection?.mode,
    connection?.remoteKind,
    connection?.remoteIdentity || connection?.remoteHost || (connection?.mode === 'remote' ? connection.baseUrl : '')
  ])
}

function invalidateScope(): void {
  assignmentGeneration += 1
  scopeGeneration += 1
  dismissNotification(NOTIFICATION_ID)
}

let connection = connectionIdentity()
$activeGatewayProfile.listen(invalidateScope)
$connection.listen(value => {
  // Reconnects and reminted websocket tickets do not change ownership.
  if (!value) {
    return
  }

  const next = connectionIdentity()

  if (connection !== next) {
    connection = next
    invalidateScope()
  }
})

export function beginCronModelImpactAssignment() {
  return { generation: ++assignmentGeneration, profile: profileIdentity(), scopeGeneration }
}

type Assignment = ReturnType<typeof beginCronModelImpactAssignment>

function validJob(value: unknown): value is CronModelImpactJob {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const job = value as Partial<CronModelImpactJob>

  const validText = (text: unknown, max: number): text is string =>
    typeof text === 'string' &&
    text.trim() === text &&
    text.length > 0 &&
    [...text].length <= max &&
    !/\p{C}/u.test(text)

  return (
    validText(job.id, 256) &&
    validText(job.name, 120) &&
    Array.isArray(job.drifted_axes) &&
    job.drifted_axes.length > 0 &&
    job.drifted_axes.length <= 2 &&
    new Set(job.drifted_axes).size === job.drifted_axes.length &&
    job.drifted_axes.every(axis => axis === 'model' || axis === 'provider')
  )
}

function parseImpact(value: unknown): CronModelImpact | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const impact = value as Partial<CronModelImpact>

  if (
    typeof impact.available !== 'boolean' ||
    !Number.isSafeInteger(impact.affected_count) ||
    (impact.affected_count ?? -1) < 0 ||
    typeof impact.truncated !== 'boolean' ||
    !Array.isArray(impact.jobs) ||
    impact.jobs.length > MAX_JOBS ||
    !impact.jobs.every(validJob)
  ) {
    return null
  }

  const count = impact.affected_count as number
  const ids = impact.jobs.map(job => job.id)

  if (
    new Set(ids).size !== ids.length ||
    (!impact.truncated && count !== impact.jobs.length) ||
    (impact.truncated && (impact.jobs.length !== MAX_JOBS || count <= impact.jobs.length))
  ) {
    return null
  }

  return impact as CronModelImpact
}

export function publishCronModelImpact(value: unknown, assignment: Assignment): void {
  const currentScope = () => assignment.profile === profileIdentity() && assignment.scopeGeneration === scopeGeneration

  if (!currentScope() || assignment.generation !== assignmentGeneration) {
    return
  }

  const impact = parseImpact(value)

  // Missing/malformed/unavailable is not evidence that an existing warning has gone away.
  if (!impact?.available) {
    return
  }

  if (impact.affected_count === 0) {
    dismissNotification(NOTIFICATION_ID)

    return
  }

  const visible = impact.jobs.slice(0, 3).map(job => job.name)
  const remaining = impact.affected_count - visible.length

  notify({
    id: NOTIFICATION_ID,
    kind: 'info',
    title: translateNow('cron.modelImpact.title'),
    message: translateNow('cron.modelImpact.message', impact.affected_count),
    detail:
      remaining > 0 ? translateNow('cron.modelImpact.detailMore', visible.join(', '), remaining) : visible.join(', '),
    action: {
      label: translateNow('cron.modelImpact.review'),
      onClick: () => {
        if (currentScope()) {
          requestCronReview()
        }
      }
    }
  })
}
