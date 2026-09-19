import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { capabilityScoped } from '@/api/client'
import { getHermesConfigRecord, type ProfileScope, profileScopeKey } from '@/hermes'
import { queryClient, writeCache } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import type { HermesConfigRecord } from '@/types/hermes'

// Shared prefix for broad invalidation; every record belongs to a concrete owner.
export const HERMES_CONFIG_KEY = ['hermes-config-record'] as const

export function hermesConfigScope(profile?: ProfileScope) {
  const scope = capabilityScoped(profile ?? undefined)

  // An object pin keeps later refetches off the new ambient route, including
  // when the captured connection was untagged (not an explicit 'local' pin).
  return { connectionId: scope.connectionId ?? null, profile: scope.profile ?? 'default', priority: scope.priority }
}

export function useHermesConfigScope(profile?: ProfileScope) {
  const activeProfile = useStore($activeGatewayProfile)
  const connection = useStore($connection)

  // The API's ambient route mirrors these stores; keep both as dependencies
  // even though the resolver reads the route through capabilityScoped.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => hermesConfigScope(profile), [profile, activeProfile, connection])
}

export const hermesConfigKey = (profile?: ProfileScope) =>
  [...HERMES_CONFIG_KEY, profileScopeKey(hermesConfigScope(profile))] as const

// Paint only this owner's cache, then revalidate on mount. Both key and
// request capture the same owner, even if an inactive query is refetched later.
export function useHermesConfigRecord(profile?: ProfileScope) {
  const scope = useHermesConfigScope(profile)

  const query = useQuery({
    queryKey: hermesConfigKey(scope),
    queryFn: () => getHermesConfigRecord(scope),
    staleTime: 0
  })

  return { ...query, scope }
}

// Capture before awaiting a mutation so success and rollback stay with its owner.
export const hermesConfigCacheWriter = (profile?: ProfileScope) =>
  writeCache<HermesConfigRecord>(hermesConfigKey(profile))

export const invalidateHermesConfig = (profile?: ProfileScope) =>
  queryClient.invalidateQueries({ queryKey: profile == null ? HERMES_CONFIG_KEY : hermesConfigKey(profile) })
