import { useStore } from '@nanostores/react'
import { replaceEqualDeep, useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { capabilityScoped } from '@/api/client'
import {
  getHermesConfigRecord,
  peekConfigReadOrigin,
  type ProfileScope,
  profileScopeKey,
  retainConfigReadOrigin
} from '@/hermes'
import { queryClient } from '@/lib/query-client'
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

// Both key and request capture the owner, even for later inactive refetches.
export const useHermesConfigRecord = (profile?: ProfileScope) => {
  const scope = useHermesConfigScope(profile)
  const query = useQuery({
    queryKey: hermesConfigKey(scope),
    queryFn: () => getHermesConfigRecord(scope),
    staleTime: 0,
    // Keep structural sharing so an unchanged refetch (every consumer mount at
    // staleTime 0, every invalidate) yields the SAME object and consumers'
    // memos/autosave effects don't re-arm. The read origin lives in a WeakMap
    // keyed by the record, so re-stamp whatever object survives the merge with
    // the origin of the NEW fetch (`next`, bound by getHermesConfigRecord) —
    // otherwise a retained object would keep routing writes to the gateway
    // that served the previous GET.
    structuralSharing: (previous: unknown, next: unknown) =>
      retainConfigReadOrigin(
        replaceEqualDeep(previous as HermesConfigRecord | undefined, next as HermesConfigRecord),
        next as object
      )
  })

  // Attach `writeScope` as a lazy getter instead of spreading `query`: useQuery
  // hands back a tracked-props Proxy, and spreading enumerates EVERY key, which
  // subscribes each consumer to fetchStatus/dataUpdatedAt/… churn. The getter
  // reads `query.data` through the proxy, so only `data` is tracked.
  //
  // `undefined`, never `null`: callers hand this straight to saveHermesConfig
  // with sparse `setNested({}, …)` patches, so the WeakMap misses and the
  // fallback is capabilityScoped(writeScope) → profileScoped(writeScope).
  // profileScoped(undefined) keeps the app-wide `_apiProfile`; profileScoped
  // (null) drops it and would write the PRIMARY profile before the first
  // GET resolves.
  Object.defineProperty(query, 'writeScope', {
    get: () => peekConfigReadOrigin(query.data) ?? undefined,
    configurable: true,
    enumerable: false
  })

  Object.defineProperty(query, 'scope', { value: scope, configurable: true, enumerable: false })

  return query as typeof query & { scope: typeof scope; writeScope: ReturnType<typeof peekConfigReadOrigin> }
}

// Capture before awaiting a mutation so success and rollback stay with its owner.
const writeHermesConfigCache =
  (key: ReturnType<typeof hermesConfigKey>) =>
  (
    next:
      HermesConfigRecord | undefined | ((previous: HermesConfigRecord | undefined) => HermesConfigRecord | undefined)
  ) =>
    void queryClient.setQueryData<HermesConfigRecord>(key, previous => {
      const record = typeof next === 'function' ? next(previous) : next

      // setQueryData also runs the hook's structuralSharing (query.setData →
      // replaceData), but that pass stamps the origin of `record` (the NEW
      // value), which optimistic patches do not carry — and it only applies once
      // the observer has built the query. So the previous record's origin is
      // carried over explicitly here.
      return record ? retainConfigReadOrigin(record, previous) : record
    })

export const hermesConfigCacheWriter = (profile?: ProfileScope) => writeHermesConfigCache(hermesConfigKey(profile))

export const invalidateHermesConfig = (profile?: ProfileScope) =>
  queryClient.invalidateQueries({ queryKey: profile == null ? HERMES_CONFIG_KEY : hermesConfigKey(profile) })
