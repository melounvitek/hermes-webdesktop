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
import { $activeConnectionId } from '@/store/connections'
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

// Slot for one gateway inside the existing config-record cache. The id is
// `$activeConnectionId` — the resolved descriptor identity the rest of the app
// already uses — not a second cache. A bare root key let a settings save after
// a gateway switch paint the previous machine's record and PUT it onto the
// other config.yaml. An owner that already names a connection keeps
// profileScopeKey's suffix; an untagged owner is namespaced the same way so
// two gateways' `default` profiles do not share a row.
export const hermesConfigKey = (
  profile?: ProfileScope,
  connectionId: null | string | undefined = $activeConnectionId.get()
) => {
  const scope = hermesConfigScope(profile)
  const active = (connectionId ?? '').trim()

  return [
    ...HERMES_CONFIG_KEY,
    profileScopeKey(scope.connectionId || !active ? scope : { ...scope, connectionId: active })
  ] as const
}

// staleTime 0 → serve cache instantly, background-revalidate on every mount.
// Both key and request capture the owner, even for later inactive refetches;
// the cache slot is additionally the active gateway's.
export const useHermesConfigRecord = (profile?: ProfileScope) => {
  const scope = useHermesConfigScope(profile)
  // Reactive read, not a store getter: under the React Compiler a value with
  // no reactive inputs is computed once per component instance, so a
  // getter-based key would freeze on the first gateway and keep serving its
  // record after a switch.
  const connectionId = useStore($activeConnectionId)

  const query = useQuery({
    queryKey: hermesConfigKey(scope, connectionId),
    queryFn: () => {
      // $activeConnectionId.listen invalidates profile queries in the same
      // turn it publishes the new id, before this observer moves to the new
      // key. A refetch of the slot we are leaving must not store the new
      // gateway's record there — that is the other machine's config.yaml.
      if (connectionId && $activeConnectionId.get() !== connectionId) {
        const cached = queryClient.getQueryData<HermesConfigRecord>(hermesConfigKey(scope, connectionId))

        if (cached !== undefined) {
          return cached
        }
      }

      return getHermesConfigRecord(scope)
    },
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

const writeHermesConfigCache =
  (keyFor: () => ReturnType<typeof hermesConfigKey>) =>
  (
    next:
      HermesConfigRecord | undefined | ((previous: HermesConfigRecord | undefined) => HermesConfigRecord | undefined)
  ) =>
    void queryClient.setQueryData<HermesConfigRecord>(keyFor(), previous => {
      const record = typeof next === 'function' ? next(previous) : next

      // setQueryData also runs the hook's structuralSharing (query.setData →
      // replaceData), but that pass stamps the origin of `record` (the NEW
      // value), which optimistic patches do not carry — and it only applies once
      // the observer has built the query. So the previous record's origin is
      // carried over explicitly here.
      return record ? retainConfigReadOrigin(record, previous) : record
    })

// Capture the owner before awaiting a mutation so success and rollback stay
// with it. Only an untagged owner's gateway slot is resolved at WRITE time, so
// a writer memoized by a long-lived settings panel lands on whichever gateway
// is active when the save happens — the same row its query reads.
export const hermesConfigCacheWriter = (profile?: ProfileScope) => {
  const scope = hermesConfigScope(profile)

  return writeHermesConfigCache(() => hermesConfigKey(scope))
}

export const invalidateHermesConfig = (profile?: ProfileScope) =>
  queryClient.invalidateQueries({ queryKey: profile == null ? HERMES_CONFIG_KEY : hermesConfigKey(profile) })
