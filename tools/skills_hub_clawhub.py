"""Skills Hub ClawHub adapter (clawhub.ai HTTP API)."""

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from tools.skills_hub_models import (
    GuardedFetchMixin, SkillBundle, SkillMeta, SkillSource, _cache_metas, _cached_metas,
    _validate_bundle_rel_path,
)

logger = logging.getLogger("tools.skills_hub")


class ClawHubSource(GuardedFetchMixin, SkillSource):
    """ClawHub (clawhub.ai) HTTP API. Every skill is community trust — the
    ClawHavoc incident (341 malicious skills, Feb 2026) showed their vetting
    is insufficient."""

    BASE_URL = "https://clawhub.ai/api/v1"

    # Wall-clock budget for a full catalog walk: 50k+ skills, sequential
    # (~250 requests each under timeout=30), so unbounded it blocks for minutes.
    CATALOG_WALK_BUDGET_SECONDS = 12

    _SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*$")

    def source_id(self) -> str:
        return "clawhub"

    def trust_level_for(self, identifier: str) -> str:
        return "community"

    # -- payload helpers ---------------------------------------------------

    @staticmethod
    def _normalize_tags(tags: Any) -> List[str]:
        if isinstance(tags, list):
            return [str(t) for t in tags]
        if isinstance(tags, dict):
            return [str(k) for k in tags if str(k) != "latest"]
        return []

    @staticmethod
    def _coerce_skill_payload(data: Any) -> Optional[Dict[str, Any]]:
        """Flatten ``{"skill": {...}, "latestVersion", "owner"}`` listing shapes."""
        if not isinstance(data, dict):
            return None
        nested = data.get("skill")
        if isinstance(nested, dict):
            merged = dict(nested)
            latest_version = data.get("latestVersion")
            if latest_version is not None and "latestVersion" not in merged:
                merged["latestVersion"] = latest_version
            # owner is needed for building valid detail URLs.
            if "owner" in data and "owner" not in merged:
                merged["owner"] = data["owner"]
            return merged
        return data

    @staticmethod
    def _owner_from_payload(data: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        owner = data.get("owner")
        if isinstance(owner, dict):
            handle = owner.get("handle")
            if isinstance(handle, str) and handle.strip():
                return handle.strip()
        if isinstance(owner, str) and owner.strip():
            return owner.strip()
        return None

    @classmethod
    def _owner_matches(cls, expected_owner: Optional[str], data: Optional[Dict[str, Any]]) -> bool:
        if not expected_owner:
            return True
        actual = cls._owner_from_payload(data)
        return not actual or actual.lower() == expected_owner.lower()

    @classmethod
    def _item_to_meta(cls, item: Dict[str, Any]) -> Optional[SkillMeta]:
        """Listing item -> SkillMeta (None without a slug)."""
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug:
            return None
        owner = cls._owner_from_payload(item)
        return SkillMeta(
            name=item.get("displayName") or item.get("name") or slug,
            description=item.get("summary") or item.get("description") or "",
            source="clawhub",
            identifier=slug,
            trust_level="community",
            tags=cls._normalize_tags(item.get("tags", [])),
            extra={"owner": owner} if owner else {},
        )

    def _skill_detail(self, identifier: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """``(slug, payload)`` for an identifier, or None when unparsable,
        missing, or owned by someone other than the ``@owner`` requested."""
        parsed = self._parse_identifier(identifier)
        if parsed is None:
            return None
        slug, expected_owner = parsed
        data = self._coerce_skill_payload(self._get_json(f"{self.BASE_URL}/skills/{slug}"))
        if not isinstance(data, dict) or not self._owner_matches(expected_owner, data):
            return None
        return slug, data

    # -- search / ranking --------------------------------------------------

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        return [term for term in re.split(r"[^a-z0-9]+", query.lower()) if term]

    @classmethod
    def _search_score(cls, query: str, meta: SkillMeta) -> int:
        query_norm = query.strip().lower()
        if not query_norm:
            return 1

        identifier = (meta.identifier or "").lower()
        name = (meta.name or "").lower()
        description = (meta.description or "").lower()
        query_terms = cls._query_terms(query_norm)
        identifier_terms = cls._query_terms(identifier)
        name_terms = cls._query_terms(name)
        normalized_identifier = " ".join(identifier_terms)
        normalized_name = " ".join(name_terms)

        checks = (
            (140, query_norm == identifier),
            (130, query_norm == name),
            (125, normalized_identifier == query_norm),
            (120, normalized_name == query_norm),
            (95, normalized_identifier.startswith(query_norm)),
            (90, normalized_name.startswith(query_norm)),
            (70, bool(query_terms) and identifier_terms[: len(query_terms)] == query_terms),
            (65, bool(query_terms) and name_terms[: len(query_terms)] == query_terms),
            (40, query_norm in identifier),
            (35, query_norm in name),
            (10, query_norm in description),
        )
        score = sum(points for points, hit in checks if hit)
        for term in query_terms:
            score += 15 * (term in identifier_terms) + 12 * (term in name_terms) + 3 * (term in description)
        return score

    @staticmethod
    def _dedupe_results(results: List[SkillMeta]) -> List[SkillMeta]:
        seen: set[str] = set()
        deduped: List[SkillMeta] = []
        for result in results:
            key = (result.identifier or result.name).lower()
            if key not in seen:
                seen.add(key)
                deduped.append(result)
        return deduped

    def _exact_slug_meta(self, query: str) -> Optional[SkillMeta]:
        query = query.strip()
        parsed = self._parse_identifier(query)
        query_terms = self._query_terms(query)
        candidates: List[str] = []

        if parsed:
            candidates.append(parsed[0])
        elif "/" not in query and self._SLUG_RE.fullmatch(query):
            candidates.append(query)

        if query_terms:
            base_slug = "-".join(query_terms)
            if len(query_terms) >= 2:
                candidates.extend(
                    f"{base_slug}-{suffix}" for suffix in ("agent", "skill", "tool", "assistant", "playbook")
                )
            candidates.append(base_slug)

        for candidate in dict.fromkeys(candidates):
            meta = self.inspect(candidate)
            if meta:
                return meta
        return None

    def _finalize_search_results(self, query: str, results: List[SkillMeta], limit: int) -> List[SkillMeta]:
        query_norm = query.strip()
        if not query_norm:
            return self._dedupe_results(results)[:limit]

        filtered = [meta for meta in results if self._search_score(query_norm, meta) > 0]
        filtered.sort(
            key=lambda meta: (
                -self._search_score(query_norm, meta),
                meta.name.lower(),
                meta.identifier.lower(),
            )
        )
        filtered = self._dedupe_results(filtered)

        exact = self._exact_slug_meta(query_norm)
        if exact:
            filtered = [meta for meta in filtered if self._search_score(query_norm, meta) >= 20]
            filtered = self._dedupe_results([exact] + filtered)

        if filtered:
            return filtered[:limit]

        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", query_norm):
            return []

        return self._dedupe_results(results)[:limit]

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        query = query.strip()

        if query:
            if len(self._query_terms(query)) >= 2:
                direct = self._exact_slug_meta(query)
                if direct:
                    return [direct]

            results = self._search_catalog(query, limit=limit)
            if results:
                return results
        else:
            # Empty query: paginating catalog walker. A disk-cached full catalog
            # is returned whole (caller paginates); on a cold cache the walk is
            # bounded to `limit` so browse renders page one without walking
            # 50k+ skills (max_items=0 = unbounded, offline index builder only).
            catalog = self._load_catalog_index(max_items=limit if limit > 0 else 0)
            if catalog:
                deduped = self._dedupe_results(catalog)
                return deduped[:limit] if limit > 0 else deduped

        # Catalog miss / walker failure: best-effort lightweight listing API.
        cache_key = f"clawhub_search_listing_v1_{hashlib.md5(query.encode()).hexdigest()}_{limit}"
        cached = _cached_metas(cache_key)
        if cached is not None:
            return self._finalize_search_results(query, cached, limit)

        data = self._get_json(f"{self.BASE_URL}/skills", timeout=15,
                              params={"search": query, "limit": limit})
        if data is None:
            return []
        skills_data = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(skills_data, list):
            return []

        results = [m for m in (self._item_to_meta(item) for item in skills_data[:limit]) if m]
        final_results = self._finalize_search_results(query, results, limit)
        _cache_metas(cache_key, final_results)
        return final_results

    @classmethod
    def _parse_identifier(cls, identifier: str) -> Optional[Tuple[str, Optional[str]]]:
        """``(slug, expected_owner)`` for a bare slug, ``clawhub/<slug>``,
        ``@owner/slug``, or the URL path ``owner/skills/slug``.

        GitHub-style ``owner/repo/skill`` identifiers are NOT ClawHub's —
        claiming them by last segment would install a same-named skill from a
        different author.
        """
        raw = (identifier or "").strip()
        if not raw:
            return None
        had_at = raw.startswith("@")
        ident = raw[1:] if had_at else raw
        if ident.startswith("clawhub/"):
            ident = ident[len("clawhub/"):]
        parts = [part for part in ident.split("/") if part]
        owner = slug = None
        if len(parts) == 1:
            slug = parts[0]
        elif len(parts) == 2 and had_at:
            owner, slug = parts
        elif len(parts) == 3 and parts[1].lower() == "skills":
            owner, _, slug = parts
        else:
            return None
        if not cls._SLUG_RE.fullmatch(slug) or (owner is not None and not cls._SLUG_RE.fullmatch(owner)):
            return None
        return slug, owner

    # -- fetch / inspect ---------------------------------------------------

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        detail = self._skill_detail(identifier)
        if detail is None:
            return None
        slug, skill_data = detail

        latest_version = self._resolve_latest_version(slug, skill_data)
        if not latest_version:
            logger.warning("ClawHub fetch failed for %s: could not resolve latest version", slug)
            return None

        # Primary: ZIP bundle from /download. Fallback: version metadata with
        # inline/raw content (files may sit under version_data["version"]).
        files = self._download_zip(slug, latest_version)
        if "SKILL.md" not in files:
            version_data = self._get_json(f"{self.BASE_URL}/skills/{slug}/versions/{latest_version}")
            if isinstance(version_data, dict):
                files = self._extract_files(version_data) or files
                if "SKILL.md" not in files:
                    nested = version_data.get("version", {})
                    if isinstance(nested, dict):
                        files = self._extract_files(nested) or files

        if "SKILL.md" not in files:
            logger.warning(
                "ClawHub fetch for %s resolved version %s but could not retrieve file content",
                slug,
                latest_version,
            )
            return None

        return SkillBundle(
            name=slug,
            files=files,
            source="clawhub",
            identifier=slug,
            trust_level="community",
        )

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        detail = self._skill_detail(identifier)
        if detail is None:
            return None
        slug, data = detail
        return self._item_to_meta({**data, "slug": data.get("slug") or slug})

    def _search_catalog(self, query: str, limit: int = 10) -> List[SkillMeta]:
        cache_key = f"clawhub_search_catalog_v1_{hashlib.md5(f'{query}|{limit}'.encode()).hexdigest()}"
        cached = _cached_metas(cache_key)
        if cached is not None:
            return cached[:limit]

        catalog = self._load_catalog_index()
        if not catalog:
            return []

        results = self._finalize_search_results(query, catalog, limit)
        _cache_metas(cache_key, results)
        return results

    def _load_catalog_index(self, max_items: int = 0) -> List[SkillMeta]:
        """Walk the ClawHub catalog via cursor pagination.

        ``max_items`` stops the walk early once that many distinct skills are
        gathered (browse's cold-start fallback renders one page); ``0`` walks
        to exhaustion (offline index builder). Only a COMPLETE walk (cursor
        exhausted or page cap) is written to the shared ``clawhub_catalog_v1``
        cache — a walk cut by ``max_items`` or the wall-clock budget would
        poison it with a partial slice.
        """
        cache_key = "clawhub_catalog_v1"
        cached = _cached_metas(cache_key)
        if cached is not None:
            return cached

        cursor: Optional[str] = None
        results: List[SkillMeta] = []
        seen: set[str] = set()
        # 750 pages * 200/page = 150k ceiling over the ~50k catalog; a safety
        # rail against an infinite-cursor loop, normally ended by nextCursor=None.
        max_pages = 750
        # Wall-clock budget applies to interactive browse only: the index builder
        # (max_items=0) must walk everything or it trips the deploy health floor.
        deadline = (
            time.monotonic() + self.CATALOG_WALK_BUDGET_SECONDS
            if max_items > 0
            else None
        )
        partial = False

        for _ in range(max_pages):
            if deadline is not None and time.monotonic() > deadline:
                partial = True
                break
            params: Dict[str, Any] = {"limit": 200}
            if cursor:
                params["cursor"] = cursor

            data = self._get_json(f"{self.BASE_URL}/skills", timeout=30, params=params)
            items = data.get("items", []) if isinstance(data, dict) else []
            if not isinstance(items, list) or not items:
                break

            for item in items:
                slug = item.get("slug")
                if not isinstance(slug, str) or not slug or slug in seen:
                    continue
                seen.add(slug)
                meta = self._item_to_meta(item)
                if meta:
                    results.append(meta)

            cursor = data.get("nextCursor") if isinstance(data, dict) else None
            if not isinstance(cursor, str) or not cursor:
                break

            if max_items > 0 and len(results) >= max_items:
                partial = True
                break

        if not partial:
            _cache_metas(cache_key, results)
        return results

    def _get_json(self, url: str, timeout: int = 20, **kwargs) -> Optional[Any]:
        try:
            resp = httpx.get(url, timeout=timeout, **kwargs)
            if resp.status_code != 200:
                return None
            return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return None

    def _resolve_latest_version(self, slug: str, skill_data: Dict[str, Any]) -> Optional[str]:
        latest = skill_data.get("latestVersion")
        if isinstance(latest, dict):
            version = latest.get("version")
            if isinstance(version, str) and version:
                return version

        tags = skill_data.get("tags")
        if isinstance(tags, dict):
            latest_tag = tags.get("latest")
            if isinstance(latest_tag, str) and latest_tag:
                return latest_tag

        versions_data = self._get_json(f"{self.BASE_URL}/skills/{slug}/versions")
        if isinstance(versions_data, list) and versions_data:
            first = versions_data[0]
            if isinstance(first, dict):
                version = first.get("version")
                if isinstance(version, str) and version:
                    return version
        return None

    def _fetch_owner_handle(self, slug: str) -> Optional[str]:
        """Owner handle from the detail API (the listing API lacks it), or None.

        Bounded retry: 3 attempts total. 429 honours ``Retry-After`` else
        exponential backoff (2s -> 4s); 5xx and transport errors back off;
        other 4xx means the resource doesn't exist — no retry.
        """
        url = f"{self.BASE_URL}/skills/{slug}"
        max_attempts = 3
        backoff_base = 2.0  # seconds

        for attempt in range(max_attempts):
            delay = backoff_base * (2 ** attempt)
            try:
                resp = httpx.get(url, timeout=20)
            except (httpx.HTTPError, OSError):
                reason = "transport error"
            else:
                if resp.status_code == 200:
                    try:
                        raw = resp.json()
                    except (json.JSONDecodeError, ValueError):
                        return None
                    data = self._coerce_skill_payload(raw)
                    return self._owner_from_payload(data) if isinstance(data, dict) else None
                if resp.status_code == 429:
                    retry_after_raw = resp.headers.get("Retry-After")
                    try:
                        delay = float(retry_after_raw) if retry_after_raw else delay
                    except (TypeError, ValueError):
                        pass
                    reason = "HTTP 429"
                elif 500 <= resp.status_code < 600:
                    reason = f"HTTP {resp.status_code}"
                else:
                    return None  # 4xx (non-429): doesn't exist / bad request
            if attempt >= max_attempts - 1:
                return None
            logger.debug(
                "_fetch_owner_handle(%s): %s on attempt %d/%d, retrying in %.1fs",
                slug, reason, attempt + 1, max_attempts, delay,
            )
            time.sleep(delay)

        return None

    def enrich_owners(self, skills: List[SkillMeta], max_workers: int = 30) -> int:
        """Batch-fetch owner handles for ClawHub skills missing ``extra["owner"]``
        (in-place; returns the number enriched). For the offline index builder:
        the full 50k catalog takes ~5–10 min at 30 workers.

        Safety rails: aborts after 50 consecutive failures (systemic outage),
        per-request 429 backoff, progress log every 1000 skills.
        """
        needs_enrichment = [
            s for s in skills
            if s.source == "clawhub" and not (s.extra or {}).get("owner")
        ]
        if not needs_enrichment:
            return 0

        enriched = 0
        consecutive_failures = 0
        max_consecutive_failures = 50
        processed = 0
        import threading
        lock = threading.Lock()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._fetch_owner_handle, s.identifier): s for s in needs_enrichment}
            for future in as_completed(futures):
                meta = futures[future]
                processed += 1
                try:
                    handle = future.result()
                except Exception:
                    handle = None
                with lock:
                    if handle:
                        if not meta.extra:
                            meta.extra = {}
                        meta.extra["owner"] = handle
                        enriched += 1
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                if processed % 1000 == 0:
                    logger.info(
                        "ClawHub owner enrichment: %d/%d processed, %d enriched",
                        processed, len(needs_enrichment), enriched,
                    )

                with lock:
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(
                            "ClawHub owner enrichment: %d consecutive failures — "
                            "aborting early (%d/%d processed, %d enriched). "
                            "The ClawHub API may be down or rate-limited.",
                            max_consecutive_failures, processed,
                            len(needs_enrichment), enriched,
                        )
                        for f in futures:
                            f.cancel()
                        break

        return enriched

    def _extract_files(self, version_data: Dict[str, Any]) -> Dict[str, str]:
        files: Dict[str, str] = {}
        file_list = version_data.get("files")

        if isinstance(file_list, dict):
            return {k: v for k, v in file_list.items() if isinstance(v, str)}

        if not isinstance(file_list, list):
            return files

        for file_meta in file_list:
            if not isinstance(file_meta, dict):
                continue

            fname = file_meta.get("path") or file_meta.get("name")
            if not fname or not isinstance(fname, str):
                continue

            inline_content = file_meta.get("content")
            if isinstance(inline_content, str):
                files[fname] = inline_content
                continue

            raw_url = file_meta.get("rawUrl") or file_meta.get("downloadUrl") or file_meta.get("url")
            if isinstance(raw_url, str) and raw_url.startswith("http"):
                content = self._fetch_text(raw_url)
                if content is not None:
                    files[fname] = content

        return files

    def _download_zip(self, slug: str, version: str) -> Dict[str, str]:
        """Download the skill ZIP from /download and extract its text files."""
        import io
        import zipfile

        files: Dict[str, str] = {}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = httpx.get(
                    f"{self.BASE_URL}/download",
                    params={"slug": slug, "version": version},
                    timeout=30,
                    follow_redirects=True,
                )
                if resp.status_code == 429:
                    try:
                        retry_after = int(resp.headers.get("retry-after", "5"))
                    except (ValueError, TypeError):
                        retry_after = 5
                    retry_after = min(retry_after, 15)  # Cap wait time
                    logger.debug(
                        "ClawHub download rate-limited for %s, retrying in %ds (attempt %d/%d)",
                        slug, retry_after, attempt + 1, max_retries,
                    )
                    time.sleep(retry_after)
                    continue
                if resp.status_code != 200:
                    logger.debug("ClawHub ZIP download for %s v%s returned %s", slug, version, resp.status_code)
                    return files

                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        try:
                            name = _validate_bundle_rel_path(info.filename)
                        except ValueError:
                            logger.debug("Skipping unsafe ZIP member path: %s", info.filename)
                            continue
                        if info.file_size > 500_000:  # skip large binaries
                            logger.debug("Skipping large file in ZIP: %s (%d bytes)", name, info.file_size)
                            continue
                        try:
                            files[name] = zf.read(info.filename).decode("utf-8")
                        except (UnicodeDecodeError, KeyError):
                            logger.debug("Skipping non-text file in ZIP: %s", name)
                            continue

                return files

            except zipfile.BadZipFile:
                logger.warning("ClawHub returned invalid ZIP for %s v%s", slug, version)
                return files
            except httpx.HTTPError as exc:
                logger.debug("ClawHub ZIP download failed for %s v%s: %s", slug, version, exc)
                return files

        logger.debug("ClawHub ZIP download exhausted retries for %s v%s", slug, version)
        return files
