from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import OrderedDict, defaultdict

from pydantic import ValidationError
from redis.exceptions import RedisError

from app.config import get_settings
from app.core.time import utc_now_naive
from app.db.connection import get_redis
from app.skills.models import (
    ResolvedSkillRef,
    SessionSkillSelection,
    SkillContentRecord,
    SkillMatchResult,
    SkillMetaRecord,
    SkillRecord,
)
from app.skills.parser import tokenize_query
from app.skills.service import (
    get_skill_content,
    get_skill_content_batch,
    get_skill_meta_by_id,
    list_skill_meta,
)


class SkillRegistry:
    def __init__(self, owner_id: str | None = None) -> None:
        self._owner_id = owner_id
        self._lock = asyncio.Lock()
        self._loaded = False
        self._meta_by_id: dict[str, SkillMetaRecord] = {}
        self._slug_to_id: dict[str, str] = {}
        self._term_index: dict[str, set[str]] = defaultdict(set)
        self._fallback_ids: list[str] = []
        self._compiled_patterns: dict[tuple[str, int], list[re.Pattern[str] | str]] = {}
        self._content_l1: OrderedDict[tuple[str, int], SkillContentRecord] = OrderedDict()
        self._revision: int = 0
        self._version_token: str = "empty"

    async def load(self) -> None:
        if self._owner_id is None:
            # A registry without a principal is used only during process startup.
            # Never load another owner's data into a process-global cache.
            self._rebuild_indexes([])
            return
        async with self._lock:
            metas = await list_skill_meta(self._owner_id, include_deleted=False)
            self._rebuild_indexes(metas)
            self._loaded = True

    async def ensure_loaded(self) -> None:
        if not self._loaded:
            await self.load()

    async def upsert(self, skill_record: SkillRecord) -> None:
        async with self._lock:
            previous_meta = self._meta_by_id.get(skill_record.id)
            if previous_meta is not None:
                self._deindex_meta(previous_meta)
            self._index_meta(skill_record.meta)
            self._remember_content(skill_record.content)
            self._bump_revision()
            await self._cache_content_l2(skill_record.content)
            await self._cache_meta_l2(skill_record.meta)

    async def remove(self, skill_id: str) -> None:
        async with self._lock:
            meta = self._meta_by_id.pop(skill_id, None)
            if meta:
                self._deindex_meta(meta)
            self._drop_content_l1(skill_id)
            self._bump_revision()
            await self._invalidate_skill_l2(skill_id)
            await self._invalidate_meta_l2(skill_id)

    def list_runtime_skills(self) -> list[SkillMetaRecord]:
        metas = list(self._meta_by_id.values())
        metas.sort(key=lambda item: item.metadata.priority, reverse=True)
        return metas

    def get_meta_by_id(self, skill_id: str) -> SkillMetaRecord | None:
        return self._meta_by_id.get(skill_id)

    async def get_skill_record(self, skill_id: str) -> SkillRecord:
        self._require_owner()
        meta = self.get_meta_by_id(skill_id)
        if meta is None:
            meta = await self._get_cached_meta_l2(skill_id)
        if meta is None:
            meta = await get_skill_meta_by_id(skill_id, self._owner_id)
        if meta is not None and not meta.is_deleted and meta.metadata.enabled:
            async with self._lock:
                if skill_id not in self._meta_by_id:
                    self._index_meta(meta)
                    self._bump_revision()
                    await self._cache_meta_l2(meta)
        content = await self._get_content(meta.id, meta.version)
        return SkillRecord(meta=meta, content=content)

    async def resolve_for_session(
        self,
        *,
        query: str,
        manually_enabled_skill_ids: list[str] | None = None,
        manually_disabled_skill_ids: list[str] | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> SessionSkillSelection:
        self._require_owner()
        if not self._loaded and not self._meta_by_id:
            await self.load()
        enabled_overrides = list(dict.fromkeys(manually_enabled_skill_ids or []))
        disabled_overrides = set(manually_disabled_skill_ids or [])
        match_cache_key = self._match_cache_key(
            query=query,
            tenant_id=tenant_id,
            project_id=project_id,
            enabled_overrides=enabled_overrides,
            disabled_overrides=sorted(disabled_overrides),
        )
        cached = await self._get_match_cache(match_cache_key)
        if cached is not None:
            return cached

        visible_skills = self._visible_skills(tenant_id=tenant_id, project_id=project_id)
        visible_ids = {skill.id for skill in visible_skills}
        candidates = self._coarse_candidates(query, visible_skills)
        candidate_ids = [skill.id for skill in candidates]
        match_results = self._precision_match(query, candidates)
        auto_matched_ids = [result.skill_id for result in match_results]

        selected_by_id: dict[str, SkillMetaRecord] = {}
        for skill_id in enabled_overrides:
            meta = self._meta_by_id.get(skill_id)
            if meta and skill_id in visible_ids and skill_id not in disabled_overrides:
                selected_by_id[skill_id] = meta

        for result in match_results:
            if result.skill_id in disabled_overrides:
                continue
            meta = self._meta_by_id.get(result.skill_id)
            if meta:
                selected_by_id[result.skill_id] = meta

        ranked_selected = sorted(
            selected_by_id.values(),
            key=lambda item: (item.id not in enabled_overrides, -item.metadata.priority, item.metadata.slug),
        )
        top_k = get_settings().skills.match_top_k
        truncated_skill_count = max(0, len(ranked_selected) - top_k)
        effective_metas = ranked_selected[:top_k]
        effective_ids = [meta.id for meta in effective_metas]
        content_map = await self._get_content_for_metas(effective_metas)
        resolved_skills = [
            self._resolved_ref_from_meta(meta, content_map[(meta.id, meta.version)], match_results)
            for meta in effective_metas
            if (meta.id, meta.version) in content_map
        ]
        effective_prompt_sections = _merge_prompt_sections(resolved_skills)
        effective_agent_hints = _merge_agent_hints(resolved_skills)
        effective_tool_allowlist = _merge_tool_allowlist(effective_metas)

        selection = SessionSkillSelection(
            auto_matched_skill_ids=auto_matched_ids,
            manually_enabled_skill_ids=enabled_overrides,
            manually_disabled_skill_ids=sorted(disabled_overrides),
            effective_skill_ids=effective_ids,
            effective_tool_allowlist=effective_tool_allowlist,
            match_results=match_results,
            effective_prompt_sections=effective_prompt_sections,
            effective_agent_hints=effective_agent_hints,
            resolved_skills=resolved_skills,
            candidate_skill_ids=candidate_ids,
            scope_filters={
                "tenant_id": tenant_id,
                "project_id": project_id,
            },
            truncated_skill_count=truncated_skill_count,
            snapshot_created_at=utc_now_naive().isoformat(),
            match_cache_key=match_cache_key,
            registry_revision=self._revision,
        )
        await self._set_match_cache(match_cache_key, selection)
        return selection

    def _rebuild_indexes(self, metas: list[SkillMetaRecord]) -> None:
        active_metas = [
            meta for meta in metas
            if not meta.is_deleted and meta.metadata.enabled
        ]
        self._meta_by_id = {meta.id: meta for meta in active_metas}
        self._slug_to_id = {meta.metadata.slug: meta.id for meta in active_metas}
        self._term_index = defaultdict(set)
        self._fallback_ids = []

        for meta in active_metas:
            if meta.coarse_terms:
                for term in meta.coarse_terms:
                    self._term_index[term].add(meta.id)
            else:
                self._fallback_ids.append(meta.id)

        self._fallback_ids.sort(
            key=lambda skill_id: self._meta_by_id[skill_id].metadata.priority,
            reverse=True,
        )
        self._bump_revision()

    def _index_meta(self, meta: SkillMetaRecord) -> None:
        if meta.is_deleted or not meta.metadata.enabled:
            return
        self._meta_by_id[meta.id] = meta
        self._slug_to_id[meta.metadata.slug] = meta.id
        if meta.coarse_terms:
            for term in meta.coarse_terms:
                self._term_index[term].add(meta.id)
            if meta.id in self._fallback_ids:
                self._fallback_ids.remove(meta.id)
            return
        if meta.id not in self._fallback_ids:
            self._fallback_ids.append(meta.id)
            self._fallback_ids.sort(
                key=lambda skill_id: self._meta_by_id[skill_id].metadata.priority,
                reverse=True,
            )

    def _deindex_meta(self, meta: SkillMetaRecord) -> None:
        self._slug_to_id.pop(meta.metadata.slug, None)
        for term in meta.coarse_terms:
            bucket = self._term_index.get(term)
            if bucket is None:
                continue
            bucket.discard(meta.id)
            if not bucket:
                self._term_index.pop(term, None)
        if meta.id in self._fallback_ids:
            self._fallback_ids.remove(meta.id)
        self._compiled_patterns = {
            key: value
            for key, value in self._compiled_patterns.items()
            if key[0] != meta.id
        }

    def _bump_revision(self) -> None:
        self._revision += 1
        self._version_token = self._compute_version_token()

    def _compute_version_token(self) -> str:
        if not self._meta_by_id:
            return "empty"
        payload = "|".join(
            f"{meta.id}:{meta.version}"
            for meta in sorted(self._meta_by_id.values(), key=lambda item: item.id)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _visible_skills(self, *, tenant_id: str | None, project_id: str | None) -> list[SkillMetaRecord]:
        visible: list[SkillMetaRecord] = []
        normalized_tenant = tenant_id.strip().lower() if tenant_id else None
        normalized_project = project_id.strip().lower() if project_id else None

        for meta in self._meta_by_id.values():
            if self._is_visible_in_scope(
                meta,
                tenant_id=normalized_tenant,
                project_id=normalized_project,
            ):
                visible.append(meta)
        visible.sort(key=lambda item: item.metadata.priority, reverse=True)
        return visible

    def _is_visible_in_scope(
        self,
        meta: SkillMetaRecord,
        *,
        tenant_id: str | None,
        project_id: str | None,
    ) -> bool:
        if meta.is_deleted or not meta.metadata.enabled:
            return False
        if (
            meta.metadata.scope == "tenant"
            and meta.metadata.tenant_id
            and meta.metadata.tenant_id != tenant_id
        ):
            return False
        if meta.metadata.scope == "project":
            if meta.metadata.tenant_id and meta.metadata.tenant_id != tenant_id:
                return False
            if meta.metadata.project_id and meta.metadata.project_id != project_id:
                return False
        return True

    def _coarse_candidates(self, query: str, visible_skills: list[SkillMetaRecord]) -> list[SkillMetaRecord]:
        visible_ids = {skill.id for skill in visible_skills}
        query_terms = tokenize_query(query)
        candidate_ids: set[str] = set()
        for term in query_terms:
            candidate_ids.update(self._term_index.get(term, set()))
        candidate_ids &= visible_ids

        floor = get_settings().skills.coarse_candidate_floor
        fallback_limit = get_settings().skills.broad_fallback_limit
        if len(candidate_ids) < floor:
            for skill_id in self._fallback_ids[:fallback_limit]:
                if skill_id in visible_ids:
                    candidate_ids.add(skill_id)
            for meta in visible_skills:
                if len(candidate_ids) >= floor:
                    break
                candidate_ids.add(meta.id)

        candidates = [self._meta_by_id[skill_id] for skill_id in candidate_ids if skill_id in self._meta_by_id]
        candidates.sort(key=lambda item: item.metadata.priority, reverse=True)
        return candidates[:fallback_limit]

    def _precision_match(self, query: str, candidates: list[SkillMetaRecord]) -> list[SkillMatchResult]:
        lower_query = query.lower()
        query_terms = set(tokenize_query(query))
        results: list[SkillMatchResult] = []

        for meta in candidates:
            regex_reason = self._match_reason(lower_query, meta)
            term_hits = len(query_terms.intersection(set(meta.coarse_terms)))
            keyword_reason = f"term:{term_hits}" if term_hits > 0 else None
            if not regex_reason and not keyword_reason:
                continue

            score = float(meta.metadata.priority)
            if regex_reason:
                score += 100.0
            score += term_hits * 10.0
            results.append(SkillMatchResult(
                skill_id=meta.id,
                slug=meta.metadata.slug,
                name=meta.metadata.name,
                reason=regex_reason or keyword_reason or "priority_fallback",
                priority=meta.metadata.priority,
                allowed_tools=meta.metadata.allowed_tools,
                score=score,
            ))

        results.sort(key=lambda item: (item.score, item.priority), reverse=True)
        return results

    def _match_reason(self, query: str, meta: SkillMetaRecord) -> str | None:
        compiled = self._compiled_pattern_list(meta)
        for matcher in compiled:
            if isinstance(matcher, re.Pattern):
                if matcher.search(query):
                    return f"regex:{matcher.pattern}"
            else:
                if matcher in query:
                    return f"keyword:{matcher}"
        return None

    def _compiled_pattern_list(self, meta: SkillMetaRecord) -> list[re.Pattern[str] | str]:
        cache_key = (meta.id, meta.version)
        cached = self._compiled_patterns.get(cache_key)
        if cached is not None:
            return cached

        compiled: list[re.Pattern[str] | str] = []
        for pattern in meta.metadata.trigger_patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                compiled.append(pattern.lower())
        self._compiled_patterns[cache_key] = compiled
        return compiled

    def _resolved_ref_from_meta(
        self,
        meta: SkillMetaRecord,
        content: SkillContentRecord,
        match_results: list[SkillMatchResult],
    ) -> ResolvedSkillRef:
        result_lookup = {result.skill_id: result for result in match_results}
        match_reason = result_lookup.get(meta.id).reason if meta.id in result_lookup else "manual_enable"
        return ResolvedSkillRef(
            skill_id=meta.id,
            slug=meta.metadata.slug,
            name=meta.metadata.name,
            version=meta.version,
            priority=meta.metadata.priority,
            matched_reason=match_reason,
            allowed_tools=meta.metadata.allowed_tools,
            prompt_sections={
                "overview": content.overview,
                "prompt": content.prompt,
                "constraints": content.constraints,
            },
            agent_hints=_agent_hints_from_meta(meta),
        )

    async def _get_content_for_metas(self, metas: list[SkillMetaRecord]) -> dict[tuple[str, int], SkillContentRecord]:
        refs = [(meta.id, meta.version) for meta in metas]
        results: dict[tuple[str, int], SkillContentRecord] = {}
        missing_refs: list[tuple[str, int]] = []

        for ref in refs:
            cached = self._content_l1.get(ref)
            if cached is not None:
                self._content_l1.move_to_end(ref)
                results[ref] = cached
                continue
            l2 = await self._get_cached_content_l2(*ref)
            if l2 is not None:
                self._remember_content(l2)
                results[ref] = l2
                continue
            missing_refs.append(ref)

        if missing_refs:
            loaded = await get_skill_content_batch(missing_refs, self._owner_id)
            for ref, content in loaded.items():
                self._remember_content(content)
                results[ref] = content
                await self._cache_content_l2(content)

        return results

    async def _get_content(self, skill_id: str, version: int) -> SkillContentRecord:
        ref = (skill_id, version)
        cached = self._content_l1.get(ref)
        if cached is not None:
            self._content_l1.move_to_end(ref)
            return cached
        l2 = await self._get_cached_content_l2(skill_id, version)
        if l2 is not None:
            self._remember_content(l2)
            return l2
        self._require_owner()
        content = await get_skill_content(skill_id, version, self._owner_id)
        self._remember_content(content)
        await self._cache_content_l2(content)
        return content

    def _remember_content(self, content: SkillContentRecord) -> None:
        ref = (content.skill_id, content.version)
        self._content_l1[ref] = content
        self._content_l1.move_to_end(ref)
        max_size = get_settings().skills.content_l1_cache_size
        while len(self._content_l1) > max_size:
            self._content_l1.popitem(last=False)

    def _drop_content_l1(self, skill_id: str) -> None:
        self._content_l1 = OrderedDict(
            (key, value) for key, value in self._content_l1.items()
            if key[0] != skill_id
        )

    def _match_cache_key(
        self,
        *,
        query: str,
        tenant_id: str | None,
        project_id: str | None,
        enabled_overrides: list[str],
        disabled_overrides: list[str],
    ) -> str:
        payload = json.dumps(
            {
                "query": query.strip().lower(),
                "tenant_id": tenant_id,
                "project_id": project_id,
                "enabled_overrides": enabled_overrides,
                "disabled_overrides": disabled_overrides,
                "version_token": self._version_token,
                "revision": self._revision,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"skill_match:{digest}"

    async def _get_match_cache(self, cache_key: str) -> SessionSkillSelection | None:
        redis_client = await self._get_redis_safe()
        if redis_client is None:
            return None
        try:
            value = await redis_client.get(cache_key)
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return None
        if not value:
            return None
        try:
            return SessionSkillSelection.model_validate(json.loads(value))
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return None

    async def _set_match_cache(self, cache_key: str, selection: SessionSkillSelection) -> None:
        redis_client = await self._get_redis_safe()
        if redis_client is None:
            return
        try:
            await redis_client.set(
                cache_key,
                selection.model_dump_json(),
                ex=get_settings().skills.match_cache_ttl,
            )
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return

    async def _cache_content_l2(self, content: SkillContentRecord) -> None:
        redis_client = await self._get_redis_safe()
        if redis_client is None:
            return
        try:
            await redis_client.set(
                self._content_cache_key(content.skill_id, content.version),
                content.model_dump_json(),
                ex=get_settings().skills.content_cache_ttl,
            )
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return

    async def _cache_meta_l2(self, meta: SkillMetaRecord) -> None:
        redis_client = await self._get_redis_safe()
        if redis_client is None:
            return
        versioned_key = self._meta_cache_key(meta.id, meta.version)
        pointer_key = self._meta_pointer_key(meta.id)
        ttl = get_settings().skills.content_cache_ttl
        try:
            await redis_client.set(versioned_key, meta.model_dump_json(), ex=ttl)
            await redis_client.set(pointer_key, str(meta.version), ex=ttl)
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return

    async def _get_cached_meta_l2(self, skill_id: str) -> SkillMetaRecord | None:
        redis_client = await self._get_redis_safe()
        if redis_client is None:
            return None
        try:
            version = await redis_client.get(self._meta_pointer_key(skill_id))
            if not version:
                return None
            value = await redis_client.get(self._meta_cache_key(skill_id, int(version)))
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return None
        if not value:
            return None
        try:
            return SkillMetaRecord.model_validate(json.loads(value))
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return None

    async def _get_cached_content_l2(self, skill_id: str, version: int) -> SkillContentRecord | None:
        redis_client = await self._get_redis_safe()
        if redis_client is None:
            return None
        try:
            value = await redis_client.get(self._content_cache_key(skill_id, version))
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return None
        if not value:
            return None
        try:
            return SkillContentRecord.model_validate(json.loads(value))
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return None

    async def _invalidate_skill_l2(self, skill_id: str) -> None:
        redis_client = await self._get_redis_safe()
        if redis_client is None:
            return
        try:
            keys = await redis_client.keys(f"skill_content:{self._owner_id}:{skill_id}:*")
            if keys:
                await redis_client.delete(*keys)
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return

    async def _invalidate_meta_l2(self, skill_id: str) -> None:
        redis_client = await self._get_redis_safe()
        if redis_client is None:
            return
        try:
            keys = await redis_client.keys(f"skill_meta:{self._owner_id}:{skill_id}:*")
            if keys:
                await redis_client.delete(*keys)
            await redis_client.delete(self._meta_pointer_key(skill_id))
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return

    async def _get_redis_safe(self):
        try:
            return await get_redis()
        except (OSError, RedisError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            return None

    def _content_cache_key(self, skill_id: str, version: int) -> str:
        return f"skill_content:{self._owner_id}:{skill_id}:{version}"

    def _meta_cache_key(self, skill_id: str, version: int) -> str:
        return f"skill_meta:{self._owner_id}:{skill_id}:{version}"

    def _meta_pointer_key(self, skill_id: str) -> str:
        return f"skill_meta_current:{self._owner_id}:{skill_id}"

    def _require_owner(self) -> None:
        if self._owner_id is None:
            raise RuntimeError("skill registry requires an owner")


def _merge_tool_allowlist(metas: list[SkillMetaRecord]) -> list[str]:
    merged: list[str] = []
    for meta in metas:
        for tool_name in meta.metadata.allowed_tools:
            if tool_name not in merged:
                merged.append(tool_name)
    return merged


def _merge_prompt_sections(resolved_skills: list[ResolvedSkillRef]) -> dict[str, str]:
    sections = {"overview": [], "prompt": [], "constraints": []}
    for item in resolved_skills:
        for key, values in sections.items():
            value = item.prompt_sections.get(key, "")
            if value and value.strip():
                values.append(value.strip())
    return {key: "\n\n".join(values).strip() for key, values in sections.items()}


def _merge_agent_hints(resolved_skills: list[ResolvedSkillRef]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for item in resolved_skills:
        for key, values in item.agent_hints.items():
            bucket = merged.setdefault(key, [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)
    return merged


def _agent_hints_from_meta(meta: SkillMetaRecord) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for key, value in meta.metadata.agent_hints.model_dump().items():
        if isinstance(value, str) and value.strip():
            result[key] = [value.strip()]
    return result


_startup_skill_registry = SkillRegistry()
_skill_registries: dict[str, SkillRegistry] = {}


def get_skill_registry(owner_id: str | None = None) -> SkillRegistry:
    """Return an owner-scoped registry; startup gets an intentionally empty registry."""
    if owner_id is None:
        return _startup_skill_registry
    registry = _skill_registries.get(owner_id)
    if registry is None:
        registry = SkillRegistry(owner_id)
        _skill_registries[owner_id] = registry
    return registry
